use std::{
    io::Cursor,
    net::{IpAddr, Ipv4Addr, Ipv6Addr},
    path::PathBuf,
    time::{Duration, Instant},
};

use anyhow::Context;
use askama::Template;
use axum::{
    Json, Router,
    body::Bytes,
    extract::{DefaultBodyLimit, Multipart, Path as AxumPath, Request, State},
    http::{HeaderName, HeaderValue, Method, StatusCode, header},
    middleware::{self, Next},
    response::{Html, IntoResponse, Response},
    routing::{get, post},
};
use image::ImageReader;
use log::{debug, info, trace, warn};
use serde::Deserialize;
use serde_json::{Value, json};
use time::OffsetDateTime;
use tokio::fs;
use tower_http::{
    cors::{AllowOrigin, CorsLayer},
    timeout::TimeoutLayer,
};
use uuid::Uuid;

use crate::{
    config::Config,
    jobs::{EnqueueError, enqueue_record},
    state::AppState,
    templates::IndexTemplate,
    types::{
        CASCADED_TASK_PROMPTS, ErrorResponse, HealthResponse, JobRecord, JobStatus,
        MetricsResponse, QueueResponse, ReadinessResponse, SINGLE_TASK_PROMPTS, TaskSpec,
        WorkerHealth,
    },
    util::{guess_extension, image_format_content_type, sha256_hex},
};

pub fn router(state: AppState) -> Router {
    let cors_allowed_origins = state.config.cors_allowed_origins.clone();
    let request_timeout_seconds = state.config.request_timeout_seconds;
    let mut router = Router::new()
        .route("/", get(index))
        .route("/health", get(health))
        .route("/ready", get(readiness))
        .route("/metrics", get(metrics))
        .route("/openapi.json", get(openapi))
        .route("/v1/infer", post(submit_inference))
        .route("/infer-form", post(submit_inference_form))
        .route("/v1/infer/path", post(submit_inference_path))
        .route("/v1/jobs/{id}", get(get_job))
        .layer(DefaultBodyLimit::max(state.config.body_limit_bytes))
        .layer(middleware::from_fn(log_request))
        .with_state(state);

    if request_timeout_seconds > 0 {
        router = router.layer(TimeoutLayer::with_status_code(
            StatusCode::REQUEST_TIMEOUT,
            Duration::from_secs(request_timeout_seconds),
        ));
    }

    let router = if cors_allowed_origins.is_empty() {
        router
    } else {
        router.layer(cors_layer(&cors_allowed_origins))
    };

    router.layer(middleware::from_fn(add_security_headers))
}

#[derive(Debug, thiserror::Error)]
pub enum ApiError {
    #[error("{0}")]
    BadRequest(String),
    #[error("job not found")]
    NotFound,
    #[error("{0}")]
    Forbidden(String),
    #[error("{0}")]
    ServiceUnavailable(String),
    #[error(transparent)]
    Internal(#[from] anyhow::Error),
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = match &self {
            ApiError::BadRequest(_) => StatusCode::BAD_REQUEST,
            ApiError::NotFound => StatusCode::NOT_FOUND,
            ApiError::Forbidden(_) => StatusCode::FORBIDDEN,
            ApiError::ServiceUnavailable(_) => StatusCode::SERVICE_UNAVAILABLE,
            ApiError::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
        };
        let body = Json(ErrorResponse {
            code: self.code().to_string(),
            message: self.to_string(),
        });
        (status, body).into_response()
    }
}

impl ApiError {
    fn code(&self) -> &'static str {
        match self {
            ApiError::BadRequest(_) => "bad_request",
            ApiError::NotFound => "not_found",
            ApiError::Forbidden(_) => "forbidden",
            ApiError::ServiceUnavailable(_) => "service_unavailable",
            ApiError::Internal(_) => "internal_error",
        }
    }
}

#[derive(Debug, Deserialize)]
struct LocalPathRequest {
    image_path: PathBuf,
    task_type: Option<String>,
    task_prompt: Option<String>,
    text_input: Option<String>,
    webhook_url: Option<String>,
    task: Option<String>,
}

struct UploadedImage {
    filename: Option<String>,
    content_type: Option<String>,
    bytes: Bytes,
}

async fn index() -> Result<Html<String>, ApiError> {
    render_index(None, None)
}

async fn health(State(state): State<AppState>) -> Json<HealthResponse> {
    let workers = worker_health(&state);
    debug!(
        "health check worker_ready={} workers={} ready_workers={} failed_workers={} queued={} model_path={} model_variant={}",
        workers.ready > 0,
        workers.expected,
        workers.ready,
        workers.failed,
        state.queue_tx.len(),
        state.config.model_path.display(),
        state.config.model_variant.as_str()
    );

    Json(HealthResponse {
        status: "ok",
        ready: workers.ready > 0,
        workers,
        queued: state.queue_tx.len(),
        model_path: state.config.model_path.clone(),
        model_variant: state.config.model_variant,
    })
}

async fn readiness(State(state): State<AppState>) -> Response {
    let workers = worker_health(&state);
    let ready = workers.ready > 0;
    let response = ReadinessResponse {
        status: if ready { "ready" } else { "not_ready" },
        ready,
        workers,
        queued: state.queue_tx.len(),
    };
    let status = if ready {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };

    (status, Json(response)).into_response()
}

async fn metrics(State(state): State<AppState>) -> Json<MetricsResponse> {
    let workers = worker_health(&state);
    let metrics = state.metrics.snapshot();
    let completed = metrics.jobs_succeeded + metrics.jobs_failed;
    let average_inference_ms =
        (completed > 0).then_some(metrics.total_inference_ms as f64 / completed as f64);

    Json(MetricsResponse {
        workers,
        queued: state.queue_tx.len(),
        jobs_started: metrics.jobs_started,
        jobs_succeeded: metrics.jobs_succeeded,
        jobs_failed: metrics.jobs_failed,
        jobs_timed_out: metrics.jobs_timed_out,
        worker_restarts: metrics.worker_restarts,
        total_inference_ms: metrics.total_inference_ms as u128,
        average_inference_ms,
        model_load_ms: metrics.model_load_ms.map(|value| value as u128),
        retained_jobs: state.jobs.read().await.len(),
    })
}

async fn openapi() -> Json<Value> {
    Json(openapi_document())
}

async fn submit_inference(
    State(state): State<AppState>,
    multipart: Multipart,
) -> Result<Json<QueueResponse>, ApiError> {
    Ok(Json(submit_multipart_job(&state, multipart).await?))
}

async fn submit_inference_form(
    State(state): State<AppState>,
    multipart: Multipart,
) -> Result<Html<String>, ApiError> {
    match submit_multipart_job(&state, multipart).await {
        Ok(response) => render_index(Some(response), None),
        Err(err) => render_index(None, Some(err.to_string())),
    }
}

async fn submit_multipart_job(
    state: &AppState,
    mut multipart: Multipart,
) -> Result<QueueResponse, ApiError> {
    debug!("multipart inference submission started");
    ensure_workers_ready(state)?;

    let mut image: Option<UploadedImage> = None;
    let mut task_type: Option<String> = None;
    let mut task_prompt: Option<String> = None;
    let mut text_input: Option<String> = None;
    let mut webhook_url: Option<String> = None;

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|err| ApiError::BadRequest(format!("failed to read multipart field: {err}")))?
    {
        let name = field.name().unwrap_or_default().to_string();
        trace!("multipart field received name={}", name);
        match name.as_str() {
            "image" | "file" => {
                let filename = field.file_name().map(str::to_string);
                let content_type = field.content_type().map(str::to_string);
                let bytes = field.bytes().await.map_err(|err| {
                    ApiError::BadRequest(format!("failed to read image field: {err}"))
                })?;
                debug!(
                    "image field read filename={:?} content_type={:?} bytes={}",
                    filename,
                    content_type,
                    bytes.len()
                );
                image = Some(UploadedImage {
                    filename,
                    content_type,
                    bytes,
                });
            }
            "task_type" | "task_type_selector" => {
                task_type = Some(field.text().await.map_err(|err| {
                    ApiError::BadRequest(format!("failed to read task_type field: {err}"))
                })?);
            }
            "task_prompt" | "task" => {
                task_prompt = Some(field.text().await.map_err(|err| {
                    ApiError::BadRequest(format!("failed to read task field: {err}"))
                })?);
            }
            "text_input" => {
                text_input = Some(field.text().await.map_err(|err| {
                    ApiError::BadRequest(format!("failed to read text_input field: {err}"))
                })?);
            }
            "webhook_url" => {
                webhook_url = Some(field.text().await.map_err(|err| {
                    ApiError::BadRequest(format!("failed to read webhook_url field: {err}"))
                })?);
            }
            _ => {}
        }
    }
    let task = task_spec_from_request(task_type, task_prompt, text_input, None)?;
    let webhook_url = validate_webhook_url(&state.config, webhook_url)?;

    let image =
        image.ok_or_else(|| ApiError::BadRequest("multipart field `image` is required".into()))?;
    if image.bytes.is_empty() {
        warn!("rejecting empty image upload");
        return Err(ApiError::BadRequest("uploaded image is empty".into()));
    }
    validate_image_bytes(state, image.content_type.as_deref(), &image.bytes)?;

    let id = Uuid::new_v4();
    let now = OffsetDateTime::now_utc();
    let extension = guess_extension(image.content_type.as_deref(), &image.bytes);
    let image_path = state.config.images_dir.join(format!("{id}.{extension}"));
    fs::write(&image_path, &image.bytes)
        .await
        .with_context(|| format!("failed to save uploaded image to {}", image_path.display()))?;
    debug!(
        "uploaded image saved job_id={} path={} bytes={} extension={}",
        id,
        image_path.display(),
        image.bytes.len(),
        extension
    );

    let image_sha256 = sha256_hex(&image.bytes);
    let record = JobRecord {
        id,
        status: JobStatus::Queued,
        created_at: now,
        updated_at: now,
        image_path: image_path.clone(),
        filename: image.filename,
        content_type: image.content_type,
        image_sha256,
        image_bytes: image.bytes.len(),
        input_kind: "upload".to_string(),
        source_path: None,
        task_type: task.task_type_name().to_string(),
        task_prompt: task.task_prompt_name().to_string(),
        text_input: task.text_input,
        webhook_url,
        result: None,
        error: None,
    };

    enqueue_record(state, record).await.map_err(ApiError::from)
}

async fn submit_inference_path(
    State(state): State<AppState>,
    Json(request): Json<LocalPathRequest>,
) -> Result<Json<QueueResponse>, ApiError> {
    ensure_workers_ready(&state)?;
    debug!(
        "local path inference submission started image_path={}",
        request.image_path.display()
    );

    if request.image_path.as_os_str().is_empty() {
        return Err(ApiError::BadRequest("`image_path` is required".into()));
    }

    let image_path = fs::canonicalize(&request.image_path).await.map_err(|err| {
        ApiError::BadRequest(format!(
            "failed to resolve image path {}: {err}",
            request.image_path.display()
        ))
    })?;
    ensure_local_path_allowed(&state, &image_path).await?;
    let metadata = fs::metadata(&image_path).await.map_err(|err| {
        ApiError::BadRequest(format!(
            "failed to inspect image path {}: {err}",
            image_path.display()
        ))
    })?;
    if !metadata.is_file() {
        return Err(ApiError::BadRequest(format!(
            "image path is not a regular file: {}",
            image_path.display()
        )));
    }

    let bytes = fs::read(&image_path).await.map_err(|err| {
        ApiError::BadRequest(format!(
            "failed to read image path {}: {err}",
            image_path.display()
        ))
    })?;
    if bytes.is_empty() {
        warn!("rejecting empty local image path={}", image_path.display());
        return Err(ApiError::BadRequest("local image file is empty".into()));
    }
    let content_type = image::guess_format(&bytes)
        .ok()
        .and_then(image_format_content_type)
        .map(str::to_string);
    validate_image_bytes(&state, content_type.as_deref(), &bytes)?;

    let id = Uuid::new_v4();
    let task = task_spec_from_request(
        request.task_type,
        request.task_prompt,
        request.text_input,
        request.task,
    )?;
    let webhook_url = validate_webhook_url(&state.config, request.webhook_url)?;
    let filename = image_path
        .file_name()
        .map(|filename| filename.to_string_lossy().into_owned());
    let image_sha256 = sha256_hex(&bytes);

    debug!(
        "local image accepted job_id={} path={} bytes={} content_type={:?}",
        id,
        image_path.display(),
        bytes.len(),
        content_type
    );

    let record = JobRecord {
        id,
        status: JobStatus::Queued,
        created_at: OffsetDateTime::now_utc(),
        updated_at: OffsetDateTime::now_utc(),
        image_path: image_path.clone(),
        filename,
        content_type,
        image_sha256,
        image_bytes: bytes.len(),
        input_kind: "local_path".to_string(),
        source_path: Some(image_path),
        task_type: task.task_type_name().to_string(),
        task_prompt: task.task_prompt_name().to_string(),
        text_input: task.text_input,
        webhook_url,
        result: None,
        error: None,
    };

    Ok(Json(enqueue_record(&state, record).await?))
}

impl From<EnqueueError> for ApiError {
    fn from(err: EnqueueError) -> Self {
        match err {
            EnqueueError::InvalidTask(source) => ApiError::BadRequest(source.to_string()),
            EnqueueError::QueueFull => {
                ApiError::ServiceUnavailable("inference queue is full".to_string())
            }
            EnqueueError::QueueClosed => {
                ApiError::ServiceUnavailable("inference queue is closed".to_string())
            }
            EnqueueError::Persist(source) => ApiError::Internal(source),
        }
    }
}

async fn get_job(
    State(state): State<AppState>,
    AxumPath(id): AxumPath<Uuid>,
) -> Result<Json<JobRecord>, ApiError> {
    let jobs = state.jobs.read().await;
    let record = jobs.get(&id).cloned().ok_or(ApiError::NotFound)?;
    debug!("job status read job_id={} status={:?}", id, record.status);
    Ok(Json(record))
}

async fn log_request(request: Request, next: Next) -> Response {
    let method = request.method().clone();
    let uri = request.uri().clone();
    let version = request.version();
    let content_length = request
        .headers()
        .get(axum::http::header::CONTENT_LENGTH)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<usize>().ok());
    let started = Instant::now();

    debug!(
        "http request started method={} uri={} version={:?} content_length={:?}",
        method, uri, version, content_length
    );

    let response = next.run(request).await;
    info!(
        "http request finished method={} uri={} status={} elapsed_ms={}",
        method,
        uri,
        response.status(),
        started.elapsed().as_millis()
    );

    response
}

async fn add_security_headers(request: Request, next: Next) -> Response {
    let mut response = next.run(request).await;
    let headers = response.headers_mut();

    headers.insert(
        header::X_CONTENT_TYPE_OPTIONS,
        HeaderValue::from_static("nosniff"),
    );
    headers.insert(
        HeaderName::from_static("x-frame-options"),
        HeaderValue::from_static("DENY"),
    );
    headers.insert(
        HeaderName::from_static("referrer-policy"),
        HeaderValue::from_static("no-referrer"),
    );
    headers.insert(
        HeaderName::from_static("cross-origin-resource-policy"),
        HeaderValue::from_static("same-origin"),
    );
    headers.insert(
        HeaderName::from_static("permissions-policy"),
        HeaderValue::from_static("camera=(), microphone=(), geolocation=()"),
    );
    headers.insert(
        HeaderName::from_static("content-security-policy"),
        HeaderValue::from_static(
            "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; form-action 'self'; frame-ancestors 'none'; base-uri 'self'",
        ),
    );

    response
}

fn task_spec_from_request(
    task_type: Option<String>,
    task_prompt: Option<String>,
    text_input: Option<String>,
    task_alias: Option<String>,
) -> Result<TaskSpec, ApiError> {
    TaskSpec::from_strings(task_type, task_prompt.or(task_alias), text_input)
        .map_err(|err| ApiError::BadRequest(err.to_string()))
}

fn validate_webhook_url(
    config: &Config,
    webhook_url: Option<String>,
) -> Result<Option<String>, ApiError> {
    let Some(webhook_url) = webhook_url.map(|value| value.trim().to_string()) else {
        return Ok(None);
    };
    if webhook_url.is_empty() {
        return Ok(None);
    }

    let parsed = reqwest::Url::parse(&webhook_url)
        .map_err(|err| ApiError::BadRequest(format!("invalid webhook_url: {err}")))?;
    match parsed.scheme() {
        "http" | "https" => {}
        scheme => {
            return Err(ApiError::BadRequest(format!(
                "unsupported webhook_url scheme `{scheme}`; expected http or https"
            )));
        }
    }
    if !parsed.username().is_empty() || parsed.password().is_some() {
        return Err(ApiError::BadRequest(
            "webhook_url must not include credentials".to_string(),
        ));
    }
    if parsed.fragment().is_some() {
        return Err(ApiError::BadRequest(
            "webhook_url must not include a fragment".to_string(),
        ));
    }
    if !config.allow_private_webhook_urls {
        validate_public_webhook_host(&parsed)?;
    }
    Ok(Some(parsed.to_string()))
}

fn validate_public_webhook_host(url: &reqwest::Url) -> Result<(), ApiError> {
    let Some(host) = url.host_str() else {
        return Err(ApiError::BadRequest(
            "webhook_url must include a host".to_string(),
        ));
    };

    if let Ok(ip) = host.parse::<IpAddr>() {
        match ip {
            IpAddr::V4(ip) if ipv4_is_private_or_local(ip) => {
                return Err(private_webhook_url_error());
            }
            IpAddr::V6(ip) if ipv6_is_private_or_local(ip) => {
                return Err(private_webhook_url_error());
            }
            IpAddr::V4(_) | IpAddr::V6(_) => {}
        }
    } else {
        let host = host.trim_end_matches('.').to_ascii_lowercase();
        if host == "localhost" || host.ends_with(".localhost") {
            return Err(private_webhook_url_error());
        }
    }

    Ok(())
}

fn private_webhook_url_error() -> ApiError {
    ApiError::BadRequest(
        "webhook_url targets a private or local address; set allow_private_webhook_urls for trusted deployments".to_string(),
    )
}

fn ipv4_is_private_or_local(ip: Ipv4Addr) -> bool {
    ip.is_private()
        || ip.is_loopback()
        || ip.is_link_local()
        || ip.is_broadcast()
        || ip.is_documentation()
        || ip.is_unspecified()
        || ip.octets()[0] == 0
}

fn ipv6_is_private_or_local(ip: Ipv6Addr) -> bool {
    ip.is_loopback()
        || ip.is_unspecified()
        || ip.is_unique_local()
        || ip.is_unicast_link_local()
        || matches!(ip.to_ipv4_mapped().map(IpAddr::V4), Some(IpAddr::V4(ip)) if ipv4_is_private_or_local(ip))
}

fn render_index(
    response: Option<QueueResponse>,
    error: Option<String>,
) -> Result<Html<String>, ApiError> {
    let template = IndexTemplate {
        single_task_prompts: SINGLE_TASK_PROMPTS,
        cascaded_task_prompts: CASCADED_TASK_PROMPTS,
        queued: response.is_some(),
        job_id: response
            .as_ref()
            .map(|response| response.id.to_string())
            .unwrap_or_default(),
        status_url: response
            .as_ref()
            .map(|response| response.status_url.clone())
            .unwrap_or_default(),
        error: error.unwrap_or_default(),
    };

    template.render().map(Html).map_err(|err| {
        ApiError::Internal(anyhow::anyhow!("failed to render index template: {err}"))
    })
}

fn ensure_workers_ready(state: &AppState) -> Result<(), ApiError> {
    if state.workers.is_ready() {
        Ok(())
    } else {
        Err(ApiError::ServiceUnavailable(
            "model workers are not ready".to_string(),
        ))
    }
}

async fn ensure_local_path_allowed(
    state: &AppState,
    image_path: &std::path::Path,
) -> Result<(), ApiError> {
    if !state.config.allow_local_paths {
        return Err(ApiError::Forbidden(
            "local path inference is disabled".to_string(),
        ));
    }

    for root in &state.config.local_path_roots {
        match fs::canonicalize(root).await {
            Ok(root) if image_path.starts_with(&root) && image_path != root => return Ok(()),
            Ok(_) => {}
            Err(err) => {
                warn!(
                    "configured local path root could not be resolved root={} error={}",
                    root.display(),
                    err
                );
            }
        }
    }

    Err(ApiError::Forbidden(format!(
        "image path is outside configured local path roots: {}",
        image_path.display()
    )))
}

fn validate_image_bytes(
    state: &AppState,
    content_type: Option<&str>,
    bytes: &[u8],
) -> Result<(), ApiError> {
    if let Some(content_type) = content_type
        && !content_type.starts_with("image/")
    {
        return Err(ApiError::BadRequest(format!(
            "unsupported content type `{content_type}`; expected an image"
        )));
    }

    let format = image::guess_format(bytes)
        .map_err(|_| ApiError::BadRequest("unsupported or invalid image format".to_string()))?;
    if image_format_content_type(format).is_none() {
        return Err(ApiError::BadRequest(format!(
            "unsupported image format: {format:?}"
        )));
    }

    let reader = ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|err| ApiError::BadRequest(format!("failed to identify image format: {err}")))?;
    let (width, height) = reader
        .into_dimensions()
        .map_err(|err| ApiError::BadRequest(format!("failed to read image dimensions: {err}")))?;

    if width > state.config.max_image_width || height > state.config.max_image_height {
        return Err(ApiError::BadRequest(format!(
            "image dimensions {}x{} exceed configured limit {}x{}",
            width, height, state.config.max_image_width, state.config.max_image_height
        )));
    }

    Ok(())
}

fn worker_health(state: &AppState) -> WorkerHealth {
    let snapshot = state.workers.snapshot();
    WorkerHealth {
        expected: snapshot.expected,
        ready: snapshot.ready,
        failed: snapshot.failed,
    }
}

fn openapi_document() -> Value {
    json!({
        "openapi": "3.1.0",
        "info": {
            "title": "Florence-2 Base Inference Server",
            "version": env!("CARGO_PKG_VERSION"),
        },
        "paths": {
            "/health": {
                "get": {
                    "summary": "Liveness check",
                    "responses": {
                        "200": {
                            "description": "Process is alive",
                            "content": { "application/json": { "schema": { "$ref": "#/components/schemas/HealthResponse" } } }
                        }
                    }
                }
            },
            "/ready": {
                "get": {
                    "summary": "Readiness check",
                    "responses": {
                        "200": { "description": "At least one worker is ready" },
                        "503": { "description": "No model worker is ready" }
                    }
                }
            },
            "/metrics": {
                "get": {
                    "summary": "Runtime metrics snapshot",
                    "responses": {
                        "200": {
                            "description": "Metrics snapshot",
                            "content": { "application/json": { "schema": { "$ref": "#/components/schemas/MetricsResponse" } } }
                        }
                    }
                }
            },
            "/v1/infer": {
                "post": {
                    "summary": "Queue inference from a multipart image upload",
                    "requestBody": {
                        "required": true,
                        "content": { "multipart/form-data": { "schema": { "$ref": "#/components/schemas/UploadInferenceRequest" } } }
                    },
                    "responses": {
                        "200": { "description": "Job queued", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/QueueResponse" } } } },
                        "400": { "$ref": "#/components/responses/BadRequest" },
                        "503": { "$ref": "#/components/responses/ServiceUnavailable" }
                    }
                }
            },
            "/v1/infer/path": {
                "post": {
                    "summary": "Queue inference from a server-side image path",
                    "requestBody": {
                        "required": true,
                        "content": { "application/json": { "schema": { "$ref": "#/components/schemas/LocalPathRequest" } } }
                    },
                    "responses": {
                        "200": { "description": "Job queued", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/QueueResponse" } } } },
                        "400": { "$ref": "#/components/responses/BadRequest" },
                        "403": { "$ref": "#/components/responses/Forbidden" },
                        "503": { "$ref": "#/components/responses/ServiceUnavailable" }
                    }
                }
            },
            "/v1/jobs/{id}": {
                "get": {
                    "summary": "Fetch a job record",
                    "parameters": [{
                        "name": "id",
                        "in": "path",
                        "required": true,
                        "schema": { "type": "string", "format": "uuid" }
                    }],
                    "responses": {
                        "200": { "description": "Job record", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/JobRecord" } } } },
                        "404": { "$ref": "#/components/responses/NotFound" }
                    }
                }
            }
        },
        "components": {
            "responses": {
                "BadRequest": { "description": "Bad request", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/ErrorResponse" } } } },
                "Forbidden": { "description": "Forbidden", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/ErrorResponse" } } } },
                "NotFound": { "description": "Not found", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/ErrorResponse" } } } },
                "ServiceUnavailable": { "description": "Service unavailable", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/ErrorResponse" } } } }
            },
            "schemas": {
                "ErrorResponse": {
                    "type": "object",
                    "required": ["code", "message"],
                    "properties": {
                        "code": { "type": "string" },
                        "message": { "type": "string" }
                    }
                },
                "QueueResponse": {
                    "type": "object",
                    "required": ["id", "status", "status_url"],
                    "properties": {
                        "id": { "type": "string", "format": "uuid" },
                        "status": { "type": "string" },
                        "status_url": { "type": "string" }
                    }
                },
                "LocalPathRequest": {
                    "type": "object",
                    "required": ["image_path"],
                    "properties": {
                        "image_path": { "type": "string" },
                        "task_type": { "type": "string" },
                        "task_prompt": { "type": "string" },
                        "text_input": { "type": ["string", "null"] },
                        "webhook_url": { "type": ["string", "null"], "format": "uri" },
                        "task": { "type": "string", "deprecated": true }
                    }
                },
                "UploadInferenceRequest": {
                    "type": "object",
                    "required": ["image"],
                    "properties": {
                        "image": { "type": "string", "format": "binary" },
                        "task_type": { "type": "string" },
                        "task_prompt": { "type": "string" },
                        "text_input": { "type": "string" },
                        "webhook_url": { "type": "string", "format": "uri" }
                    }
                },
                "WorkerHealth": {
                    "type": "object",
                    "required": ["expected", "ready", "failed"],
                    "properties": {
                        "expected": { "type": "integer" },
                        "ready": { "type": "integer" },
                        "failed": { "type": "integer" }
                    }
                },
                "HealthResponse": { "type": "object" },
                "MetricsResponse": { "type": "object" },
                "JobRecord": { "type": "object" }
            }
        }
    })
}

fn cors_layer(allowed_origins: &[String]) -> CorsLayer {
    let origins = allowed_origins
        .iter()
        .filter_map(|origin| match HeaderValue::from_str(origin) {
            Ok(origin) => Some(origin),
            Err(err) => {
                warn!("invalid CORS origin ignored origin={origin:?} error={err}");
                None
            }
        })
        .collect::<Vec<_>>();

    CorsLayer::new()
        .allow_origin(AllowOrigin::list(origins))
        .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
        .allow_headers([header::CONTENT_TYPE])
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, net::SocketAddr, sync::Arc};

    use async_channel::bounded;
    use axum::body::Body;
    use tokio::sync::RwLock;
    use tower::ServiceExt;

    use super::*;
    use crate::{
        config::{Config, ModelVariant},
        state::{AppMetrics, WorkerPoolState},
        types::{TaskPrompt, TaskType},
    };

    #[test]
    fn request_task_defaults_to_single_caption() {
        let task = task_spec_from_request(None, None, None, None).unwrap();

        assert_eq!(task.task_type, TaskType::Single);
        assert_eq!(task.task_prompt, TaskPrompt::Caption);
        assert_eq!(task.text_input, None);
    }

    #[test]
    fn request_task_prefers_task_prompt_over_alias() {
        let task = task_spec_from_request(
            Some("Single task".to_string()),
            Some("OCR".to_string()),
            Some("  text to trim  ".to_string()),
            Some("Caption".to_string()),
        )
        .unwrap();

        assert_eq!(task.task_prompt, TaskPrompt::Ocr);
        assert_eq!(task.text_input.as_deref(), Some("text to trim"));
    }

    #[test]
    fn request_task_uses_task_alias_when_prompt_is_absent() {
        let task = task_spec_from_request(
            Some("Single task".to_string()),
            None,
            None,
            Some("OCR with Region".to_string()),
        )
        .unwrap();

        assert_eq!(task.task_prompt, TaskPrompt::OcrWithRegion);
    }

    #[test]
    fn request_task_rejects_invalid_prompt_for_task_type() {
        let err = task_spec_from_request(
            Some("Single task".to_string()),
            Some("Caption + Grounding".to_string()),
            None,
            None,
        )
        .unwrap_err();

        assert!(matches!(err, ApiError::BadRequest(_)));
        assert!(
            err.to_string()
                .contains("unsupported task_prompt `Caption + Grounding`")
        );
    }

    #[test]
    fn index_template_contains_task_options_and_error() {
        let html = render_index(None, Some("bad request".to_string()))
            .unwrap()
            .0;

        assert!(html.contains("Florence-2 Inference"));
        assert!(html.contains("OCR with Region"));
        assert!(html.contains("Caption + Grounding"));
        assert!(html.contains("bad request"));
    }

    #[test]
    fn api_errors_have_stable_codes() {
        assert_eq!(ApiError::NotFound.code(), "not_found");
        assert_eq!(
            ApiError::ServiceUnavailable("not ready".to_string()).code(),
            "service_unavailable"
        );
    }

    #[test]
    fn openapi_document_describes_core_paths() {
        let document = openapi_document();

        assert_eq!(document["openapi"], "3.1.0");
        assert!(document["paths"]["/v1/infer"].is_object());
        assert!(document["paths"]["/metrics"].is_object());
        assert!(document["components"]["schemas"]["ErrorResponse"].is_object());
    }

    #[test]
    fn validates_supported_image_bytes() {
        let state = test_state(false, Vec::new());

        validate_image_bytes(&state, Some("image/png"), ONE_BY_ONE_PNG).unwrap();
    }

    #[test]
    fn rejects_non_image_content_type() {
        let state = test_state(false, Vec::new());
        let err = validate_image_bytes(&state, Some("text/plain"), ONE_BY_ONE_PNG).unwrap_err();

        assert!(matches!(err, ApiError::BadRequest(_)));
        assert!(err.to_string().contains("unsupported content type"));
    }

    #[test]
    fn validates_optional_webhook_url() {
        let state = test_state(false, Vec::new());
        let url =
            validate_webhook_url(&state.config, Some(" http://example.com/hook ".to_string()))
                .unwrap();

        assert_eq!(url.as_deref(), Some("http://example.com/hook"));
        assert_eq!(
            validate_webhook_url(&state.config, Some("  ".to_string())).unwrap(),
            None
        );
    }

    #[test]
    fn rejects_unsupported_webhook_url_scheme() {
        let state = test_state(false, Vec::new());
        let err =
            validate_webhook_url(&state.config, Some("file:///tmp/hook".to_string())).unwrap_err();

        assert!(matches!(err, ApiError::BadRequest(_)));
        assert!(err.to_string().contains("expected http or https"));
    }

    #[test]
    fn rejects_webhook_url_credentials() {
        let state = test_state(false, Vec::new());
        let err = validate_webhook_url(
            &state.config,
            Some("https://user:secret@example.com/hook".to_string()),
        )
        .unwrap_err();

        assert!(matches!(err, ApiError::BadRequest(_)));
        assert!(err.to_string().contains("must not include credentials"));
    }

    #[test]
    fn rejects_private_webhook_urls_by_default() {
        let state = test_state(false, Vec::new());
        let err = validate_webhook_url(&state.config, Some("http://127.0.0.1/hook".to_string()))
            .unwrap_err();

        assert!(matches!(err, ApiError::BadRequest(_)));
        assert!(err.to_string().contains("private or local address"));
    }

    #[test]
    fn allows_private_webhook_urls_when_configured() {
        let mut state = test_state(false, Vec::new());
        Arc::get_mut(&mut state.config)
            .unwrap()
            .allow_private_webhook_urls = true;

        let url =
            validate_webhook_url(&state.config, Some("http://127.0.0.1/hook".to_string())).unwrap();

        assert_eq!(url.as_deref(), Some("http://127.0.0.1/hook"));
    }

    #[tokio::test]
    async fn metrics_response_reports_counters() {
        let state = test_state(false, Vec::new());
        state.metrics.record_job_started();
        state.metrics.record_job_succeeded(42);

        let Json(response) = metrics(State(state)).await;

        assert_eq!(response.jobs_started, 1);
        assert_eq!(response.jobs_succeeded, 1);
        assert_eq!(response.total_inference_ms, 42);
        assert_eq!(response.average_inference_ms, Some(42.0));
    }

    #[tokio::test]
    async fn request_timeout_layer_returns_408() {
        let app = Router::new()
            .route(
                "/slow",
                get(|| async {
                    tokio::time::sleep(Duration::from_millis(20)).await;
                    "ok"
                }),
            )
            .layer(TimeoutLayer::with_status_code(
                StatusCode::REQUEST_TIMEOUT,
                Duration::from_millis(1),
            ));

        let response = app
            .oneshot(Request::builder().uri("/slow").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::REQUEST_TIMEOUT);
    }

    #[tokio::test]
    async fn security_headers_are_added_to_responses() {
        let app = Router::new()
            .route("/ok", get(|| async { "ok" }))
            .layer(middleware::from_fn(add_security_headers));

        let response = app
            .oneshot(Request::builder().uri("/ok").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(
            response.headers().get(header::X_CONTENT_TYPE_OPTIONS),
            Some(&HeaderValue::from_static("nosniff"))
        );
        assert_eq!(
            response
                .headers()
                .get(HeaderName::from_static("x-frame-options")),
            Some(&HeaderValue::from_static("DENY"))
        );
        assert!(
            response
                .headers()
                .contains_key(HeaderName::from_static("content-security-policy"))
        );
    }

    #[tokio::test]
    async fn local_path_endpoint_is_disabled_by_default() {
        let state = test_state(false, Vec::new());
        let path = std::env::temp_dir().join(format!("florence2-api-test-{}", Uuid::new_v4()));
        tokio::fs::write(&path, b"image").await.unwrap();
        let path = tokio::fs::canonicalize(&path).await.unwrap();

        let err = ensure_local_path_allowed(&state, &path).await.unwrap_err();

        assert!(matches!(err, ApiError::Forbidden(_)));
        assert!(err.to_string().contains("local path inference is disabled"));

        tokio::fs::remove_file(path).await.unwrap();
    }

    #[tokio::test]
    async fn local_path_endpoint_allows_files_under_configured_roots() {
        let root = std::env::temp_dir().join(format!("florence2-api-root-{}", Uuid::new_v4()));
        tokio::fs::create_dir_all(&root).await.unwrap();
        let image_path = root.join("image.bin");
        tokio::fs::write(&image_path, b"image").await.unwrap();
        let image_path = tokio::fs::canonicalize(&image_path).await.unwrap();
        let state = test_state(true, vec![root.clone()]);

        ensure_local_path_allowed(&state, &image_path)
            .await
            .unwrap();

        tokio::fs::remove_dir_all(root).await.unwrap();
    }

    fn test_state(allow_local_paths: bool, local_path_roots: Vec<PathBuf>) -> AppState {
        let (queue_tx, _queue_rx) = bounded(1);
        AppState {
            config: Arc::new(Config {
                addr: SocketAddr::from(([127, 0, 0, 1], 3000)),
                model_path: PathBuf::from("Florence-2-base/onnx/vision_encoder.onnx"),
                model_variant: ModelVariant::Fp32,
                data_dir: PathBuf::from("data"),
                images_dir: PathBuf::from("data/images"),
                metadata_dir: PathBuf::from("data/metadata"),
                submissions_jsonl: PathBuf::from("data/metadata/submissions.jsonl"),
                results_jsonl: PathBuf::from("data/metadata/results.jsonl"),
                allow_local_paths,
                local_path_roots,
                cors_allowed_origins: Vec::new(),
                job_retention_limit: 1000,
                metadata_retention_limit: 10_000,
                max_image_width: 8192,
                max_image_height: 8192,
                workers: 1,
                queue_size: 1,
                body_limit_bytes: 1024,
                request_timeout_seconds: 60,
                rust_log: "info".to_string(),
                max_new_tokens: 1,
                job_timeout_seconds: 300,
                webhook_timeout_seconds: 10,
                webhook_connect_timeout_seconds: 5,
                allow_private_webhook_urls: false,
                execution_providers: vec!["cpu".to_string()],
            }),
            queue_tx,
            jobs: Arc::new(RwLock::new(HashMap::new())),
            workers: Arc::new(WorkerPoolState::new(1)),
            metrics: Arc::new(AppMetrics::default()),
        }
    }

    const ONE_BY_ONE_PNG: &[u8] = &[
        137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82, 0, 0, 0, 1, 0, 0, 0, 1, 8, 6,
        0, 0, 0, 31, 21, 196, 137, 0, 0, 0, 13, 73, 68, 65, 84, 120, 156, 99, 0, 1, 0, 0, 5, 0, 1,
        13, 10, 45, 180, 0, 0, 0, 0, 73, 69, 78, 68, 174, 66, 96, 130,
    ];
}
