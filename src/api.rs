mod docs;
mod security;
mod system;

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
    extract::{DefaultBodyLimit, Multipart, Path as AxumPath, Request, State, multipart::Field},
    http::StatusCode,
    middleware::{self, Next},
    response::{Html, IntoResponse, Response},
    routing::{get, post},
};
use image::ImageReader;
use log::{debug, info, trace, warn};
use serde::Deserialize;
use serde_json::Value;
use time::OffsetDateTime;
use tokio::fs;
use tower_http::timeout::TimeoutLayer;
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

use self::{
    docs::{cors_layer, openapi_document},
    security::{add_security_headers, rate_limit, require_api_key},
    system::system_usage,
};

pub fn router(state: AppState) -> Router {
    let cors_allowed_origins = state.config.cors_allowed_origins.clone();
    let request_timeout_seconds = state.config.request_timeout_seconds;
    let state_for_middleware = state.clone();
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
        .layer(middleware::from_fn_with_state(
            state_for_middleware.clone(),
            rate_limit,
        ))
        .layer(middleware::from_fn_with_state(
            state_for_middleware.clone(),
            require_api_key,
        ))
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

    router
        .layer(middleware::from_fn_with_state(
            state_for_middleware,
            track_request_metrics,
        ))
        .layer(middleware::from_fn(add_security_headers))
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

#[derive(Default)]
struct MultipartInferenceRequest {
    image: Option<UploadedImage>,
    task_type: Option<String>,
    task_prompt: Option<String>,
    text_input: Option<String>,
    webhook_url: Option<String>,
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
    let average_request_ms = (metrics.http_requests_total > 0)
        .then_some(metrics.total_request_ms as f64 / metrics.http_requests_total as f64);
    let (process_memory_rss_bytes, gpu_memory) = system_usage().await;

    Json(MetricsResponse {
        workers,
        queued: state.queue_tx.len(),
        http_requests_total: metrics.http_requests_total,
        http_requests_failed: metrics.http_requests_failed,
        total_request_ms: metrics.total_request_ms as u128,
        average_request_ms,
        jobs_started: metrics.jobs_started,
        jobs_succeeded: metrics.jobs_succeeded,
        jobs_failed: metrics.jobs_failed,
        jobs_timed_out: metrics.jobs_timed_out,
        worker_restarts: metrics.worker_restarts,
        total_inference_ms: metrics.total_inference_ms as u128,
        average_inference_ms,
        model_load_ms: metrics.model_load_ms.map(|value| value as u128),
        webhook_failures: metrics.webhook_failures,
        cleanup_failures: metrics.cleanup_failures,
        retained_jobs: state.jobs.read().await.len(),
        process_memory_rss_bytes,
        gpu_memory,
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
    let request = read_multipart_inference_request(&mut multipart).await?;
    let record = uploaded_job_record(state, request).await?;

    enqueue_record(state, record).await.map_err(ApiError::from)
}

async fn read_multipart_inference_request(
    multipart: &mut Multipart,
) -> Result<MultipartInferenceRequest, ApiError> {
    let mut request = MultipartInferenceRequest::default();
    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|err| ApiError::BadRequest(format!("failed to read multipart field: {err}")))?
    {
        let name = field.name().unwrap_or_default().to_string();
        trace!("multipart field received name={}", name);
        apply_multipart_field(&mut request, name.as_str(), field).await?;
    }

    Ok(request)
}

async fn apply_multipart_field(
    request: &mut MultipartInferenceRequest,
    name: &str,
    field: Field<'_>,
) -> Result<(), ApiError> {
    match name {
        "image" | "file" => request.image = Some(read_uploaded_image(field).await?),
        "task_type" | "task_type_selector" => {
            request.task_type = Some(read_text_field(field, "task_type").await?);
        }
        "task_prompt" | "task" => {
            request.task_prompt = Some(read_text_field(field, "task").await?);
        }
        "text_input" => request.text_input = Some(read_text_field(field, "text_input").await?),
        "webhook_url" => request.webhook_url = Some(read_text_field(field, "webhook_url").await?),
        _ => {}
    }

    Ok(())
}

async fn read_uploaded_image(field: Field<'_>) -> Result<UploadedImage, ApiError> {
    let filename = field.file_name().map(str::to_string);
    let content_type = field.content_type().map(str::to_string);
    let bytes = field
        .bytes()
        .await
        .map_err(|err| ApiError::BadRequest(format!("failed to read image field: {err}")))?;
    debug!(
        "image field read filename={:?} content_type={:?} bytes={}",
        filename,
        content_type,
        bytes.len()
    );
    Ok(UploadedImage {
        filename,
        content_type,
        bytes,
    })
}

async fn read_text_field(field: Field<'_>, field_name: &str) -> Result<String, ApiError> {
    field
        .text()
        .await
        .map_err(|err| ApiError::BadRequest(format!("failed to read {field_name} field: {err}")))
}

async fn uploaded_job_record(
    state: &AppState,
    request: MultipartInferenceRequest,
) -> Result<JobRecord, ApiError> {
    let task = task_spec_from_request(
        request.task_type,
        request.task_prompt,
        request.text_input,
        None,
    )?;
    let webhook_url = validate_webhook_url(&state.config, request.webhook_url)?;
    let image = request
        .image
        .ok_or_else(|| ApiError::BadRequest("multipart field `image` is required".into()))?;
    if image.bytes.is_empty() {
        warn!("rejecting empty image upload");
        return Err(ApiError::BadRequest("uploaded image is empty".into()));
    }
    validate_image_bytes(state, image.content_type.as_deref(), &image.bytes)?;

    let id = Uuid::new_v4();
    let now = OffsetDateTime::now_utc();
    let image_path = save_uploaded_image(&state.config, id, &image).await?;
    let image_sha256 = sha256_hex(&image.bytes);
    Ok(JobRecord {
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
    })
}

async fn save_uploaded_image(
    config: &Config,
    id: Uuid,
    image: &UploadedImage,
) -> Result<PathBuf, ApiError> {
    let extension = guess_extension(image.content_type.as_deref(), &image.bytes);
    let image_path = config.images_dir.join(format!("{id}.{extension}"));
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
    Ok(image_path)
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

    let record = local_path_job_record(&state, request).await?;

    Ok(Json(enqueue_record(&state, record).await?))
}

async fn local_path_job_record(
    state: &AppState,
    request: LocalPathRequest,
) -> Result<JobRecord, ApiError> {
    let image_path = validated_local_image_path(state, &request.image_path).await?;
    let bytes = read_local_image(&image_path).await?;
    let content_type = image::guess_format(&bytes)
        .ok()
        .and_then(image_format_content_type)
        .map(str::to_string);
    validate_image_bytes(state, content_type.as_deref(), &bytes)?;

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

    debug!(
        "local image accepted job_id={} path={} bytes={} content_type={:?}",
        id,
        image_path.display(),
        bytes.len(),
        content_type
    );

    let now = OffsetDateTime::now_utc();
    Ok(JobRecord {
        id,
        status: JobStatus::Queued,
        created_at: now,
        updated_at: now,
        image_path: image_path.clone(),
        filename,
        content_type,
        image_sha256: sha256_hex(&bytes),
        image_bytes: bytes.len(),
        input_kind: "local_path".to_string(),
        source_path: Some(image_path),
        task_type: task.task_type_name().to_string(),
        task_prompt: task.task_prompt_name().to_string(),
        text_input: task.text_input,
        webhook_url,
        result: None,
        error: None,
    })
}

async fn validated_local_image_path(
    state: &AppState,
    requested_path: &PathBuf,
) -> Result<PathBuf, ApiError> {
    if requested_path.as_os_str().is_empty() {
        return Err(ApiError::BadRequest("`image_path` is required".into()));
    }

    let image_path = fs::canonicalize(requested_path).await.map_err(|err| {
        ApiError::BadRequest(format!(
            "failed to resolve image path {}: {err}",
            requested_path.display()
        ))
    })?;
    ensure_local_path_allowed(state, &image_path).await?;
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
    Ok(image_path)
}

async fn read_local_image(image_path: &PathBuf) -> Result<Vec<u8>, ApiError> {
    let bytes = fs::read(image_path).await.map_err(|err| {
        ApiError::BadRequest(format!(
            "failed to read image path {}: {err}",
            image_path.display()
        ))
    })?;
    if bytes.is_empty() {
        warn!("rejecting empty local image path={}", image_path.display());
        return Err(ApiError::BadRequest("local image file is empty".into()));
    }
    Ok(bytes)
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

async fn track_request_metrics(
    State(state): State<AppState>,
    request: Request,
    next: Next,
) -> Response {
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
    let elapsed = started.elapsed();
    let failed = response.status().is_client_error() || response.status().is_server_error();
    state
        .metrics
        .record_http_request(elapsed.as_millis(), failed);
    info!(
        "http request finished method={} uri={} status={} elapsed_ms={}",
        method,
        uri,
        response.status(),
        elapsed.as_millis()
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
        if ip_is_private_or_local(ip) {
            return Err(private_webhook_url_error());
        }
    } else {
        let host = host.trim_end_matches('.').to_ascii_lowercase();
        if host == "localhost" || host.ends_with(".localhost") {
            return Err(private_webhook_url_error());
        }
    }

    Ok(())
}

fn ip_is_private_or_local(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(ip) => ipv4_is_private_or_local(ip),
        IpAddr::V6(ip) => ipv6_is_private_or_local(ip),
    }
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

#[cfg(test)]
mod tests;
