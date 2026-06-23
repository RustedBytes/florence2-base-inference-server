use std::{path::PathBuf, time::Instant};

use anyhow::Context;
use askama::Template;
use axum::{
    Json, Router,
    body::Bytes,
    extract::{DefaultBodyLimit, Multipart, Path as AxumPath, Request, State},
    http::{HeaderValue, Method, StatusCode, header},
    middleware::{self, Next},
    response::{Html, IntoResponse, Response},
    routing::{get, post},
};
use log::{debug, info, trace, warn};
use serde::{Deserialize, Serialize};
use time::OffsetDateTime;
use tokio::fs;
use tower_http::cors::{AllowOrigin, CorsLayer};
use uuid::Uuid;

use crate::{
    jobs::{EnqueueError, enqueue_record},
    state::AppState,
    templates::IndexTemplate,
    types::{
        CASCADED_TASK_PROMPTS, HealthResponse, JobRecord, JobStatus, QueueResponse,
        ReadinessResponse, SINGLE_TASK_PROMPTS, TaskSpec, WorkerHealth,
    },
    util::{guess_extension, image_format_content_type, sha256_hex},
};

pub fn router(state: AppState, body_limit_bytes: usize) -> Router {
    let cors_allowed_origins = state.config.cors_allowed_origins.clone();
    let router = Router::new()
        .route("/", get(index))
        .route("/health", get(health))
        .route("/ready", get(readiness))
        .route("/v1/infer", post(submit_inference))
        .route("/infer-form", post(submit_inference_form))
        .route("/v1/infer/path", post(submit_inference_path))
        .route("/v1/jobs/{id}", get(get_job))
        .layer(DefaultBodyLimit::max(body_limit_bytes))
        .layer(middleware::from_fn(log_request))
        .with_state(state);

    if cors_allowed_origins.is_empty() {
        router
    } else {
        router.layer(cors_layer(&cors_allowed_origins))
    }
}

#[derive(Debug, Serialize)]
struct ErrorBody {
    error: String,
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
        let status = match self {
            ApiError::BadRequest(_) => StatusCode::BAD_REQUEST,
            ApiError::NotFound => StatusCode::NOT_FOUND,
            ApiError::Forbidden(_) => StatusCode::FORBIDDEN,
            ApiError::ServiceUnavailable(_) => StatusCode::SERVICE_UNAVAILABLE,
            ApiError::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
        };
        let body = Json(ErrorBody {
            error: self.to_string(),
        });
        (status, body).into_response()
    }
}

#[derive(Debug, Deserialize)]
struct LocalPathRequest {
    image_path: PathBuf,
    task_type: Option<String>,
    task_prompt: Option<String>,
    text_input: Option<String>,
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
            _ => {}
        }
    }
    let task = task_spec_from_request(task_type, task_prompt, text_input, None)?;

    let image =
        image.ok_or_else(|| ApiError::BadRequest("multipart field `image` is required".into()))?;
    if image.bytes.is_empty() {
        warn!("rejecting empty image upload");
        return Err(ApiError::BadRequest("uploaded image is empty".into()));
    }

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

    let id = Uuid::new_v4();
    let task = task_spec_from_request(
        request.task_type,
        request.task_prompt,
        request.text_input,
        request.task,
    )?;
    let filename = image_path
        .file_name()
        .map(|filename| filename.to_string_lossy().into_owned());
    let content_type = image::guess_format(&bytes)
        .ok()
        .and_then(image_format_content_type)
        .map(str::to_string);
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

fn task_spec_from_request(
    task_type: Option<String>,
    task_prompt: Option<String>,
    text_input: Option<String>,
    task_alias: Option<String>,
) -> Result<TaskSpec, ApiError> {
    TaskSpec::from_strings(task_type, task_prompt.or(task_alias), text_input)
        .map_err(|err| ApiError::BadRequest(err.to_string()))
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

fn worker_health(state: &AppState) -> WorkerHealth {
    let snapshot = state.workers.snapshot();
    WorkerHealth {
        expected: snapshot.expected,
        ready: snapshot.ready,
        failed: snapshot.failed,
    }
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
    use tokio::sync::RwLock;

    use super::*;
    use crate::{
        config::{Config, ModelVariant},
        state::WorkerPoolState,
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
                workers: 1,
                queue_size: 1,
                body_limit_bytes: 1024,
                rust_log: "info".to_string(),
                max_new_tokens: 1,
                execution_providers: vec!["cpu".to_string()],
            }),
            queue_tx,
            jobs: Arc::new(RwLock::new(HashMap::new())),
            workers: Arc::new(WorkerPoolState::new(1)),
        }
    }
}
