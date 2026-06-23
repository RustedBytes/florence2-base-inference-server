use std::{path::PathBuf, time::Instant};

use anyhow::Context;
use askama::Template;
use axum::{
    Json, Router,
    body::Bytes,
    extract::{DefaultBodyLimit, Multipart, Path as AxumPath, Request, State},
    http::StatusCode,
    middleware::{self, Next},
    response::{Html, IntoResponse, Response},
    routing::{get, post},
};
use log::{debug, info, trace, warn};
use serde::{Deserialize, Serialize};
use time::OffsetDateTime;
use tokio::fs;
use uuid::Uuid;

use crate::{
    jobs::enqueue_record,
    state::AppState,
    templates::IndexTemplate,
    types::{HealthResponse, JobRecord, JobStatus, QueueResponse, TaskSpec},
    util::{guess_extension, image_format_content_type, sha256_hex},
};

const TASK_TYPE_SINGLE: &str = "Single task";
const TASK_TYPE_CASCASED: &str = "Cascased task";

const SINGLE_TASK_PROMPTS: &[&str] = &[
    "Caption",
    "Detailed Caption",
    "More Detailed Caption",
    "Object Detection",
    "Dense Region Caption",
    "Region Proposal",
    "Caption to Phrase Grounding",
    "Referring Expression Segmentation",
    "Region to Segmentation",
    "Open Vocabulary Detection",
    "Region to Category",
    "Region to Description",
    "OCR",
    "OCR with Region",
];

const CASCASED_TASK_PROMPTS: &[&str] = &[
    "Caption + Grounding",
    "Detailed Caption + Grounding",
    "More Detailed Caption + Grounding",
];

pub fn router(state: AppState, body_limit_bytes: usize) -> Router {
    Router::new()
        .route("/", get(index))
        .route("/health", get(health))
        .route("/v1/infer", post(submit_inference))
        .route("/infer-form", post(submit_inference_form))
        .route("/v1/infer/path", post(submit_inference_path))
        .route("/v1/jobs/{id}", get(get_job))
        .layer(DefaultBodyLimit::max(body_limit_bytes))
        .layer(middleware::from_fn(log_request))
        .with_state(state)
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
    #[error(transparent)]
    Internal(#[from] anyhow::Error),
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = match self {
            ApiError::BadRequest(_) => StatusCode::BAD_REQUEST,
            ApiError::NotFound => StatusCode::NOT_FOUND,
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
    debug!(
        "health check workers={} queued={} model_path={} model_variant={}",
        state.config.workers,
        state.queue_tx.len(),
        state.config.model_path.display(),
        state.config.model_variant.as_str()
    );

    Json(HealthResponse {
        status: "ok",
        workers: state.config.workers,
        queued: state.queue_tx.len(),
        model_path: state.config.model_path.clone(),
        model_variant: state.config.model_variant,
    })
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

    let mut image: Option<UploadedImage> = None;
    let mut task = TaskSpec::default();

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
                task.task_type = field.text().await.map_err(|err| {
                    ApiError::BadRequest(format!("failed to read task_type field: {err}"))
                })?;
            }
            "task_prompt" | "task" => {
                task.task_prompt = field.text().await.map_err(|err| {
                    ApiError::BadRequest(format!("failed to read task field: {err}"))
                })?;
            }
            "text_input" => {
                let text_input = field.text().await.map_err(|err| {
                    ApiError::BadRequest(format!("failed to read text_input field: {err}"))
                })?;
                task.text_input = normalize_optional_text(text_input);
            }
            _ => {}
        }
    }
    validate_task_spec(&task)?;

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
        task_type: task.task_type,
        task_prompt: task.task_prompt,
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
        task_type: task.task_type,
        task_prompt: task.task_prompt,
        text_input: task.text_input,
        result: None,
        error: None,
    };

    Ok(Json(enqueue_record(&state, record).await?))
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
    let mut task = TaskSpec {
        task_type: task_type.unwrap_or_else(|| TaskSpec::default().task_type),
        task_prompt: task_prompt
            .or(task_alias)
            .unwrap_or_else(|| TaskSpec::default().task_prompt),
        text_input: text_input.and_then(normalize_optional_text),
    };

    task.task_type = task.task_type.trim().to_string();
    task.task_prompt = task.task_prompt.trim().to_string();
    validate_task_spec(&task)?;
    Ok(task)
}

fn validate_task_spec(task: &TaskSpec) -> Result<(), ApiError> {
    let allowed_prompts = match task.task_type.as_str() {
        TASK_TYPE_SINGLE => SINGLE_TASK_PROMPTS,
        TASK_TYPE_CASCASED => CASCASED_TASK_PROMPTS,
        other => {
            return Err(ApiError::BadRequest(format!(
                "unsupported task_type `{other}`; expected `{TASK_TYPE_SINGLE}` or `{TASK_TYPE_CASCASED}`"
            )));
        }
    };

    if !allowed_prompts.contains(&task.task_prompt.as_str()) {
        return Err(ApiError::BadRequest(format!(
            "unsupported task_prompt `{}` for task_type `{}`",
            task.task_prompt, task.task_type
        )));
    }

    Ok(())
}

fn normalize_optional_text(text: String) -> Option<String> {
    let text = text.trim().to_string();
    (!text.is_empty()).then_some(text)
}

fn render_index(
    response: Option<QueueResponse>,
    error: Option<String>,
) -> Result<Html<String>, ApiError> {
    let template = IndexTemplate {
        single_task_prompts: SINGLE_TASK_PROMPTS,
        cascased_task_prompts: CASCASED_TASK_PROMPTS,
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
