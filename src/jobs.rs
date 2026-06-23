use std::{
    collections::HashMap,
    io::ErrorKind,
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant},
};

use anyhow::{Context, anyhow};
use async_channel::Receiver;
use log::{debug, error, info, warn};
use serde::Serialize;
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::{
    config::Config,
    inference::FlorenceWorker,
    state::{AppMetrics, AppState, WorkerPoolState},
    types::{InferenceMetadata, JobRecord, JobStatus, QueueResponse, TaskSpec},
    util::{append_jsonl, hmac_sha256_hex},
};

#[derive(Debug, Clone)]
pub struct JobRequest {
    pub id: Uuid,
    pub image_path: PathBuf,
    pub task: TaskSpec,
}

#[derive(Clone)]
pub struct WebhookClient {
    http: reqwest::Client,
    max_attempts: usize,
    initial_backoff: Duration,
    signing_secret: Option<String>,
    dead_letter_jsonl: PathBuf,
}

impl WebhookClient {
    pub fn from_config(config: &Config) -> anyhow::Result<Self> {
        Self::new(
            config.webhook_timeout_seconds,
            config.webhook_connect_timeout_seconds,
            config.webhook_max_attempts,
            Duration::from_millis(config.webhook_initial_backoff_ms),
            config.webhook_signing_secret.clone(),
            config.webhooks_dead_letter_jsonl.clone(),
        )
    }

    fn new(
        timeout_seconds: u64,
        connect_timeout_seconds: u64,
        max_attempts: usize,
        initial_backoff: Duration,
        signing_secret: Option<String>,
        dead_letter_jsonl: PathBuf,
    ) -> anyhow::Result<Self> {
        let mut builder = reqwest::Client::builder().redirect(reqwest::redirect::Policy::none());
        if timeout_seconds > 0 {
            builder = builder.timeout(Duration::from_secs(timeout_seconds));
        }
        if connect_timeout_seconds > 0 {
            builder = builder.connect_timeout(Duration::from_secs(connect_timeout_seconds));
        }

        let http = builder
            .build()
            .context("failed to build webhook HTTP client")?;

        Ok(Self {
            http,
            max_attempts: max_attempts.max(1),
            initial_backoff,
            signing_secret,
            dead_letter_jsonl,
        })
    }

    async fn send(&self, event: &WebhookEvent, attempt: usize) -> anyhow::Result<()> {
        let record = &event.job;
        let Some(webhook_url) = record.webhook_url.as_deref() else {
            return Ok(());
        };
        let redacted_url = redacted_webhook_url(webhook_url);
        let body = serde_json::to_vec(event).context("failed to serialize webhook event")?;

        let mut request = self
            .http
            .post(webhook_url)
            .header(reqwest::header::CONTENT_TYPE, "application/json")
            .header("x-florence-event-id", event.event_id.to_string())
            .header("x-florence-event-type", event.event_type)
            .header("x-florence-delivery-attempt", attempt.to_string())
            .body(body.clone());
        if let Some(secret) = &self.signing_secret {
            request = request.header(
                "x-florence-signature",
                format!("sha256={}", hmac_sha256_hex(secret, &body)),
            );
        }

        let response = request
            .send()
            .await
            .with_context(|| format!("failed to send webhook request to {redacted_url}"))?;

        if !response.status().is_success() {
            return Err(anyhow!(
                "webhook endpoint returned HTTP {}",
                response.status()
            ));
        }

        Ok(())
    }

    async fn deliver(&self, record: &JobRecord) -> WebhookDeliveryResult {
        if record.webhook_url.is_none() {
            return WebhookDeliveryResult::Skipped;
        }

        let event = WebhookEvent::from_job(record.clone());
        let mut last_error = None;

        for attempt in 1..=self.max_attempts {
            match self.send(&event, attempt).await {
                Ok(()) => {
                    return WebhookDeliveryResult::Delivered {
                        event_id: event.event_id,
                        attempts: attempt,
                    };
                }
                Err(err) => {
                    last_error = Some(err.to_string());
                    if attempt < self.max_attempts && !self.initial_backoff.is_zero() {
                        tokio::time::sleep(backoff_delay(self.initial_backoff, attempt)).await;
                    }
                }
            }
        }

        let error = last_error.unwrap_or_else(|| "webhook delivery failed".to_string());
        let dead_letter = WebhookDeadLetter {
            event,
            attempts: self.max_attempts,
            failed_at: time::OffsetDateTime::now_utc(),
            error: error.clone(),
        };
        match append_jsonl(&self.dead_letter_jsonl, &dead_letter).await {
            Ok(()) => WebhookDeliveryResult::Failed {
                event_id: dead_letter.event.event_id,
                attempts: dead_letter.attempts,
                error,
            },
            Err(err) => WebhookDeliveryResult::DeadLetterFailed {
                event_id: dead_letter.event.event_id,
                attempts: dead_letter.attempts,
                error,
                dead_letter_error: err.to_string(),
            },
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct WebhookEvent {
    pub event_id: Uuid,
    pub event_type: &'static str,
    #[serde(with = "time::serde::rfc3339")]
    pub created_at: time::OffsetDateTime,
    pub job: JobRecord,
}

impl WebhookEvent {
    fn from_job(job: JobRecord) -> Self {
        Self {
            event_id: Uuid::new_v4(),
            event_type: "job.completed",
            created_at: time::OffsetDateTime::now_utc(),
            job,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct WebhookDeadLetter {
    pub event: WebhookEvent,
    pub attempts: usize,
    #[serde(with = "time::serde::rfc3339")]
    pub failed_at: time::OffsetDateTime,
    pub error: String,
}

enum WebhookDeliveryResult {
    Skipped,
    Delivered {
        event_id: Uuid,
        attempts: usize,
    },
    Failed {
        event_id: Uuid,
        attempts: usize,
        error: String,
    },
    DeadLetterFailed {
        event_id: Uuid,
        attempts: usize,
        error: String,
        dead_letter_error: String,
    },
}

pub async fn load_jobs(config: &Config) -> anyhow::Result<HashMap<Uuid, JobRecord>> {
    let mut jobs = HashMap::new();
    load_job_records(&config.submissions_jsonl, &mut jobs).await?;
    load_job_records(&config.results_jsonl, &mut jobs).await?;

    let mut recovered = Vec::new();
    for record in jobs.values_mut() {
        if matches!(record.status, JobStatus::Queued | JobStatus::Running) {
            record.status = JobStatus::Failed;
            record.updated_at = time::OffsetDateTime::now_utc();
            record.error = Some("server restarted before job reached a terminal state".to_string());
            recovered.push(record.clone());
        }
    }

    for record in recovered {
        append_jsonl(&config.results_jsonl, &record)
            .await
            .with_context(|| {
                format!(
                    "failed to append recovered job state to {}",
                    config.results_jsonl.display()
                )
            })?;
    }

    compact_metadata(config, &jobs).await?;
    retain_recent_jobs(&mut jobs, config.job_retention_limit);

    Ok(jobs)
}

async fn load_job_records(path: &Path, jobs: &mut HashMap<Uuid, JobRecord>) -> anyhow::Result<()> {
    let contents = match tokio::fs::read_to_string(path).await {
        Ok(contents) => contents,
        Err(err) if err.kind() == ErrorKind::NotFound => return Ok(()),
        Err(err) => {
            return Err(err).with_context(|| format!("failed to read {}", path.display()));
        }
    };

    for (line_number, line) in contents.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let record = serde_json::from_str::<JobRecord>(line).with_context(|| {
            format!(
                "failed to parse job record at {}:{}",
                path.display(),
                line_number + 1
            )
        })?;
        jobs.insert(record.id, record);
    }

    Ok(())
}

async fn compact_metadata(config: &Config, jobs: &HashMap<Uuid, JobRecord>) -> anyhow::Result<()> {
    if config.metadata_retention_limit == 0 {
        return Ok(());
    }

    let records = recent_records(jobs, config.metadata_retention_limit);
    let (submissions, results): (Vec<_>, Vec<_>) = records
        .into_iter()
        .partition(|record| matches!(record.status, JobStatus::Queued | JobStatus::Running));

    write_jsonl_records(&config.submissions_jsonl, &submissions).await?;
    write_jsonl_records(&config.results_jsonl, &results).await?;
    Ok(())
}

async fn write_jsonl_records(path: &Path, records: &[JobRecord]) -> anyhow::Result<()> {
    let temp_path = path.with_extension("jsonl.tmp");
    let mut bytes = Vec::with_capacity(records.len().saturating_mul(256));
    for record in records {
        serde_json::to_writer(&mut bytes, record)?;
        bytes.push(b'\n');
    }
    tokio::fs::write(&temp_path, bytes)
        .await
        .with_context(|| format!("failed to write {}", temp_path.display()))?;
    tokio::fs::rename(&temp_path, path)
        .await
        .with_context(|| format!("failed to replace {}", path.display()))?;
    Ok(())
}

fn retain_recent_jobs(jobs: &mut HashMap<Uuid, JobRecord>, limit: usize) {
    if jobs.len() <= limit {
        return;
    }
    if limit == 0 {
        jobs.clear();
        return;
    }

    let keep = recent_records(jobs, limit)
        .into_iter()
        .map(|record| record.id)
        .collect::<std::collections::HashSet<_>>();
    jobs.retain(|id, _| keep.contains(id));
}

fn recent_records(jobs: &HashMap<Uuid, JobRecord>, limit: usize) -> Vec<JobRecord> {
    let mut records = jobs.values().cloned().collect::<Vec<_>>();
    records.sort_by_key(|record| record.updated_at);
    records.reverse();
    records.truncate(limit);
    records.reverse();
    records
}

#[derive(Debug, thiserror::Error)]
pub enum EnqueueError {
    #[error("job record contains an invalid task specification")]
    InvalidTask(#[source] crate::types::TaskSpecError),
    #[error("inference queue is full")]
    QueueFull,
    #[error("inference queue is closed")]
    QueueClosed,
    #[error(transparent)]
    Persist(#[from] anyhow::Error),
}

pub async fn enqueue_record(
    state: &AppState,
    record: JobRecord,
) -> Result<QueueResponse, EnqueueError> {
    let id = record.id;
    let request = job_request_from_record(&record)?;

    if state.queue_tx.is_full() {
        return Err(EnqueueError::QueueFull);
    }

    {
        let mut jobs = state.jobs.write().await;
        jobs.insert(id, record.clone());
    }

    append_jsonl(&state.config.submissions_jsonl, &record)
        .await
        .map_err(EnqueueError::Persist)?;

    match state.queue_tx.try_send(request) {
        Ok(()) => {}
        Err(async_channel::TrySendError::Full(_)) => {
            rollback_unqueued_record(
                state,
                id,
                record,
                "inference queue became full before the job could be queued",
            )
            .await?;
            return Err(EnqueueError::QueueFull);
        }
        Err(async_channel::TrySendError::Closed(_)) => {
            rollback_unqueued_record(
                state,
                id,
                record,
                "inference queue closed before the job could be queued",
            )
            .await?;
            return Err(EnqueueError::QueueClosed);
        }
    }

    log_queued_job(&record, state.queue_tx.len());

    Ok(queue_response(id))
}

fn job_request_from_record(record: &JobRecord) -> Result<JobRequest, EnqueueError> {
    let task = TaskSpec::from_strings(
        Some(record.task_type.clone()),
        Some(record.task_prompt.clone()),
        record.text_input.clone(),
    )
    .map_err(EnqueueError::InvalidTask)?;
    Ok(JobRequest {
        id: record.id,
        image_path: record.image_path.clone(),
        task,
    })
}

fn log_queued_job(record: &JobRecord, queued: usize) {
    info!(
        "job queued job_id={} task_type={} task_prompt={} text_input_present={} input_kind={} image_path={} image_bytes={} sha256={} queued={}",
        record.id,
        record.task_type,
        record.task_prompt,
        record.text_input.is_some(),
        record.input_kind,
        record.image_path.display(),
        record.image_bytes,
        record.image_sha256,
        queued
    );
}

fn queue_response(id: Uuid) -> QueueResponse {
    QueueResponse {
        id,
        status: JobStatus::Queued,
        status_url: format!("/v1/jobs/{id}"),
    }
}

async fn rollback_unqueued_record(
    state: &AppState,
    id: Uuid,
    record: JobRecord,
    error: &str,
) -> Result<(), EnqueueError> {
    state.jobs.write().await.remove(&id);
    persist_unqueued_record(&state.config.results_jsonl, record, error).await
}

async fn persist_unqueued_record(
    results_jsonl: &Path,
    mut record: JobRecord,
    error: &str,
) -> Result<(), EnqueueError> {
    record.status = JobStatus::Failed;
    record.updated_at = time::OffsetDateTime::now_utc();
    record.error = Some(error.to_string());
    append_jsonl(results_jsonl, &record)
        .await
        .map_err(EnqueueError::Persist)
}

pub fn start_workers(
    config: Arc<Config>,
    jobs: Arc<RwLock<HashMap<Uuid, JobRecord>>>,
    workers: Arc<WorkerPoolState>,
    metrics: Arc<AppMetrics>,
    webhooks: Arc<WebhookClient>,
    queue_rx: Receiver<JobRequest>,
) {
    info!("starting model worker pool workers={}", config.workers);
    for worker_id in 0..config.workers {
        debug!("spawning model worker worker_id={}", worker_id);
        tokio::spawn(worker_loop(
            worker_id,
            WorkerRuntime {
                config: Arc::clone(&config),
                jobs: Arc::clone(&jobs),
                workers: Arc::clone(&workers),
                metrics: Arc::clone(&metrics),
                webhooks: Arc::clone(&webhooks),
                queue_rx: queue_rx.clone(),
            },
        ));
    }
}

struct WorkerRuntime {
    config: Arc<Config>,
    jobs: Arc<RwLock<HashMap<Uuid, JobRecord>>>,
    workers: Arc<WorkerPoolState>,
    metrics: Arc<AppMetrics>,
    webhooks: Arc<WebhookClient>,
    queue_rx: Receiver<JobRequest>,
}

async fn worker_loop(worker_id: usize, runtime: WorkerRuntime) {
    loop {
        let Some(mut worker) = initialize_worker(
            worker_id,
            Arc::clone(&runtime.config),
            &runtime.workers,
            &runtime.metrics,
        )
        .await
        else {
            return;
        };

        let should_restart = process_worker_queue(worker_id, &runtime, &mut worker).await;

        runtime.workers.mark_stopped();
        if !should_restart {
            return;
        }
    }
}

async fn process_worker_queue(
    worker_id: usize,
    runtime: &WorkerRuntime,
    worker: &mut FlorenceWorker,
) -> bool {
    while let Ok(request) = runtime.queue_rx.recv().await {
        if process_worker_request(worker_id, runtime, request, worker).await {
            return true;
        }
    }

    false
}

async fn process_worker_request(
    worker_id: usize,
    runtime: &WorkerRuntime,
    request: JobRequest,
    worker: &mut FlorenceWorker,
) -> bool {
    let job_span = tracing::info_span!(
        "inference_job",
        job_id = %request.id,
        worker_id,
        task_type = request.task.task_type_name(),
        task_prompt = request.task.task_prompt_name(),
    );
    job_span.in_scope(|| {
        debug!(
            "worker received job worker_id={} job_id={} task_type={} task_prompt={} text_input_present={} image_path={}",
            worker_id,
            request.id,
            request.task.task_type_name(),
            request.task.task_prompt_name(),
            request.task.text_input.is_some(),
            request.image_path.display()
        );
    });
    runtime.metrics.record_job_started();
    let job_started = Instant::now();
    mark_running(&runtime.jobs, request.id).await;

    let join_handle = spawn_inference(worker_id, worker, &request, job_span);
    let result = wait_for_inference(&runtime.config, join_handle).await;
    let elapsed_ms = job_started.elapsed().as_millis();

    handle_inference_result(runtime, request.id, elapsed_ms, result, worker).await
}

fn spawn_inference(
    worker_id: usize,
    worker: &mut FlorenceWorker,
    request: &JobRequest,
    job_span: tracing::Span,
) -> BlockingInferenceJoin {
    tokio::task::spawn_blocking({
        let image_path = request.image_path.clone();
        let task = request.task.clone();
        let mut worker = worker.take_for_blocking();
        move || {
            job_span.in_scope(|| {
                let result = worker.infer(&image_path, &task);
                (worker_id, worker, result)
            })
        }
    })
}

async fn handle_inference_result(
    runtime: &WorkerRuntime,
    job_id: Uuid,
    elapsed_ms: u128,
    result: InferenceRunResult,
    worker: &mut FlorenceWorker,
) -> bool {
    match result {
        InferenceRunResult::Completed(result) => {
            handle_completed_inference(runtime, job_id, elapsed_ms, *result, worker).await
        }
        InferenceRunResult::TimedOut => {
            runtime.metrics.record_job_timed_out(elapsed_ms);
            runtime.metrics.record_worker_restart();
            finish_job(
                &runtime.config,
                &runtime.jobs,
                &runtime.metrics,
                &runtime.webhooks,
                job_id,
                Err(anyhow!(
                    "job timed out after {} seconds",
                    runtime.config.job_timeout_seconds
                )),
            )
            .await;
            true
        }
    }
}

async fn handle_completed_inference(
    runtime: &WorkerRuntime,
    job_id: Uuid,
    elapsed_ms: u128,
    result: BlockingInferenceResult,
    worker: &mut FlorenceWorker,
) -> bool {
    match result {
        Ok((_, returned_worker, Ok(metadata))) => {
            *worker = returned_worker;
            runtime.metrics.record_job_succeeded(elapsed_ms);
            finish_job(
                &runtime.config,
                &runtime.jobs,
                &runtime.metrics,
                &runtime.webhooks,
                job_id,
                Ok(metadata),
            )
            .await;
            false
        }
        Ok((_, returned_worker, Err(err))) => {
            *worker = returned_worker;
            runtime.metrics.record_job_failed(elapsed_ms);
            finish_job(
                &runtime.config,
                &runtime.jobs,
                &runtime.metrics,
                &runtime.webhooks,
                job_id,
                Err(err),
            )
            .await;
            false
        }
        Err(err) => {
            runtime.metrics.record_job_failed(elapsed_ms);
            runtime.metrics.record_worker_restart();
            finish_job(
                &runtime.config,
                &runtime.jobs,
                &runtime.metrics,
                &runtime.webhooks,
                job_id,
                Err(anyhow!("worker task failed: {err}")),
            )
            .await;
            true
        }
    }
}

async fn initialize_worker(
    worker_id: usize,
    config: Arc<Config>,
    workers: &WorkerPoolState,
    metrics: &AppMetrics,
) -> Option<FlorenceWorker> {
    let started = Instant::now();
    let worker = tokio::task::spawn_blocking({
        let config = Arc::clone(&config);
        move || FlorenceWorker::new(worker_id, config)
    })
    .await;

    match worker {
        Ok(Ok(worker)) => {
            metrics.record_model_load(started.elapsed().as_millis());
            workers.mark_ready();
            info!("model worker ready worker_id={}", worker_id);
            Some(worker)
        }
        Ok(Err(err)) => {
            workers.mark_failed();
            error!(
                "failed to initialize model worker worker_id={} error={}",
                worker_id, err
            );
            None
        }
        Err(err) => {
            workers.mark_failed();
            error!(
                "model worker initialization panicked worker_id={} error={}",
                worker_id, err
            );
            None
        }
    }
}

type BlockingInferenceJoin =
    tokio::task::JoinHandle<(usize, FlorenceWorker, anyhow::Result<InferenceMetadata>)>;
type BlockingInferenceResult =
    Result<(usize, FlorenceWorker, anyhow::Result<InferenceMetadata>), tokio::task::JoinError>;

enum InferenceRunResult {
    Completed(Box<BlockingInferenceResult>),
    TimedOut,
}

async fn wait_for_inference(
    config: &Config,
    join_handle: BlockingInferenceJoin,
) -> InferenceRunResult {
    if config.job_timeout_seconds == 0 {
        return InferenceRunResult::Completed(Box::new(join_handle.await));
    }

    match tokio::time::timeout(Duration::from_secs(config.job_timeout_seconds), join_handle).await {
        Ok(result) => InferenceRunResult::Completed(Box::new(result)),
        Err(_) => InferenceRunResult::TimedOut,
    }
}

async fn mark_running(jobs: &RwLock<HashMap<Uuid, JobRecord>>, id: Uuid) {
    let mut jobs = jobs.write().await;
    if let Some(record) = jobs.get_mut(&id) {
        record.status = JobStatus::Running;
        record.updated_at = time::OffsetDateTime::now_utc();
        info!(
            "job running job_id={} image_path={}",
            id,
            record.image_path.display()
        );
    } else {
        warn!("job missing while marking running job_id={}", id);
    }
}

async fn finish_job(
    config: &Config,
    jobs: &RwLock<HashMap<Uuid, JobRecord>>,
    metrics: &AppMetrics,
    webhooks: &WebhookClient,
    id: Uuid,
    result: anyhow::Result<InferenceMetadata>,
) {
    let mut final_record = None;
    {
        let mut jobs = jobs.write().await;
        if let Some(record) = jobs.get_mut(&id) {
            record.updated_at = time::OffsetDateTime::now_utc();
            match result {
                Ok(metadata) => {
                    record.status = JobStatus::Succeeded;
                    record.result = Some(metadata);
                    record.error = None;
                    info!("job succeeded job_id={}", id);
                }
                Err(err) => {
                    record.status = JobStatus::Failed;
                    record.error = Some(err.to_string());
                    warn!("job failed job_id={} error={}", id, err);
                }
            }
            final_record = Some(record.clone());
        }
    }

    if let Some(record) = final_record {
        if let Err(err) = append_jsonl(&config.results_jsonl, &record).await {
            error!(
                "failed to append result metadata job_id={} path={} error={}",
                id,
                config.results_jsonl.display(),
                err
            );
        } else {
            debug!(
                "result metadata appended job_id={} path={}",
                id,
                config.results_jsonl.display()
            );
        }
        send_webhook(metrics, webhooks, &record).await;
        cleanup_job_artifacts(config, metrics, &record).await;
        {
            let mut jobs = jobs.write().await;
            retain_recent_jobs(&mut jobs, config.job_retention_limit);
        }
    } else {
        warn!("job missing while finishing job_id={}", id);
    }
}

async fn send_webhook(metrics: &AppMetrics, webhooks: &WebhookClient, record: &JobRecord) {
    let Some(webhook_url) = record.webhook_url.as_deref() else {
        return;
    };
    let redacted_url = redacted_webhook_url(webhook_url);

    match webhooks.deliver(record).await {
        WebhookDeliveryResult::Skipped => {}
        WebhookDeliveryResult::Delivered { event_id, attempts } => info!(
            "webhook delivered job_id={} event_id={} url={} attempts={}",
            record.id, event_id, redacted_url, attempts
        ),
        WebhookDeliveryResult::Failed {
            event_id,
            attempts,
            error,
        } => {
            metrics.record_webhook_failure();
            warn!(
                "webhook delivery failed job_id={} event_id={} url={} attempts={} error={}",
                record.id, event_id, redacted_url, attempts, error
            );
        }
        WebhookDeliveryResult::DeadLetterFailed {
            event_id,
            attempts,
            error,
            dead_letter_error,
        } => {
            metrics.record_webhook_failure();
            warn!(
                "webhook delivery failed and dead-letter append failed job_id={} event_id={} url={} attempts={} error={} dead_letter_error={}",
                record.id, event_id, redacted_url, attempts, error, dead_letter_error
            );
        }
    }
}

fn redacted_webhook_url(webhook_url: &str) -> String {
    let Ok(mut url) = reqwest::Url::parse(webhook_url) else {
        return "<invalid webhook url>".to_string();
    };
    url.set_query(None);
    url.set_fragment(None);
    if url.set_username("").is_err() {
        warn!("failed to redact webhook URL username");
        return "<redacted webhook url>".to_string();
    }
    if url.set_password(None).is_err() {
        warn!("failed to redact webhook URL password");
        return "<redacted webhook url>".to_string();
    }
    url.to_string()
}

async fn cleanup_job_artifacts(config: &Config, metrics: &AppMetrics, record: &JobRecord) {
    if !should_delete_image(config, record) {
        return;
    }

    match tokio::fs::remove_file(&record.image_path).await {
        Ok(()) => {
            info!(
                "uploaded image cleaned up job_id={} path={}",
                record.id,
                record.image_path.display()
            );
        }
        Err(err) if err.kind() == ErrorKind::NotFound => {
            debug!(
                "uploaded image already removed job_id={} path={}",
                record.id,
                record.image_path.display()
            );
        }
        Err(err) => {
            metrics.record_cleanup_failure();
            warn!(
                "failed to clean up uploaded image job_id={} path={} error={}",
                record.id,
                record.image_path.display(),
                err
            );
        }
    }
}

fn should_delete_image(config: &Config, record: &JobRecord) -> bool {
    record.input_kind == "upload" && path_is_inside(&record.image_path, &config.images_dir)
}

fn path_is_inside(path: &Path, directory: &Path) -> bool {
    path.starts_with(directory) && path != directory
}

fn backoff_delay(initial_backoff: Duration, attempt: usize) -> Duration {
    let multiplier = 1_u32.checked_shl((attempt - 1).min(16) as u32).unwrap_or(1);
    initial_backoff.saturating_mul(multiplier)
}

#[cfg(test)]
mod tests;
