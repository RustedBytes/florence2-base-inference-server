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
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::{
    config::Config,
    inference::FlorenceWorker,
    state::{AppMetrics, AppState, WorkerPoolState},
    types::{InferenceMetadata, JobRecord, JobStatus, QueueResponse, TaskSpec},
    util::append_jsonl,
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
}

impl WebhookClient {
    pub fn from_config(config: &Config) -> anyhow::Result<Self> {
        Self::new(
            config.webhook_timeout_seconds,
            config.webhook_connect_timeout_seconds,
        )
    }

    fn new(timeout_seconds: u64, connect_timeout_seconds: u64) -> anyhow::Result<Self> {
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

        Ok(Self { http })
    }

    async fn send(&self, record: &JobRecord) -> anyhow::Result<()> {
        let Some(webhook_url) = record.webhook_url.as_deref() else {
            return Ok(());
        };
        let redacted_url = redacted_webhook_url(webhook_url);

        let response = self
            .http
            .post(webhook_url)
            .json(record)
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
    let mut bytes = Vec::new();
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
    let image_path = record.image_path.clone();
    let task = TaskSpec::from_strings(
        Some(record.task_type.clone()),
        Some(record.task_prompt.clone()),
        record.text_input.clone(),
    )
    .map_err(EnqueueError::InvalidTask)?;
    let request = JobRequest {
        id,
        image_path,
        task,
    };

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
            state.jobs.write().await.remove(&id);
            persist_unqueued_record(
                &state.config.results_jsonl,
                record,
                "inference queue became full before the job could be queued",
            )
            .await?;
            return Err(EnqueueError::QueueFull);
        }
        Err(async_channel::TrySendError::Closed(_)) => {
            state.jobs.write().await.remove(&id);
            persist_unqueued_record(
                &state.config.results_jsonl,
                record,
                "inference queue closed before the job could be queued",
            )
            .await?;
            return Err(EnqueueError::QueueClosed);
        }
    }

    info!(
        "job queued job_id={} task_type={} task_prompt={} text_input_present={} input_kind={} image_path={} image_bytes={} sha256={} queued={}",
        id,
        record.task_type,
        record.task_prompt,
        record.text_input.is_some(),
        record.input_kind,
        record.image_path.display(),
        record.image_bytes,
        record.image_sha256,
        state.queue_tx.len()
    );

    Ok(QueueResponse {
        id,
        status: JobStatus::Queued,
        status_url: format!("/v1/jobs/{id}"),
    })
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
        let config = Arc::clone(&config);
        let jobs = Arc::clone(&jobs);
        let workers = Arc::clone(&workers);
        let metrics = Arc::clone(&metrics);
        let webhooks = Arc::clone(&webhooks);
        let queue_rx = queue_rx.clone();
        tokio::spawn(async move {
            loop {
                let Some(mut worker) =
                    initialize_worker(worker_id, Arc::clone(&config), &workers, &metrics).await
                else {
                    return;
                };

                let mut restart_worker = false;
                while let Ok(request) = queue_rx.recv().await {
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
                    metrics.record_job_started();
                    let job_started = Instant::now();
                    mark_running(&jobs, request.id).await;

                    let join_handle = tokio::task::spawn_blocking({
                        let image_path = request.image_path.clone();
                        let task = request.task.clone();
                        let mut worker = worker.take_for_blocking();
                        let job_span = job_span.clone();
                        move || {
                            job_span.in_scope(|| {
                                let result = worker.infer(&image_path, &task);
                                (worker_id, worker, result)
                            })
                        }
                    });

                    let result = wait_for_inference(&config, join_handle).await;
                    let elapsed_ms = job_started.elapsed().as_millis();

                    match result {
                        InferenceRunResult::Completed(result) => match *result {
                            Ok((_, returned_worker, Ok(metadata))) => {
                                worker = returned_worker;
                                metrics.record_job_succeeded(elapsed_ms);
                                finish_job(&config, &jobs, &webhooks, request.id, Ok(metadata))
                                    .await;
                            }
                            Ok((_, returned_worker, Err(err))) => {
                                worker = returned_worker;
                                metrics.record_job_failed(elapsed_ms);
                                finish_job(&config, &jobs, &webhooks, request.id, Err(err)).await;
                            }
                            Err(err) => {
                                metrics.record_job_failed(elapsed_ms);
                                metrics.record_worker_restart();
                                finish_job(
                                    &config,
                                    &jobs,
                                    &webhooks,
                                    request.id,
                                    Err(anyhow!("worker task failed: {err}")),
                                )
                                .await;
                                restart_worker = true;
                                break;
                            }
                        },
                        InferenceRunResult::TimedOut => {
                            metrics.record_job_timed_out(elapsed_ms);
                            metrics.record_worker_restart();
                            finish_job(
                                &config,
                                &jobs,
                                &webhooks,
                                request.id,
                                Err(anyhow!(
                                    "job timed out after {} seconds",
                                    config.job_timeout_seconds
                                )),
                            )
                            .await;
                            restart_worker = true;
                            break;
                        }
                    }
                }

                workers.mark_stopped();
                if !restart_worker {
                    return;
                }
            }
        });
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
        send_webhook(webhooks, &record).await;
        cleanup_job_artifacts(config, &record).await;
        {
            let mut jobs = jobs.write().await;
            retain_recent_jobs(&mut jobs, config.job_retention_limit);
        }
    } else {
        warn!("job missing while finishing job_id={}", id);
    }
}

async fn send_webhook(webhooks: &WebhookClient, record: &JobRecord) {
    let Some(webhook_url) = record.webhook_url.as_deref() else {
        return;
    };
    let redacted_url = redacted_webhook_url(webhook_url);

    match webhooks.send(record).await {
        Ok(()) => info!(
            "webhook delivered job_id={} url={}",
            record.id, redacted_url
        ),
        Err(err) => warn!(
            "webhook delivery failed job_id={} url={} error={}",
            record.id, redacted_url, err
        ),
    }
}

fn redacted_webhook_url(webhook_url: &str) -> String {
    let Ok(mut url) = reqwest::Url::parse(webhook_url) else {
        return "<invalid webhook url>".to_string();
    };
    url.set_query(None);
    url.set_fragment(None);
    let _ = url.set_username("");
    let _ = url.set_password(None);
    url.to_string()
}

async fn cleanup_job_artifacts(config: &Config, record: &JobRecord) {
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

#[cfg(test)]
mod tests {
    use std::net::SocketAddr;

    use async_channel::bounded;
    use time::OffsetDateTime;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    use super::*;
    use crate::{
        config::ModelVariant,
        state::{AppMetrics, WorkerPoolState},
    };

    fn test_config() -> Config {
        Config {
            addr: SocketAddr::from(([127, 0, 0, 1], 3000)),
            model_path: PathBuf::from("Florence-2-base/onnx/vision_encoder.onnx"),
            model_variant: ModelVariant::Fp32,
            data_dir: PathBuf::from("data"),
            images_dir: PathBuf::from("data/images"),
            metadata_dir: PathBuf::from("data/metadata"),
            submissions_jsonl: PathBuf::from("data/metadata/submissions.jsonl"),
            results_jsonl: PathBuf::from("data/metadata/results.jsonl"),
            allow_local_paths: false,
            local_path_roots: Vec::new(),
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
        }
    }

    fn job_record(input_kind: &str, image_path: PathBuf) -> JobRecord {
        let now = OffsetDateTime::now_utc();

        JobRecord {
            id: Uuid::nil(),
            status: JobStatus::Succeeded,
            created_at: now,
            updated_at: now,
            image_path,
            filename: None,
            content_type: None,
            image_sha256: String::new(),
            image_bytes: 0,
            input_kind: input_kind.to_string(),
            source_path: None,
            task_type: "Single task".to_string(),
            task_prompt: "Caption".to_string(),
            text_input: None,
            webhook_url: None,
            result: None,
            error: None,
        }
    }

    fn temp_config(name: &str) -> Config {
        let mut config = test_config();
        let dir = std::env::temp_dir().join(format!(
            "florence2-base-inference-server-{name}-{}",
            Uuid::new_v4()
        ));
        std::fs::create_dir_all(dir.join("metadata")).unwrap();
        config.data_dir = dir.clone();
        config.images_dir = dir.join("images");
        config.metadata_dir = dir.join("metadata");
        config.submissions_jsonl = config.metadata_dir.join("submissions.jsonl");
        config.results_jsonl = config.metadata_dir.join("results.jsonl");
        config
    }

    #[test]
    fn deletes_only_uploaded_images_under_images_dir() {
        let config = test_config();

        assert!(should_delete_image(
            &config,
            &job_record("upload", PathBuf::from("data/images/job.png"))
        ));
        assert!(!should_delete_image(
            &config,
            &job_record("local_path", PathBuf::from("data/images/job.png"))
        ));
        assert!(!should_delete_image(
            &config,
            &job_record("upload", PathBuf::from("/tmp/source.png"))
        ));
        assert!(!should_delete_image(
            &config,
            &job_record("upload", PathBuf::from("data/images"))
        ));
    }

    #[tokio::test]
    async fn load_jobs_recovers_non_terminal_jobs_as_failed() {
        let config = temp_config("recover");
        let mut record = job_record("upload", config.images_dir.join("job.png"));
        record.status = JobStatus::Running;

        append_jsonl(&config.submissions_jsonl, &record)
            .await
            .unwrap();

        let jobs = load_jobs(&config).await.unwrap();
        let recovered = jobs.get(&record.id).unwrap();

        assert_eq!(recovered.status, JobStatus::Failed);
        assert_eq!(
            recovered.error.as_deref(),
            Some("server restarted before job reached a terminal state")
        );
        assert!(
            std::fs::read_to_string(&config.results_jsonl)
                .unwrap()
                .contains("server restarted before job reached a terminal state")
        );

        std::fs::remove_dir_all(&config.data_dir).unwrap();
    }

    #[tokio::test]
    async fn load_jobs_applies_retention_and_compacts_metadata() {
        let mut config = temp_config("retention");
        config.job_retention_limit = 2;
        config.metadata_retention_limit = 2;

        for minutes in 0..3 {
            let mut record = job_record("upload", config.images_dir.join(format!("{minutes}.png")));
            record.id = Uuid::new_v4();
            record.status = JobStatus::Succeeded;
            record.updated_at += time::Duration::minutes(minutes);
            append_jsonl(&config.results_jsonl, &record).await.unwrap();
        }

        let jobs = load_jobs(&config).await.unwrap();
        let compacted = std::fs::read_to_string(&config.results_jsonl).unwrap();

        assert_eq!(jobs.len(), 2);
        assert_eq!(compacted.lines().count(), 2);

        std::fs::remove_dir_all(&config.data_dir).unwrap();
    }

    #[tokio::test]
    async fn enqueue_record_rejects_full_queue_without_persisting() {
        let config = Arc::new(temp_config("queue-full"));
        let (queue_tx, queue_rx) = bounded(1);
        queue_tx
            .try_send(JobRequest {
                id: Uuid::new_v4(),
                image_path: PathBuf::from("busy.png"),
                task: TaskSpec::default(),
            })
            .unwrap();
        let state = AppState {
            config: Arc::clone(&config),
            queue_tx,
            jobs: Arc::new(RwLock::new(HashMap::new())),
            workers: Arc::new(WorkerPoolState::new(1)),
            metrics: Arc::new(AppMetrics::default()),
        };
        let record = job_record("upload", config.images_dir.join("job.png"));
        let err = enqueue_record(&state, record.clone()).await.unwrap_err();

        assert!(matches!(err, EnqueueError::QueueFull));
        assert!(state.jobs.read().await.get(&record.id).is_none());
        assert!(!config.submissions_jsonl.exists());

        std::fs::remove_dir_all(&config.data_dir).unwrap();
        drop(queue_rx);
    }

    #[tokio::test]
    async fn post_webhook_sends_final_job_record() {
        let (url, server) = spawn_webhook_receiver().await;
        let mut record = job_record("upload", PathBuf::from("data/images/job.png"));
        record.id = Uuid::new_v4();
        record.webhook_url = Some(url.clone());

        WebhookClient::new(10, 5)
            .unwrap()
            .send(&record)
            .await
            .unwrap();

        let request = server.await.unwrap();
        let request = String::from_utf8(request).unwrap();
        let (_, body) = request.split_once("\r\n\r\n").unwrap();
        let body = serde_json::from_str::<serde_json::Value>(body).unwrap();

        assert!(request.starts_with("POST /hook HTTP/1.1"));
        assert_eq!(body["id"], record.id.to_string());
        assert_eq!(body["status"], "succeeded");
        assert_eq!(body["webhook_url"], url);
    }

    #[test]
    fn redacted_webhook_url_removes_sensitive_parts() {
        let redacted =
            redacted_webhook_url("https://user:secret@example.com/hook?token=secret#frag");

        assert_eq!(redacted, "https://example.com/hook");
    }

    #[tokio::test]
    async fn finish_job_delivers_webhook_with_terminal_record() {
        let config = temp_config("finish-webhook");
        let (url, server) = spawn_webhook_receiver().await;
        let mut record = job_record("upload", config.images_dir.join("job.png"));
        record.id = Uuid::new_v4();
        record.status = JobStatus::Running;
        record.webhook_url = Some(url);
        let id = record.id;
        let jobs = RwLock::new(HashMap::from([(id, record)]));

        let webhooks = WebhookClient::new(10, 5).unwrap();
        finish_job(
            &config,
            &jobs,
            &webhooks,
            id,
            Err(anyhow!("inference failed")),
        )
        .await;

        let request = server.await.unwrap();
        let request = String::from_utf8(request).unwrap();
        let (_, body) = request.split_once("\r\n\r\n").unwrap();
        let body = serde_json::from_str::<serde_json::Value>(body).unwrap();

        assert_eq!(body["id"], id.to_string());
        assert_eq!(body["status"], "failed");
        assert_eq!(body["error"], "inference failed");

        std::fs::remove_dir_all(&config.data_dir).unwrap();
    }

    async fn spawn_webhook_receiver() -> (String, tokio::task::JoinHandle<Vec<u8>>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/hook", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut request = Vec::new();
            let mut buffer = [0_u8; 1024];

            loop {
                let n = stream.read(&mut buffer).await.unwrap();
                assert!(n > 0, "client closed before sending full request");
                request.extend_from_slice(&buffer[..n]);

                if request_is_complete(&request) {
                    break;
                }
            }

            stream
                .write_all(b"HTTP/1.1 204 No Content\r\nContent-Length: 0\r\n\r\n")
                .await
                .unwrap();
            request
        });

        (url, server)
    }

    fn request_is_complete(request: &[u8]) -> bool {
        let Some(header_end) = request
            .windows(4)
            .position(|window| window == b"\r\n\r\n")
            .map(|position| position + 4)
        else {
            return false;
        };

        let headers = String::from_utf8_lossy(&request[..header_end]);
        let content_length = headers
            .lines()
            .find_map(|line| line.split_once(':'))
            .filter(|(name, _)| name.eq_ignore_ascii_case("content-length"))
            .and_then(|(_, value)| value.trim().parse::<usize>().ok())
            .unwrap_or(0);

        request.len() >= header_end + content_length
    }
}
