use std::{
    collections::HashMap,
    io::ErrorKind,
    path::{Path, PathBuf},
    sync::Arc,
};

use anyhow::{Context, anyhow};
use async_channel::Receiver;
use log::{debug, error, info, warn};
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::{
    config::Config,
    inference::FlorenceWorker,
    state::{AppState, WorkerPoolState},
    types::{InferenceMetadata, JobRecord, JobStatus, QueueResponse, TaskSpec},
    util::append_jsonl,
};

#[derive(Debug, Clone)]
pub struct JobRequest {
    pub id: Uuid,
    pub image_path: PathBuf,
    pub task: TaskSpec,
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
    queue_rx: Receiver<JobRequest>,
) {
    info!("starting model worker pool workers={}", config.workers);
    for worker_id in 0..config.workers {
        debug!("spawning model worker worker_id={}", worker_id);
        let config = Arc::clone(&config);
        let jobs = Arc::clone(&jobs);
        let workers = Arc::clone(&workers);
        let queue_rx = queue_rx.clone();
        tokio::spawn(async move {
            let worker = tokio::task::spawn_blocking({
                let config = Arc::clone(&config);
                move || FlorenceWorker::new(worker_id, config)
            })
            .await;

            let mut worker = match worker {
                Ok(Ok(worker)) => worker,
                Ok(Err(err)) => {
                    workers.mark_failed();
                    error!(
                        "failed to initialize model worker worker_id={} error={}",
                        worker_id, err
                    );
                    return;
                }
                Err(err) => {
                    workers.mark_failed();
                    error!(
                        "model worker initialization panicked worker_id={} error={}",
                        worker_id, err
                    );
                    return;
                }
            };

            workers.mark_ready();
            info!("model worker ready worker_id={}", worker_id);
            while let Ok(request) = queue_rx.recv().await {
                debug!(
                    "worker received job worker_id={} job_id={} task_type={} task_prompt={} text_input_present={} image_path={}",
                    worker_id,
                    request.id,
                    request.task.task_type_name(),
                    request.task.task_prompt_name(),
                    request.task.text_input.is_some(),
                    request.image_path.display()
                );
                mark_running(&jobs, request.id).await;

                let result = tokio::task::spawn_blocking({
                    let image_path = request.image_path.clone();
                    let task = request.task.clone();
                    let worker_id = worker_id;
                    // ORT inference is CPU/GPU-bound and may block. Move the
                    // worker into a blocking task, then return it to this loop.
                    let mut worker = worker.take_for_blocking();
                    move || {
                        let result = worker.infer(&image_path, &task);
                        (worker_id, worker, result)
                    }
                })
                .await;

                match result {
                    Ok((_, returned_worker, Ok(metadata))) => {
                        worker = returned_worker;
                        finish_job(&config, &jobs, request.id, Ok(metadata)).await;
                    }
                    Ok((_, returned_worker, Err(err))) => {
                        worker = returned_worker;
                        finish_job(&config, &jobs, request.id, Err(err)).await;
                    }
                    Err(err) => {
                        finish_job(
                            &config,
                            &jobs,
                            request.id,
                            Err(anyhow!("worker task failed: {err}")),
                        )
                        .await;
                        break;
                    }
                }
            }
            workers.mark_stopped();
        });
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
        cleanup_job_artifacts(config, &record).await;
    } else {
        warn!("job missing while finishing job_id={}", id);
    }
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

    use super::*;
    use crate::{config::ModelVariant, state::WorkerPoolState};

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
            workers: 1,
            queue_size: 1,
            body_limit_bytes: 1024,
            rust_log: "info".to_string(),
            max_new_tokens: 1,
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
        };
        let record = job_record("upload", config.images_dir.join("job.png"));
        let err = enqueue_record(&state, record.clone()).await.unwrap_err();

        assert!(matches!(err, EnqueueError::QueueFull));
        assert!(state.jobs.read().await.get(&record.id).is_none());
        assert!(!config.submissions_jsonl.exists());

        std::fs::remove_dir_all(&config.data_dir).unwrap();
        drop(queue_rx);
    }
}
