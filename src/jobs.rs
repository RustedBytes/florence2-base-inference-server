use std::{collections::HashMap, path::PathBuf, sync::Arc};

use anyhow::{Context, anyhow};
use async_channel::Receiver;
use log::{debug, error, info, warn};
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::{
    config::Config,
    inference::FlorenceWorker,
    state::AppState,
    types::{InferenceMetadata, JobRecord, JobStatus, QueueResponse, TaskSpec},
    util::append_jsonl,
};

#[derive(Debug, Clone)]
pub struct JobRequest {
    pub id: Uuid,
    pub image_path: PathBuf,
    pub task: TaskSpec,
}

pub async fn enqueue_record(state: &AppState, record: JobRecord) -> anyhow::Result<QueueResponse> {
    let id = record.id;
    let image_path = record.image_path.clone();
    let task = TaskSpec::from_strings(
        Some(record.task_type.clone()),
        Some(record.task_prompt.clone()),
        record.text_input.clone(),
    )
    .context("job record contains an invalid task specification")?;

    {
        let mut jobs = state.jobs.write().await;
        jobs.insert(id, record.clone());
    }
    append_jsonl(&state.config.submissions_jsonl, &record).await?;

    state
        .queue_tx
        .send(JobRequest {
            id,
            image_path,
            task,
        })
        .await
        .context("inference queue is closed")?;

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

pub fn start_workers(
    config: Arc<Config>,
    jobs: Arc<RwLock<HashMap<Uuid, JobRecord>>>,
    queue_rx: Receiver<JobRequest>,
) {
    info!("starting model worker pool workers={}", config.workers);
    for worker_id in 0..config.workers {
        debug!("spawning model worker worker_id={}", worker_id);
        let config = Arc::clone(&config);
        let jobs = Arc::clone(&jobs);
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
                    error!(
                        "failed to initialize model worker worker_id={} error={}",
                        worker_id, err
                    );
                    return;
                }
                Err(err) => {
                    error!(
                        "model worker initialization panicked worker_id={} error={}",
                        worker_id, err
                    );
                    return;
                }
            };

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
    } else {
        warn!("job missing while finishing job_id={}", id);
    }
}
