use std::{
    collections::HashMap,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use async_channel::Sender;
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::{config::Config, jobs::JobRequest, types::JobRecord};

#[derive(Clone)]
pub struct AppState {
    pub config: Arc<Config>,
    pub queue_tx: Sender<JobRequest>,
    pub jobs: Arc<RwLock<HashMap<Uuid, JobRecord>>>,
    pub workers: Arc<WorkerPoolState>,
    pub metrics: Arc<AppMetrics>,
}

#[derive(Debug)]
pub struct WorkerPoolState {
    expected: usize,
    ready: AtomicUsize,
    failed: AtomicUsize,
}

#[derive(Debug, Default)]
pub struct AppMetrics {
    jobs_started: AtomicUsize,
    jobs_succeeded: AtomicUsize,
    jobs_failed: AtomicUsize,
    jobs_timed_out: AtomicUsize,
    worker_restarts: AtomicUsize,
    total_inference_ms: AtomicUsize,
    model_load_ms: AtomicUsize,
}

#[derive(Debug, Clone, Copy)]
pub struct MetricsSnapshot {
    pub jobs_started: usize,
    pub jobs_succeeded: usize,
    pub jobs_failed: usize,
    pub jobs_timed_out: usize,
    pub worker_restarts: usize,
    pub total_inference_ms: usize,
    pub model_load_ms: Option<usize>,
}

#[derive(Debug, Clone, Copy)]
pub struct WorkerPoolSnapshot {
    pub expected: usize,
    pub ready: usize,
    pub failed: usize,
}

impl WorkerPoolState {
    pub fn new(expected: usize) -> Self {
        Self {
            expected,
            ready: AtomicUsize::new(0),
            failed: AtomicUsize::new(0),
        }
    }

    pub fn mark_ready(&self) {
        self.ready.fetch_add(1, Ordering::Relaxed);
    }

    pub fn mark_failed(&self) {
        self.failed.fetch_add(1, Ordering::Relaxed);
    }

    pub fn mark_stopped(&self) {
        let _ = self
            .ready
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |ready| {
                ready.checked_sub(1)
            });
    }

    pub fn snapshot(&self) -> WorkerPoolSnapshot {
        WorkerPoolSnapshot {
            expected: self.expected,
            ready: self.ready.load(Ordering::Relaxed),
            failed: self.failed.load(Ordering::Relaxed),
        }
    }

    pub fn is_ready(&self) -> bool {
        self.snapshot().ready > 0
    }
}

impl AppMetrics {
    pub fn record_job_started(&self) {
        self.jobs_started.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_job_succeeded(&self, elapsed_ms: u128) {
        self.jobs_succeeded.fetch_add(1, Ordering::Relaxed);
        self.total_inference_ms.fetch_add(
            usize::try_from(elapsed_ms).unwrap_or(usize::MAX),
            Ordering::Relaxed,
        );
    }

    pub fn record_job_failed(&self, elapsed_ms: u128) {
        self.jobs_failed.fetch_add(1, Ordering::Relaxed);
        self.total_inference_ms.fetch_add(
            usize::try_from(elapsed_ms).unwrap_or(usize::MAX),
            Ordering::Relaxed,
        );
    }

    pub fn record_job_timed_out(&self, elapsed_ms: u128) {
        self.jobs_timed_out.fetch_add(1, Ordering::Relaxed);
        self.record_job_failed(elapsed_ms);
    }

    pub fn record_worker_restart(&self) {
        self.worker_restarts.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_model_load(&self, elapsed_ms: u128) {
        self.model_load_ms.store(
            usize::try_from(elapsed_ms).unwrap_or(usize::MAX),
            Ordering::Relaxed,
        );
    }

    pub fn snapshot(&self) -> MetricsSnapshot {
        let model_load_ms = self.model_load_ms.load(Ordering::Relaxed);
        MetricsSnapshot {
            jobs_started: self.jobs_started.load(Ordering::Relaxed),
            jobs_succeeded: self.jobs_succeeded.load(Ordering::Relaxed),
            jobs_failed: self.jobs_failed.load(Ordering::Relaxed),
            jobs_timed_out: self.jobs_timed_out.load(Ordering::Relaxed),
            worker_restarts: self.worker_restarts.load(Ordering::Relaxed),
            total_inference_ms: self.total_inference_ms.load(Ordering::Relaxed),
            model_load_ms: (model_load_ms > 0).then_some(model_load_ms),
        }
    }
}
