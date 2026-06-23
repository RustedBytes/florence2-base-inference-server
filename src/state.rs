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
}

#[derive(Debug)]
pub struct WorkerPoolState {
    expected: usize,
    ready: AtomicUsize,
    failed: AtomicUsize,
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
