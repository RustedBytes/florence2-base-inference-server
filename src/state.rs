use std::{collections::HashMap, sync::Arc};

use async_channel::Sender;
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::{config::Config, jobs::JobRequest, types::JobRecord};

#[derive(Clone)]
pub struct AppState {
    pub config: Arc<Config>,
    pub queue_tx: Sender<JobRequest>,
    pub jobs: Arc<RwLock<HashMap<Uuid, JobRecord>>>,
}
