use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use time::OffsetDateTime;
use uuid::Uuid;

use crate::config::ModelVariant;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobStatus {
    Queued,
    Running,
    Succeeded,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobRecord {
    pub id: Uuid,
    pub status: JobStatus,
    #[serde(with = "time::serde::rfc3339")]
    pub created_at: OffsetDateTime,
    #[serde(with = "time::serde::rfc3339")]
    pub updated_at: OffsetDateTime,
    pub image_path: PathBuf,
    pub filename: Option<String>,
    pub content_type: Option<String>,
    pub image_sha256: String,
    pub image_bytes: usize,
    pub input_kind: String,
    pub source_path: Option<PathBuf>,
    pub task_type: String,
    pub task_prompt: String,
    pub text_input: Option<String>,
    pub result: Option<InferenceMetadata>,
    pub error: Option<String>,
}

#[derive(Debug, Clone)]
pub struct TaskSpec {
    pub task_type: String,
    pub task_prompt: String,
    pub text_input: Option<String>,
}

impl Default for TaskSpec {
    fn default() -> Self {
        Self {
            task_type: "Single task".to_string(),
            task_prompt: "Caption".to_string(),
            text_input: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceMetadata {
    pub backend: String,
    pub model_path: PathBuf,
    pub model_variant: ModelVariant,
    pub input_name: String,
    pub input_dtype: String,
    pub original_width: u32,
    pub original_height: u32,
    pub processed_width: u32,
    pub processed_height: u32,
    pub elapsed_ms: u128,
    pub task_token: String,
    pub prompt_text: String,
    pub generated_text: String,
    pub generated_tokens: usize,
    pub result: Value,
    pub generations: Vec<GenerationMetadata>,
    pub outputs: Vec<TensorMetadata>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationMetadata {
    pub task_token: String,
    pub prompt_text: String,
    pub generated_text: String,
    pub generated_tokens: usize,
    pub result: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorMetadata {
    pub name: String,
    pub shape: Vec<i64>,
    pub elements: usize,
    pub mean: Option<f32>,
    pub min: Option<f32>,
    pub max: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueResponse {
    pub id: Uuid,
    pub status: JobStatus,
    pub status_url: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: &'static str,
    pub workers: usize,
    pub queued: usize,
    pub model_path: PathBuf,
    pub model_variant: ModelVariant,
}
