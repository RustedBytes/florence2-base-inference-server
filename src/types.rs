use std::{fmt, path::PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use time::OffsetDateTime;
use uuid::Uuid;

use crate::config::ModelVariant;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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
    pub webhook_url: Option<String>,
    pub result: Option<InferenceMetadata>,
    pub error: Option<String>,
}

#[derive(Debug, Clone)]
pub struct TaskSpec {
    pub task_type: TaskType,
    pub task_prompt: TaskPrompt,
    pub text_input: Option<String>,
}

impl Default for TaskSpec {
    fn default() -> Self {
        Self {
            task_type: TaskType::Single,
            task_prompt: TaskPrompt::Caption,
            text_input: None,
        }
    }
}

impl TaskSpec {
    pub fn new(
        task_type: TaskType,
        task_prompt: TaskPrompt,
        text_input: Option<String>,
    ) -> Result<Self, TaskSpecError> {
        if !task_type.prompts().contains(&task_prompt) {
            return Err(TaskSpecError::UnsupportedTaskPrompt {
                task_type,
                task_prompt: task_prompt.as_str().to_string(),
            });
        }

        Ok(Self {
            task_type,
            task_prompt,
            text_input,
        })
    }

    pub fn from_strings(
        task_type: Option<String>,
        task_prompt: Option<String>,
        text_input: Option<String>,
    ) -> Result<Self, TaskSpecError> {
        let default = Self::default();
        let task_type = task_type
            .map(|value| TaskType::parse(&value))
            .transpose()?
            .unwrap_or(default.task_type);
        let task_prompt = task_prompt
            .map(|value| TaskPrompt::parse_for_type(task_type, &value))
            .transpose()?
            .unwrap_or(default.task_prompt);
        let text_input = text_input.and_then(normalize_optional_text);

        Self::new(task_type, task_prompt, text_input)
    }

    pub fn task_type_name(&self) -> &'static str {
        self.task_type.as_str()
    }

    pub fn task_prompt_name(&self) -> &'static str {
        self.task_prompt.as_str()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskType {
    Single,
    Cascaded,
}

impl TaskType {
    pub fn parse(value: &str) -> Result<Self, TaskSpecError> {
        match value.trim() {
            "Single task" => Ok(Self::Single),
            // The original Florence Space label is misspelled as "Cascased".
            // Accept the corrected spelling too, but keep the old wire value.
            "Cascased task" | "Cascaded task" => Ok(Self::Cascaded),
            other => Err(TaskSpecError::UnsupportedTaskType {
                task_type: other.to_string(),
            }),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Single => "Single task",
            Self::Cascaded => "Cascased task",
        }
    }

    pub fn prompts(self) -> &'static [TaskPrompt] {
        match self {
            Self::Single => SINGLE_TASK_PROMPTS,
            Self::Cascaded => CASCADED_TASK_PROMPTS,
        }
    }
}

impl fmt::Display for TaskType {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskPrompt {
    Caption,
    DetailedCaption,
    MoreDetailedCaption,
    ObjectDetection,
    DenseRegionCaption,
    RegionProposal,
    CaptionToPhraseGrounding,
    ReferringExpressionSegmentation,
    RegionToSegmentation,
    OpenVocabularyDetection,
    RegionToCategory,
    RegionToDescription,
    Ocr,
    OcrWithRegion,
    CaptionGrounding,
    DetailedCaptionGrounding,
    MoreDetailedCaptionGrounding,
}

impl TaskPrompt {
    pub fn parse_for_type(task_type: TaskType, value: &str) -> Result<Self, TaskSpecError> {
        let value = value.trim();
        task_type
            .prompts()
            .iter()
            .copied()
            .find(|prompt| prompt.as_str() == value)
            .ok_or_else(|| TaskSpecError::UnsupportedTaskPrompt {
                task_type,
                task_prompt: value.to_string(),
            })
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Caption => "Caption",
            Self::DetailedCaption => "Detailed Caption",
            Self::MoreDetailedCaption => "More Detailed Caption",
            Self::ObjectDetection => "Object Detection",
            Self::DenseRegionCaption => "Dense Region Caption",
            Self::RegionProposal => "Region Proposal",
            Self::CaptionToPhraseGrounding => "Caption to Phrase Grounding",
            Self::ReferringExpressionSegmentation => "Referring Expression Segmentation",
            Self::RegionToSegmentation => "Region to Segmentation",
            Self::OpenVocabularyDetection => "Open Vocabulary Detection",
            Self::RegionToCategory => "Region to Category",
            Self::RegionToDescription => "Region to Description",
            Self::Ocr => "OCR",
            Self::OcrWithRegion => "OCR with Region",
            Self::CaptionGrounding => "Caption + Grounding",
            Self::DetailedCaptionGrounding => "Detailed Caption + Grounding",
            Self::MoreDetailedCaptionGrounding => "More Detailed Caption + Grounding",
        }
    }

    pub fn single_task_token(self) -> Option<&'static str> {
        match self {
            Self::Caption => Some("<CAPTION>"),
            Self::DetailedCaption => Some("<DETAILED_CAPTION>"),
            Self::MoreDetailedCaption => Some("<MORE_DETAILED_CAPTION>"),
            Self::ObjectDetection => Some("<OD>"),
            Self::DenseRegionCaption => Some("<DENSE_REGION_CAPTION>"),
            Self::RegionProposal => Some("<REGION_PROPOSAL>"),
            Self::CaptionToPhraseGrounding => Some("<CAPTION_TO_PHRASE_GROUNDING>"),
            Self::ReferringExpressionSegmentation => Some("<REFERRING_EXPRESSION_SEGMENTATION>"),
            Self::RegionToSegmentation => Some("<REGION_TO_SEGMENTATION>"),
            Self::OpenVocabularyDetection => Some("<OPEN_VOCABULARY_DETECTION>"),
            Self::RegionToCategory => Some("<REGION_TO_CATEGORY>"),
            Self::RegionToDescription => Some("<REGION_TO_DESCRIPTION>"),
            Self::Ocr => Some("<OCR>"),
            Self::OcrWithRegion => Some("<OCR_WITH_REGION>"),
            Self::CaptionGrounding
            | Self::DetailedCaptionGrounding
            | Self::MoreDetailedCaptionGrounding => None,
        }
    }

    pub fn cascaded_caption_token(self) -> Option<&'static str> {
        match self {
            Self::CaptionGrounding => Some("<CAPTION>"),
            Self::DetailedCaptionGrounding => Some("<DETAILED_CAPTION>"),
            Self::MoreDetailedCaptionGrounding => Some("<MORE_DETAILED_CAPTION>"),
            _ => None,
        }
    }
}

impl fmt::Display for TaskPrompt {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

pub const SINGLE_TASK_PROMPTS: &[TaskPrompt] = &[
    TaskPrompt::Caption,
    TaskPrompt::DetailedCaption,
    TaskPrompt::MoreDetailedCaption,
    TaskPrompt::ObjectDetection,
    TaskPrompt::DenseRegionCaption,
    TaskPrompt::RegionProposal,
    TaskPrompt::CaptionToPhraseGrounding,
    TaskPrompt::ReferringExpressionSegmentation,
    TaskPrompt::RegionToSegmentation,
    TaskPrompt::OpenVocabularyDetection,
    TaskPrompt::RegionToCategory,
    TaskPrompt::RegionToDescription,
    TaskPrompt::Ocr,
    TaskPrompt::OcrWithRegion,
];

pub const CASCADED_TASK_PROMPTS: &[TaskPrompt] = &[
    TaskPrompt::CaptionGrounding,
    TaskPrompt::DetailedCaptionGrounding,
    TaskPrompt::MoreDetailedCaptionGrounding,
];

#[derive(Debug, thiserror::Error)]
pub enum TaskSpecError {
    #[error(
        "unsupported task_type `{task_type}`; expected `Single task`, `Cascased task`, or `Cascaded task`"
    )]
    UnsupportedTaskType { task_type: String },
    #[error("unsupported task_prompt `{task_prompt}` for task_type `{task_type}`")]
    UnsupportedTaskPrompt {
        task_type: TaskType,
        task_prompt: String,
    },
}

fn normalize_optional_text(text: String) -> Option<String> {
    let text = text.trim().to_string();
    (!text.is_empty()).then_some(text)
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
    pub ready: bool,
    pub workers: WorkerHealth,
    pub queued: usize,
    pub model_path: PathBuf,
    pub model_variant: ModelVariant,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub code: String,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadinessResponse {
    pub status: &'static str,
    pub ready: bool,
    pub workers: WorkerHealth,
    pub queued: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerHealth {
    pub expected: usize,
    pub ready: usize,
    pub failed: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsResponse {
    pub workers: WorkerHealth,
    pub queued: usize,
    pub http_requests_total: usize,
    pub http_requests_failed: usize,
    pub total_request_ms: u128,
    pub average_request_ms: Option<f64>,
    pub jobs_started: usize,
    pub jobs_succeeded: usize,
    pub jobs_failed: usize,
    pub jobs_timed_out: usize,
    pub worker_restarts: usize,
    pub total_inference_ms: u128,
    pub average_inference_ms: Option<f64>,
    pub model_load_ms: Option<u128>,
    pub webhook_failures: usize,
    pub cleanup_failures: usize,
    pub retained_jobs: usize,
    pub process_memory_rss_bytes: Option<u64>,
    pub gpu_memory: Option<GpuMemoryResponse>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuMemoryResponse {
    pub used_bytes: u64,
    pub total_bytes: u64,
}

#[cfg(test)]
mod tests;
