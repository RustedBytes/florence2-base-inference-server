mod inference;
mod job;
mod responses;
mod task;

pub use inference::{GenerationMetadata, InferenceMetadata, TensorMetadata};
pub use job::{JobRecord, JobStatus};
pub use responses::{
    ErrorResponse, GpuMemoryResponse, HealthResponse, MetricsResponse, QueueResponse,
    ReadinessResponse, WorkerHealth,
};
pub use task::{
    CASCADED_TASK_PROMPTS, SINGLE_TASK_PROMPTS, TaskPrompt, TaskSpec, TaskSpecError, TaskType,
};

#[cfg(test)]
mod tests;
