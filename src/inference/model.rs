use std::path::{Path, PathBuf};

use anyhow::anyhow;
use log::warn;
use ort::{
    ep::{self, ExecutionProviderDispatch},
    session::{
        Session,
        builder::{AutoDevicePolicy, GraphOptimizationLevel},
    },
    value::{TensorElementType, ValueType},
};

use crate::config::ModelVariant;

#[derive(Debug, Clone)]
pub(super) struct FlorenceModelPaths {
    pub(super) model_dir: PathBuf,
    pub(super) vision_encoder: PathBuf,
    pub(super) embed_tokens: PathBuf,
    pub(super) encoder_model: PathBuf,
    pub(super) decoder_model: PathBuf,
    pub(super) decoder_model_merged: PathBuf,
}

impl FlorenceModelPaths {
    pub(super) fn from_vision_path(vision_encoder: &Path, variant: ModelVariant) -> Self {
        let onnx_dir = vision_encoder
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));
        let model_dir = onnx_dir
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| onnx_dir.clone());
        let suffix = model_suffix(vision_encoder, variant);

        Self {
            model_dir,
            vision_encoder: vision_encoder.to_path_buf(),
            embed_tokens: onnx_dir.join(format!("embed_tokens{suffix}.onnx")),
            encoder_model: onnx_dir.join(format!("encoder_model{suffix}.onnx")),
            decoder_model: onnx_dir.join(format!("decoder_model{suffix}.onnx")),
            decoder_model_merged: onnx_dir.join(format!("decoder_model_merged{suffix}.onnx")),
        }
    }

    pub(super) fn ensure_exists(&self) -> anyhow::Result<()> {
        for path in [
            &self.vision_encoder,
            &self.embed_tokens,
            &self.encoder_model,
            &self.decoder_model,
            &self.decoder_model_merged,
        ] {
            if !path.exists() {
                return Err(anyhow!("model file does not exist: {}", path.display()));
            }
        }
        if !self.model_dir.join("tokenizer.json").exists() {
            return Err(anyhow!(
                "tokenizer file does not exist: {}",
                self.model_dir.join("tokenizer.json").display()
            ));
        }
        Ok(())
    }
}

pub(super) fn load_session(path: &Path, execution_providers: &[String]) -> anyhow::Result<Session> {
    let requested_eps = execution_provider_dispatches(execution_providers);
    let use_auto_device = execution_providers
        .iter()
        .any(|provider| matches!(provider.as_str(), "auto" | "autodevice"));
    let cpu_only = execution_providers
        .iter()
        .all(|provider| provider.as_str() == "cpu");
    let has_requested_eps = !requested_eps.is_empty();

    // Keep explicit EP registration opt-in. CoreML can register on Apple
    // Silicon but still spend a long time compiling unsupported dynamic graphs.
    let mut builder = Session::builder()
        .map_err(|err| anyhow!("failed to create ONNX session builder: {err}"))?
        .with_optimization_level(GraphOptimizationLevel::All)
        .map_err(|err| anyhow!("failed to set ONNX graph optimization level: {err}"))?
        .with_prepacking(true)
        .map_err(|err| anyhow!("failed to enable ONNX prepacking: {err}"))?
        .with_memory_pattern(false)
        .map_err(|err| anyhow!("failed to configure ONNX memory pattern optimization: {err}"))?;

    if execution_providers
        .iter()
        .any(|provider| provider.as_str() == "cuda")
    {
        builder = builder
            .with_device_allocated_initializers()
            .map_err(|err| anyhow!("failed to enable ONNX device-allocated initializers: {err}"))?;
    }

    if has_requested_eps {
        builder = builder
            .with_execution_providers(&requested_eps)
            .map_err(|err| {
                anyhow!(
                    "failed to register ONNX execution providers for {}: {err}",
                    path.display()
                )
            })?;
    }

    if use_auto_device || (!has_requested_eps && !cpu_only) {
        builder = builder
            .with_auto_device(AutoDevicePolicy::MaxPerformance)
            .map_err(|err| anyhow!("failed to enable ONNX Runtime auto device selection: {err}"))?;
    }

    builder
        .commit_from_file(path)
        .map_err(|err| anyhow!("failed to load ONNX model {}: {err}", path.display()))
}

pub(super) fn image_input_metadata(
    session: &Session,
) -> anyhow::Result<(String, TensorElementType)> {
    let input = session
        .inputs()
        .iter()
        .find(|input| input.name() == "pixel_values")
        .or_else(|| session.inputs().first())
        .ok_or_else(|| anyhow!("ONNX model has no inputs"))?;

    let ValueType::Tensor { ty, .. } = input.dtype() else {
        return Err(anyhow!(
            "ONNX image input `{}` is not a tensor: {:?}",
            input.name(),
            input.dtype()
        ));
    };

    Ok((input.name().to_string(), *ty))
}

pub(super) fn execution_provider_dispatches(
    execution_providers: &[String],
) -> Vec<ExecutionProviderDispatch> {
    execution_providers
        .iter()
        .filter_map(|provider| execution_provider_dispatch(provider))
        .collect()
}

fn execution_provider_dispatch(provider: &str) -> Option<ExecutionProviderDispatch> {
    match provider {
        "coreml" => Some(coreml_execution_provider(
            ep::coreml::ComputeUnits::All,
            true,
        )),
        "coremlgpu" => Some(coreml_execution_provider(
            ep::coreml::ComputeUnits::CPUAndGPU,
            true,
        )),
        "coremlnpu" | "coremlane" | "ane" | "npu" => Some(coreml_execution_provider(
            ep::coreml::ComputeUnits::CPUAndNeuralEngine,
            false,
        )),
        "cuda" => cuda_execution_provider(),
        "xnnpack" => Some(ep::XNNPACK::default().build()),
        "auto" | "autodevice" | "cpu" => None,
        other => {
            warn!("unknown execution provider `{other}` ignored");
            None
        }
    }
}

fn coreml_execution_provider(
    compute_units: ep::coreml::ComputeUnits,
    low_precision_accumulation_on_gpu: bool,
) -> ExecutionProviderDispatch {
    ep::CoreML::default()
        .with_compute_units(compute_units)
        .with_model_format(ep::coreml::ModelFormat::MLProgram)
        .with_low_precision_accumulation_on_gpu(low_precision_accumulation_on_gpu)
        .build()
}

#[cfg(feature = "cuda")]
fn cuda_execution_provider() -> Option<ExecutionProviderDispatch> {
    Some(
        ep::CUDA::default()
            .with_tf32(true)
            .with_prefer_nhwc(true)
            .with_conv_max_workspace(true)
            .build(),
    )
}

#[cfg(not(feature = "cuda"))]
fn cuda_execution_provider() -> Option<ExecutionProviderDispatch> {
    warn!("CUDA execution provider requested but this binary was built without `--features cuda`");
    None
}

fn model_suffix(vision_encoder: &Path, variant: ModelVariant) -> String {
    if !matches!(variant, ModelVariant::Custom) {
        return match variant {
            ModelVariant::Fp32 => "",
            ModelVariant::Fp16 => "_fp16",
            ModelVariant::Int8 => "_int8",
            ModelVariant::Uint8 => "_uint8",
            ModelVariant::Quantized => "_quantized",
            ModelVariant::Q4 => "_q4",
            ModelVariant::Q4F16 => "_q4f16",
            ModelVariant::Bnb4 => "_bnb4",
            ModelVariant::Custom => unreachable!(),
        }
        .to_string();
    }

    vision_encoder
        .file_stem()
        .and_then(|stem| stem.to_str())
        .and_then(|stem| stem.strip_prefix("vision_encoder"))
        .unwrap_or("")
        .to_string()
}
