use std::{env, path::PathBuf};

use anyhow::anyhow;
use serde::{Deserialize, Serialize};

use super::defaults::DEFAULT_MODEL_PATH;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelVariant {
    Fp32,
    Fp16,
    Int8,
    Uint8,
    Quantized,
    Q4,
    Q4F16,
    Bnb4,
    Custom,
}

impl ModelVariant {
    pub fn default_model_path(self) -> PathBuf {
        if matches!(self, ModelVariant::Fp32 | ModelVariant::Custom) {
            return PathBuf::from(DEFAULT_MODEL_PATH);
        }

        let file_name = match self {
            ModelVariant::Fp16 => "vision_encoder_fp16.onnx",
            ModelVariant::Int8 => "vision_encoder_int8.onnx",
            ModelVariant::Uint8 => "vision_encoder_uint8.onnx",
            ModelVariant::Quantized => "vision_encoder_quantized.onnx",
            ModelVariant::Q4 => "vision_encoder_q4.onnx",
            ModelVariant::Q4F16 => "vision_encoder_q4f16.onnx",
            ModelVariant::Bnb4 => "vision_encoder_bnb4.onnx",
            ModelVariant::Fp32 | ModelVariant::Custom => unreachable!(),
        };

        PathBuf::from("Florence-2-base/onnx").join(file_name)
    }

    pub fn as_str(self) -> &'static str {
        match self {
            ModelVariant::Fp32 => "fp32",
            ModelVariant::Fp16 => "fp16",
            ModelVariant::Int8 => "int8",
            ModelVariant::Uint8 => "uint8",
            ModelVariant::Quantized => "quantized",
            ModelVariant::Q4 => "q4",
            ModelVariant::Q4F16 => "q4f16",
            ModelVariant::Bnb4 => "bnb4",
            ModelVariant::Custom => "custom",
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) enum ModelPathSelection {
    ExplicitPath,
    VariantDefaultPath,
}

pub(super) fn parse_model_variant(
    file_value: Option<String>,
    model_path_selection: ModelPathSelection,
) -> anyhow::Result<ModelVariant> {
    let Some(raw) = env::var("MODEL_VARIANT").ok().or(file_value) else {
        return Ok(match model_path_selection {
            ModelPathSelection::ExplicitPath => ModelVariant::Custom,
            ModelPathSelection::VariantDefaultPath => ModelVariant::Fp32,
        });
    };

    parse_model_variant_value(&raw)
}

pub(super) fn parse_model_variant_value(raw: &str) -> anyhow::Result<ModelVariant> {
    let normalized = raw.trim().to_ascii_lowercase().replace(['-', '_'], "");
    match normalized.as_str() {
        "fp32" | "float32" | "f32" => Ok(ModelVariant::Fp32),
        "fp16" | "float16" | "f16" => Ok(ModelVariant::Fp16),
        "int8" | "i8" => Ok(ModelVariant::Int8),
        "uint8" | "u8" => Ok(ModelVariant::Uint8),
        "quantized" | "quant" => Ok(ModelVariant::Quantized),
        "q4" => Ok(ModelVariant::Q4),
        "q4f16" | "q4fp16" => Ok(ModelVariant::Q4F16),
        "bnb4" | "bitsandbytes4" => Ok(ModelVariant::Bnb4),
        "custom" => Ok(ModelVariant::Custom),
        _ => Err(anyhow!(
            "unsupported model variant `{raw}`; expected one of fp32, fp16, int8, uint8, quantized, q4, q4f16, bnb4, custom"
        )),
    }
}
