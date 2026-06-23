use anyhow::anyhow;
use half::f16;
use ort::value::{DynValue, Shape, TensorElementType, ValueType};

use crate::types::TensorMetadata;

#[derive(Debug, Clone)]
pub(super) struct TensorData {
    pub(super) shape: Vec<i64>,
    pub(super) data: Vec<f32>,
}

impl TensorData {
    pub(super) fn seq_len(&self) -> anyhow::Result<usize> {
        let seq_len =
            self.shape.get(1).copied().ok_or_else(|| {
                anyhow!("tensor shape {:?} has no sequence dimension", self.shape)
            })?;
        usize::try_from(seq_len)
            .map_err(|_| anyhow!("tensor sequence length is invalid: {seq_len}"))
    }
}

pub(super) fn extract_output_tensor(value: &DynValue, name: &str) -> anyhow::Result<TensorData> {
    match value.dtype() {
        ValueType::Tensor {
            ty: TensorElementType::Float32,
            ..
        } => {
            let (shape, data) = value
                .try_extract_tensor::<f32>()
                .map_err(|err| anyhow!("output `{name}` is not an f32 tensor: {err}"))?;
            Ok(TensorData {
                shape: shape.iter().copied().collect(),
                data: data.to_vec(),
            })
        }
        ValueType::Tensor {
            ty: TensorElementType::Float16,
            ..
        } => {
            let (shape, data) = value
                .try_extract_tensor::<f16>()
                .map_err(|err| anyhow!("output `{name}` is not an f16 tensor: {err}"))?;
            Ok(TensorData {
                shape: shape.iter().copied().collect(),
                data: data.iter().map(|value| value.to_f32()).collect(),
            })
        }
        other => Err(anyhow!("output `{name}` has unsupported dtype: {other:?}")),
    }
}

pub(super) fn concat_embeddings(
    image_features: &TensorData,
    text_embeds: &TensorData,
    hidden_size: usize,
) -> anyhow::Result<TensorData> {
    let image_seq_len = image_features.seq_len()?;
    let text_seq_len = text_embeds.seq_len()?;
    let image_values = image_seq_len
        .checked_mul(hidden_size)
        .ok_or_else(|| anyhow!("image embedding shape is too large"))?;
    let text_values = text_seq_len
        .checked_mul(hidden_size)
        .ok_or_else(|| anyhow!("text embedding shape is too large"))?;
    if image_features.data.len() != image_values {
        return Err(anyhow!(
            "image feature data length {} does not match shape {:?}",
            image_features.data.len(),
            image_features.shape
        ));
    }
    if text_embeds.data.len() != text_values {
        return Err(anyhow!(
            "text embedding data length {} does not match shape {:?}",
            text_embeds.data.len(),
            text_embeds.shape
        ));
    }

    let mut data = Vec::with_capacity(image_features.data.len() + text_embeds.data.len());
    data.extend_from_slice(&image_features.data);
    data.extend_from_slice(&text_embeds.data);
    Ok(TensorData {
        shape: vec![1, (image_seq_len + text_seq_len) as i64, hidden_size as i64],
        data,
    })
}

pub(super) fn argmax_last_token(logits: &TensorData) -> anyhow::Result<i64> {
    if logits.shape.len() != 3 {
        return Err(anyhow!(
            "logits tensor has invalid shape {:?}",
            logits.shape
        ));
    }
    let seq_len = usize::try_from(logits.shape[1])
        .map_err(|_| anyhow!("logits sequence length is invalid: {}", logits.shape[1]))?;
    let vocab_size = usize::try_from(logits.shape[2])
        .map_err(|_| anyhow!("logits vocabulary size is invalid: {}", logits.shape[2]))?;
    if seq_len == 0 || vocab_size == 0 {
        return Err(anyhow!(
            "logits tensor has empty sequence or vocabulary dimension"
        ));
    }
    let start = (seq_len - 1)
        .checked_mul(vocab_size)
        .ok_or_else(|| anyhow!("logits shape is too large"))?;
    let end = start
        .checked_add(vocab_size)
        .ok_or_else(|| anyhow!("logits shape is too large"))?;
    let row = logits
        .data
        .get(start..end)
        .ok_or_else(|| anyhow!("logits data length does not match shape {:?}", logits.shape))?;
    let (idx, _) = row
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .ok_or_else(|| anyhow!("failed to select next token from logits"))?;
    Ok(idx as i64)
}

pub(super) fn tensor_metadata_f32(name: &str, shape: &Shape, data: &[f32]) -> TensorMetadata {
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut sum = 0.0_f64;

    for &value in data {
        min = min.min(value);
        max = max.max(value);
        sum += value as f64;
    }

    TensorMetadata {
        name: name.to_string(),
        shape: shape.iter().copied().collect(),
        elements: data.len(),
        mean: (!data.is_empty()).then_some((sum / data.len() as f64) as f32),
        min: (!data.is_empty()).then_some(min),
        max: (!data.is_empty()).then_some(max),
    }
}
