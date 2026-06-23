use std::{
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};

use anyhow::{Context, anyhow};
use half::f16;
use image::{DynamicImage, GenericImageView, ImageReader, imageops::FilterType};
use log::{debug, info, trace, warn};
use ort::{
    environment::Environment,
    session::{
        Session,
        builder::{AutoDevicePolicy, GraphOptimizationLevel},
    },
    value::{DynValue, Shape, Tensor, TensorElementType, ValueType},
};
use serde_json::{Value, json};
use tokenizers::Tokenizer;

use crate::{
    config::{Config, ModelVariant},
    types::{GenerationMetadata, InferenceMetadata, TaskSpec, TensorMetadata},
};

const IMAGE_SIDE: u32 = 768;
const HIDDEN_SIZE: i64 = 768;
const DECODER_START_TOKEN_ID: i64 = 2;
const EOS_TOKEN_ID: i64 = 2;

pub struct FlorenceWorker {
    id: usize,
    model_paths: FlorenceModelPaths,
    model_variant: ModelVariant,
    vision_encoder: Option<Session>,
    embed_tokens: Option<Session>,
    encoder_model: Option<Session>,
    decoder_model: Option<Session>,
    decoder_with_past_model: Option<Session>,
    tokenizer: Tokenizer,
    backend: String,
    input_name: String,
    input_dtype: TensorElementType,
    max_new_tokens: usize,
}

impl FlorenceWorker {
    pub fn new(id: usize, config: Arc<Config>) -> anyhow::Result<Self> {
        let started = Instant::now();
        let model_paths =
            FlorenceModelPaths::from_vision_path(&config.model_path, config.model_variant);
        model_paths.ensure_exists()?;

        info!(
            "initializing model worker worker_id={} vision_encoder={} embed_tokens={} encoder_model={} decoder_model={} decoder_with_past_model={}",
            id,
            model_paths.vision_encoder.display(),
            model_paths.embed_tokens.display(),
            model_paths.encoder_model.display(),
            model_paths.decoder_model.display(),
            model_paths.decoder_with_past_model.display()
        );

        let env =
            Environment::current().context("failed to initialize ONNX Runtime environment")?;
        let devices = env
            .devices()
            .map(|device| {
                let ep = device.ep().unwrap_or("unknown").to_string();
                let ty = format!("{:?}", device.ty());
                format!("{ep}:{ty}")
            })
            .collect::<Vec<_>>();

        if devices.is_empty() {
            warn!("ONNX Runtime did not report accelerator devices; CPU fallback will be used");
        } else {
            info!(
                "ONNX Runtime detected devices worker_id={} devices={:?}",
                id, devices
            );
        }

        let session_started = Instant::now();
        let vision_encoder = load_session(&model_paths.vision_encoder)?;
        let embed_tokens = load_session(&model_paths.embed_tokens)?;
        let encoder_model = load_session(&model_paths.encoder_model)?;
        let decoder_model = load_session(&model_paths.decoder_model)?;
        let decoder_with_past_model = load_session(&model_paths.decoder_with_past_model)?;
        let (input_name, input_dtype) = image_input_metadata(&vision_encoder)?;
        debug!(
            "ONNX image input detected worker_id={} input_name={} input_dtype={}",
            id, input_name, input_dtype
        );
        info!(
            "ONNX sessions loaded worker_id={} elapsed_ms={}",
            id,
            session_started.elapsed().as_millis()
        );

        let tokenizer_path = model_paths.model_dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|err| {
            anyhow!(
                "failed to load tokenizer from {}: {err}",
                tokenizer_path.display()
            )
        })?;

        let worker = Self {
            id,
            model_paths,
            model_variant: config.model_variant,
            vision_encoder: Some(vision_encoder),
            embed_tokens: Some(embed_tokens),
            encoder_model: Some(encoder_model),
            decoder_model: Some(decoder_model),
            decoder_with_past_model: Some(decoder_with_past_model),
            tokenizer,
            backend: format!("ort:auto:max_performance:{}", devices.join(",")),
            input_name,
            input_dtype,
            max_new_tokens: config.max_new_tokens,
        };

        info!(
            "model worker initialized worker_id={} backend={} max_new_tokens={} elapsed_ms={}",
            id,
            worker.backend,
            worker.max_new_tokens,
            started.elapsed().as_millis()
        );

        Ok(worker)
    }

    pub fn take_for_blocking(&mut self) -> Self {
        Self {
            id: self.id,
            model_paths: self.model_paths.clone(),
            model_variant: self.model_variant,
            vision_encoder: self.vision_encoder.take(),
            embed_tokens: self.embed_tokens.take(),
            encoder_model: self.encoder_model.take(),
            decoder_model: self.decoder_model.take(),
            decoder_with_past_model: self.decoder_with_past_model.take(),
            tokenizer: self.tokenizer.clone(),
            backend: self.backend.clone(),
            input_name: self.input_name.clone(),
            input_dtype: self.input_dtype,
            max_new_tokens: self.max_new_tokens,
        }
    }

    pub fn infer(
        &mut self,
        image_path: &Path,
        task: &TaskSpec,
    ) -> anyhow::Result<InferenceMetadata> {
        let total_started = Instant::now();
        info!(
            "inference started worker_id={} task_type={} task_prompt={} text_input_present={} image_path={}",
            self.id,
            task.task_type,
            task.task_prompt,
            task.text_input.is_some(),
            image_path.display()
        );

        let decode_started = Instant::now();
        let image = ImageReader::open(image_path)
            .with_context(|| format!("failed to open image {}", image_path.display()))?
            .with_guessed_format()?
            .decode()
            .context("failed to decode image")?;

        let (original_width, original_height) = image.dimensions();
        debug!(
            "image decoded worker_id={} image_path={} width={} height={} elapsed_ms={}",
            self.id,
            image_path.display(),
            original_width,
            original_height,
            decode_started.elapsed().as_millis()
        );

        let preprocess_started = Instant::now();
        let input = preprocess_image(image)?;
        debug!(
            "image preprocessed worker_id={} output_f32_values={} elapsed_ms={}",
            self.id,
            input.len(),
            preprocess_started.elapsed().as_millis()
        );

        let vision_started = Instant::now();
        let image_features = self.run_vision_encoder(input)?;
        let vision_elapsed_ms = vision_started.elapsed().as_millis();
        let mut output_metadata = vec![tensor_metadata_f32(
            "image_features",
            &Shape::from(image_features.shape.clone()),
            &image_features.data,
        )];
        debug!(
            "vision encoder finished worker_id={} shape={:?} elapsed_ms={}",
            self.id, image_features.shape, vision_elapsed_ms
        );

        let generations = self.run_task_plan(&image_features, task)?;
        let final_generation = generations
            .last()
            .cloned()
            .ok_or_else(|| anyhow!("task produced no generations"))?;
        let result = combine_generation_results(&generations);
        let elapsed_ms = total_started.elapsed().as_millis();

        output_metadata.push(TensorMetadata {
            name: "generated_token_ids".to_string(),
            shape: vec![1, final_generation.generated_tokens as i64],
            elements: final_generation.generated_tokens,
            mean: None,
            min: None,
            max: None,
        });

        info!(
            "Florence generation finished worker_id={} model_variant={} task_token={} generated_tokens={} elapsed_ms={}",
            self.id,
            self.model_variant.as_str(),
            final_generation.task_token,
            final_generation.generated_tokens,
            elapsed_ms
        );

        Ok(InferenceMetadata {
            backend: self.backend.clone(),
            model_path: self.model_paths.vision_encoder.clone(),
            model_variant: self.model_variant,
            input_name: self.input_name.clone(),
            input_dtype: self.input_dtype.to_string(),
            original_width,
            original_height,
            processed_width: IMAGE_SIDE,
            processed_height: IMAGE_SIDE,
            elapsed_ms,
            task_token: final_generation.task_token,
            prompt_text: final_generation.prompt_text,
            generated_text: final_generation.generated_text,
            generated_tokens: final_generation.generated_tokens,
            result,
            generations,
            outputs: output_metadata,
        })
    }

    fn run_task_plan(
        &mut self,
        image_features: &TensorData,
        task: &TaskSpec,
    ) -> anyhow::Result<Vec<GenerationMetadata>> {
        if task.task_type == "Cascased task" {
            let caption_token = match task.task_prompt.as_str() {
                "Caption + Grounding" => "<CAPTION>",
                "Detailed Caption + Grounding" => "<DETAILED_CAPTION>",
                "More Detailed Caption + Grounding" => "<MORE_DETAILED_CAPTION>",
                other => return Err(anyhow!("unsupported cascased task prompt `{other}`")),
            };
            let caption = ResolvedTask::new(caption_token, None)?;
            let first = self.generate_once(image_features, &caption)?;
            let grounding_input = generation_text_for_input(&first);
            let grounding =
                ResolvedTask::new("<CAPTION_TO_PHRASE_GROUNDING>", Some(grounding_input))?;
            let second = self.generate_once(image_features, &grounding)?;
            return Ok(vec![first, second]);
        }

        let resolved = ResolvedTask::from_task_spec(task)?;
        self.generate_once(image_features, &resolved)
            .map(|step| vec![step])
    }

    fn generate_once(
        &mut self,
        image_features: &TensorData,
        task: &ResolvedTask,
    ) -> anyhow::Result<GenerationMetadata> {
        let prompt_ids = self.encode_prompt(&task.prompt_text)?;
        let text_embeds = self.run_embed_tokens(&prompt_ids)?;
        let encoder_inputs = concat_embeddings(image_features, &text_embeds)?;
        let encoder_attention_mask = vec![1_i64; encoder_inputs.seq_len()?];
        let encoder_hidden_states =
            self.run_encoder_model(encoder_inputs, encoder_attention_mask.clone())?;

        let mut generated_ids = vec![DECODER_START_TOKEN_ID];
        for _ in 0..self.max_new_tokens {
            let decoder_embeds = self.run_embed_tokens(&generated_ids)?;
            let logits = self.run_decoder_model(
                decoder_embeds,
                &encoder_hidden_states,
                &encoder_attention_mask,
            )?;
            let next_token_id = argmax_last_token(&logits)?;
            generated_ids.push(next_token_id);
            if next_token_id == EOS_TOKEN_ID {
                break;
            }
        }

        let generated_u32 = generated_ids
            .iter()
            .filter_map(|id| u32::try_from(*id).ok())
            .collect::<Vec<_>>();
        let raw_text = self
            .tokenizer
            .decode(&generated_u32, false)
            .map_err(|err| anyhow!("failed to decode generated tokens: {err}"))?;
        let generated_text = clean_generated_text(&raw_text);
        let result = json!({ task.task_token.clone(): generated_text.clone() });

        debug!(
            "generation step completed worker_id={} task_token={} prompt_ids={} generated_tokens={}",
            self.id,
            task.task_token,
            prompt_ids.len(),
            generated_ids.len()
        );

        Ok(GenerationMetadata {
            task_token: task.task_token.clone(),
            prompt_text: task.prompt_text.clone(),
            generated_text,
            generated_tokens: generated_ids.len(),
            result,
        })
    }

    fn encode_prompt(&self, prompt_text: &str) -> anyhow::Result<Vec<i64>> {
        let encoding = self
            .tokenizer
            .encode(prompt_text, true)
            .map_err(|err| anyhow!("failed to tokenize prompt `{prompt_text}`: {err}"))?;
        let ids = encoding
            .get_ids()
            .iter()
            .map(|id| i64::from(*id))
            .collect::<Vec<_>>();
        if ids.is_empty() {
            return Err(anyhow!("prompt tokenized to an empty input"));
        }
        trace!(
            "prompt tokenized worker_id={} prompt={:?} token_count={}",
            self.id,
            prompt_text,
            ids.len()
        );
        Ok(ids)
    }

    fn run_vision_encoder(&mut self, input: Vec<f32>) -> anyhow::Result<TensorData> {
        let session = self
            .vision_encoder
            .as_mut()
            .ok_or_else(|| anyhow!("worker {} vision encoder session is unavailable", self.id))?;
        let tensor_started = Instant::now();
        let outputs = match self.input_dtype {
            TensorElementType::Float32 => {
                let input = Tensor::<f32>::from_array((
                    Shape::from([1, 3, IMAGE_SIDE as i64, IMAGE_SIDE as i64]),
                    input,
                ))
                .map_err(|err| anyhow!("failed to create f32 image input tensor: {err}"))?;
                trace!(
                    "image tensor created worker_id={} input_name={} dtype=f32 shape=[1,3,{},{}] elapsed_ms={}",
                    self.id,
                    self.input_name,
                    IMAGE_SIDE,
                    IMAGE_SIDE,
                    tensor_started.elapsed().as_millis()
                );
                session
                    .run(ort::inputs![self.input_name.as_str() => input])
                    .map_err(|err| anyhow!("vision encoder ONNX inference failed: {err}"))?
            }
            TensorElementType::Float16 => {
                let input = input.into_iter().map(f16::from_f32).collect::<Vec<_>>();
                let input = Tensor::<f16>::from_array((
                    Shape::from([1, 3, IMAGE_SIDE as i64, IMAGE_SIDE as i64]),
                    input,
                ))
                .map_err(|err| anyhow!("failed to create f16 image input tensor: {err}"))?;
                trace!(
                    "image tensor created worker_id={} input_name={} dtype=f16 shape=[1,3,{},{}] elapsed_ms={}",
                    self.id,
                    self.input_name,
                    IMAGE_SIDE,
                    IMAGE_SIDE,
                    tensor_started.elapsed().as_millis()
                );
                session
                    .run(ort::inputs![self.input_name.as_str() => input])
                    .map_err(|err| anyhow!("vision encoder ONNX inference failed: {err}"))?
            }
            dtype => {
                return Err(anyhow!(
                    "unsupported image input dtype `{dtype}` for model {}; expected f32 or f16",
                    self.model_paths.vision_encoder.display()
                ));
            }
        };

        extract_output_tensor(&outputs[0], "image_features")
    }

    fn run_embed_tokens(&mut self, input_ids: &[i64]) -> anyhow::Result<TensorData> {
        let session = self
            .embed_tokens
            .as_mut()
            .ok_or_else(|| anyhow!("worker {} embed_tokens session is unavailable", self.id))?;
        let input = Tensor::<i64>::from_array((
            Shape::from([1, input_ids.len() as i64]),
            input_ids.to_vec(),
        ))
        .map_err(|err| anyhow!("failed to create input_ids tensor: {err}"))?;
        let outputs = session
            .run(ort::inputs!["input_ids" => input])
            .map_err(|err| anyhow!("embed_tokens ONNX inference failed: {err}"))?;
        extract_output_tensor(&outputs[0], "inputs_embeds")
    }

    fn run_encoder_model(
        &mut self,
        inputs_embeds: TensorData,
        attention_mask: Vec<i64>,
    ) -> anyhow::Result<TensorData> {
        let session = self
            .encoder_model
            .as_mut()
            .ok_or_else(|| anyhow!("worker {} encoder_model session is unavailable", self.id))?;
        let seq_len = inputs_embeds.seq_len()?;
        let inputs_embeds = Tensor::<f32>::from_array((
            Shape::from([1, seq_len as i64, HIDDEN_SIZE]),
            inputs_embeds.data,
        ))
        .map_err(|err| anyhow!("failed to create encoder inputs_embeds tensor: {err}"))?;
        let attention_mask =
            Tensor::<i64>::from_array((Shape::from([1, seq_len as i64]), attention_mask))
                .map_err(|err| anyhow!("failed to create encoder attention_mask tensor: {err}"))?;

        let outputs = session
            .run(ort::inputs![
                "inputs_embeds" => inputs_embeds,
                "attention_mask" => attention_mask,
            ])
            .map_err(|err| anyhow!("encoder_model ONNX inference failed: {err}"))?;
        extract_output_tensor(&outputs[0], "last_hidden_state")
    }

    fn run_decoder_model(
        &mut self,
        inputs_embeds: TensorData,
        encoder_hidden_states: &TensorData,
        encoder_attention_mask: &[i64],
    ) -> anyhow::Result<TensorData> {
        let session = self
            .decoder_model
            .as_mut()
            .ok_or_else(|| anyhow!("worker {} decoder_model session is unavailable", self.id))?;
        let decoder_seq_len = inputs_embeds.seq_len()?;
        let encoder_seq_len = encoder_hidden_states.seq_len()?;
        let inputs_embeds = Tensor::<f32>::from_array((
            Shape::from([1, decoder_seq_len as i64, HIDDEN_SIZE]),
            inputs_embeds.data,
        ))
        .map_err(|err| anyhow!("failed to create decoder inputs_embeds tensor: {err}"))?;
        let encoder_hidden_states = Tensor::<f32>::from_array((
            Shape::from([1, encoder_seq_len as i64, HIDDEN_SIZE]),
            encoder_hidden_states.data.clone(),
        ))
        .map_err(|err| anyhow!("failed to create decoder encoder_hidden_states tensor: {err}"))?;
        let encoder_attention_mask = Tensor::<i64>::from_array((
            Shape::from([1, encoder_seq_len as i64]),
            encoder_attention_mask.to_vec(),
        ))
        .map_err(|err| anyhow!("failed to create decoder encoder_attention_mask tensor: {err}"))?;

        let outputs = session
            .run(ort::inputs![
                "inputs_embeds" => inputs_embeds,
                "encoder_hidden_states" => encoder_hidden_states,
                "encoder_attention_mask" => encoder_attention_mask,
            ])
            .map_err(|err| anyhow!("decoder_model ONNX inference failed: {err}"))?;
        extract_output_tensor(&outputs[0], "logits")
    }
}

#[derive(Debug, Clone)]
struct FlorenceModelPaths {
    model_dir: PathBuf,
    vision_encoder: PathBuf,
    embed_tokens: PathBuf,
    encoder_model: PathBuf,
    decoder_model: PathBuf,
    decoder_with_past_model: PathBuf,
}

impl FlorenceModelPaths {
    fn from_vision_path(vision_encoder: &Path, variant: ModelVariant) -> Self {
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
            decoder_with_past_model: onnx_dir.join(format!("decoder_with_past_model{suffix}.onnx")),
        }
    }

    fn ensure_exists(&self) -> anyhow::Result<()> {
        for path in [
            &self.vision_encoder,
            &self.embed_tokens,
            &self.encoder_model,
            &self.decoder_model,
            &self.decoder_with_past_model,
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

#[derive(Debug, Clone)]
struct ResolvedTask {
    task_token: String,
    prompt_text: String,
}

impl ResolvedTask {
    fn from_task_spec(task: &TaskSpec) -> anyhow::Result<Self> {
        let task_token = match task.task_prompt.as_str() {
            "Caption" => "<CAPTION>",
            "Detailed Caption" => "<DETAILED_CAPTION>",
            "More Detailed Caption" => "<MORE_DETAILED_CAPTION>",
            "Object Detection" => "<OD>",
            "Dense Region Caption" => "<DENSE_REGION_CAPTION>",
            "Region Proposal" => "<REGION_PROPOSAL>",
            "Caption to Phrase Grounding" => "<CAPTION_TO_PHRASE_GROUNDING>",
            "Referring Expression Segmentation" => "<REFERRING_EXPRESSION_SEGMENTATION>",
            "Region to Segmentation" => "<REGION_TO_SEGMENTATION>",
            "Open Vocabulary Detection" => "<OPEN_VOCABULARY_DETECTION>",
            "Region to Category" => "<REGION_TO_CATEGORY>",
            "Region to Description" => "<REGION_TO_DESCRIPTION>",
            "OCR" => "<OCR>",
            "OCR with Region" => "<OCR_WITH_REGION>",
            other => return Err(anyhow!("unsupported task prompt `{other}`")),
        };
        Self::new(task_token, task.text_input.clone())
    }

    fn new(task_token: &str, text_input: Option<String>) -> anyhow::Result<Self> {
        let prompt_text = prompt_text_for_task(task_token, text_input.as_deref())?;
        Ok(Self {
            task_token: task_token.to_string(),
            prompt_text,
        })
    }
}

#[derive(Debug, Clone)]
struct TensorData {
    shape: Vec<i64>,
    data: Vec<f32>,
}

impl TensorData {
    fn seq_len(&self) -> anyhow::Result<usize> {
        let seq_len =
            self.shape.get(1).copied().ok_or_else(|| {
                anyhow!("tensor shape {:?} has no sequence dimension", self.shape)
            })?;
        usize::try_from(seq_len)
            .map_err(|_| anyhow!("tensor sequence length is invalid: {seq_len}"))
    }
}

fn load_session(path: &Path) -> anyhow::Result<Session> {
    Session::builder()
        .map_err(|err| anyhow!("failed to create ONNX session builder: {err}"))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|err| anyhow!("failed to set ONNX graph optimization level: {err}"))?
        .with_auto_device(AutoDevicePolicy::MaxPerformance)
        .map_err(|err| anyhow!("failed to enable ONNX Runtime auto device selection: {err}"))?
        .commit_from_file(path)
        .map_err(|err| anyhow!("failed to load ONNX model {}: {err}", path.display()))
}

fn preprocess_image(image: DynamicImage) -> anyhow::Result<Vec<f32>> {
    trace!(
        "resizing and normalizing image target_width={} target_height={}",
        IMAGE_SIDE, IMAGE_SIDE
    );
    let resized = image
        .resize_exact(IMAGE_SIDE, IMAGE_SIDE, FilterType::CatmullRom)
        .to_rgb8();
    let pixels = resized.as_raw();
    let plane = (IMAGE_SIDE * IMAGE_SIDE) as usize;
    let mut chw = vec![0.0_f32; 3 * plane];
    let mean = [0.485_f32, 0.456, 0.406];
    let std = [0.229_f32, 0.224, 0.225];

    for y in 0..IMAGE_SIDE as usize {
        for x in 0..IMAGE_SIDE as usize {
            let src = (y * IMAGE_SIDE as usize + x) * 3;
            let dst = y * IMAGE_SIDE as usize + x;
            for channel in 0..3 {
                let value = pixels[src + channel] as f32 / 255.0;
                chw[channel * plane + dst] = (value - mean[channel]) / std[channel];
            }
        }
    }

    Ok(chw)
}

fn image_input_metadata(session: &Session) -> anyhow::Result<(String, TensorElementType)> {
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

fn extract_output_tensor(value: &DynValue, name: &str) -> anyhow::Result<TensorData> {
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

fn concat_embeddings(
    image_features: &TensorData,
    text_embeds: &TensorData,
) -> anyhow::Result<TensorData> {
    let image_seq_len = image_features.seq_len()?;
    let text_seq_len = text_embeds.seq_len()?;
    let image_values = image_seq_len
        .checked_mul(HIDDEN_SIZE as usize)
        .ok_or_else(|| anyhow!("image embedding shape is too large"))?;
    let text_values = text_seq_len
        .checked_mul(HIDDEN_SIZE as usize)
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
        shape: vec![1, (image_seq_len + text_seq_len) as i64, HIDDEN_SIZE],
        data,
    })
}

fn argmax_last_token(logits: &TensorData) -> anyhow::Result<i64> {
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

fn prompt_text_for_task(task_token: &str, text_input: Option<&str>) -> anyhow::Result<String> {
    match task_token {
        "<OCR>" => Ok("What is the text in the image?".to_string()),
        "<OCR_WITH_REGION>" => Ok("What is the text in the image, with regions?".to_string()),
        "<CAPTION>" => Ok("What does the image describe?".to_string()),
        "<DETAILED_CAPTION>" => Ok("Describe in detail what is shown in the image.".to_string()),
        "<MORE_DETAILED_CAPTION>" => {
            Ok("Describe with a paragraph what is shown in the image.".to_string())
        }
        "<OD>" => Ok("Locate the objects with category name in the image.".to_string()),
        "<DENSE_REGION_CAPTION>" => {
            Ok("Locate the objects in the image, with their descriptions.".to_string())
        }
        "<REGION_PROPOSAL>" => Ok("Locate the region proposals in the image.".to_string()),
        "<CAPTION_TO_PHRASE_GROUNDING>" => Ok(format!(
            "Locate the phrases in the caption: {}",
            required_text_input(task_token, text_input)?
        )),
        "<REFERRING_EXPRESSION_SEGMENTATION>" => Ok(format!(
            "Locate {} in the image with mask",
            required_text_input(task_token, text_input)?
        )),
        "<REGION_TO_SEGMENTATION>" => Ok(format!(
            "What is the polygon mask of region {}",
            required_text_input(task_token, text_input)?
        )),
        "<OPEN_VOCABULARY_DETECTION>" => Ok(format!(
            "Locate {} in the image.",
            required_text_input(task_token, text_input)?
        )),
        "<REGION_TO_CATEGORY>" => Ok(format!(
            "What is the region {}?",
            required_text_input(task_token, text_input)?
        )),
        "<REGION_TO_DESCRIPTION>" => Ok(format!(
            "What does the region {} describe?",
            required_text_input(task_token, text_input)?
        )),
        other => Err(anyhow!("unsupported Florence task token `{other}`")),
    }
}

fn required_text_input<'a>(
    task_token: &str,
    text_input: Option<&'a str>,
) -> anyhow::Result<&'a str> {
    let text_input = text_input
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| anyhow!("task {task_token} requires text_input"))?;
    Ok(text_input)
}

fn generation_text_for_input(generation: &GenerationMetadata) -> String {
    generation
        .result
        .get(&generation.task_token)
        .and_then(Value::as_str)
        .unwrap_or(&generation.generated_text)
        .to_string()
}

fn combine_generation_results(generations: &[GenerationMetadata]) -> Value {
    let mut output = serde_json::Map::new();
    for generation in generations {
        if let Some(value) = generation.result.get(&generation.task_token) {
            output.insert(generation.task_token.clone(), value.clone());
        }
    }
    Value::Object(output)
}

fn clean_generated_text(raw: &str) -> String {
    raw.replace("<s>", "")
        .replace("</s>", "")
        .replace("<pad>", "")
        .trim()
        .to_string()
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

fn tensor_metadata_f32(name: &str, shape: &Shape, data: &[f32]) -> TensorMetadata {
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
