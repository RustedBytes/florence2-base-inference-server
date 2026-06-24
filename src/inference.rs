mod model;
mod postprocess;
mod tensor;

use std::{path::Path, sync::Arc, time::Instant};

use anyhow::{Context, anyhow};
use half::f16;
use image::{DynamicImage, GenericImageView, ImageReader, imageops::FilterType};
use log::{debug, info, trace, warn};
use ort::{
    environment::Environment,
    session::{Session, SessionInputValue, SessionOutputs},
    value::{DynValue, Shape, Tensor, TensorElementType},
};
use serde_json::Value;
use tokenizers::Tokenizer;

use crate::{
    config::{Config, ModelVariant},
    types::{GenerationMetadata, InferenceMetadata, TaskSpec, TaskType, TensorMetadata},
};

use self::{
    model::{FlorenceModelPaths, image_input_metadata, load_session},
    postprocess::{clean_generated_text, post_process_generation},
    tensor::{
        TensorData, argmax_last_token, concat_embeddings, extract_output_tensor,
        tensor_metadata_f32,
    },
};

const IMAGE_SIDE: u32 = 768;
const HIDDEN_SIZE: i64 = 768;
const DECODER_LAYERS: usize = 6;
const DECODER_START_TOKEN_ID: i64 = 2;
const EOS_TOKEN_ID: i64 = 2;

pub fn validate_model_artifacts(model_path: &Path, variant: ModelVariant) -> anyhow::Result<()> {
    FlorenceModelPaths::from_vision_path(model_path, variant).ensure_exists()
}

pub struct FlorenceWorker {
    id: usize,
    model_paths: FlorenceModelPaths,
    model_variant: ModelVariant,
    // Florence generation is split across exported ONNX graphs. Keep each
    // session alive inside one worker so a queue slot owns a full model set.
    sessions: FlorenceSessions,
    tokenizer: Tokenizer,
    backend: String,
    image_input: ImageInput,
    max_new_tokens: usize,
    execution_providers: Vec<String>,
}

struct FlorenceSessions {
    vision_encoder: Option<Session>,
    embed_tokens: Option<Session>,
    encoder_model: Option<Session>,
    decoder_model: Option<Session>,
    decoder_model_merged: Option<Session>,
}

struct ImageInput {
    input_name: String,
    input_dtype: TensorElementType,
}

struct DecoderRun {
    logits: TensorData,
    cache: DecoderCache,
}

struct DecoderCache {
    layers: Vec<DecoderLayerCache>,
}

struct DecoderLayerCache {
    decoder_key: Arc<DynValue>,
    decoder_value: Arc<DynValue>,
    encoder_key: Arc<DynValue>,
    encoder_value: Arc<DynValue>,
}

impl FlorenceWorker {
    pub fn new(id: usize, config: Arc<Config>) -> anyhow::Result<Self> {
        let started = Instant::now();
        let model_paths =
            FlorenceModelPaths::from_vision_path(&config.model_path, config.model_variant);
        model_paths.ensure_exists()?;

        info!(
            "initializing model worker worker_id={} vision_encoder={} embed_tokens={} encoder_model={} decoder_model={} decoder_model_merged={}",
            id,
            model_paths.vision_encoder.display(),
            model_paths.embed_tokens.display(),
            model_paths.encoder_model.display(),
            model_paths.decoder_model.display(),
            model_paths.decoder_model_merged.display()
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
            warn!("ONNX Runtime did not report hardware devices");
        } else {
            info!(
                "ONNX Runtime detected hardware devices worker_id={} devices={:?}",
                id, devices
            );
        }
        let backend = runtime_backend_summary(&config.execution_providers, &devices);

        let session_started = Instant::now();
        let vision_encoder =
            load_session(&model_paths.vision_encoder, &config.execution_providers)?;
        let embed_tokens = load_session(&model_paths.embed_tokens, &config.execution_providers)?;
        let encoder_model = load_session(&model_paths.encoder_model, &config.execution_providers)?;
        let decoder_model = load_session(&model_paths.decoder_model, &config.execution_providers)?;
        let decoder_model_merged = load_session(
            &model_paths.decoder_model_merged,
            &config.execution_providers,
        )?;
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
            sessions: FlorenceSessions {
                vision_encoder: Some(vision_encoder),
                embed_tokens: Some(embed_tokens),
                encoder_model: Some(encoder_model),
                decoder_model: Some(decoder_model),
                decoder_model_merged: Some(decoder_model_merged),
            },
            tokenizer,
            backend,
            image_input: ImageInput {
                input_name,
                input_dtype,
            },
            max_new_tokens: config.max_new_tokens,
            execution_providers: config.execution_providers.clone(),
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
        // ONNX sessions are consumed by a blocking thread for inference and
        // returned afterward. Option::take prevents two threads from touching
        // the same session handle at once.
        Self {
            id: self.id,
            model_paths: self.model_paths.clone(),
            model_variant: self.model_variant,
            sessions: FlorenceSessions {
                vision_encoder: self.sessions.vision_encoder.take(),
                embed_tokens: self.sessions.embed_tokens.take(),
                encoder_model: self.sessions.encoder_model.take(),
                decoder_model: self.sessions.decoder_model.take(),
                decoder_model_merged: self.sessions.decoder_model_merged.take(),
            },
            tokenizer: self.tokenizer.clone(),
            backend: self.backend.clone(),
            image_input: ImageInput {
                input_name: self.image_input.input_name.clone(),
                input_dtype: self.image_input.input_dtype,
            },
            max_new_tokens: self.max_new_tokens,
            execution_providers: self.execution_providers.clone(),
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
            task.task_type_name(),
            task.task_prompt_name(),
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

        let generations =
            self.run_task_plan(&image_features, task, (original_width, original_height))?;
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
            input_name: self.image_input.input_name.clone(),
            input_dtype: self.image_input.input_dtype.to_string(),
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
        image_size: (u32, u32),
    ) -> anyhow::Result<Vec<GenerationMetadata>> {
        if task.task_type == TaskType::Cascaded {
            // Cascaded HF Space tasks first produce a caption, then reuse that
            // caption as text input for phrase grounding.
            let caption_token = task.task_prompt.cascaded_caption_token().ok_or_else(|| {
                anyhow!("unsupported cascaded task prompt `{}`", task.task_prompt)
            })?;
            let caption = ResolvedTask::new(caption_token, None)?;
            let first = self.generate_once(image_features, &caption, image_size)?;
            let grounding_input = generation_text_for_input(&first);
            let grounding =
                ResolvedTask::new("<CAPTION_TO_PHRASE_GROUNDING>", Some(grounding_input))?;
            let second = self.generate_once(image_features, &grounding, image_size)?;
            return Ok(vec![first, second]);
        }

        let resolved = ResolvedTask::from_task_spec(task)?;
        self.generate_once(image_features, &resolved, image_size)
            .map(|step| vec![step])
    }

    fn generate_once(
        &mut self,
        image_features: &TensorData,
        task: &ResolvedTask,
        image_size: (u32, u32),
    ) -> anyhow::Result<GenerationMetadata> {
        let prompt_ids = self.encode_prompt(&task.prompt_text)?;
        let text_embeds = self.run_embed_tokens(&prompt_ids)?;
        // The ONNX encoder expects the image tokens and prompt tokens in one
        // embedding sequence, matching Florence's processor output.
        let encoder_inputs = concat_embeddings(image_features, &text_embeds, HIDDEN_SIZE as usize)?;
        let encoder_attention_mask = vec![1_i64; encoder_inputs.seq_len()?];
        let encoder_hidden_states =
            self.run_encoder_model(encoder_inputs, encoder_attention_mask.clone())?;

        let mut generated_ids = vec![DECODER_START_TOKEN_ID];
        let mut decoder_cache = None;
        for _ in 0..self.max_new_tokens {
            let decoder_run = if let Some(cache) = decoder_cache.as_ref() {
                let last_token_id = generated_ids
                    .last()
                    .copied()
                    .ok_or_else(|| anyhow!("decoder token sequence is empty"))?;
                let decoder_embeds = self.run_embed_tokens(&[last_token_id])?;
                self.run_cached_decoder_model(
                    decoder_embeds,
                    &encoder_hidden_states,
                    &encoder_attention_mask,
                    cache,
                )?
            } else {
                let decoder_embeds = self.run_embed_tokens(&generated_ids)?;
                self.run_decoder_model(
                    decoder_embeds,
                    &encoder_hidden_states,
                    &encoder_attention_mask,
                )?
            };
            let next_token_id = argmax_last_token(&decoder_run.logits)?;
            decoder_cache = Some(decoder_run.cache);
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
        // Florence embeds structure such as boxes and polygons directly in the
        // decoded text via special tokens; convert those into JSON here.
        let result = post_process_generation(&task.task_token, &generated_text, image_size);

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
        let session =
            self.sessions.vision_encoder.as_mut().ok_or_else(|| {
                anyhow!("worker {} vision encoder session is unavailable", self.id)
            })?;
        let tensor_started = Instant::now();
        let outputs = match self.image_input.input_dtype {
            TensorElementType::Float32 => {
                let input = Tensor::<f32>::from_array((
                    Shape::from([1, 3, IMAGE_SIDE as i64, IMAGE_SIDE as i64]),
                    input,
                ))
                .map_err(|err| anyhow!("failed to create f32 image input tensor: {err}"))?;
                trace!(
                    "image tensor created worker_id={} input_name={} dtype=f32 shape=[1,3,{},{}] elapsed_ms={}",
                    self.id,
                    self.image_input.input_name,
                    IMAGE_SIDE,
                    IMAGE_SIDE,
                    tensor_started.elapsed().as_millis()
                );
                session
                    .run(ort::inputs![self.image_input.input_name.as_str() => input])
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
                    self.image_input.input_name,
                    IMAGE_SIDE,
                    IMAGE_SIDE,
                    tensor_started.elapsed().as_millis()
                );
                session
                    .run(ort::inputs![self.image_input.input_name.as_str() => input])
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
            .sessions
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
        let session =
            self.sessions.encoder_model.as_mut().ok_or_else(|| {
                anyhow!("worker {} encoder_model session is unavailable", self.id)
            })?;
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
    ) -> anyhow::Result<DecoderRun> {
        let session =
            self.sessions.decoder_model.as_mut().ok_or_else(|| {
                anyhow!("worker {} decoder_model session is unavailable", self.id)
            })?;
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

        let mut outputs = session
            .run(ort::inputs![
                "inputs_embeds" => inputs_embeds,
                "encoder_hidden_states" => encoder_hidden_states,
                "encoder_attention_mask" => encoder_attention_mask,
            ])
            .map_err(|err| anyhow!("decoder_model ONNX inference failed: {err}"))?;
        let logits = extract_named_output_tensor(&outputs, "logits")?;
        let cache = DecoderCache::from_decoder_outputs(&mut outputs)?;
        Ok(DecoderRun { logits, cache })
    }

    fn run_cached_decoder_model(
        &mut self,
        inputs_embeds: TensorData,
        encoder_hidden_states: &TensorData,
        encoder_attention_mask: &[i64],
        cache: &DecoderCache,
    ) -> anyhow::Result<DecoderRun> {
        let session = self.sessions.decoder_model_merged.as_mut().ok_or_else(|| {
            anyhow!(
                "worker {} decoder_model_merged session is unavailable",
                self.id
            )
        })?;
        let decoder_seq_len = inputs_embeds.seq_len()?;
        let encoder_seq_len = encoder_hidden_states.seq_len()?;
        let inputs_embeds = Tensor::<f32>::from_array((
            Shape::from([1, decoder_seq_len as i64, HIDDEN_SIZE]),
            inputs_embeds.data,
        ))
        .map_err(|err| anyhow!("failed to create cached decoder inputs_embeds tensor: {err}"))?;
        let encoder_hidden_states = Tensor::<f32>::from_array((
            Shape::from([1, encoder_seq_len as i64, HIDDEN_SIZE]),
            encoder_hidden_states.data.clone(),
        ))
        .map_err(|err| {
            anyhow!("failed to create cached decoder encoder_hidden_states tensor: {err}")
        })?;
        let encoder_attention_mask = Tensor::<i64>::from_array((
            Shape::from([1, encoder_seq_len as i64]),
            encoder_attention_mask.to_vec(),
        ))
        .map_err(|err| {
            anyhow!("failed to create cached decoder encoder_attention_mask tensor: {err}")
        })?;
        let use_cache_branch = Tensor::<bool>::from_array((Shape::from([1_i64]), vec![true]))
            .map_err(|err| {
                anyhow!("failed to create cached decoder branch selector tensor: {err}")
            })?;

        let mut inputs: Vec<(String, SessionInputValue<'_>)> =
            Vec::with_capacity(4 + DECODER_LAYERS * 4);
        inputs.push(("inputs_embeds".to_string(), inputs_embeds.into()));
        inputs.push((
            "encoder_hidden_states".to_string(),
            encoder_hidden_states.into(),
        ));
        inputs.push((
            "encoder_attention_mask".to_string(),
            encoder_attention_mask.into(),
        ));
        inputs.push(("use_cache_branch".to_string(), use_cache_branch.into()));

        for (layer_index, layer) in cache.layers.iter().enumerate() {
            inputs.push((
                past_cache_name(layer_index, CacheKind::DecoderKey),
                layer.decoder_key.as_ref().into(),
            ));
            inputs.push((
                past_cache_name(layer_index, CacheKind::DecoderValue),
                layer.decoder_value.as_ref().into(),
            ));
            inputs.push((
                past_cache_name(layer_index, CacheKind::EncoderKey),
                layer.encoder_key.as_ref().into(),
            ));
            inputs.push((
                past_cache_name(layer_index, CacheKind::EncoderValue),
                layer.encoder_value.as_ref().into(),
            ));
        }

        let mut outputs = session
            .run(inputs)
            .map_err(|err| anyhow!("decoder_model_merged cached ONNX inference failed: {err}"))?;
        let logits = extract_named_output_tensor(&outputs, "logits")?;
        let cache = cache.with_updated_decoder_outputs(&mut outputs)?;
        Ok(DecoderRun { logits, cache })
    }
}

impl DecoderCache {
    fn from_decoder_outputs(outputs: &mut SessionOutputs<'_>) -> anyhow::Result<Self> {
        let mut layers = Vec::with_capacity(DECODER_LAYERS);
        for layer_index in 0..DECODER_LAYERS {
            layers.push(DecoderLayerCache {
                decoder_key: take_output_value(
                    outputs,
                    &present_cache_name(layer_index, CacheKind::DecoderKey),
                )?,
                decoder_value: take_output_value(
                    outputs,
                    &present_cache_name(layer_index, CacheKind::DecoderValue),
                )?,
                encoder_key: take_output_value(
                    outputs,
                    &present_cache_name(layer_index, CacheKind::EncoderKey),
                )?,
                encoder_value: take_output_value(
                    outputs,
                    &present_cache_name(layer_index, CacheKind::EncoderValue),
                )?,
            });
        }
        Ok(Self { layers })
    }

    fn with_updated_decoder_outputs(
        &self,
        outputs: &mut SessionOutputs<'_>,
    ) -> anyhow::Result<Self> {
        let mut layers = Vec::with_capacity(self.layers.len());
        for (layer_index, previous) in self.layers.iter().enumerate() {
            layers.push(DecoderLayerCache {
                decoder_key: take_output_value(
                    outputs,
                    &present_cache_name(layer_index, CacheKind::DecoderKey),
                )?,
                decoder_value: take_output_value(
                    outputs,
                    &present_cache_name(layer_index, CacheKind::DecoderValue),
                )?,
                encoder_key: Arc::clone(&previous.encoder_key),
                encoder_value: Arc::clone(&previous.encoder_value),
            });
        }
        Ok(Self { layers })
    }
}

#[derive(Debug, Clone, Copy)]
enum CacheKind {
    DecoderKey,
    DecoderValue,
    EncoderKey,
    EncoderValue,
}

fn present_cache_name(layer_index: usize, kind: CacheKind) -> String {
    let suffix = cache_name_suffix(kind);
    format!("present.{layer_index}.{suffix}")
}

fn past_cache_name(layer_index: usize, kind: CacheKind) -> String {
    let suffix = cache_name_suffix(kind);
    format!("past_key_values.{layer_index}.{suffix}")
}

fn cache_name_suffix(kind: CacheKind) -> &'static str {
    match kind {
        CacheKind::DecoderKey => "decoder.key",
        CacheKind::DecoderValue => "decoder.value",
        CacheKind::EncoderKey => "encoder.key",
        CacheKind::EncoderValue => "encoder.value",
    }
}

fn extract_named_output_tensor(
    outputs: &SessionOutputs<'_>,
    name: &str,
) -> anyhow::Result<TensorData> {
    let value = outputs
        .get(name)
        .ok_or_else(|| anyhow!("ONNX output `{name}` is missing"))?;
    extract_output_tensor(value, name)
}

fn take_output_value(
    outputs: &mut SessionOutputs<'_>,
    name: &str,
) -> anyhow::Result<Arc<DynValue>> {
    outputs
        .remove(name)
        .map(Arc::new)
        .ok_or_else(|| anyhow!("ONNX output `{name}` is missing"))
}

fn runtime_backend_summary(execution_providers: &[String], devices: &[String]) -> String {
    let providers = if execution_providers.is_empty() {
        "default".to_string()
    } else {
        execution_providers.join(",")
    };
    let devices = if devices.is_empty() {
        "none_reported".to_string()
    } else {
        devices.join(",")
    };

    format!("ort:execution_providers={providers};detected_devices={devices}")
}

#[derive(Debug, Clone)]
struct ResolvedTask {
    task_token: String,
    prompt_text: String,
}

impl ResolvedTask {
    fn from_task_spec(task: &TaskSpec) -> anyhow::Result<Self> {
        let task_token = task
            .task_prompt
            .single_task_token()
            .ok_or_else(|| anyhow!("unsupported single task prompt `{}`", task.task_prompt))?;
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

#[cfg(test)]
mod tests;
