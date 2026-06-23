use std::{env, fs as std_fs, net::SocketAddr, path::PathBuf};

use anyhow::{Context, anyhow};
use log::debug;
use serde::{Deserialize, Serialize};
use tokio::fs;

const DEFAULT_ADDR: &str = "127.0.0.1:3000";
const DEFAULT_CONFIG_PATH: &str = "config.toml";
const DEFAULT_MODEL_PATH: &str = "Florence-2-base/onnx/vision_encoder.onnx";
const DEFAULT_DATA_DIR: &str = "data";
const DEFAULT_QUEUE_SIZE: usize = 128;
const DEFAULT_BODY_LIMIT_BYTES: usize = 32 * 1024 * 1024;
const DEFAULT_RUST_LOG: &str = "info,ort=warn";
const DEFAULT_MAX_NEW_TOKENS: usize = 256;
const DEFAULT_EXECUTION_PROVIDER: &str = "auto";

pub struct Config {
    pub addr: SocketAddr,
    pub model_path: PathBuf,
    pub model_variant: ModelVariant,
    pub data_dir: PathBuf,
    pub images_dir: PathBuf,
    pub metadata_dir: PathBuf,
    pub submissions_jsonl: PathBuf,
    pub results_jsonl: PathBuf,
    pub allow_local_paths: bool,
    pub local_path_roots: Vec<PathBuf>,
    pub workers: usize,
    pub queue_size: usize,
    pub body_limit_bytes: usize,
    pub rust_log: String,
    pub max_new_tokens: usize,
    pub execution_providers: Vec<String>,
}

impl Config {
    pub fn load(config_path: Option<PathBuf>) -> anyhow::Result<Self> {
        let file_config = FileConfig::load(config_path)?;
        let server = file_config.server.unwrap_or_default();
        let model = file_config.model.unwrap_or_default();
        let queue = file_config.queue.unwrap_or_default();
        let generation = file_config.generation.unwrap_or_default();
        let runtime = file_config.runtime.unwrap_or_default();
        let logging = file_config.logging.unwrap_or_default();

        let data_dir = path_setting("DATA_DIR", server.data_dir, DEFAULT_DATA_DIR);
        let metadata_dir = data_dir.join("metadata");
        let allow_local_paths = bool_setting("ALLOW_LOCAL_PATHS", server.allow_local_paths, false)?;
        let local_path_roots = path_list_setting("LOCAL_PATH_ROOTS", server.local_path_roots);
        if allow_local_paths && local_path_roots.is_empty() {
            return Err(anyhow!(
                "local path inference requires at least one configured local_path_roots entry"
            ));
        }
        let model_path_override = env_path("MODEL_PATH").or(model.path);
        let model_variant = parse_model_variant(model.variant, model_path_override.is_some())?;
        let model_path = model_path_override.unwrap_or_else(|| model_variant.default_model_path());
        let addr = string_setting("BIND_ADDR", server.bind_addr, DEFAULT_ADDR)
            .parse()
            .context("BIND_ADDR must be a socket address, for example 127.0.0.1:3000")?;

        Ok(Self {
            addr,
            model_path,
            model_variant,
            images_dir: data_dir.join("images"),
            submissions_jsonl: metadata_dir.join("submissions.jsonl"),
            results_jsonl: metadata_dir.join("results.jsonl"),
            allow_local_paths,
            local_path_roots,
            metadata_dir,
            data_dir,
            workers: usize_setting(
                "MODEL_POOL_SIZE",
                queue.model_pool_size,
                default_worker_count(),
            )?
            .max(1),
            queue_size: usize_setting("QUEUE_SIZE", queue.queue_size, DEFAULT_QUEUE_SIZE)?,
            body_limit_bytes: usize_setting(
                "BODY_LIMIT_BYTES",
                queue.body_limit_bytes,
                DEFAULT_BODY_LIMIT_BYTES,
            )?,
            rust_log: string_setting("RUST_LOG", logging.rust_log, DEFAULT_RUST_LOG),
            max_new_tokens: usize_setting(
                "MAX_NEW_TOKENS",
                generation.max_new_tokens,
                DEFAULT_MAX_NEW_TOKENS,
            )?
            .max(1),
            execution_providers: execution_providers_setting(runtime.execution_providers),
        })
    }

    pub async fn ensure_dirs(&self) -> anyhow::Result<()> {
        if !self.model_path.exists() {
            return Err(anyhow!(
                "model file does not exist: {}",
                self.model_path.display()
            ));
        }
        debug!(
            "ensuring data directories images_dir={} metadata_dir={}",
            self.images_dir.display(),
            self.metadata_dir.display()
        );
        fs::create_dir_all(&self.images_dir).await?;
        fs::create_dir_all(&self.metadata_dir).await?;
        Ok(())
    }
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct FileConfig {
    server: Option<ServerConfig>,
    model: Option<ModelConfig>,
    queue: Option<QueueConfig>,
    generation: Option<GenerationConfig>,
    runtime: Option<RuntimeConfig>,
    logging: Option<LoggingConfig>,
}

impl FileConfig {
    fn load(config_path: Option<PathBuf>) -> anyhow::Result<Self> {
        // CLI path wins over CONFIG_PATH. Missing default config.toml is OK so
        // the binary can still run with built-in defaults.
        let has_cli_path = config_path.is_some();
        let has_env_path = env::var_os("CONFIG_PATH").is_some();
        let config_path = config_path
            .or_else(|| env_path("CONFIG_PATH"))
            .unwrap_or_else(|| PathBuf::from(DEFAULT_CONFIG_PATH));
        let has_explicit_path = has_cli_path || has_env_path;

        if !config_path.exists() {
            if has_explicit_path {
                return Err(anyhow!(
                    "config file does not exist: {}",
                    config_path.display()
                ));
            }

            return Ok(Self::default());
        }

        let contents = std_fs::read_to_string(&config_path)
            .with_context(|| format!("failed to read config file {}", config_path.display()))?;
        toml::from_str(&contents)
            .with_context(|| format!("failed to parse TOML config {}", config_path.display()))
    }
}

#[derive(Debug, Default, Deserialize)]
struct ServerConfig {
    bind_addr: Option<String>,
    data_dir: Option<PathBuf>,
    allow_local_paths: Option<bool>,
    local_path_roots: Option<Vec<PathBuf>>,
}

#[derive(Debug, Default, Deserialize)]
struct ModelConfig {
    variant: Option<String>,
    path: Option<PathBuf>,
}

#[derive(Debug, Default, Deserialize)]
struct QueueConfig {
    model_pool_size: Option<usize>,
    queue_size: Option<usize>,
    body_limit_bytes: Option<usize>,
}

#[derive(Debug, Default, Deserialize)]
struct GenerationConfig {
    max_new_tokens: Option<usize>,
}

#[derive(Debug, Default, Deserialize)]
struct RuntimeConfig {
    execution_providers: Option<Vec<String>>,
}

#[derive(Debug, Default, Deserialize)]
struct LoggingConfig {
    rust_log: Option<String>,
}

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

fn env_path(key: &str) -> Option<PathBuf> {
    env::var_os(key).map(PathBuf::from)
}

fn path_setting(key: &str, file_value: Option<PathBuf>, default: &str) -> PathBuf {
    env_path(key)
        .or(file_value)
        .unwrap_or_else(|| PathBuf::from(default))
}

fn string_setting(key: &str, file_value: Option<String>, default: &str) -> String {
    env::var(key)
        .ok()
        .or(file_value)
        .unwrap_or_else(|| default.into())
}

fn usize_setting(key: &str, file_value: Option<usize>, default: usize) -> anyhow::Result<usize> {
    match env::var(key) {
        Ok(value) => value
            .parse()
            .map_err(|err| anyhow!("{key} has invalid value `{value}`: {err}")),
        Err(_) => Ok(file_value.unwrap_or(default)),
    }
}

fn bool_setting(key: &str, file_value: Option<bool>, default: bool) -> anyhow::Result<bool> {
    match env::var(key) {
        Ok(value) => match value.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Ok(true),
            "0" | "false" | "no" | "off" => Ok(false),
            _ => Err(anyhow!(
                "{key} has invalid value `{value}`; expected true or false"
            )),
        },
        Err(_) => Ok(file_value.unwrap_or(default)),
    }
}

fn path_list_setting(key: &str, file_value: Option<Vec<PathBuf>>) -> Vec<PathBuf> {
    env::var_os(key)
        .map(|value| {
            env::split_paths(&value)
                .filter(|path| !path.as_os_str().is_empty())
                .collect::<Vec<_>>()
        })
        .or(file_value)
        .unwrap_or_default()
}

fn execution_providers_setting(file_value: Option<Vec<String>>) -> Vec<String> {
    // Keep provider names normalized once here; inference can then match simple
    // strings without accepting every spelling variant again.
    let values = env::var("EXECUTION_PROVIDERS")
        .ok()
        .map(|value| {
            value
                .split(',')
                .map(str::to_string)
                .collect::<Vec<String>>()
        })
        .or(file_value)
        .unwrap_or_else(|| vec![DEFAULT_EXECUTION_PROVIDER.to_string()]);

    let normalized = values
        .into_iter()
        .map(|value| value.trim().to_ascii_lowercase().replace(['-', '_'], ""))
        .filter(|value| !value.is_empty())
        .collect::<Vec<_>>();

    if normalized.is_empty() {
        vec![DEFAULT_EXECUTION_PROVIDER.to_string()]
    } else {
        normalized
    }
}

fn parse_model_variant(
    file_value: Option<String>,
    has_model_path_override: bool,
) -> anyhow::Result<ModelVariant> {
    let Some(raw) = env::var("MODEL_VARIANT").ok().or(file_value) else {
        return Ok(if has_model_path_override {
            ModelVariant::Custom
        } else {
            ModelVariant::Fp32
        });
    };

    parse_model_variant_value(&raw)
}

fn parse_model_variant_value(raw: &str) -> anyhow::Result<ModelVariant> {
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

fn default_worker_count() -> usize {
    std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1)
        .clamp(1, 4)
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        time::{SystemTime, UNIX_EPOCH},
    };

    use super::*;

    #[test]
    fn parses_supported_model_variant_aliases() {
        let cases = [
            ("fp32", "fp32"),
            ("float-16", "fp16"),
            ("UINT_8", "uint8"),
            ("quant", "quantized"),
            ("q4_fp16", "q4f16"),
            ("bitsandbytes4", "bnb4"),
            ("custom", "custom"),
        ];

        for (raw, expected) in cases {
            let variant = parse_model_variant_value(raw).unwrap();
            assert_eq!(variant.as_str(), expected);
        }
    }

    #[test]
    fn rejects_unknown_model_variant() {
        let err = parse_model_variant_value("fp12").unwrap_err();

        assert!(err.to_string().contains("unsupported model variant `fp12`"));
    }

    #[test]
    fn normalizes_execution_provider_names() {
        let providers = execution_providers_setting(Some(vec![
            " CoreML-GPU ".to_string(),
            "CUDA".to_string(),
            "xnn_pack".to_string(),
        ]));

        assert_eq!(providers, vec!["coremlgpu", "cuda", "xnnpack"]);
    }

    #[test]
    fn model_variant_selects_matching_default_model_path() {
        let cases = [
            (
                ModelVariant::Fp32,
                "Florence-2-base/onnx/vision_encoder.onnx",
            ),
            (
                ModelVariant::Fp16,
                "Florence-2-base/onnx/vision_encoder_fp16.onnx",
            ),
            (
                ModelVariant::Q4F16,
                "Florence-2-base/onnx/vision_encoder_q4f16.onnx",
            ),
        ];

        for (variant, expected) in cases {
            assert_eq!(variant.default_model_path(), PathBuf::from(expected));
        }
    }

    #[test]
    fn explicit_missing_config_path_is_an_error() {
        let path = unique_temp_path("missing-config.toml");
        let err = FileConfig::load(Some(path.clone())).unwrap_err();

        assert!(
            err.to_string()
                .contains(&format!("config file does not exist: {}", path.display()))
        );
    }

    #[test]
    fn loads_file_config_from_toml() {
        let path = unique_temp_path("config.toml");
        fs::write(
            &path,
            r#"
[server]
bind_addr = "127.0.0.1:9999"
data_dir = "tmp-data"

[model]
variant = "q4-f16"

[queue]
model_pool_size = 2
queue_size = 8
body_limit_bytes = 4096

[generation]
max_new_tokens = 32

[runtime]
execution_providers = ["coreml-gpu", "xnnpack"]

[logging]
rust_log = "debug"
"#,
        )
        .unwrap();

        let config = FileConfig::load(Some(path.clone())).unwrap();
        fs::remove_file(path).unwrap();

        let server = config.server.unwrap();
        let model = config.model.unwrap();
        let queue = config.queue.unwrap();
        let generation = config.generation.unwrap();
        let runtime = config.runtime.unwrap();
        let logging = config.logging.unwrap();

        assert_eq!(server.bind_addr.as_deref(), Some("127.0.0.1:9999"));
        assert_eq!(server.data_dir, Some(PathBuf::from("tmp-data")));
        assert_eq!(model.variant.as_deref(), Some("q4-f16"));
        assert_eq!(queue.model_pool_size, Some(2));
        assert_eq!(queue.queue_size, Some(8));
        assert_eq!(queue.body_limit_bytes, Some(4096));
        assert_eq!(generation.max_new_tokens, Some(32));
        assert_eq!(
            runtime.execution_providers,
            Some(vec!["coreml-gpu".to_string(), "xnnpack".to_string()])
        );
        assert_eq!(logging.rust_log.as_deref(), Some("debug"));
    }

    fn unique_temp_path(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();

        env::temp_dir().join(format!("florence2-base-inference-server-{nanos}-{name}"))
    }
}
