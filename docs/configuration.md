# Configuration

Copy the sample TOML config:

```bash
cp config.example.toml config.toml
```

The server reads `config.toml` by default. Use another file with:

```bash
cargo run -- --config /path/to/config.toml
```

`CONFIG_PATH=/path/to/config.toml cargo run` is also supported when no `--config` argument is provided.

Sample config:

```toml
[server]
bind_addr = "127.0.0.1:3000"
data_dir = "data"
allow_local_paths = false
local_path_roots = []

[model]
variant = "fp32"
# path = "Florence-2-base/onnx/vision_encoder.onnx"

[queue]
model_pool_size = 1
queue_size = 128
body_limit_bytes = 33554432

[generation]
max_new_tokens = 256

[runtime]
execution_providers = ["auto"]

[logging]
rust_log = "info,ort=warn"
```

Supported `model.variant` values:

- `fp32`
- `fp16`
- `int8`
- `uint8`
- `quantized`
- `q4`
- `q4f16`
- `bnb4`
- `custom`

Supported `runtime.execution_providers` values:

- `auto`: ORT auto-device policy, with CPU fallback
- `cpu`: CPU only
- `coreml`: CoreML with all Apple compute units
- `coreml_gpu`: CoreML CPU+GPU
- `coreml_npu`: CoreML CPU+Neural Engine
- `xnnpack`: XNNPACK when available in the ORT build
- `cuda`: NVIDIA CUDA, only when built with the Cargo `cuda` feature

`auto` is the default. On Apple Silicon, CoreML may register successfully but still compile only parts of Florence's dynamic ONNX graphs and can emit unbounded-dimension warnings during session load.

Build with CUDA support:

```bash
cargo run --features cuda
```

Then request CUDA explicitly:

```toml
[runtime]
execution_providers = ["cuda", "cpu"]
```

or with the environment:

```bash
EXECUTION_PROVIDERS=cuda,cpu cargo run --features cuda
```

Environment variables override TOML values when set:

- `BIND_ADDR`: bind address, default `127.0.0.1:3000`
- `DATA_DIR`: image and JSONL metadata directory, default `data`
- `ALLOW_LOCAL_PATHS`: set to `true` to enable `/v1/infer/path`
- `LOCAL_PATH_ROOTS`: platform-separated allowed roots for `/v1/infer/path`
- `MODEL_POOL_SIZE`: number of model workers
- `QUEUE_SIZE`: queued job capacity
- `BODY_LIMIT_BYTES`: multipart upload limit
- `MODEL_VARIANT`: model variant
- `MODEL_PATH`: explicit ONNX model path, overrides `MODEL_VARIANT` path selection
- `MAX_NEW_TOKENS`: maximum decoder tokens per generation
- `EXECUTION_PROVIDERS`: comma-separated provider list, for example `auto` or `coreml,auto`
- `RUST_LOG`: logging level, for example `debug`
- `CONFIG_PATH`: explicit TOML config path when `--config` is not set

`/v1/infer/path` is disabled by default. To enable it safely:

```toml
[server]
allow_local_paths = true
local_path_roots = ["/srv/florence-inputs"]
```

Only files under the configured roots are accepted. The server rejects startup configuration where local paths are enabled without at least one root.
