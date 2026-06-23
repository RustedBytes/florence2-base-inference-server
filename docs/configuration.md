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
cors_allowed_origins = []

[model]
variant = "fp32"
# path = "Florence-2-base/onnx/vision_encoder.onnx"

[queue]
model_pool_size = 1
queue_size = 128
body_limit_bytes = 33554432
request_timeout_seconds = 60

[retention]
job_retention_limit = 1000
metadata_retention_limit = 10000

[validation]
max_image_width = 8192
max_image_height = 8192

[generation]
max_new_tokens = 256
job_timeout_seconds = 300
webhook_timeout_seconds = 10
webhook_connect_timeout_seconds = 5

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
- `CORS_ALLOWED_ORIGINS`: comma-separated origins allowed by browser CORS checks
- `MODEL_POOL_SIZE`: number of model workers
- `QUEUE_SIZE`: queued job capacity
- `BODY_LIMIT_BYTES`: multipart upload limit
- `REQUEST_TIMEOUT_SECONDS`: whole HTTP request timeout; set to `0` to disable timeout enforcement
- `JOB_RETENTION_LIMIT`: maximum in-memory job records kept queryable through `/v1/jobs/{id}`
- `METADATA_RETENTION_LIMIT`: maximum latest metadata records kept when JSONL files are compacted at startup; set to `0` to disable compaction
- `MAX_IMAGE_WIDTH`: maximum accepted image width before decode
- `MAX_IMAGE_HEIGHT`: maximum accepted image height before decode
- `MODEL_VARIANT`: model variant
- `MODEL_PATH`: explicit ONNX model path, overrides `MODEL_VARIANT` path selection
- `MAX_NEW_TOKENS`: maximum decoder tokens per generation
- `JOB_TIMEOUT_SECONDS`: per-job inference timeout; set to `0` to disable timeout enforcement
- `WEBHOOK_TIMEOUT_SECONDS`: total outbound webhook request timeout; set to `0` to disable
- `WEBHOOK_CONNECT_TIMEOUT_SECONDS`: outbound webhook connection timeout; set to `0` to disable
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

CORS response headers are disabled by default. Configure explicit allowed origins for browser clients:

```toml
[server]
cors_allowed_origins = ["http://localhost:5173"]
```

Only `GET`, `POST`, and `OPTIONS` methods are allowed by the CORS layer.

Request validation happens before the image is decoded for inference:

- `queue.body_limit_bytes` limits multipart upload size.
- `queue.request_timeout_seconds` limits total HTTP request handling time and returns `408 Request Timeout`.
- `validation.max_image_width` and `validation.max_image_height` reject oversized image dimensions.
- Only supported image formats with `image/*` content types are accepted.

Jobs are marked failed if inference exceeds `generation.job_timeout_seconds`. A timed-out blocking inference task may finish in the background, so the worker slot is restarted before it accepts more work.

Webhook delivery uses `generation.webhook_timeout_seconds` for the full request and `generation.webhook_connect_timeout_seconds` for establishing the connection. Webhook timeout failures are logged and do not change the completed job result.
