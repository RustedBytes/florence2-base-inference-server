# Florence 2 Base Inference Server

Minimal Axum server for running Florence-2-base ONNX image-to-text inference in a background worker queue.

## Run

```bash
cargo run
```

Default bind address is `127.0.0.1:3000`.

Open the browser UI:

```bash
open http://127.0.0.1:3000/
```

The browser UI template is embedded into the binary.

## Endpoints

Health:

```bash
curl http://127.0.0.1:3000/health
```

Upload image:

```bash
curl -s \
  -F image=@/path/to/image.png \
  -F task_type='Single task' \
  -F task_prompt='Caption' \
  -F text_input='' \
  http://127.0.0.1:3000/v1/infer
```

Use local server-side image path:

```bash
curl -s -X POST http://127.0.0.1:3000/v1/infer/path \
  -H 'content-type: application/json' \
  -d '{"image_path":"/path/to/image.png","task_type":"Single task","task_prompt":"Caption","text_input":null}'
```

Check job:

```bash
curl http://127.0.0.1:3000/v1/jobs/<job-id>
```

## Task Fields

Requests use the same task fields as the Florence-2 Space:

- `task_type`: `Single task` or `Cascased task`
- `task_prompt`: defaults to `Caption`
- `text_input`: optional text for prompts that need it

Single task prompts:

- `Caption`
- `Detailed Caption`
- `More Detailed Caption`
- `Object Detection`
- `Dense Region Caption`
- `Region Proposal`
- `Caption to Phrase Grounding`
- `Referring Expression Segmentation`
- `Region to Segmentation`
- `Open Vocabulary Detection`
- `Region to Category`
- `Region to Description`
- `OCR`
- `OCR with Region`

Cascased task prompts:

- `Caption + Grounding`
- `Detailed Caption + Grounding`
- `More Detailed Caption + Grounding`

## Configuration

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

[model]
variant = "fp32"
# path = "Florence-2-base/onnx/vision_encoder.onnx"

[queue]
model_pool_size = 1
queue_size = 128
body_limit_bytes = 33554432

[generation]
max_new_tokens = 256

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

Environment variables override TOML values when set:

- `BIND_ADDR`: bind address, default `127.0.0.1:3000`
- `DATA_DIR`: image and JSONL metadata directory, default `data`
- `MODEL_POOL_SIZE`: number of model workers
- `QUEUE_SIZE`: queued job capacity
- `BODY_LIMIT_BYTES`: multipart upload limit
- `MODEL_VARIANT`: model variant
- `MODEL_PATH`: explicit ONNX model path, overrides `MODEL_VARIANT` path selection
- `MAX_NEW_TOKENS`: maximum decoder tokens per generation
- `RUST_LOG`: logging level, for example `debug`
- `CONFIG_PATH`: explicit TOML config path when `--config` is not set

Metadata is written to:

- `data/metadata/submissions.jsonl`
- `data/metadata/results.jsonl`

Inference results include generated Florence text and a JSON object keyed by the Florence task token.
