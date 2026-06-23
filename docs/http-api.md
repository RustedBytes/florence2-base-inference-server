# HTTP API

Health:

```bash
curl http://127.0.0.1:3000/health
```

Readiness:

```bash
curl http://127.0.0.1:3000/ready
```

`/health` is a liveness endpoint. `/ready` returns `200` only after at least one model worker has initialized successfully; otherwise it returns `503`.

Metrics snapshot:

```bash
curl http://127.0.0.1:3000/metrics
```

OpenAPI document:

```bash
curl http://127.0.0.1:3000/openapi.json
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

Local-path inference is disabled by default. Enable `server.allow_local_paths` and configure `server.local_path_roots` before using this endpoint.

Check job:

```bash
curl http://127.0.0.1:3000/v1/jobs/<job-id>
```

Errors use a stable JSON shape:

```json
{
  "code": "bad_request",
  "message": "uploaded image is empty"
}
```

Known error codes are `bad_request`, `not_found`, `forbidden`, `service_unavailable`, and `internal_error`.

Inference submissions return `503` when no model worker is ready, when the queue is full, or when the queue has closed. Uploads are rejected before inference if they exceed the configured body limit, have an unsupported content type or format, or exceed the configured image dimensions.
