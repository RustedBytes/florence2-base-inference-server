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
  -F webhook_url='https://example.com/florence-webhook' \
  http://127.0.0.1:3000/v1/infer
```

Use local server-side image path:

```bash
curl -s -X POST http://127.0.0.1:3000/v1/infer/path \
  -H 'content-type: application/json' \
  -d '{"image_path":"/path/to/image.png","task_type":"Single task","task_prompt":"Caption","text_input":null,"webhook_url":"https://example.com/florence-webhook"}'
```

Local-path inference is disabled by default. Enable `server.allow_local_paths` and configure `server.local_path_roots` before using this endpoint.

`webhook_url` is optional for both submission endpoints. When set, it must use `http` or `https`, must not include credentials or fragments, and rejects local/private literal IP addresses by default. After the job reaches `succeeded` or `failed`, the server sends a `POST` request to that URL with the final `JobRecord` JSON body, including `status`, `result`, and `error`. Webhook delivery is best-effort: failed callbacks are logged and do not change the job result.

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

Requests return `408` if they exceed the configured `queue.request_timeout_seconds` limit. Inference submissions return `503` when no model worker is ready, when the queue is full, or when the queue has closed. Uploads are rejected before inference if they exceed the configured body limit, have an unsupported content type or format, or exceed the configured image dimensions.
