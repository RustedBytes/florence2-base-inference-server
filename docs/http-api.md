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

Inference submissions return `503` when no model worker is ready, when the queue is full, or when the queue has closed.
