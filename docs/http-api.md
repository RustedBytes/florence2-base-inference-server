# HTTP API

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
