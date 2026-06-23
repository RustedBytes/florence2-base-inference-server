# Metadata and Cleanup

Metadata is written to:

- `data/metadata/submissions.jsonl`
- `data/metadata/results.jsonl`

On startup, the server reloads these JSONL files so completed job records remain available through `/v1/jobs/{id}` after a restart. Any job that was still `queued` or `running` when the previous process exited is recovered as `failed` and appended to `results.jsonl`.

Uploaded image files in `data/images/` are removed after each job finishes. Images submitted through `/v1/infer/path` are treated as caller-owned files and are not deleted.

Inference results include generated Florence text and a JSON object keyed by the Florence task token.
