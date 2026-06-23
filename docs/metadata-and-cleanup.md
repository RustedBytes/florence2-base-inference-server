# Metadata and Cleanup

Metadata is written to:

- `data/metadata/submissions.jsonl`
- `data/metadata/results.jsonl`

Uploaded image files in `data/images/` are removed after each job finishes. Images submitted through `/v1/infer/path` are treated as caller-owned files and are not deleted.

Inference results include generated Florence text and a JSON object keyed by the Florence task token.
