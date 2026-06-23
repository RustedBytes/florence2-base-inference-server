# Model Files

Download the Florence-2-base ONNX files from:

```text
https://huggingface.co/onnx-community/Florence-2-base
```

By default, the server expects the model repository contents under `Florence-2-base/`, including `Florence-2-base/onnx/vision_encoder.onnx` and the matching tokenizer files.

Startup validates the selected vision encoder, matching ONNX graph files, and `tokenizer.json` before the server accepts traffic.
