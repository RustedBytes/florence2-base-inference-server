# Running the Server

Start the server:

```bash
cargo run
```

Default bind address is `127.0.0.1:3000`.

Open the browser UI:

```bash
open http://127.0.0.1:3000/
```

The browser UI template is embedded into the binary.

## Podman

Build the container image:

```bash
podman build -f Containerfile -t florence2-base-inference-server .
```

Run it with the downloaded model files mounted at the default path:

```bash
podman volume create florence2-data
podman run --rm \
  -p 3000:3000 \
  -v "$PWD/Florence-2-base:/app/Florence-2-base:ro" \
  -v florence2-data:/app/data:U \
  florence2-base-inference-server
```

The image does not include ONNX model files. Download them separately and mount the `Florence-2-base/` directory into `/app/Florence-2-base`.

To use a custom config file:

```bash
podman run --rm \
  -p 3000:3000 \
  -v "$PWD/Florence-2-base:/app/Florence-2-base:ro" \
  -v "$PWD/config.toml:/app/config.toml:ro" \
  -v florence2-data:/app/data:U \
  florence2-base-inference-server
```

## Compose

Build and run with Compose:

```bash
podman compose -f compose.yml up --build
```

Docker Compose works with the same file:

```bash
docker compose -f compose.yml up --build
```

The Compose file mounts `./Florence-2-base` into the container read-only and stores runtime data in the `florence2-data` named volume.
