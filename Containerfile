FROM docker.io/library/rust:1-bookworm AS builder

ARG APP_NAME=florence2-base-inference-server

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        libssl-dev \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

COPY Cargo.toml Cargo.lock ./
COPY src ./src

RUN set -eux; \
    cargo build --release --locked; \
    mkdir -p /out/bin /out/lib; \
    cp "target/release/${APP_NAME}" /out/bin/; \
    find target/release target/release/deps \
        -maxdepth 1 \
        \( -type f -o -type l \) \
        \( -name '*.so' -o -name '*.so.*' \) \
        -exec cp -L '{}' /out/lib/ \;

FROM docker.io/library/debian:bookworm-slim AS runtime

ARG APP_NAME=florence2-base-inference-server

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        libssl3 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --uid 10001 --home-dir /app --create-home --shell /usr/sbin/nologin florence

WORKDIR /app

COPY --from=builder /out/bin/${APP_NAME} /usr/local/bin/${APP_NAME}
COPY --from=builder /out/lib/ /usr/local/lib/
COPY config.example.toml /app/config.example.toml

RUN mkdir -p /app/data /app/Florence-2-base \
    && chown -R florence:florence /app

ENV BIND_ADDR=0.0.0.0:3000 \
    DATA_DIR=/app/data \
    LD_LIBRARY_PATH=/usr/local/lib \
    RUST_LOG=info,ort=warn

USER florence

EXPOSE 3000
VOLUME ["/app/data", "/app/Florence-2-base"]
STOPSIGNAL SIGTERM

ENTRYPOINT ["florence2-base-inference-server"]
