mod api;
mod config;
mod inference;
mod jobs;
mod state;
mod templates;
mod types;
mod util;

use std::{collections::HashMap, path::PathBuf, sync::Arc};

use anyhow::Context;
use clap::Parser;
use env_logger::Env;
use log::{debug, info};
use state::AppState;
use tokio::sync::RwLock;

use crate::{config::Config, jobs::start_workers};

#[derive(Debug, Parser)]
#[command(author, version, about)]
struct Cli {
    /// TOML config file path. Overrides CONFIG_PATH when set.
    #[arg(short, long, value_name = "FILE")]
    config: Option<PathBuf>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let config = Arc::new(Config::load(cli.config)?);

    env_logger::Builder::from_env(Env::default().default_filter_or(&config.rust_log))
        .format_timestamp_millis()
        .init();

    config.ensure_dirs().await?;
    debug!(
        "config loaded addr={} model_path={} model_variant={} data_dir={} images_dir={} metadata_dir={} workers={} queue_size={} body_limit_bytes={} max_new_tokens={} rust_log={}",
        config.addr,
        config.model_path.display(),
        config.model_variant.as_str(),
        config.data_dir.display(),
        config.images_dir.display(),
        config.metadata_dir.display(),
        config.workers,
        config.queue_size,
        config.body_limit_bytes,
        config.max_new_tokens,
        config.rust_log
    );

    let (queue_tx, queue_rx) = async_channel::bounded(config.queue_size);
    let state = AppState {
        config: Arc::clone(&config),
        queue_tx,
        jobs: Arc::new(RwLock::new(HashMap::new())),
    };

    start_workers(Arc::clone(&config), Arc::clone(&state.jobs), queue_rx);

    let app = api::router(state, config.body_limit_bytes);
    let listener = tokio::net::TcpListener::bind(config.addr)
        .await
        .context("failed to bind TCP listener")?;

    info!(
        "server listening addr={} workers={} model={} model_variant={} data_dir={} queue_size={} body_limit_bytes={} max_new_tokens={}",
        config.addr,
        config.workers,
        config.model_path.display(),
        config.model_variant.as_str(),
        config.data_dir.display(),
        config.queue_size,
        config.body_limit_bytes,
        config.max_new_tokens
    );

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    Ok(())
}

async fn shutdown_signal() {
    if let Err(err) = tokio::signal::ctrl_c().await {
        log::error!("failed to install ctrl-c handler error={}", err);
    }
}
