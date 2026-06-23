use std::path::Path;

use image::ImageFormat;
use log::{debug, trace};
use serde::Serialize;
use sha2::{Digest, Sha256};
use tokio::fs;

pub async fn append_jsonl<T: Serialize>(path: &Path, value: &T) -> anyhow::Result<()> {
    let mut line = serde_json::to_vec(value)?;
    line.push(b'\n');
    trace!(
        "appending JSONL path={} bytes={}",
        path.display(),
        line.len()
    );

    use tokio::io::AsyncWriteExt;
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .await?;
    file.write_all(&line).await?;
    file.flush().await?;
    debug!(
        "JSONL append flushed path={} bytes={}",
        path.display(),
        line.len()
    );
    Ok(())
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

pub fn guess_extension(content_type: Option<&str>, bytes: &[u8]) -> &'static str {
    match content_type {
        Some("image/png") => "png",
        Some("image/jpeg") => "jpg",
        Some("image/webp") => "webp",
        Some("image/bmp") => "bmp",
        Some("image/tiff") => "tiff",
        _ => image::guess_format(bytes)
            .ok()
            .and_then(image_format_extension)
            .unwrap_or("img"),
    }
}

pub fn image_format_content_type(format: ImageFormat) -> Option<&'static str> {
    match format {
        ImageFormat::Png => Some("image/png"),
        ImageFormat::Jpeg => Some("image/jpeg"),
        ImageFormat::WebP => Some("image/webp"),
        ImageFormat::Bmp => Some("image/bmp"),
        ImageFormat::Tiff => Some("image/tiff"),
        _ => None,
    }
}

fn image_format_extension(format: ImageFormat) -> Option<&'static str> {
    match format {
        ImageFormat::Png => Some("png"),
        ImageFormat::Jpeg => Some("jpg"),
        ImageFormat::WebP => Some("webp"),
        ImageFormat::Bmp => Some("bmp"),
        ImageFormat::Tiff => Some("tiff"),
        _ => None,
    }
}
