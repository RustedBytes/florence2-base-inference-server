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

#[cfg(test)]
mod tests {
    use std::{
        fs,
        time::{SystemTime, UNIX_EPOCH},
    };

    use serde_json::json;

    use super::*;

    const PNG_HEADER: &[u8] = b"\x89PNG\r\n\x1a\n";

    #[test]
    fn hashes_bytes_as_lowercase_sha256_hex() {
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn content_type_takes_precedence_for_extension() {
        assert_eq!(guess_extension(Some("image/jpeg"), PNG_HEADER), "jpg");
    }

    #[test]
    fn guesses_extension_from_image_bytes() {
        assert_eq!(guess_extension(None, PNG_HEADER), "png");
    }

    #[test]
    fn unknown_extension_falls_back_to_img() {
        assert_eq!(guess_extension(None, b"not an image"), "img");
    }

    #[test]
    fn maps_supported_image_formats_to_content_types() {
        assert_eq!(
            image_format_content_type(ImageFormat::Png),
            Some("image/png")
        );
        assert_eq!(
            image_format_content_type(ImageFormat::Jpeg),
            Some("image/jpeg")
        );
        assert_eq!(image_format_content_type(ImageFormat::Gif), None);
    }

    #[tokio::test]
    async fn append_jsonl_writes_one_json_value_per_line() {
        let path = unique_temp_path("append-jsonl.jsonl");

        append_jsonl(&path, &json!({ "id": 1 })).await.unwrap();
        append_jsonl(&path, &json!({ "id": 2 })).await.unwrap();

        let contents = fs::read_to_string(&path).unwrap();
        fs::remove_file(path).unwrap();

        assert_eq!(contents, "{\"id\":1}\n{\"id\":2}\n");
    }

    fn unique_temp_path(name: &str) -> std::path::PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();

        std::env::temp_dir().join(format!("florence2-base-inference-server-{nanos}-{name}"))
    }
}
