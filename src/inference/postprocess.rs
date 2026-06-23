use serde_json::{Value, json};

const MAX_LOCATION_TOKEN: u16 = 999;

pub(super) fn post_process_generation(
    task_token: &str,
    generated_text: &str,
    image_size: (u32, u32),
) -> Value {
    if task_token != "<OCR_WITH_REGION>" {
        return json!({ task_token: generated_text });
    }

    // OCR-with-region returns text followed by Florence location tokens.
    // Preserve the raw text and add pixel-space boxes for API consumers.
    let items = parse_ocr_regions(generated_text, image_size);
    if items.is_empty() {
        return json!({ task_token: generated_text });
    }

    json!({
        task_token: {
            "raw": generated_text,
            "items": items,
        }
    })
}

fn parse_ocr_regions(generated_text: &str, image_size: (u32, u32)) -> Vec<Value> {
    let mut items = Vec::new();
    let mut cursor = 0;

    while let Some(relative_loc_start) = generated_text[cursor..].find("<loc_") {
        let loc_start = cursor + relative_loc_start;
        let text = generated_text[cursor..loc_start].trim().to_string();
        let mut loc_tokens = Vec::new();
        let mut loc_cursor = loc_start;

        // A single OCR item is encoded as text plus a contiguous run of
        // <loc_N> tokens. Four pairs form the usual quadrilateral.
        while let Some((loc, next_cursor)) = parse_loc_token_at(generated_text, loc_cursor) {
            loc_tokens.push(loc);
            loc_cursor = next_cursor;
        }

        if !loc_tokens.is_empty()
            && let Some(item) = ocr_region_item(text, loc_tokens, image_size)
        {
            items.push(item);
        }

        cursor = loc_cursor;
    }

    items
}

fn parse_loc_token_at(input: &str, cursor: usize) -> Option<(u16, usize)> {
    let remaining = input.get(cursor..)?;
    let remaining = remaining.strip_prefix("<loc_")?;
    let end = remaining.find('>')?;
    let loc = remaining[..end].parse::<u16>().ok()?;
    Some((
        loc.min(MAX_LOCATION_TOKEN),
        cursor + "<loc_".len() + end + 1,
    ))
}

fn ocr_region_item(text: String, loc_tokens: Vec<u16>, image_size: (u32, u32)) -> Option<Value> {
    let points = loc_tokens
        .chunks_exact(2)
        .map(|pair| {
            let x = scale_loc(pair[0], image_size.0);
            let y = scale_loc(pair[1], image_size.1);
            (x, y)
        })
        .collect::<Vec<_>>();

    if points.len() < 2 {
        return None;
    }

    let (mut x_min, mut y_min) = points[0];
    let (mut x_max, mut y_max) = points[0];
    for &(x, y) in &points[1..] {
        x_min = x_min.min(x);
        y_min = y_min.min(y);
        x_max = x_max.max(x);
        y_max = y_max.max(y);
    }

    let polygon = points
        .iter()
        .map(|(x, y)| json!({ "x": x, "y": y }))
        .collect::<Vec<_>>();

    Some(json!({
        "text": text,
        "bbox": {
            "x_min": x_min,
            "y_min": y_min,
            "x_max": x_max,
            "y_max": y_max,
            "width": round3(x_max - x_min),
            "height": round3(y_max - y_min),
        },
        "bbox_xyxy": [x_min, y_min, x_max, y_max],
        "polygon": polygon,
        "loc_tokens": loc_tokens,
    }))
}

fn scale_loc(value: u16, image_side: u32) -> f64 {
    // Florence location bins are normalized to 0..999, not image pixels.
    round3((f64::from(value) / 999.0) * f64::from(image_side))
}

fn round3(value: f64) -> f64 {
    (value * 1000.0).round() / 1000.0
}

pub(super) fn clean_generated_text(raw: &str) -> String {
    raw.replace("<s>", "")
        .replace("</s>", "")
        .replace("<pad>", "")
        .trim()
        .to_string()
}
