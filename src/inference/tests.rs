use super::model::execution_provider_dispatches;
use super::*;

#[test]
fn parses_ocr_region_location_tokens_into_bbox() {
    let generated =
        "let x = 5:<loc_213><loc_402><loc_789><loc_402><loc_789><loc_503><loc_213><loc_502>";

    let result = post_process_generation("<OCR_WITH_REGION>", generated, (1254, 1254));
    let item = &result["<OCR_WITH_REGION>"]["items"][0];

    assert_eq!(item["text"], "let x = 5:");
    assert_eq!(item["loc_tokens"][0], 213);
    assert_eq!(item["loc_tokens"][7], 502);
    assert_eq!(item["bbox"]["x_min"], 267.369);
    assert_eq!(item["bbox"]["y_min"], 504.613);
    assert_eq!(item["bbox"]["x_max"], 990.396);
    assert_eq!(item["bbox"]["y_max"], 631.393);
    assert_eq!(item["polygon"].as_array().unwrap().len(), 4);
}

#[test]
fn cuda_provider_is_feature_gated() {
    let providers = vec!["cuda".to_string()];
    let dispatches = execution_provider_dispatches(&providers);

    assert_eq!(dispatches.len(), usize::from(cfg!(feature = "cuda")));
}

#[test]
fn runtime_backend_summary_keeps_provider_order_visible() {
    let providers = vec!["cuda".to_string(), "cpu".to_string()];
    let devices = vec!["CPUExecutionProvider:CPU".to_string()];

    let summary = runtime_backend_summary(&providers, &devices);

    assert_eq!(
        summary,
        "ort:execution_providers=cuda,cpu;detected_devices=CPUExecutionProvider:CPU"
    );
}

#[test]
fn runtime_backend_summary_handles_missing_devices() {
    let providers = vec!["auto".to_string()];

    let summary = runtime_backend_summary(&providers, &[]);

    assert_eq!(
        summary,
        "ort:execution_providers=auto;detected_devices=none_reported"
    );
}
