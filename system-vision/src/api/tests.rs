// SPDX-FileCopyrightText: 2026 Conecsa
//
// SPDX-License-Identifier: AGPL-3.0-only

//! Unit tests for the pure classes-text parser.
use super::*;
use wasm_bindgen_test::*;

#[wasm_bindgen_test]
fn splits_one_class_per_line() {
    assert_eq!(parse_classes_text("person\ncar\ndog"), vec!["person", "car", "dog"]);
}

#[wasm_bindgen_test]
fn trims_whitespace_around_each_class() {
    assert_eq!(parse_classes_text("  person \n\tcar\t"), vec!["person", "car"]);
}

#[wasm_bindgen_test]
fn drops_blank_and_whitespace_only_lines() {
    assert_eq!(parse_classes_text("person\n\n   \ncar\n"), vec!["person", "car"]);
}

#[wasm_bindgen_test]
fn handles_crlf_line_endings() {
    assert_eq!(parse_classes_text("person\r\ncar\r\n"), vec!["person", "car"]);
}

#[wasm_bindgen_test]
fn empty_or_whitespace_input_yields_no_classes() {
    assert!(parse_classes_text("").is_empty());
    assert!(parse_classes_text(" \n\t\n").is_empty());
}

fn job(status: &str) -> TrainingJobStatus {
    serde_json::from_value(serde_json::json!({ "status": status })).unwrap()
}

#[wasm_bindgen_test]
fn training_job_is_active_only_while_it_owns_the_gpu() {
    for s in ["preparing", "training", "uploading"] {
        assert!(job(s).is_active(), "{s} must count as active");
    }
    for s in ["idle", "done", "failed", "cancelled", ""] {
        assert!(!job(s).is_active(), "{s:?} must not count as active");
    }
}

// ── camera source ────────────────────────────────────────────────────────────

use super::{camera_source_body, CameraDevicesResponse, CameraHealth, SOURCE_NETWORK};
use serde_json::json;

#[wasm_bindgen_test]
fn a_camera_response_without_source_fields_reads_as_a_local_camera() {
    // A backend that predates the remote camera must still deserialize.
    let resp: CameraDevicesResponse = serde_json::from_value(json!({
        "devices": [], "current_device": "/dev/video0", "current_index": 0,
        "current_width": 1280, "current_height": 720, "current_framerate": 30,
        "current_auto_exposure": true, "current_exposure_time": 156,
        "current_rgb_red": 128, "current_rgb_green": 128, "current_rgb_blue": 128,
        "current_gamma": 100,
    }))
    .unwrap();
    assert_eq!(resp.current_source, "local");
    assert_eq!(resp.current_network_host, "");
    assert_eq!(resp.current_network_port, 0);
    assert!(!resp.network_token_set);
    assert_eq!(resp.camera_detail, "");
}

#[wasm_bindgen_test]
fn an_omitted_token_is_absent_from_the_request_not_empty() {
    // Absent means "keep the stored token"; the device rejects an empty one.
    let body = camera_source_body("network", Some("192.0.2.1"), Some(8080), None);
    assert_eq!(body, json!({"source": "network", "network_host": "192.0.2.1", "network_port": 8080}));
    assert!(body.get("network_token").is_none());

    let body = camera_source_body("network", Some("192.0.2.1"), Some(8080), Some("ABCD-1234"));
    assert_eq!(body["network_token"], "ABCD-1234");

    assert_eq!(camera_source_body("local", None, None, None), json!({"source": "local"}));
}

#[wasm_bindgen_test]
fn camera_health_event_defaults_its_optional_fields() {
    let health: CameraHealth = serde_json::from_value(json!({"status": "no_camera"})).unwrap();
    assert_eq!(health.status, "no_camera");
    assert_eq!(health.detail, "");
    assert_eq!(health.source, "");
    let full: CameraHealth = serde_json::from_value(
        json!({"status": "no_camera", "detail": "unauthorized", "source": "network"}),
    )
    .unwrap();
    assert_eq!(full.detail, "unauthorized");
    assert_eq!(full.source, SOURCE_NETWORK);
}

#[wasm_bindgen_test]
fn access_point_status_defaults_everything_but_what_the_device_sent() {
    use super::ApStatus;
    let st: ApStatus = serde_json::from_value(json!({"active": false, "ssid": "conecsa-000001"})).unwrap();
    assert!(!st.active);
    assert_eq!(st.ssid, "conecsa-000001");
    assert!(st.stations.is_empty() && st.channels.is_empty());
    assert_eq!(st.join_deadline_remaining_secs, 0);
    let live: ApStatus = serde_json::from_value(json!({
        "active": true, "ssid": "conecsa-000001", "frequency_mhz": 5180, "address": "10.98.76.1", "prefix": 24,
        "stations": [{"address": "10.98.76.11", "hostname": "pixel", "signal": -50}],
        "join_deadline_remaining_secs": 0, "wired_ready": true, "message": "", "channels": [36, 40, 44]
    }))
    .unwrap();
    assert_eq!(live.stations[0].address, "10.98.76.11");
    assert_eq!(live.channels, vec![36, 40, 44]);
}
