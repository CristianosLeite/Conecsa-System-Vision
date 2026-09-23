// SPDX-FileCopyrightText: 2026 Conecsa
//
// SPDX-License-Identifier: AGPL-3.0-only

//! Unit tests for the frontend serde models (run in a headless browser).
use super::*;

#[wasm_bindgen_test]
fn only_a_faces_package_is_an_enrollment_package() {
    let model = |name: &str| ModelInfo {
        name: name.into(),
        size: 1,
        modified: 0.0,
        is_active: false,
        task: "face".into(),
    };
    assert!(model("staff.faces").is_enrollment_package());
    assert!(model("staff.FACES").is_enrollment_package());
    assert!(!model("staff.engine").is_enrollment_package());
    assert!(!model("faces.pt").is_enrollment_package());
}
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
fn system_status_round_trip() {
    let status = SystemStatus {
        is_running: true,
        model: "yolo26s".into(),
        confidence_threshold: 0.75,
        overlay_threshold: 0.45,
        acceleration_type: "TensorRT".into(),
        camera_connected: true,
        task: Some("detect".into()),
        segment_max_instances: Some(32),
        face_match_threshold: None,
        face_min_size_px: None,
        face_max_faces: None,
        stats: PerformanceStats {
            fps: 30.0,
            inference_time: 12.5,
            detections: 3,
            frames_with_detections: 100,
        },
        protocols: ProtocolInfo { http_port: 80 },
    };
    let json = serde_json::to_string(&status).unwrap();
    let back: SystemStatus = serde_json::from_str(&json).unwrap();
    assert_eq!(back.model, "yolo26s");
    assert_eq!(back.stats.fps, 30.0);
    assert_eq!(back.protocols.http_port, 80);
}

#[wasm_bindgen_test]
fn a_classification_snapshot_has_no_bbox_and_carries_candidates() {
    let json = r##"{
        "task": "classify", "total": 1, "frame": null, "raw_frame": null,
        "detections": [{"class_name": "dog", "color": "#00ff00", "confidence": 0.8, "area": null}],
        "candidates": [{"class_id": 1, "class_name": "dog", "confidence": 0.8},
                       {"class_id": 0, "class_name": "cat", "confidence": 0.2}]
    }"##;
    let snapshot: Snapshot = serde_json::from_str(json).unwrap();
    let top = snapshot.top_class().unwrap();
    assert_eq!(top.class_name, "dog");
    assert_eq!(top.color.as_deref(), Some("#00ff00"));
    assert!(top.bbox.is_none());
    let names: Vec<_> = snapshot
        .candidates
        .unwrap()
        .into_iter()
        .map(|c| c.class_name)
        .collect();
    assert_eq!(names, ["dog", "cat"]);
}

#[wasm_bindgen_test]
fn a_detection_snapshot_keeps_its_boxes_and_has_no_candidates() {
    let json = r#"{
        "task": "detect", "total": 1,
        "detections": [{"class_name": "cap", "color": null, "confidence": 0.9, "area": null,
                        "bbox": [0.1, 0.2, 0.3, 0.4]}]
    }"#;
    let snapshot: Snapshot = serde_json::from_str(json).unwrap();
    assert_eq!(snapshot.detections[0].bbox, Some([0.1, 0.2, 0.3, 0.4]));
    assert!(snapshot.candidates.is_none());
}

#[wasm_bindgen_test]
fn a_classification_frame_without_a_class_has_no_top_class() {
    let json = r#"{"task": "classify", "total": 0, "detections": [],
                   "candidates": [{"class_id": 0, "class_name": "cat", "confidence": 0.3}]}"#;
    let snapshot: Snapshot = serde_json::from_str(json).unwrap();
    assert!(snapshot.top_class().is_none());
    assert_eq!(snapshot.total, 0);
}

#[wasm_bindgen_test]
fn system_status_defaults_missing_acceleration_type() {
    let json = r#"{
        "is_running": false, "model": "m",
        "confidence_threshold": 0.5, "overlay_threshold": 0.4,
        "stats": {"fps": 0.0, "inference_time": 0.0, "detections": 0, "frames_with_detections": 0},
        "protocols": {}
    }"#;
    let status: SystemStatus = serde_json::from_str(json).unwrap();
    assert_eq!(status.acceleration_type, "");
    assert_eq!(status.protocols.http_port, 0); // ProtocolInfo default
    // Older firmware without the field must not make the UI claim the camera
    // is gone (which would also disable Start Detection).
    assert!(status.camera_connected);
    // …nor claim an application type it never reported.
    assert_eq!(status.task, None);
    assert_eq!(status.face_match_threshold, None);
}

#[wasm_bindgen_test]
fn model_info_round_trip() {
    let info = ModelInfo {
        name: "weights.engine".into(),
        size: 1024,
        modified: 1_700_000_000.0,
        is_active: true,
        task: "detect".into(),
    };
    let back: ModelInfo = serde_json::from_str(&serde_json::to_string(&info).unwrap()).unwrap();
    assert_eq!(back.name, "weights.engine");
    assert!(back.is_active);
}

#[wasm_bindgen_test]
fn model_info_without_task_is_a_detection_model() {
    let json = r#"{"name": "old.engine", "size": 1, "modified": 0.0, "is_active": false}"#;
    let info: ModelInfo = serde_json::from_str(json).unwrap();
    assert_eq!(info.task, "detect");
}

#[wasm_bindgen_test]
fn task_ids_round_trip() {
    for task in Task::ALL {
        assert_eq!(Task::parse(task.id()), Some(task));
    }
    assert_eq!(Task::ALL.len(), 4);
    assert_eq!(Task::parse("face"), Some(Task::Face));
    assert_eq!(Task::parse("pose"), None);
    assert_eq!(Task::parse("Detect"), None);
}

#[wasm_bindgen_test]
fn classification_and_face_label_whole_images() {
    assert!(Task::Classify.uses_image_class());
    assert!(Task::Face.uses_image_class());
    assert!(!Task::Detect.uses_image_class());
    assert!(!Task::Segment.uses_image_class());
}

#[wasm_bindgen_test]
fn a_face_snapshot_has_the_detection_shape() {
    let json = r##"{
        "task": "face", "total": 2,
        "detections": [
            {"class_name": "Ana", "color": "#00ff00", "confidence": 0.71, "area": null,
             "bbox": [0.1, 0.2, 0.3, 0.4]},
            {"class_name": "unknown", "color": "#888888", "confidence": 0.2, "area": null,
             "bbox": [0.5, 0.2, 0.6, 0.4]}
        ]
    }"##;
    let snapshot: Snapshot = serde_json::from_str(json).unwrap();
    assert_eq!(snapshot.total, 2);
    assert_eq!(snapshot.detections[0].class_name, "Ana");
    assert!(snapshot.detections.iter().all(|d| d.bbox.is_some()));
    assert!(snapshot.candidates.is_none());
}

#[wasm_bindgen_test]
fn system_status_reads_the_face_settings_when_present() {
    let json = r#"{
        "is_running": true, "model": "staff", "task": "face",
        "confidence_threshold": 0.6, "overlay_threshold": 0.3,
        "face_match_threshold": 0.363, "face_min_size_px": 40, "face_max_faces": 5,
        "stats": {"fps": 0.0, "inference_time": 0.0, "detections": 0, "frames_with_detections": 0},
        "protocols": {}
    }"#;
    let status: SystemStatus = serde_json::from_str(json).unwrap();
    assert_eq!(status.face_match_threshold, Some(0.363));
    assert_eq!(status.face_min_size_px, Some(40));
    assert_eq!(status.face_max_faces, Some(5));
}

#[wasm_bindgen_test]
fn app_state_from_the_reported_task() {
    assert_eq!(AppState::from_task(None), AppState::Unset);
    assert_eq!(AppState::from_task(Some("")), AppState::Unset);
    assert_eq!(AppState::from_task(Some("detect")), AppState::Set(Task::Detect));
    // A newer device's task is unsupported here, never "unset".
    assert_eq!(AppState::from_task(Some("pose")), AppState::Unsupported("pose".into()));
}

#[wasm_bindgen_test]
fn only_unset_and_unsupported_show_the_selector() {
    assert!(!AppState::Loading.gated());
    assert!(!AppState::Error.gated());
    assert!(!AppState::Set(Task::Detect).gated());
    assert!(AppState::Unset.gated());
    assert!(AppState::Unsupported("pose".into()).gated());
    assert!(!AppState::Loading.is_known() && !AppState::Error.is_known());
}

#[wasm_bindgen_test]
fn application_info_supported_tasks() {
    let json = r#"{"task": null, "supported_tasks": ["detect"], "migrated": false}"#;
    let info: ApplicationInfo = serde_json::from_str(json).unwrap();
    assert_eq!(info.task, None);
    assert!(info.supports(Task::Detect));
    assert!(!info.supports(Task::Classify));
    let legacy: ApplicationInfo = serde_json::from_str("{}").unwrap();
    assert!(!legacy.supports(Task::Detect));
}

#[wasm_bindgen_test]
fn detection_config_tuple_resolution() {
    let cfg = DetectionConfig {
        capture_device: "/dev/media0".into(),
        capture_resolution: (1920, 1080),
        capture_framerate: 30,
        model_path: "/data/models/weights.engine".into(),
        confidence_threshold: 0.6,
    };
    let back: DetectionConfig =
        serde_json::from_str(&serde_json::to_string(&cfg).unwrap()).unwrap();
    assert_eq!(back.capture_resolution, (1920, 1080));
}

#[wasm_bindgen_test]
fn system_metrics_optional_gpu_fields_default_to_none() {
    let json = r#"{
        "cpu_usage": 10.0, "ram_usage": 20.0, "ram_total": 100, "ram_used": 20,
        "disk_usage": 30.0, "disk_total": 200, "disk_used": 60, "temperature": 45.0
    }"#;
    let m: SystemMetrics = serde_json::from_str(json).unwrap();
    assert_eq!(m.temperature, Some(45.0));
    assert!(m.gpu_usage.is_none());
    assert!(m.gpu_temperature.is_none());
}
