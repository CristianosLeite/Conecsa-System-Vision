// SPDX-FileCopyrightText: 2026 Conecsa
//
// SPDX-License-Identifier: AGPL-3.0-only

//! Detection control (HTTP/protobuf) and the latest result (JSON).

use prost::Message;

use crate::api::wasm32::http::{fetch_api, fetch_protobuf};
use crate::models::Snapshot;
use crate::proto::detection;

/// GET /api/v1/detections/snapshot — the latest result, without frames.
///
/// `passive=true`: the classification panel polls this while it is open, and
/// it must never count as the hub's heartbeat for the device's offline buffer,
/// even when the device is open inside the hub.
pub async fn get_snapshot() -> Result<Snapshot, String> {
    fetch_api::<Snapshot>(
        "/api/v1/detections/snapshot?include_frame=false&passive=true",
        "GET",
        None,
    )
    .await
}

pub async fn start_detection() -> Result<(), String> {
    let request = detection::StartDetectionRequest {};
    let request_bytes = request.encode_to_vec();
    let response: detection::StartDetectionResponse =
        fetch_protobuf("/api/v1/start", "POST", Some(&request_bytes)).await?;
    if response.success {
        Ok(())
    } else {
        Err(response.message)
    }
}

pub async fn stop_detection() -> Result<(), String> {
    let request = detection::StopDetectionRequest {};
    let request_bytes = request.encode_to_vec();
    let response: detection::StopDetectionResponse =
        fetch_protobuf("/api/v1/stop", "POST", Some(&request_bytes)).await?;
    if response.success {
        Ok(())
    } else {
        Err(response.message)
    }
}

/// Set the confidence threshold and save it with the active model's settings.
///
/// Goes through `PUT /api/v1/config`, as the hub's recipes do: `POST
/// /api/v1/threshold` only changes memory (it stays for short-lived Flow
/// values), so the slider would be lost on the next model load or reboot.
pub async fn set_threshold(threshold: f32) -> Result<(), String> {
    let body = serde_json::json!({ "confidence_threshold": threshold }).to_string();
    let response: serde_json::Value =
        crate::api::fetch_api("/api/v1/config", "PUT", Some(&body)).await?;
    if response.get("success").and_then(|v| v.as_bool()).unwrap_or(true) {
        Ok(())
    } else {
        Err(response
            .get("message")
            .or_else(|| response.get("error"))
            .and_then(|v| v.as_str())
            .unwrap_or("request failed")
            .to_string())
    }
}

pub async fn set_overlay_threshold(threshold: f32) -> Result<(), String> {
    let request = detection::SetThresholdRequest { threshold };
    let request_bytes = request.encode_to_vec();
    let response: detection::SetThresholdResponse =
        fetch_protobuf("/api/v1/overlay_threshold", "POST", Some(&request_bytes)).await?;
    if response.success {
        Ok(())
    } else {
        Err(response.message)
    }
}

/// Set the segmentation instance limit (1..255): the most instances per frame
/// that get a mask, saved with the active model's settings.
pub async fn set_segment_max_instances(max_instances: u32) -> Result<(), String> {
    let body = serde_json::json!({ "max_instances": max_instances }).to_string();
    let response: serde_json::Value =
        crate::api::fetch_api("/api/v1/segment/max_instances", "POST", Some(&body)).await?;
    if response.get("success").and_then(|v| v.as_bool()).unwrap_or(false) {
        Ok(())
    } else {
        Err(response
            .get("message")
            .and_then(|v| v.as_str())
            .unwrap_or("request failed")
            .to_string())
    }
}

/// Face recognition settings to change; `None` fields are left as they are.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct FaceSettings {
    /// Cosine similarity (0..1) a face needs to take a person's name.
    pub match_threshold: Option<f32>,
    /// Faces smaller than this many pixels (0..1024) are ignored.
    pub min_size_px: Option<u32>,
    /// Most faces recognized per frame (1..20).
    pub max_faces: Option<u32>,
}

/// Change the face recognition settings, saved with the active model's
/// settings. Only the fields that are set are sent.
pub async fn set_face_settings(settings: FaceSettings) -> Result<(), String> {
    let mut body = serde_json::Map::new();
    if let Some(v) = settings.match_threshold {
        body.insert("match_threshold".into(), serde_json::json!(v));
    }
    if let Some(v) = settings.min_size_px {
        body.insert("min_size_px".into(), serde_json::json!(v));
    }
    if let Some(v) = settings.max_faces {
        body.insert("max_faces".into(), serde_json::json!(v));
    }
    let body = serde_json::Value::Object(body).to_string();
    let response: serde_json::Value =
        crate::api::fetch_api("/api/v1/face/settings", "POST", Some(&body)).await?;
    if response.get("success").and_then(|v| v.as_bool()).unwrap_or(false) {
        Ok(())
    } else {
        Err(response
            .get("message")
            .and_then(|v| v.as_str())
            .unwrap_or("request failed")
            .to_string())
    }
}
