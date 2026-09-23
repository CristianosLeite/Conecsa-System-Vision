// SPDX-FileCopyrightText: 2026 Conecsa
//
// SPDX-License-Identifier: AGPL-3.0-only

//! Backend access layer (HTTP/SSE on wasm, Tauri IPC on native).

/// Detection-area HTTP client. Mirrors `/api/v1/detection-areas/*` endpoints.
use serde::{Deserialize, Serialize};

use crate::api::wasm32::http::fetch_api;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionArea {
    pub id: String,
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub is_editing: bool,
    #[serde(default = "default_shape")]
    pub shape: String,
}

fn default_shape() -> String {
    "rectangle".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionAreasResponse {
    pub areas: Vec<DetectionArea>,
}

pub async fn list_detection_areas() -> Result<DetectionAreasResponse, String> {
    fetch_api::<DetectionAreasResponse>("/api/v1/detection-areas", "GET", None).await
}

pub async fn create_detection_area() -> Result<DetectionAreasResponse, String> {
    fetch_api::<DetectionAreasResponse>("/api/v1/detection-areas", "POST", Some("{}")).await
}

pub async fn delete_detection_area(id: &str) -> Result<DetectionAreasResponse, String> {
    fetch_api::<DetectionAreasResponse>(&format!("/api/v1/detection-areas/{}", id), "DELETE", None)
        .await
}

pub async fn save_detection_area(id: &str) -> Result<DetectionAreasResponse, String> {
    fetch_api::<DetectionAreasResponse>(
        &format!("/api/v1/detection-areas/{}/save", id),
        "POST",
        Some("{}"),
    )
    .await
}

pub async fn send_area_command(id: &str, action: &str) -> Result<DetectionAreasResponse, String> {
    let body = serde_json::json!({ "action": action }).to_string();
    fetch_api::<DetectionAreasResponse>(
        &format!("/api/v1/detection-areas/{}/command", id),
        "POST",
        Some(&body),
    )
    .await
}

pub async fn edit_detection_area(id: &str) -> Result<DetectionAreasResponse, String> {
    fetch_api::<DetectionAreasResponse>(
        &format!("/api/v1/detection-areas/{}/edit", id),
        "POST",
        Some("{}"),
    )
    .await
}

pub async fn discard_detection_area(id: &str) -> Result<DetectionAreasResponse, String> {
    fetch_api::<DetectionAreasResponse>(
        &format!("/api/v1/detection-areas/{}/discard", id),
        "POST",
        Some("{}"),
    )
    .await
}

pub async fn set_area_shape(id: &str, shape: &str) -> Result<DetectionAreasResponse, String> {
    let body = serde_json::json!({ "shape": shape }).to_string();
    fetch_api::<DetectionAreasResponse>(
        &format!("/api/v1/detection-areas/{}/shape", id),
        "POST",
        Some(&body),
    )
    .await
}
