// SPDX-FileCopyrightText: 2026 Conecsa
//
// SPDX-License-Identifier: AGPL-3.0-only

//! Backend access layer (HTTP/SSE on wasm, Tauri IPC on native).

use crate::api::wasm32::http::fetch_api;
use crate::models::ApplicationInfo;

/// GET /api/v1/application — the device's application type.
pub async fn get_application() -> Result<ApplicationInfo, String> {
    fetch_api::<ApplicationInfo>("/api/v1/application", "GET", None).await
}

/// PUT /api/v1/application — switch the application type (admin). The
/// backend refuses (409) while the GPU is busy and a task it cannot run; the
/// error carries its reason.
pub async fn set_application(task: &str) -> Result<ApplicationInfo, String> {
    let body = serde_json::json!({ "task": task }).to_string();
    fetch_api::<ApplicationInfo>("/api/v1/application", "PUT", Some(&body)).await
}
