// SPDX-FileCopyrightText: 2026 Conecsa
//
// SPDX-License-Identifier: AGPL-3.0-only

//! Backend access layer (HTTP/SSE on wasm, Tauri IPC on native).

pub mod application;
pub mod camera;
pub mod classes;
pub mod detection;
pub mod detection_areas;
pub mod event_stream;
pub mod gpio;
pub mod model;
pub mod network;
pub mod power;
pub mod system;
pub mod training;
