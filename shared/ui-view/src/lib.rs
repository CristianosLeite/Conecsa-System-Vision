// SPDX-FileCopyrightText: 2026 Conecsa
//
// SPDX-License-Identifier: Apache-2.0

//! Zoomable image view shared by the Conecsa device UI and the hub: the view
//! transform math and the gesture decisions (plain Rust, tested natively), the
//! wasm32 viewport glue that applies them to DOM events, and the page-zoom
//! guard. See `README.md` for the DOM contract.

pub mod gestures;
pub mod transform;
#[cfg(target_arch = "wasm32")]
pub mod viewport;
#[cfg(target_arch = "wasm32")]
pub mod zoom_guard;
