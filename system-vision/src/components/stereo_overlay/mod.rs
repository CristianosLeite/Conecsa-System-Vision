// SPDX-FileCopyrightText: 2026 Conecsa
//
// SPDX-License-Identifier: AGPL-3.0-only

//! Stereo overlay alignment: toggle button, alignment panel, and range control.

mod alignment_panel;
mod range_control;
mod stereo_overlay;
mod toggle_button;

pub use stereo_overlay::StereoOverlay;

/// Shared `panel` id for this overlay (0 = none). Lets sibling overlays stay
/// mutually exclusive without knowing about each other.
const PANEL_ID: u8 = 1;
