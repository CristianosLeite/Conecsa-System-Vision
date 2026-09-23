// SPDX-FileCopyrightText: 2026 Conecsa
//
// SPDX-License-Identifier: AGPL-3.0-only

//! Face recognition: the people of the latest frame under the video. The
//! device burns the face boxes and names into the stream; detection areas, the
//! confidence (face score) and overlay (NMS) thresholds apply exactly as for
//! object detection. Face photos and the gallery stay on the device.

mod identity_panel;

pub use identity_panel::{identities, Identities, Identity, IdentityPanel, UNKNOWN};
