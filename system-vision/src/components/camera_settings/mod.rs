// SPDX-FileCopyrightText: 2026 Conecsa
//
// SPDX-License-Identifier: AGPL-3.0-only

mod ap_confirm_modal;
mod apply_button;
mod component;
mod device_select;
mod loading_state;
mod network_source_form;
mod resolution_controls;
mod source_select;
mod status_row;
mod stereo_toggle;

#[cfg(test)]
mod tests;

pub use component::CameraSettings;

use crate::api::{self, ApStatus};

/// Preset resolutions shown in the UI.
/// Tuples: (width, height, label, aspect-ratio badge)
const RESOLUTIONS: &[(u32, u32, &str, &str)] = &[
    (640, 640, "640 × 640", "1:1"),
    (1280, 720, "1280 × 720", "16:9"),
    (1440, 1080, "1440 × 1080", "4:3"),
    (1920, 1080, "1920 × 1080", "16:9"),
];

#[derive(Clone, Copy, PartialEq)]
struct Resolution {
    w: u32,
    h: u32,
}

type CameraFormat = (u32, u32, Vec<u32>);

/// V4L2 name marker for a side-by-side 3D camera ("3D USB Camera" & friends).
const STEREO_NAME_MARKER: &str = "3d";

/// Whether a V4L2 device name identifies a 3D (side-by-side stereo) camera.
///
/// The overlay blends the left|right halves of one frame into a single image,
/// so offering it for an ordinary camera would split a normal picture in two.
/// The camera model is the only reliable signal: resolution alone cannot tell a
/// side-by-side frame from a wide one.
fn is_stereo_camera(name: &str) -> bool {
    name.to_ascii_lowercase().contains(STEREO_NAME_MARKER)
}

/// What is wrong with the remote camera form, checked before anything is sent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NetworkFormError {
    Address,
    Port,
    Token,
}

/// The token as the operator should see it while typing: upper-case, in
/// groups of four joined by hyphens, anything else dropped.
///
/// A token is Crockford base32, so case and hyphens carry no meaning; doing
/// the formatting for the operator answers the questions "with or without
/// hyphens?" and "capitals?" before they are asked. Letters the alphabet
/// excludes (I, L, O, U) are kept so a typo stays visible and is refused by
/// the device rather than silently turned into another token.
fn format_token_input(raw: &str) -> String {
    let chars: Vec<char> = raw
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .map(|c| c.to_ascii_uppercase())
        .take(32)
        .collect();
    chars
        .chunks(4)
        .map(|group| group.iter().collect::<String>())
        .collect::<Vec<_>>()
        .join("-")
}

/// The token as the device wants it: no hyphens, upper-case.
fn normalize_token(formatted: &str) -> String {
    formatted.chars().filter(|c| c.is_ascii_alphanumeric()).map(|c| c.to_ascii_uppercase()).collect()
}

/// Validate the remote camera form into `(host, port, token)`.
///
/// A blank token is `None` — "keep the stored one" — and only acceptable when
/// the device already has one. The device validates again; this only spares
/// the operator a round trip for the obvious mistakes.
fn parse_network_form(
    host: &str,
    port: &str,
    token: &str,
    token_set: bool,
) -> Result<(String, u32, Option<String>), NetworkFormError> {
    let host = host.trim();
    if host.parse::<std::net::Ipv4Addr>().is_err() {
        return Err(NetworkFormError::Address);
    }
    let port = match port.trim().parse::<u32>() {
        Ok(port) if (1..=65_535).contains(&port) => port,
        _ => return Err(NetworkFormError::Port),
    };
    let token = normalize_token(token);
    if token.is_empty() && !token_set {
        return Err(NetworkFormError::Token);
    }
    Ok((host.to_string(), port, (!token.is_empty()).then_some(token)))
}

/// Where an address the operator has not confirmed yet came from, for the
/// hint under the field. On the remote camera's hotspot the camera is the
/// Wi-Fi gateway; on the device's own access point it is the one station that
/// joined and got a lease.
#[derive(Debug, Clone, PartialEq, Eq)]
enum AddressSuggestion {
    Gateway(String),
    Station(String),
}

/// Whether applying the remote camera should first offer the device's own
/// access point: only when the device reports it off. An unknown state (the
/// hardware agent unreachable) is no offer — Apply must keep working without it,
/// and an active access point needs nothing more.
fn offers_access_point(source: &str, status: Option<&ApStatus>) -> bool {
    source == api::SOURCE_NETWORK && status.is_some_and(|st| !st.active)
}

/// The address the access point can suggest: the one remote camera that
/// joined it and got a lease. A station without a lease has no address yet,
/// and two joined cameras are ambiguous, so neither suggests anything.
fn station_suggestion(status: Option<&ApStatus>) -> Option<String> {
    let st = status.filter(|st| st.active)?;
    match st.stations.as_slice() {
        [only] if !only.address.is_empty() => Some(only.address.clone()),
        _ => None,
    }
}
