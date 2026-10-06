// SPDX-FileCopyrightText: 2026 Conecsa
//
// SPDX-License-Identifier: AGPL-3.0-only

mod ap_confirm_modal;
mod apply_button;
mod component;
mod device_select;
mod ip_cameras_form;
mod loading_state;
mod network_source_form;
mod resolution_controls;
mod source_select;
mod status_row;
mod stereo_toggle;

#[cfg(test)]
mod tests;

pub use component::CameraSettings;

use crate::api::{self, ApStatus, IpCamera, IpCameraUpdate};

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

// ── IP cameras ───────────────────────────────────────────────────────────────

/// Limits the device enforces on an IP camera; checked here first so the
/// operator gets a precise message instead of a round trip. Like the device,
/// they count UTF-8 bytes, so an accented letter takes two.
const IP_NAME_MAX_BYTES: usize = 40;
const IP_URL_MAX_BYTES: usize = 256;
const IP_CREDENTIAL_MAX_BYTES: usize = 64;
const RTSP_SCHEME: &str = "rtsp://";

/// The URL after its `rtsp://` scheme, which the device accepts in any case.
fn strip_rtsp_scheme(url: &str) -> Option<&str> {
    let scheme = url.get(..RTSP_SCHEME.len())?;
    scheme.eq_ignore_ascii_case(RTSP_SCHEME).then(|| &url[RTSP_SCHEME.len()..])
}

/// One row of the IP camera form, as the operator is editing it.
///
/// `key` only identifies the row on this page (the keyed list needs one before
/// the device has assigned an `id`). The password is write-only: `password`
/// is what the operator typed, `password_set` whether the device has one
/// stored, and `clear_password` the explicit request to remove it.
#[derive(Debug, Clone, PartialEq, Default)]
struct IpCameraDraft {
    key: u32,
    id: Option<String>,
    name: String,
    url: String,
    username: String,
    password: String,
    password_set: bool,
    clear_password: bool,
    status: String,
    detail: String,
}

impl IpCameraDraft {
    fn from_stored(key: u32, cam: &IpCamera) -> Self {
        Self {
            key,
            id: (!cam.id.is_empty()).then(|| cam.id.clone()),
            name: cam.name.clone(),
            url: cam.url.clone(),
            username: cam.username.clone(),
            password: String::new(),
            password_set: cam.password_set,
            clear_password: false,
            status: cam.status.clone(),
            detail: cam.detail.clone(),
        }
    }
}

/// What is wrong with one IP camera row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IpFieldError {
    UrlScheme,
    UrlCredentials,
    UrlTooLong,
    NameTooLong,
    UsernameTooLong,
    PasswordTooLong,
}

/// What is wrong with the IP camera form, checked before anything is sent.
/// `Field` carries the 0-based row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IpFormError {
    Empty,
    TooMany,
    Field(usize, IpFieldError),
}

/// Whether an `rtsp://` URL carries credentials in its authority
/// (`rtsp://user:pass@host/...`): they belong in the username and password
/// fields, where the password stays write-only.
fn rtsp_url_has_credentials(url: &str) -> bool {
    let rest = strip_rtsp_scheme(url).unwrap_or("");
    let authority = rest.split(['/', '?', '#']).next().unwrap_or("");
    authority.contains('@')
}

fn check_ip_row(row: &IpCameraDraft) -> Result<IpCameraUpdate, IpFieldError> {
    let url = row.url.trim();
    let host = strip_rtsp_scheme(url).unwrap_or("");
    if host.is_empty() || host.starts_with(['/', '?', '#', ':']) {
        return Err(IpFieldError::UrlScheme);
    }
    if url.len() > IP_URL_MAX_BYTES {
        return Err(IpFieldError::UrlTooLong);
    }
    if rtsp_url_has_credentials(url) {
        return Err(IpFieldError::UrlCredentials);
    }
    let name = row.name.trim();
    if name.len() > IP_NAME_MAX_BYTES {
        return Err(IpFieldError::NameTooLong);
    }
    let username = row.username.trim();
    if username.len() > IP_CREDENTIAL_MAX_BYTES {
        return Err(IpFieldError::UsernameTooLong);
    }
    if row.password.len() > IP_CREDENTIAL_MAX_BYTES {
        return Err(IpFieldError::PasswordTooLong);
    }
    // A typed password wins; otherwise an explicit removal sends "", and
    // nothing at all keeps the stored one (a new camera has none to keep).
    let password = if !row.password.is_empty() {
        Some(row.password.clone())
    } else if row.clear_password && row.id.is_some() {
        Some(String::new())
    } else {
        None
    };
    Ok(IpCameraUpdate {
        id: row.id.clone(),
        name: name.to_string(),
        url: url.to_string(),
        username: username.to_string(),
        password,
    })
}

/// Validate the IP camera form into the complete, ordered list the device
/// takes. The device validates again.
fn parse_ip_form(rows: &[IpCameraDraft], max: usize) -> Result<Vec<IpCameraUpdate>, IpFormError> {
    if rows.is_empty() {
        return Err(IpFormError::Empty);
    }
    if rows.len() > max {
        return Err(IpFormError::TooMany);
    }
    rows.iter()
        .enumerate()
        .map(|(i, row)| check_ip_row(row).map_err(|e| IpFormError::Field(i, e)))
        .collect()
}

/// The mosaic grid `(columns, rows)` for a camera count: one camera fills
/// the frame, two sit side by side, three or four share a 2 x 2 grid.
fn mosaic_grid(count: usize) -> (usize, usize) {
    match count {
        0 | 1 => (1, 1),
        2 => (2, 1),
        _ => (2, 2),
    }
}

/// Whether applying `rows` changes what the frame shows under the detection
/// areas: a different mosaic grid, or a cell that shows another camera —
/// moved, new in that position, or the same entry pointed at another URL.
/// `applied` is the stored cameras as `(id, url)` in mosaic order when the IP
/// cameras are the applied source, and `None` for a single-frame source
/// (local or remote camera): switching from it always changes the scene.
fn mosaic_changes(applied: Option<&[(String, String)]>, rows: &[IpCameraDraft]) -> bool {
    let Some(applied) = applied else {
        return true;
    };
    if mosaic_grid(applied.len()) != mosaic_grid(rows.len()) {
        return true;
    }
    rows.iter().enumerate().any(|(i, row)| match (applied.get(i), row.id.as_deref()) {
        // A new camera in a cell that was empty does not move the others.
        (None, _) => false,
        (Some((id, url)), Some(row_id)) => id != row_id || url.trim() != row.url.trim(),
        (Some(_), None) => true,
    })
}
