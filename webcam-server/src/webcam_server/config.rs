// SPDX-FileCopyrightText: 2026 Conecsa
//
// SPDX-License-Identifier: AGPL-3.0-only

//! Camera configuration (source, index, resolution, framerate, exposure, gains,
//! remote and IP camera endpoints) sourced from environment variables and
//! updated live via the SHM config region.

use std::fmt;
use std::net::{Ipv4Addr, Ipv6Addr};

use serde::{Deserialize, Serialize};

const EXPOSURE_MIN: u32 = 1;
const EXPOSURE_MAX: u32 = 300_000;
pub const RGB_LEVEL_DEFAULT: u16 = 128;
pub const GAMMA_DEFAULT: u32 = 100;
pub const GAIN_DEFAULT: u32 = 0;
/// Longest IPv4 literal (`255.255.255.255`).
pub const NETWORK_HOST_MAX: usize = 15;
pub const NETWORK_TOKEN_MIN: usize = 8;
pub const NETWORK_TOKEN_MAX: usize = 32;
/// Most IP cameras one device takes (one SHM ring each).
pub const IP_CAMERAS_MAX: usize = 4;
/// Longest RTSP URL accepted, in bytes.
pub const IP_URL_MAX: usize = 256;
/// Longest RTSP username or password accepted, in bytes.
pub const IP_CREDENTIAL_MAX: usize = 64;
/// What `Debug` prints in place of the stream token or a camera password.
const REDACTED: &str = "<redacted>";

/// Where frames come from: a local V4L2 camera, a remote camera's MJPEG stream
/// or up to four RTSP IP cameras.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CameraSource {
    #[default]
    Local,
    Network,
    Ip,
}

impl CameraSource {
    /// Parse `CAMERA_SOURCE`. An unknown value is an error, never `Local`: a
    /// typo must not silently select a different inspection camera.
    pub fn from_env_text(text: &str) -> Result<Self, String> {
        match text.trim().to_ascii_lowercase().as_str() {
            "" | "local" => Ok(Self::Local),
            "network" => Ok(Self::Network),
            "ip" => Ok(Self::Ip),
            other => Err(format!("CAMERA_SOURCE must be local, network or ip (got {other:?})")),
        }
    }

    /// Map the wire enum; `None` for a value this build does not know.
    pub fn from_proto(value: i32) -> Option<Self> {
        match value {
            0 => Some(Self::Local),
            1 => Some(Self::Network),
            2 => Some(Self::Ip),
            _ => None,
        }
    }
}

/// Strip hyphens and upper-case, so `abcd-1234` and `ABCD1234` are one token.
pub fn normalize_token(token: &str) -> String {
    token
        .chars()
        .filter(|c| *c != '-')
        .map(|c| c.to_ascii_uppercase())
        .collect()
}

/// Validate a complete network-source state (token already normalized).
/// Error text never contains the token.
pub fn validate_network(host: &str, port: u32, token: &str) -> Result<(), String> {
    if host.len() > NETWORK_HOST_MAX || host.parse::<Ipv4Addr>().is_err() {
        return Err("network_host must be an IPv4 literal".to_string());
    }
    if !(1..=65_535).contains(&port) {
        return Err("network_port must be between 1 and 65535".to_string());
    }
    // Crockford base32: digits and upper-case letters without I, L, O, U.
    let crockford = |c: char| c.is_ascii_digit() || (c.is_ascii_uppercase() && !"ILOU".contains(c));
    if !(NETWORK_TOKEN_MIN..=NETWORK_TOKEN_MAX).contains(&token.len())
        || !token.chars().all(crockford)
    {
        return Err("network_token must be 8-32 Crockford base32 characters".to_string());
    }
    Ok(())
}

/// One RTSP IP camera. Its position in `CameraConfig::ip_cameras` is its ring.
#[derive(Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct IpCameraConfig {
    /// `rtsp://host[:port][/path]`, never with credentials in it.
    pub url: String,
    /// RTSP user; empty = no authentication (the password is then unused).
    pub username: String,
    /// RTSP password — a secret: never serialized, never printed (see `Debug`).
    #[serde(skip)]
    pub password: String,
}

// Manual so the password cannot reach a log line through `{:?}`.
impl fmt::Debug for IpCameraConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("IpCameraConfig")
            .field("url", &self.url)
            .field("username", &self.username)
            .field("password", &REDACTED)
            .finish()
    }
}

/// The part of a camera URL after a case-insensitive `rtsp://`, or `None`.
pub fn strip_rtsp_scheme(url: &str) -> Option<&str> {
    let scheme = url.get(..7)?;
    scheme.eq_ignore_ascii_case("rtsp://").then(|| &url[7..])
}

/// True for an ASCII control character (`\x00`-`\x1f`, `\x7f`).
fn is_ascii_control(c: char) -> bool {
    c.is_ascii_control()
}

/// Validate the authority (`host[:port]`) of a camera URL. Host names are
/// taken as given (resolution failures surface as "unreachable"); IP literals
/// must be addresses a camera can have; `localhost` is refused.
fn validate_ip_authority(authority: &str) -> Result<(), String> {
    const UNREACHABLE: &str = "url must name a host reachable by the device";
    let (host, port) = if let Some(rest) = authority.strip_prefix('[') {
        let (literal, after) = rest
            .split_once(']')
            .ok_or_else(|| "url is not a valid RTSP address".to_string())?;
        let ip: Ipv6Addr =
            literal.parse().map_err(|_| "url is not a valid RTSP address".to_string())?;
        if ip.is_loopback() || ip.is_unspecified() || ip.is_multicast() {
            return Err(UNREACHABLE.to_string());
        }
        let port = match after {
            "" => None,
            _ => Some(
                after
                    .strip_prefix(':')
                    .ok_or_else(|| "url is not a valid RTSP address".to_string())?,
            ),
        };
        (None, port)
    } else {
        match authority.split_once(':') {
            Some((host, port)) => (Some(host), Some(port)),
            None => (Some(authority), None),
        }
    };
    if let Some(host) = host {
        if host.is_empty() {
            return Err("url must name a host".to_string());
        }
        if host.eq_ignore_ascii_case("localhost") {
            return Err(UNREACHABLE.to_string());
        }
        if let Ok(ip) = host.parse::<Ipv4Addr>()
            && (ip.is_loopback() || ip.is_unspecified() || ip.is_multicast() || ip.is_broadcast())
        {
            return Err(UNREACHABLE.to_string());
        }
    }
    // An empty port (`host:`) means the default port, as for the API.
    if let Some(port) = port.filter(|p| !p.is_empty()) {
        let valid = port.len() <= 5
            && port.bytes().all(|b| b.is_ascii_digit())
            && port.parse::<u32>().is_ok_and(|p| (1..=65_535).contains(&p));
        if !valid {
            return Err("url port must be between 1 and 65535".to_string());
        }
    }
    Ok(())
}

/// Validate one camera URL: `rtsp://host[:port][/path][?query]`, at most
/// `IP_URL_MAX` bytes, no whitespace or control characters, no credentials in
/// it. Mirrors `validate_rtsp_url` in the inference-service, which must never
/// accept a URL this rejects. Error text never contains the URL (a rejected
/// URL may carry a password).
pub fn validate_ip_url(url: &str) -> Result<(), String> {
    if url.is_empty() {
        return Err("url is required".to_string());
    }
    if url.len() > IP_URL_MAX {
        return Err(format!("url must be at most {IP_URL_MAX} bytes"));
    }
    if url.chars().any(|c| c.is_whitespace() || is_ascii_control(c)) {
        return Err("url must not contain spaces or control characters".to_string());
    }
    let rest = strip_rtsp_scheme(url).ok_or_else(|| "url must start with rtsp://".to_string())?;
    let authority = &rest[..rest.find(['/', '?', '#']).unwrap_or(rest.len())];
    if authority.contains('@') {
        return Err(
            "url must not carry credentials; use the username and password fields".to_string(),
        );
    }
    validate_ip_authority(authority)
}

/// Validate a complete IP camera list: 1 to `IP_CAMERAS_MAX` cameras, each
/// with a valid URL and bounded credentials. Error text names the camera by
/// its 1-based position and never contains a URL, user or password.
pub fn validate_ip_cameras(cameras: &[IpCameraConfig]) -> Result<(), String> {
    if !(1..=IP_CAMERAS_MAX).contains(&cameras.len()) {
        return Err(format!("ip_cameras must hold 1 to {IP_CAMERAS_MAX} cameras"));
    }
    for (index, camera) in cameras.iter().enumerate() {
        let n = index + 1;
        validate_ip_url(&camera.url).map_err(|why| format!("camera {n}: {why}"))?;
        for (what, value) in [("username", &camera.username), ("password", &camera.password)] {
            if value.len() > IP_CREDENTIAL_MAX {
                return Err(format!("camera {n}: {what} must be at most {IP_CREDENTIAL_MAX} bytes"));
            }
            if value.chars().any(is_ascii_control) {
                return Err(format!("camera {n}: {what} must not contain control characters"));
            }
        }
    }
    Ok(())
}

/// A `CameraConfig` struct.
#[derive(Clone, Serialize, Deserialize)]
pub struct CameraConfig {
    // Active source. The network fields below are only used for `Network`.
    #[serde(default)]
    pub source: CameraSource,
    // Remote camera address (IPv4 literal) and TCP port of the MJPEG server.
    #[serde(default)]
    pub network_host: String,
    #[serde(default)]
    pub network_port: u32,
    // Stream token — a secret: never serialized, never printed (see `Debug`).
    #[serde(skip)]
    pub network_token: String,
    // IP cameras (position N = ring `<shm_name>_ip<N>`). Only used for `Ip`.
    #[serde(default)]
    pub ip_cameras: Vec<IpCameraConfig>,
    pub camera_index: u32,
    pub width: u32,
    pub height: u32,
    pub framerate: u32,
    // false = manual exposure (v4l2 auto_exposure=1)
    // true = auto/aperture-priority (v4l2 auto_exposure=3)
    pub auto_exposure: bool,
    // Exposure time in 100 us units, used only when auto_exposure = false.
    pub exposure_time: u32,
    // Software/hardware RGB levels. 128 = neutral gain.
    pub rgb_red: u16,
    pub rgb_green: u16,
    pub rgb_blue: u16,
    // Gamma correction (V4L2 gamma control, range 1–500, 100 = neutral).
    pub gamma: u32,
    // Camera analog/digital gain (V4L2 gain control).
    pub gain: u32,
    // Shared-memory segment name (without leading slash).
    #[serde(skip)]
    pub shm_name: String,
}

impl Default for CameraConfig {
    fn default() -> Self {
        let framerate: u32 = std::env::var("CAPTURE_FRAMERATE")
            .unwrap_or_else(|_| "60".to_string())
            .parse()
            .unwrap_or(60);

        Self {
            // The source comes from `from_env`, which can reject a bad value.
            source: CameraSource::Local,
            network_host: String::new(),
            network_port: 0,
            network_token: String::new(),
            ip_cameras: Vec::new(),
            camera_index: std::env::var("CAMERA_INDEX")
                .unwrap_or_else(|_| "0".to_string())
                .parse()
                .unwrap_or(0),
            width: std::env::var("CAPTURE_WIDTH")
                .unwrap_or_else(|_| "2560".to_string())
                .parse()
                .unwrap_or(2560),
            height: std::env::var("CAPTURE_HEIGHT")
                .unwrap_or_else(|_| "720".to_string())
                .parse()
                .unwrap_or(720),
            framerate,
            auto_exposure: false,
            // Default: 1/fps seconds (e.g. 333 x 100 us ~= 33 ms ~= 1/30 s)
            exposure_time: (10_000u32 / framerate.max(1)).clamp(EXPOSURE_MIN, EXPOSURE_MAX),
            rgb_red: RGB_LEVEL_DEFAULT,
            rgb_green: RGB_LEVEL_DEFAULT,
            rgb_blue: RGB_LEVEL_DEFAULT,
            gamma: GAMMA_DEFAULT,
            gain: GAIN_DEFAULT,
            shm_name: std::env::var("SHM_NAME")
                .unwrap_or_else(|_| "conecsa_frame_shm".to_string()),
        }
    }
}

// Manual so the token cannot reach a log line through `{:?}`.
impl fmt::Debug for CameraConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CameraConfig")
            .field("source", &self.source)
            .field("network_host", &self.network_host)
            .field("network_port", &self.network_port)
            .field("network_token", &REDACTED)
            .field("ip_cameras", &self.ip_cameras)
            .field("camera_index", &self.camera_index)
            .field("width", &self.width)
            .field("height", &self.height)
            .field("framerate", &self.framerate)
            .field("auto_exposure", &self.auto_exposure)
            .field("exposure_time", &self.exposure_time)
            .field("rgb_red", &self.rgb_red)
            .field("rgb_green", &self.rgb_green)
            .field("rgb_blue", &self.rgb_blue)
            .field("gamma", &self.gamma)
            .field("gain", &self.gain)
            .field("shm_name", &self.shm_name)
            .finish()
    }
}

impl CameraConfig {
    /// Defaults plus the bootstrap source from `CAMERA_SOURCE`,
    /// `CAMERA_NETWORK_HOST|PORT|TOKEN` and `CAMERA_IP_URL|USER|PASSWORD_<n>`
    /// (n = 1..4). Development/bootstrap only: once the inference-service
    /// publishes a source over SHM, that one wins.
    pub fn from_env() -> Result<Self, String> {
        Self::from_lookup(&|name| std::env::var(name).unwrap_or_default())
    }

    /// `from_env` over explicit network values (the tests must not mutate the
    /// process environment).
    #[cfg(test)]
    pub(crate) fn with_source_text(
        source: &str,
        host: &str,
        port: &str,
        token: &str,
    ) -> Result<Self, String> {
        Self::from_lookup(&|name| {
            match name {
                "CAMERA_SOURCE" => source,
                "CAMERA_NETWORK_HOST" => host,
                "CAMERA_NETWORK_PORT" => port,
                "CAMERA_NETWORK_TOKEN" => token,
                _ => "",
            }
            .to_string()
        })
    }

    /// `from_env` over any variable lookup (missing = empty).
    pub(crate) fn from_lookup(var: &dyn Fn(&str) -> String) -> Result<Self, String> {
        let mut cfg = Self {
            source: CameraSource::from_env_text(&var("CAMERA_SOURCE"))?,
            ..Self::default()
        };
        if cfg.source == CameraSource::Ip {
            // Unset slots are skipped, so the cameras keep their relative order.
            cfg.ip_cameras = (1..=IP_CAMERAS_MAX)
                .filter_map(|n| {
                    let url = var(&format!("CAMERA_IP_URL_{n}")).trim().to_string();
                    (!url.is_empty()).then(|| IpCameraConfig {
                        url,
                        username: var(&format!("CAMERA_IP_USER_{n}")).trim().to_string(),
                        password: var(&format!("CAMERA_IP_PASSWORD_{n}")),
                    })
                })
                .collect();
            validate_ip_cameras(&cfg.ip_cameras)?;
        }
        if cfg.source == CameraSource::Network {
            let (host, port, token) = (
                var("CAMERA_NETWORK_HOST"),
                var("CAMERA_NETWORK_PORT"),
                var("CAMERA_NETWORK_TOKEN"),
            );
            let port: u32 = port
                .trim()
                .parse()
                .map_err(|_| "CAMERA_NETWORK_PORT must be a number".to_string())?;
            let token = normalize_token(token.trim());
            validate_network(host.trim(), port, &token)?;
            cfg.network_host = host.trim().to_string();
            cfg.network_port = port;
            cfg.network_token = token;
        }
        Ok(cfg)
    }

    pub fn has_non_neutral_rgb_levels(&self) -> bool {
        self.rgb_red != RGB_LEVEL_DEFAULT
            || self.rgb_green != RGB_LEVEL_DEFAULT
            || self.rgb_blue != RGB_LEVEL_DEFAULT
    }
}

#[cfg(test)]
mod tests;

