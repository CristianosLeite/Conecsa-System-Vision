// SPDX-FileCopyrightText: 2026 Conecsa
//
// SPDX-License-Identifier: AGPL-3.0-only

//! Camera configuration (source, index, resolution, framerate, exposure, gains)
//! sourced from environment variables and updated live via the SHM config region.

use std::fmt;
use std::net::Ipv4Addr;

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
/// What `Debug` prints in place of the stream token.
const REDACTED: &str = "<redacted>";

/// Where frames come from: a local V4L2 camera or a remote camera's MJPEG stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CameraSource {
    #[default]
    Local,
    Network,
}

impl CameraSource {
    /// Parse `CAMERA_SOURCE`. An unknown value is an error, never `Local`: a
    /// typo must not silently select a different inspection camera.
    pub fn from_env_text(text: &str) -> Result<Self, String> {
        match text.trim().to_ascii_lowercase().as_str() {
            "" | "local" => Ok(Self::Local),
            "network" => Ok(Self::Network),
            other => Err(format!("CAMERA_SOURCE must be local or network (got {other:?})")),
        }
    }

    /// Map the wire enum; `None` for a value this build does not know.
    pub fn from_proto(value: i32) -> Option<Self> {
        match value {
            0 => Some(Self::Local),
            1 => Some(Self::Network),
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
    /// Defaults plus the bootstrap source from `CAMERA_SOURCE` and
    /// `CAMERA_NETWORK_HOST|PORT|TOKEN`. Development/bootstrap only: once the
    /// inference-service publishes a source over SHM, that one wins.
    pub fn from_env() -> Result<Self, String> {
        let var = |name: &str| std::env::var(name).unwrap_or_default();
        Self::with_source_text(
            &var("CAMERA_SOURCE"),
            &var("CAMERA_NETWORK_HOST"),
            &var("CAMERA_NETWORK_PORT"),
            &var("CAMERA_NETWORK_TOKEN"),
        )
    }

    /// `from_env` over explicit values (the tests must not mutate the process
    /// environment).
    pub(crate) fn with_source_text(
        source: &str,
        host: &str,
        port: &str,
        token: &str,
    ) -> Result<Self, String> {
        let mut cfg = Self {
            source: CameraSource::from_env_text(source)?,
            ..Self::default()
        };
        if cfg.source == CameraSource::Network {
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

