// SPDX-FileCopyrightText: 2026 Conecsa
//
// SPDX-License-Identifier: AGPL-3.0-only

//! Unit tests for `CameraConfig`.
use super::*;

#[test]
fn default_has_neutral_rgb_levels() {
    let cfg = CameraConfig::default();
    assert_eq!(cfg.rgb_red, RGB_LEVEL_DEFAULT);
    assert_eq!(cfg.rgb_green, RGB_LEVEL_DEFAULT);
    assert_eq!(cfg.rgb_blue, RGB_LEVEL_DEFAULT);
    assert!(!cfg.has_non_neutral_rgb_levels());
}

#[test]
fn any_off_neutral_channel_is_detected() {
    let cfg = CameraConfig { rgb_red: 200, ..Default::default() };
    assert!(cfg.has_non_neutral_rgb_levels());

    let cfg = CameraConfig { rgb_green: 0, ..Default::default() };
    assert!(cfg.has_non_neutral_rgb_levels());

    let cfg = CameraConfig { rgb_blue: RGB_LEVEL_DEFAULT + 1, ..Default::default() };
    assert!(cfg.has_non_neutral_rgb_levels());
}

#[test]
fn serde_round_trip_preserves_fields() {
    let cfg = CameraConfig {
        camera_index: 2,
        width: 1280,
        height: 480,
        framerate: 30,
        auto_exposure: true,
        exposure_time: 500,
        rgb_red: 140,
        gamma: 120,
        gain: 64,
        ..Default::default()
    };

    let json = serde_json::to_string(&cfg).unwrap();
    let back: CameraConfig = serde_json::from_str(&json).unwrap();

    assert_eq!(back.camera_index, 2);
    assert_eq!(back.width, 1280);
    assert_eq!(back.height, 480);
    assert_eq!(back.framerate, 30);
    assert!(back.auto_exposure);
    assert_eq!(back.exposure_time, 500);
    assert_eq!(back.rgb_red, 140);
    assert_eq!(back.gamma, 120);
    assert_eq!(back.gain, 64);
}

#[test]
fn shm_name_is_not_serialized() {
    let cfg = CameraConfig::default();
    let json = serde_json::to_string(&cfg).unwrap();
    assert!(!json.contains("shm_name"));
}

// ── source and the stream token ──────────────────────────────────────────────

/// Visibly fake stream token.
const TOKEN: &str = "TESTT0KEN123";

fn network_cfg() -> CameraConfig {
    CameraConfig::with_source_text("network", "192.0.2.1", "8080", TOKEN).unwrap()
}

#[test]
fn the_default_source_is_local() {
    let cfg = CameraConfig::with_source_text("", "", "", "").unwrap();
    assert_eq!(cfg.source, CameraSource::Local);
    assert_eq!(CameraConfig::default().source, CameraSource::Local);
}

#[test]
fn an_unknown_source_is_an_error_not_local() {
    assert!(CameraSource::from_env_text("netwrok").is_err());
    assert!(CameraConfig::with_source_text("rtsp", "", "", "").is_err());
    assert_eq!(CameraSource::from_proto(2), None);
    assert_eq!(CameraSource::from_proto(-1), None);
}

#[test]
fn a_network_source_needs_a_complete_valid_endpoint() {
    let cfg = CameraConfig::with_source_text(" Network ", " 192.0.2.1 ", "8080", "test-t0ke-n123")
        .unwrap();
    assert_eq!(cfg.source, CameraSource::Network);
    assert_eq!(cfg.network_host, "192.0.2.1");
    assert_eq!(cfg.network_port, 8080);
    assert_eq!(cfg.network_token, TOKEN, "hyphens dropped, upper-cased");

    for (host, port, token) in [
        ("", "8080", TOKEN),
        ("camera.local", "8080", TOKEN),
        ("192.0.2.1", "", TOKEN),
        ("192.0.2.1", "0", TOKEN),
        ("192.0.2.1", "70000", TOKEN),
        ("192.0.2.1", "8080", ""),
        ("192.0.2.1", "8080", "SHORT"),
    ] {
        assert!(CameraConfig::with_source_text("network", host, port, token).is_err());
    }
}

#[test]
fn validation_errors_never_echo_the_token() {
    let err = CameraConfig::with_source_text("network", "192.0.2.1", "8080", "bad token!!")
        .unwrap_err();
    assert!(!err.contains("bad token"), "{err}");
}

#[test]
fn the_token_never_reaches_debug_or_json() {
    let cfg = network_cfg();
    let debug = format!("{cfg:?} {cfg:#?}");
    assert!(!debug.contains(TOKEN), "{debug}");
    assert!(debug.contains("<redacted>"));
    assert!(debug.contains("192.0.2.1"), "the endpoint itself is printable");

    let json = serde_json::to_string(&cfg).unwrap();
    assert!(!json.contains(TOKEN), "{json}");
    assert!(!json.contains("network_token"));
    assert!(json.contains("\"source\":\"network\""));
}
