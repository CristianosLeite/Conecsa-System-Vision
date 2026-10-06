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
    assert_eq!(CameraSource::from_proto(3), None);
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

// ── IP cameras ───────────────────────────────────────────────────────────────

/// Visibly fake camera credentials.
const IP_USER: &str = "fake-user";
const IP_PASSWORD: &str = "FAKE-PASSWORD-123";

fn lookup(vars: &[(&str, &str)]) -> impl Fn(&str) -> String {
    let vars: Vec<(String, String)> =
        vars.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
    move |name| {
        vars.iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.clone())
            .unwrap_or_default()
    }
}

fn camera(url: &str) -> IpCameraConfig {
    IpCameraConfig {
        url: url.to_string(),
        username: IP_USER.to_string(),
        password: IP_PASSWORD.to_string(),
    }
}

#[test]
fn the_ip_source_parses_from_env_text_and_proto() {
    assert_eq!(CameraSource::from_env_text(" IP "), Ok(CameraSource::Ip));
    assert_eq!(CameraSource::from_proto(2), Some(CameraSource::Ip));
}

#[test]
fn the_ip_source_bootstraps_from_numbered_variables() {
    let cfg = CameraConfig::from_lookup(&lookup(&[
        ("CAMERA_SOURCE", "ip"),
        ("CAMERA_IP_URL_1", " rtsp://192.0.2.10:554/stream1 "),
        ("CAMERA_IP_USER_1", IP_USER),
        ("CAMERA_IP_PASSWORD_1", IP_PASSWORD),
        // Slot 2 unset: slot 3 moves up to position 2 (ring ip1).
        ("CAMERA_IP_URL_3", "rtsp://camera-3.example/live"),
    ]))
    .unwrap();
    assert_eq!(cfg.source, CameraSource::Ip);
    assert_eq!(cfg.ip_cameras.len(), 2);
    assert_eq!(cfg.ip_cameras[0], camera("rtsp://192.0.2.10:554/stream1"));
    assert_eq!(cfg.ip_cameras[1].url, "rtsp://camera-3.example/live");
    assert!(cfg.ip_cameras[1].username.is_empty() && cfg.ip_cameras[1].password.is_empty());
}

#[test]
fn the_ip_source_needs_at_least_one_valid_camera() {
    assert!(CameraConfig::from_lookup(&lookup(&[("CAMERA_SOURCE", "ip")])).is_err());
    let err = CameraConfig::from_lookup(&lookup(&[
        ("CAMERA_SOURCE", "ip"),
        ("CAMERA_IP_URL_1", "http://192.0.2.10/"),
    ]))
    .unwrap_err();
    assert!(err.contains("camera 1"), "{err}");
}

#[test]
fn valid_ip_urls_are_accepted() {
    // Everything the inference-service's validate_rtsp_url accepts.
    for url in [
        "rtsp://192.0.2.10",
        "rtsp://192.0.2.10/",
        "rtsp://192.0.2.10:554/cam/realmonitor?channel=1&subtype=0",
        "rtsp://198.51.100.7:65535/stream",
        "rtsp://camera-1.example.com/h264",
        "rtsp://camera.local/live",
        "rtsp://cam_era/live",
        "rtsp://CAMERA/live#x",
        "RTSP://192.0.2.10/stream",
        "rtsp://192.0.2.10:/stream",
        "rtsp://[2001:db8::1]:8554/a",
        "rtsp://[2001:db8::1]/a",
        "rtsp://192.0.2.10/caf\u{e9}",
    ] {
        assert_eq!(validate_ip_url(url), Ok(()), "{url}");
    }
}

#[test]
fn invalid_ip_urls_are_rejected() {
    let long = format!("rtsp://192.0.2.10/{}", "a".repeat(IP_URL_MAX));
    for url in [
        "",
        "192.0.2.10/stream",
        "rtsps://192.0.2.10/stream",
        "http://192.0.2.10/stream",
        "rtsp://",
        "rtsp:///stream",
        "rtsp://:554/stream",
        "rtsp://fake-user:FAKE-PASSWORD-123@192.0.2.10/stream",
        "rtsp://@192.0.2.10/stream",
        "rtsp://192.0.2.10 /stream",
        "rtsp://192.0.2.10/str\team",
        "rtsp://192.0.2.10/st\u{7f}",
        "rtsp://127.0.0.1/stream",
        "rtsp://127.8.9.10/stream",
        "rtsp://0.0.0.0/stream",
        "rtsp://224.0.0.1/stream",
        "rtsp://255.255.255.255/stream",
        "rtsp://localhost/stream",
        "rtsp://LocalHost:554/stream",
        "rtsp://[::1]/stream",
        "rtsp://[::]/stream",
        "rtsp://[ff02::1]/stream",
        "rtsp://[2001:db8::1/stream",
        "rtsp://[not-ipv6]/stream",
        "rtsp://[2001:db8::1]x/stream",
        "rtsp://192.0.2.10:0/stream",
        "rtsp://192.0.2.10:65536/stream",
        "rtsp://192.0.2.10:+554/stream",
        "rtsp://192.0.2.10:554:1/stream",
        long.as_str(),
    ] {
        assert!(validate_ip_url(url).is_err(), "{url:?} was accepted");
    }
}

#[test]
fn ip_url_errors_never_echo_the_url() {
    let err = validate_ip_url("rtsp://fake-user:FAKE-PASSWORD-123@192.0.2.10/").unwrap_err();
    assert!(!err.contains(IP_PASSWORD) && !err.contains("192.0.2.10"), "{err}");
}

#[test]
fn the_ip_camera_list_is_bounded_and_credentials_are_checked() {
    assert!(validate_ip_cameras(&[]).is_err());
    let four: Vec<_> = (1..=4).map(|n| camera(&format!("rtsp://192.0.2.{n}/s"))).collect();
    assert_eq!(validate_ip_cameras(&four), Ok(()));
    let five: Vec<_> = (1..=5).map(|n| camera(&format!("rtsp://192.0.2.{n}/s"))).collect();
    assert!(validate_ip_cameras(&five).is_err());

    let no_auth = IpCameraConfig { url: "rtsp://192.0.2.1/s".to_string(), ..Default::default() };
    assert_eq!(validate_ip_cameras(&[no_auth]), Ok(()));

    let at_max = IpCameraConfig {
        username: "u".repeat(IP_CREDENTIAL_MAX),
        password: "p".repeat(IP_CREDENTIAL_MAX),
        ..camera("rtsp://192.0.2.1/s")
    };
    assert_eq!(validate_ip_cameras(std::slice::from_ref(&at_max)), Ok(()));
    for bad in [
        IpCameraConfig { username: "u".repeat(IP_CREDENTIAL_MAX + 1), ..at_max.clone() },
        IpCameraConfig { password: "p".repeat(IP_CREDENTIAL_MAX + 1), ..at_max.clone() },
        IpCameraConfig { username: "fake\nuser".to_string(), ..at_max.clone() },
        IpCameraConfig { password: "FAKE\u{0}PASSWORD".to_string(), ..at_max.clone() },
    ] {
        let err = validate_ip_cameras(&[bad]).unwrap_err();
        assert!(err.starts_with("camera 1:"), "{err}");
        assert!(!err.contains("FAKE"), "{err}");
    }
}

#[test]
fn the_ip_password_never_reaches_debug_or_json() {
    let cfg = CameraConfig {
        source: CameraSource::Ip,
        ip_cameras: vec![camera("rtsp://192.0.2.10/stream")],
        ..Default::default()
    };
    let debug = format!("{cfg:?} {cfg:#?} {:?}", cfg.ip_cameras[0]);
    assert!(!debug.contains(IP_PASSWORD), "{debug}");
    assert!(debug.contains("rtsp://192.0.2.10/stream"));

    let json = serde_json::to_string(&cfg).unwrap();
    assert!(!json.contains(IP_PASSWORD), "{json}");
    assert!(!json.contains("password"), "{json}");
    assert!(json.contains("\"source\":\"ip\""));
}
