// SPDX-FileCopyrightText: 2026 Conecsa
//
// SPDX-License-Identifier: AGPL-3.0-only

//! Unit tests for the 3D-camera detection used to gate the stereo overlay
//! (headless browser).
use super::*;
use wasm_bindgen_test::*;

#[wasm_bindgen_test]
fn recognises_a_3d_camera_by_name() {
    assert!(is_stereo_camera("3D USB Camera"));
    assert!(is_stereo_camera("3d usb camera"));
    assert!(is_stereo_camera("USB 3D Webcam: USB 3D Webcam"));
}

#[wasm_bindgen_test]
fn ordinary_cameras_are_not_3d() {
    assert!(!is_stereo_camera("C270 HD WEBCAM"));
    assert!(!is_stereo_camera("video0"));
    assert!(!is_stereo_camera(""));
}

// ── remote camera form ────────────────────────────────────────────────────────

use super::{format_token_input, normalize_token, parse_network_form, NetworkFormError};

#[wasm_bindgen_test]
fn a_complete_form_is_accepted_and_trimmed() {
    let form = parse_network_form(" 192.0.2.1 ", " 8080 ", " abcd-1234 ", false);
    assert_eq!(form, Ok(("192.0.2.1".to_string(), 8080, Some("ABCD1234".to_string()))));
}

#[wasm_bindgen_test]
fn a_blank_token_keeps_the_stored_one_only_when_one_is_stored() {
    assert_eq!(
        parse_network_form("192.0.2.1", "8080", "", true),
        Ok(("192.0.2.1".to_string(), 8080, None)),
    );
    assert_eq!(parse_network_form("192.0.2.1", "8080", "  ", false), Err(NetworkFormError::Token));
}

#[wasm_bindgen_test]
fn the_address_must_be_an_ipv4_literal() {
    for host in ["", "camera.local", "192.0.2", "192.0.2.256", "::1"] {
        assert_eq!(parse_network_form(host, "8080", "ABCD1234", false), Err(NetworkFormError::Address));
    }
}

#[wasm_bindgen_test]
fn the_port_must_be_in_range() {
    for port in ["", "0", "65536", "-1", "http", "80.5"] {
        assert_eq!(parse_network_form("192.0.2.1", port, "ABCD1234", false), Err(NetworkFormError::Port));
    }
}

#[wasm_bindgen_test]
fn the_token_is_formatted_as_typed() {
    assert_eq!(format_token_input("abcd"), "ABCD");
    assert_eq!(format_token_input("abcd1"), "ABCD-1");
    assert_eq!(format_token_input("ABCD-1234"), "ABCD-1234");
    assert_eq!(format_token_input("abcd 1234 efgh"), "ABCD-1234-EFGH");
    assert_eq!(format_token_input("--ab--cd--"), "ABCD");
    assert_eq!(format_token_input(""), "");
    // 32 characters is the protocol maximum; extra input is dropped.
    assert_eq!(format_token_input(&"a".repeat(40)).len(), 32 + 7);
}

#[wasm_bindgen_test]
fn the_device_receives_the_token_without_hyphens() {
    assert_eq!(normalize_token("ABCD-1234-EF"), "ABCD1234EF");
    assert_eq!(normalize_token("abcd"), "ABCD");
    assert_eq!(normalize_token("-- --"), "");
}

// ── the device's access point ────────────────────────────────────────────────

use super::{offers_access_point, station_suggestion};
use crate::api::{ApStation, ApStatus};

fn ap(active: bool, stations: Vec<ApStation>) -> ApStatus {
    ApStatus {
        active,
        ssid: "conecsa-000001".into(),
        frequency_mhz: if active { 5180 } else { 0 },
        address: if active { "10.98.76.1".into() } else { String::new() },
        prefix: if active { 24 } else { 0 },
        stations,
        join_deadline_remaining_secs: 0,
        wired_ready: true,
        message: String::new(),
        channels: vec![36, 40, 44],
    }
}

fn station(address: &str) -> ApStation {
    ApStation { address: address.into(), hostname: "camera".into(), signal: -48 }
}

#[wasm_bindgen_test]
fn the_access_point_is_offered_only_for_the_remote_camera_while_it_is_off() {
    let off = ap(false, Vec::new());
    let on = ap(true, Vec::new());
    assert!(offers_access_point("network", Some(&off)));
    assert!(!offers_access_point("network", Some(&on)), "already on: nothing to offer");
    assert!(!offers_access_point("network", None), "unknown state must not block Apply");
    assert!(!offers_access_point("local", Some(&off)));
}

#[wasm_bindgen_test]
fn the_one_joined_remote_camera_with_a_lease_is_suggested() {
    assert_eq!(station_suggestion(None), None);
    assert_eq!(station_suggestion(Some(&ap(true, Vec::new()))), None);
    assert_eq!(
        station_suggestion(Some(&ap(true, vec![station("10.98.76.11")]))),
        Some("10.98.76.11".to_string()),
    );
    assert_eq!(station_suggestion(Some(&ap(true, vec![station("")]))), None, "no lease yet");
    assert_eq!(
        station_suggestion(Some(&ap(true, vec![station("10.98.76.11"), station("10.98.76.12")]))),
        None,
        "two joined cameras are ambiguous",
    );
    assert_eq!(
        station_suggestion(Some(&ap(false, vec![station("10.98.76.11")]))),
        None,
        "a stale station list on an inactive access point suggests nothing",
    );
}
