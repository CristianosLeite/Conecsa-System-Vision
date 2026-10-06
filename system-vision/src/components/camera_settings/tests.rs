// SPDX-FileCopyrightText: 2026 Conecsa
//
// SPDX-License-Identifier: AGPL-3.0-only

//! Unit tests for the Camera Settings form helpers (headless browser).
use wasm_bindgen_test::*;

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

// ── IP cameras ───────────────────────────────────────────────────────────────

use super::{
    mosaic_changes, mosaic_grid, parse_ip_form, rtsp_url_has_credentials, IpCameraDraft,
    IpFieldError, IpFormError,
};

fn ip_row(id: Option<&str>, url: &str) -> IpCameraDraft {
    IpCameraDraft {
        id: id.map(str::to_string),
        name: " Dock 1 ".into(),
        url: url.into(),
        username: "operator".into(),
        password_set: id.is_some(),
        ..Default::default()
    }
}

#[wasm_bindgen_test]
fn a_valid_ip_list_is_sent_in_order_and_trimmed() {
    let rows = vec![
        ip_row(Some("cam-0001"), " rtsp://192.0.2.10:554/cam/realmonitor?channel=1&subtype=0 "),
        ip_row(None, "rtsp://192.0.2.11/stream1"),
    ];
    let list = parse_ip_form(&rows, 4).unwrap();
    assert_eq!(list.len(), 2);
    assert_eq!(list[0].id.as_deref(), Some("cam-0001"));
    assert_eq!(list[0].name, "Dock 1");
    assert_eq!(list[0].url, "rtsp://192.0.2.10:554/cam/realmonitor?channel=1&subtype=0");
    assert_eq!(list[1].id, None);
}

#[wasm_bindgen_test]
fn an_ip_password_is_kept_replaced_or_cleared() {
    let mut row = ip_row(Some("cam-0001"), "rtsp://192.0.2.10/stream1");
    assert_eq!(parse_ip_form(&[row.clone()], 4).unwrap()[0].password, None, "blank keeps it");
    row.clear_password = true;
    assert_eq!(parse_ip_form(&[row.clone()], 4).unwrap()[0].password.as_deref(), Some(""));
    row.password = "example-only".into();
    assert_eq!(
        parse_ip_form(&[row], 4).unwrap()[0].password.as_deref(),
        Some("example-only"),
        "a typed password wins over a removal",
    );
    let mut new = ip_row(None, "rtsp://192.0.2.11/stream1");
    new.clear_password = true;
    assert_eq!(parse_ip_form(&[new], 4).unwrap()[0].password, None, "a new camera has none to clear");
}

#[wasm_bindgen_test]
fn the_ip_list_needs_one_to_max_cameras() {
    assert_eq!(parse_ip_form(&[], 4), Err(IpFormError::Empty));
    let rows = vec![ip_row(None, "rtsp://192.0.2.10/s"); 5];
    assert_eq!(parse_ip_form(&rows, 4), Err(IpFormError::TooMany));
}

#[wasm_bindgen_test]
fn an_ip_url_must_be_rtsp_with_a_host() {
    for url in ["", "http://192.0.2.10/s", "rtsp://", "rtsp:///stream", "192.0.2.10:554", "RTSP://"] {
        assert_eq!(
            parse_ip_form(&[ip_row(None, url)], 4),
            Err(IpFormError::Field(0, IpFieldError::UrlScheme)),
            "{url:?}",
        );
    }
    // The device takes the scheme in any case, so the form must too.
    assert!(parse_ip_form(&[ip_row(None, "RTSP://192.0.2.10/s")], 4).is_ok());
    assert!(rtsp_url_has_credentials("RTSP://a:b@192.0.2.10/s"));
    let long = format!("rtsp://192.0.2.10/{}", "a".repeat(250));
    assert_eq!(
        parse_ip_form(&[ip_row(None, &long)], 4),
        Err(IpFormError::Field(0, IpFieldError::UrlTooLong)),
    );
}

#[wasm_bindgen_test]
fn credentials_in_the_url_are_refused() {
    assert!(rtsp_url_has_credentials("rtsp://operator:example-only@192.0.2.10:554/s"));
    assert!(rtsp_url_has_credentials("rtsp://operator@192.0.2.10/s"));
    assert!(!rtsp_url_has_credentials("rtsp://192.0.2.10/s?user=a@b"), "only the authority counts");
    let rows = [ip_row(None, "rtsp://192.0.2.10/s"), ip_row(None, "rtsp://a:b@192.0.2.11/s")];
    assert_eq!(parse_ip_form(&rows, 4), Err(IpFormError::Field(1, IpFieldError::UrlCredentials)));
}

#[wasm_bindgen_test]
fn ip_field_lengths_are_checked() {
    let mut row = ip_row(None, "rtsp://192.0.2.10/s");
    row.name = "n".repeat(41);
    assert_eq!(parse_ip_form(&[row.clone()], 4), Err(IpFormError::Field(0, IpFieldError::NameTooLong)));
    // Limits are UTF-8 bytes, as on the device: "ç" takes two.
    row.name = "ç".repeat(20);
    assert!(parse_ip_form(&[row.clone()], 4).is_ok());
    row.name = "ç".repeat(21);
    assert_eq!(parse_ip_form(&[row.clone()], 4), Err(IpFormError::Field(0, IpFieldError::NameTooLong)));
    row.name.clear();
    row.username = "u".repeat(65);
    assert_eq!(parse_ip_form(&[row.clone()], 4), Err(IpFormError::Field(0, IpFieldError::UsernameTooLong)));
    row.username.clear();
    row.password = "p".repeat(65);
    assert_eq!(parse_ip_form(&[row], 4), Err(IpFormError::Field(0, IpFieldError::PasswordTooLong)));
}

#[wasm_bindgen_test]
fn the_mosaic_grid_follows_the_camera_count() {
    assert_eq!(mosaic_grid(1), (1, 1));
    assert_eq!(mosaic_grid(2), (2, 1));
    assert_eq!(mosaic_grid(3), (2, 2));
    assert_eq!(mosaic_grid(4), (2, 2));
}

#[wasm_bindgen_test]
fn a_mosaic_change_is_a_new_grid_or_another_camera_in_a_cell() {
    let url = "rtsp://192.0.2.10/s";
    let applied: Vec<(String, String)> =
        ["a", "b", "c"].iter().map(|id| (id.to_string(), url.to_string())).collect();
    let row = |id: Option<&str>| ip_row(id, url);
    // Same cameras, same order.
    assert!(!mosaic_changes(Some(&applied), &[row(Some("a")), row(Some("b")), row(Some("c"))]));
    // A fourth camera fills the empty cell of the same 2 x 2 grid.
    assert!(!mosaic_changes(Some(&applied), &[row(Some("a")), row(Some("b")), row(Some("c")), row(None)]));
    // Reordered, or down to a side-by-side pair.
    assert!(mosaic_changes(Some(&applied), &[row(Some("b")), row(Some("a")), row(Some("c"))]));
    assert!(mosaic_changes(Some(&applied), &[row(Some("a")), row(Some("b"))]));
    // The same entry pointed at another camera.
    let mut moved = row(Some("b"));
    moved.url = "rtsp://192.0.2.99/s".to_string();
    assert!(mosaic_changes(Some(&applied), &[row(Some("a")), moved, row(Some("c"))]));
    // From a local or remote camera the scene changes even with one camera.
    assert!(mosaic_changes(None, &[row(None)]));
    assert!(mosaic_changes(None, &[row(None), row(None)]));
}
