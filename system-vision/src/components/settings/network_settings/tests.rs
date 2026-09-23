// SPDX-FileCopyrightText: 2026 Conecsa
//
// SPDX-License-Identifier: AGPL-3.0-only

//! Unit tests for the access point panel's pure helpers (headless browser).
use super::ap_panel::remaining_secs;
use crate::components::common::access_point::passphrase_is_valid;
use wasm_bindgen_test::*;

#[wasm_bindgen_test]
fn a_passphrase_needs_8_to_63_characters() {
    assert!(!passphrase_is_valid(""));
    assert!(!passphrase_is_valid("short-7"));
    assert!(passphrase_is_valid("eight-ch"));
    assert!(passphrase_is_valid(&"x".repeat(63)));
    assert!(!passphrase_is_valid(&"x".repeat(64)));
}

#[wasm_bindgen_test]
fn the_passphrase_is_printable_ascii_like_the_device_requires() {
    assert!(passphrase_is_valid("plain words, digits 123 and $ymbols!"));
    assert!(!passphrase_is_valid("çãéíóúàê"), "accented letters are refused by the device");
    assert!(!passphrase_is_valid("tab\tinside it"), "control characters too");
}

#[wasm_bindgen_test]
fn the_countdown_subtracts_only_this_browsers_elapsed_time() {
    assert_eq!(remaining_secs(300, 10.0, 10.0), 300);
    assert_eq!(remaining_secs(300, 10.0, 130.4), 180);
    assert_eq!(remaining_secs(300, 10.0, 400.0), 0, "it never goes below zero");
}

#[wasm_bindgen_test]
fn a_clock_that_goes_backwards_never_adds_time() {
    assert_eq!(remaining_secs(120, 50.0, 20.0), 120);
    assert_eq!(remaining_secs(0, 50.0, 20.0), 0);
}
