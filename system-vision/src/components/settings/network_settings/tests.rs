// SPDX-FileCopyrightText: 2026 Conecsa
//
// SPDX-License-Identifier: AGPL-3.0-only

//! Unit tests for the access point panel's pure helpers (headless browser).
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
