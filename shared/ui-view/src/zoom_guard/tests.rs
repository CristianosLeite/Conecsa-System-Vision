// SPDX-FileCopyrightText: 2026 Conecsa
//
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for the page-zoom guard: the key set natively, the install in a
//! headless browser.
use super::*;

#[test]
fn zoom_keys_by_key_or_code() {
    assert!(is_zoom_key("+", "BracketRight"));
    assert!(is_zoom_key("=", "Equal"));
    assert!(is_zoom_key("-", "Minus"));
    assert!(is_zoom_key("0", "Digit0"));
    assert!(is_zoom_key("Unidentified", "NumpadAdd"));
    assert!(is_zoom_key("Unidentified", "NumpadSubtract"));
    assert!(is_zoom_key("Unidentified", "Numpad0"));
    assert!(!is_zoom_key("a", "KeyA"));
    assert!(!is_zoom_key("1", "Digit1"));
    assert!(!is_zoom_key("Control", "ControlLeft"));
}

#[cfg(target_arch = "wasm32")]
mod browser {
    use super::*;
    use wasm_bindgen_test::*;

    wasm_bindgen_test_configure!(run_in_browser);

    #[wasm_bindgen_test]
    fn install_is_idempotent_and_does_not_panic() {
        install();
        install();
    }
}
