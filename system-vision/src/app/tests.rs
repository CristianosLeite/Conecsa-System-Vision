//! Unit tests for the app-level URL and size-formatting helpers.
use super::*;
use wasm_bindgen_test::*;

#[wasm_bindgen_test]
fn api_base_url_is_same_origin() {
    assert_eq!(get_api_base_url(), "");
}

#[wasm_bindgen_test]
fn base_prefix_accepts_only_same_origin_paths() {
    // The manual's prefix, and root-relative prefixes, pass through without
    // a trailing slash (callers append "/api/…").
    assert_eq!(sanitize_base_prefix("."), ".");
    assert_eq!(sanitize_base_prefix("./"), ".");
    assert_eq!(sanitize_base_prefix("/device/a41f9c/"), "/device/a41f9c");
    assert_eq!(sanitize_base_prefix(" /x "), "/x");
    assert_eq!(sanitize_base_prefix("/"), "");
    // Anything that would change the origin falls back to the default.
    assert_eq!(sanitize_base_prefix("https://evil.example"), "");
    assert_eq!(sanitize_base_prefix("//evil.example/x"), "");
    assert_eq!(sanitize_base_prefix("javascript:alert(1)"), "");
}

#[wasm_bindgen_test]
fn node_red_url_is_proxied_under_flow() {
    assert_eq!(get_node_red_url(None), "/flow/");
    assert_eq!(get_node_red_url(Some("")), "/flow/");
}

#[wasm_bindgen_test]
fn node_red_url_carries_the_editor_token() {
    assert_eq!(
        get_node_red_url(Some("v1.abc.d+e")),
        "/flow/?access_token=v1.abc.d%2Be"
    );
}

#[wasm_bindgen_test]
fn bytes_below_one_kb_are_plain() {
    assert_eq!(format_size(0), "0 B");
    assert_eq!(format_size(1023), "1023 B");
}

#[wasm_bindgen_test]
fn kb_mb_gb_boundaries_switch_units() {
    assert_eq!(format_size(1024), "1.00 KB");
    assert_eq!(format_size(1024 * 1024), "1.00 MB");
    assert_eq!(format_size(1024 * 1024 * 1024), "1.00 GB");
}

#[wasm_bindgen_test]
fn fractional_sizes_round_to_two_decimals() {
    assert_eq!(format_size(1536), "1.50 KB");
    assert_eq!(format_size(5 * 1024 * 1024 + 512 * 1024), "5.50 MB");
}

#[wasm_bindgen_test]
fn sizes_above_gb_stay_in_gb() {
    assert_eq!(format_size(1024u64.pow(4)), "1024.00 GB");
}
