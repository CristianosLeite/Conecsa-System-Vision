//! Role resolution and persistence (mirrors `locale.rs`), and the hub-proxy
//! capability token.
//!
//! The device UI has no login of its own: the hub appends `?role=admin|user`
//! to the embed iframe URL (owner is mapped to admin hub-side), and direct
//! browser access falls back to the last role persisted in localStorage. When
//! neither is present the UI fails closed (restricted). The role is a
//! presentation hint — the gateway enforces it server-side from the identity
//! the hub stamps on relayed requests.
//!
//! When the UI is embedded through the hub's loopback device proxy, the hub
//! also appends `?cap=<token>`: a per-proxy capability the proxy requires on
//! every `/api` and `/enroll` request. Unlike the role it is a secret, so it
//! lives only in this process ([`proxy_cap`]) and is never persisted to
//! localStorage; fetches attach it as the `X-Conecsa-Proxy-Token` header, and
//! carriers that cannot set headers (EventSource, `<img>`, downloads) append
//! it back as `?cap=` via [`with_cap`]. Absent (direct LAN access, no hub
//! proxy) nothing is attached.

use std::sync::OnceLock;

use leptos::prelude::*;

/// localStorage key holding the last effective role.
const ROLE_KEY: &str = "conecsa.role";

/// Header carrying the proxy capability on fetches.
pub const PROXY_CAP_HEADER: &str = "X-Conecsa-Proxy-Token";

/// Query parameter carrying the proxy capability (iframe URL and header-less
/// carriers).
pub const PROXY_CAP_PARAM: &str = "cap";

static PROXY_CAP: OnceLock<Option<String>> = OnceLock::new();

/// Strict parse of the capability from a query string: non-empty, URL-safe
/// hex/token characters only, bounded length.
fn parse_cap(search: &str) -> Option<String> {
    let params = web_sys::UrlSearchParams::new_with_str(search).ok()?;
    let cap = params.get(PROXY_CAP_PARAM)?;
    let valid = !cap.is_empty()
        && cap.len() <= 128
        && cap.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_');
    valid.then_some(cap)
}

/// The hub-proxy capability token for this page load, when embedded through
/// the hub. Read once from `?cap=`; deliberately never persisted.
pub fn proxy_cap() -> Option<String> {
    PROXY_CAP
        .get_or_init(|| {
            let search = web_sys::window()?.location().search().ok()?;
            parse_cap(&search)
        })
        .clone()
}

/// Append the capability to a same-origin URL for header-less carriers
/// (EventSource, `<img>`, download links). Identity when no token is present.
pub fn with_cap(url: &str) -> String {
    match proxy_cap() {
        Some(cap) => {
            let sep = if url.contains('?') { '&' } else { '?' };
            format!("{url}{sep}{PROXY_CAP_PARAM}={cap}")
        }
        None => url.to_string(),
    }
}

/// Whether the current session may use privileged controls.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Privileged(pub bool);

/// Strict parse, like locale's `FromStr`: unknown values fall through to the
/// next source instead of silently granting or revoking access.
fn parse_role(role: &str) -> Option<bool> {
    match role {
        "admin" => Some(true),
        "user" => Some(false),
        _ => None,
    }
}

/// Explicit role requested for this page load: `?role=` (hub embed) first,
/// then localStorage.
fn explicit_privileged() -> Option<bool> {
    let window = web_sys::window()?;

    let search = window.location().search().ok()?;
    if let Ok(params) = web_sys::UrlSearchParams::new_with_str(&search) {
        if let Some(role) = params.get("role") {
            if let Some(privileged) = parse_role(&role) {
                return Some(privileged);
            }
        }
    }

    let stored = window.local_storage().ok()??.get_item(ROLE_KEY).ok()??;
    parse_role(&stored)
}

/// Resolve the role, persist it, and provide [`Privileged`] as context. Unlike
/// `init_locale` there is no competing writer, so this resolves synchronously.
/// Must run in a component body above every consumer (i.e. in `App`).
pub fn init_access() {
    let privileged = explicit_privileged().unwrap_or(false);
    if let Some(Ok(Some(storage))) = web_sys::window().map(|w| w.local_storage()) {
        let _ = storage.set_item(ROLE_KEY, if privileged { "admin" } else { "user" });
    }
    provide_context(Privileged(privileged));
}

/// Consumer helper; fails closed when the context is missing.
pub fn privileged() -> bool {
    use_context::<Privileged>().map(|p| p.0).unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    wasm_bindgen_test_configure!(run_in_browser);

    #[wasm_bindgen_test]
    fn parse_role_is_strict() {
        assert_eq!(parse_role("admin"), Some(true));
        assert_eq!(parse_role("user"), Some(false));
        assert_eq!(parse_role("owner"), None);
        assert_eq!(parse_role("Admin"), None);
        assert_eq!(parse_role(""), None);
    }

    #[wasm_bindgen_test]
    fn parse_cap_is_strict() {
        assert_eq!(parse_cap("?cap=abc123DEF"), Some("abc123DEF".to_string()));
        assert_eq!(parse_cap("?role=admin&cap=a-b_c"), Some("a-b_c".to_string()));
        assert_eq!(parse_cap("?role=admin"), None);
        assert_eq!(parse_cap("?cap="), None);
        // Injection attempts and oversized values fall through to "no token".
        assert_eq!(parse_cap("?cap=a%20b"), None);
        assert_eq!(parse_cap(&format!("?cap={}", "a".repeat(200))), None);
    }

    #[wasm_bindgen_test]
    fn with_cap_is_identity_without_a_token() {
        // The test page URL carries no ?cap=, so proxy_cap() resolves None and
        // URLs must pass through untouched.
        assert_eq!(with_cap("/api/v1/events/stream?stats=1"),
                   "/api/v1/events/stream?stats=1");
        assert_eq!(with_cap("/api/v1/training/preview"),
                   "/api/v1/training/preview");
    }

    #[wasm_bindgen_test]
    fn url_builders_never_append_after_the_capability() {
        // Regression: the preview URL once appended `?r=` AFTER with_cap,
        // producing `...?cap=<token>?r=0` — a corrupted capability the hub
        // proxy rejects forever. Every query parameter must precede the cap.
        let url = crate::api::training_preview_url(3);
        assert!(url.ends_with("/api/v1/training/preview?r=3"), "{url}");
        assert_eq!(url.matches('?').count(), 1, "{url}");
    }

    #[wasm_bindgen_test]
    fn the_capability_is_never_persisted() {
        let _ = proxy_cap();
        let storage = web_sys::window().unwrap().local_storage().unwrap().unwrap();
        for i in 0..storage.length().unwrap() {
            let key = storage.key(i).unwrap().unwrap();
            assert!(!key.contains("cap"), "capability leaked to localStorage");
        }
    }
}
