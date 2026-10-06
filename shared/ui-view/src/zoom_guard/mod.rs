// SPDX-FileCopyrightText: 2026 Conecsa
//
// SPDX-License-Identifier: Apache-2.0

//! Keeps the page from being zoomed as a whole.
//!
//! The UI is a fixed, responsive app shell: a page zoom (Ctrl+wheel, a
//! trackpad pinch — delivered as a `wheel` with `ctrlKey` — Ctrl+`+`/`-`/`0`,
//! or WebKit's touch `gesture*` events) only breaks the layout. The viewport
//! meta and `touch-action` on the app root stop the touch gestures; this
//! guard stops the keyboard and wheel ones. An image view zooms its own
//! image and stops the propagation of the Ctrl+wheel it handles, so it never
//! reaches here (see `viewport::Gestures::wheel`).
//!
//! Installed once per page from each app's entry point (not by the user-manual
//! simulator, which mounts the UI itself and stays zoomable). The listeners
//! live for the page, so their closures are leaked on purpose. `wheel` is
//! registered with `passive: false` explicitly: browsers make window-level
//! wheel listeners passive by default, which makes `preventDefault` a no-op.

use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

/// Install the guard on `window`. Safe to call when there is no window
/// (nothing happens).
pub fn install() {
    let Some(window) = web_sys::window() else {
        return;
    };
    let target: &web_sys::EventTarget = window.as_ref();
    let active = web_sys::AddEventListenerOptions::new();
    active.set_passive(false);

    let wheel = Closure::<dyn FnMut(web_sys::WheelEvent)>::new(|ev: web_sys::WheelEvent| {
        if ev.ctrl_key() || ev.meta_key() {
            ev.prevent_default();
        }
    });
    let _ = target.add_event_listener_with_callback_and_add_event_listener_options(
        "wheel",
        wheel.as_ref().unchecked_ref(),
        &active,
    );
    wheel.forget();

    let keydown = Closure::<dyn FnMut(web_sys::KeyboardEvent)>::new(|ev: web_sys::KeyboardEvent| {
        if (ev.ctrl_key() || ev.meta_key()) && is_zoom_key(&ev.key(), &ev.code()) {
            ev.prevent_default();
        }
    });
    let _ = target.add_event_listener_with_callback_and_add_event_listener_options(
        "keydown",
        keydown.as_ref().unchecked_ref(),
        &active,
    );
    keydown.forget();

    // Non-standard WebKit touch-zoom events (no `web_sys` type; a plain Event
    // is enough to cancel them).
    for name in ["gesturestart", "gesturechange", "gestureend"] {
        let gesture = Closure::<dyn FnMut(web_sys::Event)>::new(|ev: web_sys::Event| {
            ev.prevent_default();
        });
        let _ = target.add_event_listener_with_callback_and_add_event_listener_options(
            name,
            gesture.as_ref().unchecked_ref(),
            &active,
        );
        gesture.forget();
    }
}

/// The keys browsers bind to page zoom together with Ctrl/Cmd, by `key` (layout
/// dependent) or `code` (physical, covers the numeric keypad).
fn is_zoom_key(key: &str, code: &str) -> bool {
    matches!(key, "+" | "-" | "=" | "0" | "Add" | "Subtract")
        || matches!(
            code,
            "Equal" | "Minus" | "Digit0" | "NumpadAdd" | "NumpadSubtract" | "Numpad0"
        )
}

#[cfg(test)]
mod tests;
