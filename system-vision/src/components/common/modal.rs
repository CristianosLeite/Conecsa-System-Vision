// SPDX-FileCopyrightText: 2026 Conecsa
//
// SPDX-License-Identifier: AGPL-3.0-only

//! An accessible modal dialog.
//!
//! On open the first focusable element inside gets the focus (put the safe
//! action first); Tab and Shift+Tab stay inside the dialog; Esc calls
//! `on_close`; Enter activates only the focused button (native behavior, no
//! form); on close the focus returns to the element that opened it. The
//! backdrop does not close it: a confirmation must be an explicit choice.

use gloo_timers::future::TimeoutFuture;
use leptos::prelude::*;
use leptos::task::spawn_local;
use wasm_bindgen::JsCast;

const FOCUSABLE: &str = "button:not([disabled]), [href], input:not([disabled]), \
     select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex='-1'])";

/// The focusable descendants of `root`, in document order.
fn focusables(root: &web_sys::Element) -> Vec<web_sys::HtmlElement> {
    let Ok(list) = root.query_selector_all(FOCUSABLE) else {
        return Vec::new();
    };
    (0..list.length())
        .filter_map(|i| list.item(i))
        .filter_map(|node| node.dyn_into::<web_sys::HtmlElement>().ok())
        .collect()
}

fn active_element() -> Option<web_sys::HtmlElement> {
    web_sys::window()?
        .document()?
        .active_element()?
        .dyn_into::<web_sys::HtmlElement>()
        .ok()
}

#[component]
pub fn Modal(
    /// Whether the dialog is shown.
    #[prop(into)]
    open: Signal<bool>,
    /// Called on Esc; the owner decides whether it may close now.
    on_close: Callback<()>,
    /// `id` of the element that titles the dialog (`aria-labelledby`).
    #[prop(into)]
    labelled_by: String,
    children: ChildrenFn,
) -> impl IntoView {
    let dialog = NodeRef::<leptos::html::Div>::new();
    let opener = StoredValue::new_local(None::<web_sys::HtmlElement>);

    Effect::new(move |was_open: Option<bool>| {
        let is_open = open.get();
        if is_open && was_open != Some(true) {
            opener.set_value(active_element());
            // After the dialog has rendered.
            spawn_local(async move {
                TimeoutFuture::new(0).await;
                if let Some(root) = dialog.get_untracked() {
                    if let Some(first) = focusables(&root).into_iter().next() {
                        let _ = first.focus();
                    }
                }
            });
        } else if !is_open && was_open == Some(true) {
            opener.update_value(|slot| {
                if let Some(element) = slot.take() {
                    let _ = element.focus();
                }
            });
        }
        is_open
    });

    let on_keydown = move |ev: leptos::ev::KeyboardEvent| match ev.key().as_str() {
        "Escape" => {
            ev.prevent_default();
            on_close.run(());
        }
        "Tab" => {
            let Some(root) = dialog.get_untracked() else {
                return;
            };
            let items = focusables(&root);
            let (Some(first), Some(last)) = (items.first(), items.last()) else {
                ev.prevent_default();
                return;
            };
            let active = active_element();
            if ev.shift_key() {
                if active.as_ref().is_none_or(|a| a == first) {
                    ev.prevent_default();
                    let _ = last.focus();
                }
            } else if active.as_ref().is_none_or(|a| a == last) {
                ev.prevent_default();
                let _ = first.focus();
            }
        }
        _ => {}
    };

    view! {
        <Show when=move || open.get()>
            <div class="ui-modal-backdrop">
                <div
                    class="ui-card ui-card-pad ui-modal"
                    role="dialog"
                    aria-modal="true"
                    aria-labelledby=labelled_by.clone()
                    node_ref=dialog
                    on:keydown=on_keydown
                >
                    {children()}
                </div>
            </div>
        </Show>
    }
}
