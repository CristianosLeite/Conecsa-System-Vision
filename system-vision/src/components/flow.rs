// SPDX-FileCopyrightText: 2026 Conecsa
//
// SPDX-License-Identifier: AGPL-3.0-only

use crate::api::fetch_api;
use crate::app::get_node_red_url;
use crate::components::panel_header::PanelHeader;
use crate::i18n::*;
use leptos::prelude::*;
use leptos::task::spawn_local;
use serde::Deserialize;

/// `POST /api/v1/flow/token` response.
#[derive(Deserialize)]
struct FlowToken {
    token: String,
}

/// The editor iframe is opened only after an editor token was requested from
/// the gateway: Node-RED's admin API is behind `adminAuth`, and the token
/// (passed as `?access_token=`) is what the editor sends on every admin call.
/// A device without tokens configured falls back to the bare editor URL.
#[component]
pub fn Flow() -> impl IntoView {
    let i18n = use_i18n();
    let (src, set_src) = signal::<Option<String>>(None);
    Effect::new(move |_| {
        spawn_local(async move {
            let token = match fetch_api::<FlowToken>("/api/v1/flow/token", "POST", Some("{}")).await {
                Ok(t) => Some(t.token),
                Err(e) => {
                    web_sys::console::warn_1(&format!("flow editor token unavailable: {e}").into());
                    None
                }
            };
            set_src.set(Some(get_node_red_url(token.as_deref())));
        });
    });
    view! {
        <div class="app-panel app-flow-panel">
            <div class="app-flow-header">
                <PanelHeader
                    title=move || t_string!(i18n, flow::flow_editor)
                    margin_bottom=false
                    trailing=view! {
                        <span class="ui-help">"Node-RED"</span>
                    }.into_any()
                >
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z" />
                </PanelHeader>
            </div>
            {move || src.get().map(|url| view! {
                <iframe
                    src=url
                    class="app-flow-frame"
                    allow="same-origin"
                    title=move || t_string!(i18n, flow::node_red_frame_title)
                />
            })}
        </div>
    }
}
