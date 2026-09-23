// SPDX-FileCopyrightText: 2026 Conecsa
//
// SPDX-License-Identifier: AGPL-3.0-only

use leptos::prelude::*;

use crate::components::PowerButton;
use crate::i18n::*;

#[component]
pub fn Header(api_health: ReadSignal<bool>) -> impl IntoView {
    let i18n = use_i18n();
    view! {
        <header class="app-header">
            <div class="app-header-inner">
                <div class="app-brand">
                    <img src="/public/conecsa_white_logo.png" alt="CONECSA Logo" class="app-brand-logo" />
                    <div class="app-brand-copy">
                        <p class="app-brand-wordmark">
                            <span>"CONEC"</span>
                            <span class="app-brand-accent">"SA"</span>
                        </p>
                        <p class="app-brand-subtitle">"AUTOMAÇÃO"</p>
                    </div>
                </div>
                <div class="app-service-status">
                    <span class="text-sm font-medium opacity-85">{t!(i18n, common::inference_service)}</span>
                    <span class={move || if api_health.get() {
                        "app-status-pill app-status-pill-online"
                    } else {
                        "app-status-pill app-status-pill-offline"
                    }}>
                        <span class={move || if api_health.get() {
                            "app-status-dot app-status-dot-online status-dot-pulse"
                        } else {
                            "app-status-dot app-status-dot-offline"
                        }}></span>
                        {move || if api_health.get() {
                            t_string!(i18n, common::connected)
                        } else {
                            t_string!(i18n, common::disconnected)
                        }}
                    </span>
                    <PowerButton />
                </div>
            </div>
        </header>
    }
}
