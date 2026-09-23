// SPDX-FileCopyrightText: 2026 Conecsa
//
// SPDX-License-Identifier: AGPL-3.0-only

use crate::i18n::*;
use leptos::prelude::*;

#[component]
pub(super) fn NetworkSettingsTabs(
    active_tab: ReadSignal<String>,
    set_active_tab: WriteSignal<String>,
) -> impl IntoView {
    let i18n = use_i18n();
    let tab_class = move |tab: &str| {
        if active_tab.get() == tab {
            "ui-tab ui-tab-active"
        } else {
            "ui-tab"
        }
    };

    view! {
        <div role="tablist" class="ui-tabs mb-3">
            <button
                type="button"
                role="tab"
                aria-selected=move || active_tab.get() == "wired"
                class=move || tab_class("wired")
                on:click=move |_| set_active_tab.set("wired".to_string())
            >
                {t!(i18n, settings::wired_tab)}
            </button>
            <button
                type="button"
                role="tab"
                aria-selected=move || active_tab.get() == "wifi"
                class=move || tab_class("wifi")
                on:click=move |_| set_active_tab.set("wifi".to_string())
            >
                "Wi-Fi"
            </button>
            <button
                type="button"
                role="tab"
                aria-selected=move || active_tab.get() == "ap"
                class=move || tab_class("ap")
                on:click=move |_| set_active_tab.set("ap".to_string())
            >
                {t!(i18n, settings::ap_tab)}
            </button>
        </div>
    }
}
