// SPDX-FileCopyrightText: 2026 Conecsa
//
// SPDX-License-Identifier: AGPL-3.0-only

use crate::api::{SOURCE_IP, SOURCE_LOCAL, SOURCE_NETWORK};
use crate::i18n::*;
use leptos::prelude::*;

/// Local camera, remote camera or IP cameras. Only a selection: nothing reaches the device
/// until the operator applies it.
#[component]
pub(super) fn CaptureSourceSelect(
    source: ReadSignal<String>,
    set_source: WriteSignal<String>,
) -> impl IntoView {
    let i18n = use_i18n();

    view! {
        <fieldset class="camera-source-picker">
            <legend class="ui-section-title">{t!(i18n, camera::source)}</legend>
            <div class="camera-source-options">
                <label class="camera-source-card">
                    <input
                        type="radio"
                        name="camera-source"
                        class="ui-radio"
                        prop:checked=move || source.get() == SOURCE_LOCAL
                        on:change=move |_| set_source.set(SOURCE_LOCAL.to_string())
                    />
                    <span class="camera-source-copy">
                        <span class="ui-card-title">{t!(i18n, camera::source_local)}</span>
                        <span class="ui-help">{t!(i18n, camera::source_local_hint)}</span>
                    </span>
                </label>
                <label class="camera-source-card">
                    <input
                        type="radio"
                        name="camera-source"
                        class="ui-radio"
                        prop:checked=move || source.get() == SOURCE_NETWORK
                        on:change=move |_| set_source.set(SOURCE_NETWORK.to_string())
                    />
                    <span class="camera-source-copy">
                        <span class="ui-card-title">{t!(i18n, camera::source_network)}</span>
                        <span class="ui-help">{t!(i18n, camera::source_network_hint)}</span>
                    </span>
                </label>
                <label class="camera-source-card">
                    <input
                        type="radio"
                        name="camera-source"
                        class="ui-radio"
                        prop:checked=move || source.get() == SOURCE_IP
                        on:change=move |_| set_source.set(SOURCE_IP.to_string())
                    />
                    <span class="camera-source-copy">
                        <span class="ui-card-title">{t!(i18n, camera::source_ip)}</span>
                        <span class="ui-help">{t!(i18n, camera::source_ip_hint)}</span>
                    </span>
                </label>
            </div>
        </fieldset>
    }
}
