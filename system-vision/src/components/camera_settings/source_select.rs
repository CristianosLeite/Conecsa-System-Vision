// SPDX-FileCopyrightText: 2026 Conecsa
//
// SPDX-License-Identifier: AGPL-3.0-only

use crate::api::{SOURCE_LOCAL, SOURCE_NETWORK};
use crate::i18n::*;
use leptos::prelude::*;

/// Local camera or remote camera. Only a selection: nothing reaches the device
/// until the operator applies it.
#[component]
pub(super) fn CaptureSourceSelect(
    source: ReadSignal<String>,
    set_source: WriteSignal<String>,
) -> impl IntoView {
    let i18n = use_i18n();

    view! {
        <div class="camera-field">
            <span class="ui-label">{t!(i18n, camera::source)}</span>
            <div class="camera-source-options">
                <label class="ui-radio-label">
                    <input
                        type="radio"
                        name="camera-source"
                        class="ui-radio"
                        checked=move || source.get() == SOURCE_LOCAL
                        on:change=move |_| set_source.set(SOURCE_LOCAL.to_string())
                    />
                    {t!(i18n, camera::source_local)}
                </label>
                <label class="ui-radio-label">
                    <input
                        type="radio"
                        name="camera-source"
                        class="ui-radio"
                        checked=move || source.get() == SOURCE_NETWORK
                        on:change=move |_| set_source.set(SOURCE_NETWORK.to_string())
                    />
                    {t!(i18n, camera::source_network)}
                </label>
            </div>
        </div>
    }
}
