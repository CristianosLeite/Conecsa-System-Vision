// SPDX-FileCopyrightText: 2026 Conecsa
//
// SPDX-License-Identifier: AGPL-3.0-only

use super::{format_token_input, AddressSuggestion};
use crate::api::ApStatus;
use crate::i18n::*;
use leptos::prelude::*;

/// Address, port and token of the remote camera.
///
/// The token is write-only: the device never sends it back, so the field is
/// always blank on load and `token_set` says whether one is stored. Leaving it
/// blank keeps the stored token.
#[component]
pub(super) fn NetworkSourceForm(
    host: ReadSignal<String>,
    set_host: WriteSignal<String>,
    port: ReadSignal<String>,
    set_port: WriteSignal<String>,
    token: ReadSignal<String>,
    set_token: WriteSignal<String>,
    token_set: ReadSignal<bool>,
    /// Where the address in the field was suggested from, while it still holds
    /// that unconfirmed suggestion.
    suggestion: Signal<Option<AddressSuggestion>>,
    /// The device's own access point, shown while it is on: the remote camera
    /// joins it, and its address comes from there.
    ap_status: RwSignal<Option<ApStatus>>,
) -> impl IntoView {
    let i18n = use_i18n();

    // With a token stored the input is folded away behind "Change token":
    // an always-present empty field reads as "fill me in" and invites
    // re-entering a stored token. Folds back after each load.
    let (editing_token, set_editing_token) = signal(false);
    Effect::new(move |_| {
        if token_set.get() {
            set_editing_token.set(false);
        }
    });
    let show_input = move || !token_set.get() || editing_token.get();

    view! {
        <div class="camera-network-form">
            <p class="ui-help">{t!(i18n, camera::network_help)}</p>
            {move || ap_status.get().filter(|st| st.active).map(|st| {
                let joined = st.stations.len() as u32;
                view! {
                    <div class="camera-ap-line">
                        <span class="ui-badge ui-badge-success">{t!(i18n, camera::ap_on)}</span>
                        <span class="ui-value min-w-0 break-all text-xs font-mono">{st.ssid.clone()}</span>
                        <span class="ui-muted text-xs">
                            {if joined == 0 {
                                t_string!(i18n, camera::ap_active_no_station).to_string()
                            } else {
                                t_string!(i18n, camera::ap_joined_count, count = joined).to_string()
                            }}
                        </span>
                    </div>
                }
            })}

            <div class="camera-endpoint">
                <div class="camera-field">
                    <label class="ui-label" for="camera-network-host">
                        {t!(i18n, camera::network_address)}
                    </label>
                    <input
                        id="camera-network-host"
                        type="text"
                        inputmode="decimal"
                        autocomplete="off"
                        spellcheck="false"
                        class="ui-input ui-input-mono"
                        placeholder="192.0.2.1"
                        prop:value=move || host.get()
                        on:input=move |ev| set_host.set(event_target_value(&ev))
                    />
                </div>
                <div class="camera-field">
                    <label class="ui-label" for="camera-network-port">
                        {t!(i18n, camera::network_port)}
                    </label>
                    <input
                        id="camera-network-port"
                        type="number"
                        min="1"
                        max="65535"
                        class="ui-input ui-input-mono"
                        prop:value=move || port.get()
                        on:input=move |ev| set_port.set(event_target_value(&ev))
                    />
                </div>
            </div>
            {move || suggestion.get().map(|origin| view! {
                <p class="ui-help">
                    {match origin {
                        AddressSuggestion::Gateway(address) => {
                            t_string!(i18n, camera::hotspot_suggestion, address = address).to_string()
                        }
                        AddressSuggestion::Station(address) => {
                            t_string!(i18n, camera::ap_station_suggestion, address = address).to_string()
                        }
                    }}
                </p>
            })}

            <div class="camera-field">
                <div class="camera-token-head">
                    <label class="ui-label" for="camera-network-token">
                        {t!(i18n, camera::network_token)}
                    </label>
                    <span class="camera-token-actions">
                        <span class=move || if token_set.get() {
                            "ui-badge ui-badge-success"
                        } else {
                            "ui-badge ui-badge-muted"
                        }>
                            {move || if token_set.get() {
                                t_string!(i18n, camera::token_stored)
                            } else {
                                t_string!(i18n, camera::token_not_set)
                            }}
                        </span>
                        <Show when=move || token_set.get()>
                            {move || if editing_token.get() {
                                view! {
                                    <button
                                        type="button"
                                        class="ui-button ui-button-ghost ui-button-xs"
                                        on:click=move |_| {
                                            set_token.set(String::new());
                                            set_editing_token.set(false);
                                        }
                                    >
                                        {t!(i18n, camera::token_keep)}
                                    </button>
                                }.into_any()
                            } else {
                                view! {
                                    <button
                                        type="button"
                                        class="ui-button ui-button-ghost ui-button-xs"
                                        on:click=move |_| set_editing_token.set(true)
                                    >
                                        {t!(i18n, camera::token_change)}
                                    </button>
                                }.into_any()
                            }}
                        </Show>
                    </span>
                </div>
                // Visible, not masked: the operator is reading it off the remote camera's
                // screen anyway, and formatting it as they type shows that hyphens
                // and case are taken care of.
                <Show when=show_input>
                    <input
                        id="camera-network-token"
                        type="text"
                        inputmode="latin"
                        autocomplete="off"
                        autocapitalize="characters"
                        spellcheck="false"
                        class="ui-input ui-input-mono"
                        placeholder="XXXX-XXXX"
                        prop:value=move || token.get()
                        on:input=move |ev| set_token.set(format_token_input(&event_target_value(&ev)))
                    />
                    <p class="ui-help">
                        {move || if token_set.get() {
                            t_string!(i18n, camera::token_replace_hint)
                        } else {
                            t_string!(i18n, camera::token_first_hint)
                        }}
                    </p>
                </Show>
            </div>
        </div>
    }
}
