// SPDX-FileCopyrightText: 2026 Conecsa
//
// SPDX-License-Identifier: AGPL-3.0-only

use crate::api::ApStatus;
use crate::components::common::access_point::passphrase_is_valid;
use crate::components::common::modal::Modal;
use crate::i18n::*;
use leptos::prelude::*;

/// Asked when the remote camera is applied while the device's access point is
/// off: turn it on for the remote camera, or apply without it because the
/// remote camera's own hotspot is the link.
///
/// The passphrase lives only while the dialog is open. Turning the access
/// point on takes the radio off the Wi-Fi network, so the warning comes before
/// the choice and the button is refused without a wired link.
#[component]
pub(super) fn AccessPointConfirmModal(
    open: ReadSignal<bool>,
    status: RwSignal<Option<ApStatus>>,
    busy: ReadSignal<bool>,
    on_close: Callback<()>,
    on_apply_only: Callback<()>,
    /// `(passphrase, channel)`.
    on_turn_on: Callback<(String, u32)>,
) -> impl IntoView {
    let i18n = use_i18n();
    let (passphrase, set_passphrase) = signal(String::new());
    // 0 = automatic: the device picks a channel it may start on right now.
    let (channel, set_channel) = signal(0u32);

    Effect::new(move |_| {
        if !open.get() {
            set_passphrase.set(String::new());
        }
    });

    let ssid = move || status.get().map(|st| st.ssid).unwrap_or_default();
    let wired_ready = move || status.get().is_some_and(|st| st.wired_ready);
    // The device's live list: which channels may start a network changes
    // with the regulatory flags, so nothing is assumed here.
    let channels = move || status.get().map(|st| st.channels).unwrap_or_default();
    let can_turn_on =
        move || !busy.get() && wired_ready() && passphrase_is_valid(&passphrase.get());

    view! {
        <Modal open=open on_close=on_close labelled_by="camera-ap-confirm-title">
            <h3 id="camera-ap-confirm-title" class="ui-card-title mb-2">
                {t!(i18n, camera::ap_confirm_title)}
            </h3>
            <p class="ui-help text-sm">{t!(i18n, camera::ap_confirm_body)}</p>
            <div class="camera-ap-confirm">
                <div class="ui-alert ui-alert-warning text-xs">
                    {t!(i18n, settings::ap_warning)}
                </div>
                <div class="ui-row ui-row-wrap">
                    <span class="ui-label-xs">{t!(i18n, settings::ap_network)}</span>
                    <span class="ui-value min-w-0 break-all text-xs font-mono">{ssid}</span>
                </div>
                <div class="ui-row ui-row-wrap">
                    <span class="ui-label-xs">{t!(i18n, settings::ap_wired_link)}</span>
                    <span class=move || if wired_ready() {
                        "ui-badge ui-badge-success"
                    } else {
                        "ui-badge ui-badge-danger"
                    }>
                        {move || if wired_ready() {
                            t_string!(i18n, settings::ap_wired_ready)
                        } else {
                            t_string!(i18n, settings::ap_wired_missing)
                        }}
                    </span>
                </div>
                <input
                    type="password"
                    class="ui-input ui-input-sm"
                    autocomplete="new-password"
                    placeholder=move || t_string!(i18n, settings::ap_passphrase)
                    prop:value=move || passphrase.get()
                    on:input=move |ev| set_passphrase.set(event_target_value(&ev))
                />
                <div class="ui-row ui-row-wrap">
                    <span class="ui-label-xs">{t!(i18n, settings::ap_channel)}</span>
                    <select
                        class="ui-select cursor-pointer"
                        on:change=move |ev| {
                            if let Ok(ch) = event_target_value(&ev).parse::<u32>() {
                                set_channel.set(ch);
                            }
                        }
                    >
                        <option value="0" selected=move || channel.get() == 0>
                            {t!(i18n, settings::ap_channel_auto)}
                        </option>
                        {move || channels().into_iter().map(|ch| view! {
                            <option value=ch.to_string() selected=move || channel.get() == ch>
                                {ch.to_string()}
                            </option>
                        }).collect_view()}
                    </select>
                </div>
            </div>
            <div class="ui-modal-actions">
                <button
                    type="button"
                    class="ui-button ui-button-neutral ui-button-md"
                    disabled=move || busy.get()
                    on:click=move |_| on_close.run(())
                >
                    {t_string!(i18n, common::cancel)}
                </button>
                <button
                    type="button"
                    class="ui-button ui-button-neutral ui-button-md"
                    disabled=move || busy.get()
                    on:click=move |_| on_apply_only.run(())
                >
                    {t_string!(i18n, camera::ap_apply_without)}
                </button>
                <button
                    type="button"
                    class="ui-button ui-button-warning ui-button-md"
                    disabled=move || !can_turn_on()
                    on:click=move |_| on_turn_on.run((passphrase.get_untracked(), channel.get_untracked()))
                >
                    {move || if busy.get() {
                        t_string!(i18n, settings::ap_starting).to_string()
                    } else {
                        t_string!(i18n, camera::ap_turn_on_apply).to_string()
                    }}
                </button>
            </div>
        </Modal>
    }
}
