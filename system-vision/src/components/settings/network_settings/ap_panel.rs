// SPDX-FileCopyrightText: 2026 Conecsa
//
// SPDX-License-Identifier: AGPL-3.0-only

use crate::api;
use crate::api::ApStatus;
use crate::components::common::access_point::passphrase_is_valid;
use crate::i18n::*;
use leptos::prelude::*;
use leptos::task::spawn_local;

/// How often the panel re-reads the access point while it is on screen: the
/// join deadline counts down and remote cameras come and go without any event.
const POLL_MS: i32 = 3_000;

/// The device's own Wi-Fi access point, for a remote camera with no router.
///
/// Starting it takes the radio off the Wi-Fi network, so the panel says so
/// before the button and refuses to offer it without a wired link. The
/// passphrase lives in this component only: a reload means typing it again.
/// The channel list is the device's live one (the regulatory flags move), and
/// automatic, the default, leaves the choice to the device.
#[component]
pub(super) fn AccessPointPanel(
    refresh_network: ReadSignal<u32>,
    set_error_msg: WriteSignal<String>,
    set_success_msg: WriteSignal<String>,
) -> impl IntoView {
    let i18n = use_i18n();
    let status = RwSignal::new(None::<ApStatus>);
    let (passphrase, set_passphrase) = signal(String::new());
    // 0 = automatic: the device picks a channel it may start on right now.
    let (channel, set_channel) = signal(0u32);
    let (busy, set_busy) = signal(false);
    // The deadline the device reported, and how long ago it did: the panel
    // adds only its own elapsed time, never a difference between two clocks.
    let (deadline, set_deadline) = signal(None::<(u32, f64)>);

    let reload = move || {
        spawn_local(async move {
            if let Ok(st) = api::get_ap_status().await {
                let remaining = st.join_deadline_remaining_secs;
                set_deadline.set((st.active && remaining > 0).then(|| (remaining, now_secs())));
                status.set(Some(st));
            }
        });
    };

    Effect::new(move |_| {
        let _ = refresh_network.get();
        reload();
    });
    let handle = set_interval_with_handle(reload, std::time::Duration::from_millis(POLL_MS as u64));
    on_cleanup(move || {
        if let Ok(handle) = handle {
            handle.clear();
        }
    });

    let start = move |_| {
        let locale = i18n.get_locale_untracked();
        let secret = passphrase.get_untracked();
        if !passphrase_is_valid(&secret) {
            set_error_msg.set(td_string!(locale, settings::ap_passphrase_invalid).to_string());
            return;
        }
        set_busy.set(true);
        spawn_local(async move {
            match api::start_ap(secret, channel.get_untracked()).await {
                Ok(resp) if resp.success => {
                    set_passphrase.set(String::new());
                    set_success_msg.set(resp.message);
                }
                Ok(resp) => set_error_msg.set(resp.message),
                Err(e) => set_error_msg.set(td_string!(locale, settings::ap_start_failed, err = e)),
            }
            set_busy.set(false);
            reload();
        });
    };

    let stop = move |_| {
        let locale = i18n.get_locale_untracked();
        set_busy.set(true);
        spawn_local(async move {
            match api::stop_ap().await {
                Ok(resp) if resp.success => set_success_msg.set(resp.message),
                Ok(resp) => set_error_msg.set(resp.message),
                Err(e) => set_error_msg.set(td_string!(locale, settings::ap_stop_failed, err = e)),
            }
            set_busy.set(false);
            reload();
        });
    };

    let countdown = move || deadline.get().map(|(remaining, at)| remaining_secs(remaining, at, now_secs()));

    view! {
        <div class="flex flex-col gap-3">
            {move || match status.get() {
                None => view! { <div class="text-sm ui-muted">{t!(i18n, common::loading)}</div> }.into_any(),
                Some(st) if st.active => view! {
                    <div class="ui-row ui-row-wrap">
                        <span class="ui-label-xs">{t!(i18n, settings::ap_network)}</span>
                        <span class="ui-badge ui-badge-success">{t!(i18n, settings::ap_active)}</span>
                        <span class="ui-value min-w-0 break-all text-xs font-mono">{st.ssid.clone()}</span>
                    </div>
                    <div class="ui-row ui-row-wrap">
                        <span class="ui-label-xs">{t!(i18n, settings::ap_device_address)}</span>
                        <span class="ui-value text-xs font-mono">{format!("{}/{}", st.address, st.prefix)}</span>
                        <span class="ui-muted text-xs">{format!("{} MHz", st.frequency_mhz)}</span>
                    </div>
                    {move || countdown().map(|secs| view! {
                        <p class="ui-help">{t_string!(i18n, settings::ap_join_deadline, secs = secs)}</p>
                    })}
                    <div class="flex flex-col gap-1">
                        <span class="ui-label-xs">{t!(i18n, settings::ap_stations)}</span>
                        {if st.stations.is_empty() {
                            view! { <p class="ui-help">{t!(i18n, settings::ap_no_stations)}</p> }.into_any()
                        } else {
                            st.stations.iter().map(|sta| {
                                let address = if sta.address.is_empty() {
                                    t_string!(i18n, settings::ap_lease_pending).to_string()
                                } else {
                                    sta.address.clone()
                                };
                                view! {
                                    <div class="ui-row ui-row-wrap">
                                        <span class="ui-value text-xs font-mono">{address}</span>
                                        <span class="ui-muted text-xs">{sta.hostname.clone()}</span>
                                        <span class="ui-muted text-xs">{format!("{} dBm", sta.signal)}</span>
                                    </div>
                                }
                            }).collect_view().into_any()
                        }}
                    </div>
                    <p class="ui-help">{t!(i18n, settings::ap_camera_hint)}</p>
                    <button
                        class="ui-button ui-button-danger ui-button-xs w-full"
                        disabled=move || busy.get()
                        on:click=stop
                    >
                        {t!(i18n, settings::ap_stop)}
                    </button>
                }.into_any(),
                Some(st) => view! {
                    <div class="ui-alert ui-alert-warning text-xs">
                        {t!(i18n, settings::ap_warning)}
                    </div>
                    <div class="ui-row ui-row-wrap">
                        <span class="ui-label-xs">{t!(i18n, settings::ap_network)}</span>
                        <span class="ui-value min-w-0 break-all text-xs font-mono">{st.ssid.clone()}</span>
                    </div>
                    <div class="ui-row ui-row-wrap">
                        <span class="ui-label-xs">{t!(i18n, settings::ap_wired_link)}</span>
                        <span class=if st.wired_ready { "ui-badge ui-badge-success" } else { "ui-badge ui-badge-danger" }>
                            {if st.wired_ready {
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
                            {st.channels.iter().map(|ch| {
                                let ch = *ch;
                                view! {
                                    <option value=ch.to_string() selected=move || channel.get() == ch>
                                        {ch.to_string()}
                                    </option>
                                }
                            }).collect_view()}
                        </select>
                    </div>
                    {(!st.message.is_empty()).then(|| view! {
                        <p class="ui-help">{st.message.clone()}</p>
                    })}
                    <button
                        class="ui-button ui-button-primary ui-button-xs w-full"
                        disabled=move || busy.get() || !st.wired_ready
                        on:click=start
                    >
                        {move || if busy.get() {
                            t_string!(i18n, settings::ap_starting)
                        } else {
                            t_string!(i18n, settings::ap_start)
                        }}
                    </button>
                }.into_any(),
            }}
        </div>
    }
}

/// Seconds left of a join deadline the device reported as `reported` at the
/// browser time `reported_at`: only this browser's own elapsed time is
/// subtracted, never a difference between the device clock and ours, and a
/// clock that goes backwards leaves the value where it was.
pub(super) fn remaining_secs(reported: u32, reported_at: f64, now: f64) -> u32 {
    let elapsed = (now - reported_at).max(0.0) as u32;
    reported.saturating_sub(elapsed)
}

/// The browser's monotonic clock, in seconds.
fn now_secs() -> f64 {
    web_sys::window()
        .and_then(|w| w.performance())
        .map(|p| p.now() / 1000.0)
        .unwrap_or(0.0)
}
