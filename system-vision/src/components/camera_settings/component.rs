// SPDX-FileCopyrightText: 2026 Conecsa
//
// SPDX-License-Identifier: AGPL-3.0-only

use crate::api;
use crate::components::panel_header::PanelHeader;
use crate::i18n::*;
use leptos::prelude::*;
use leptos::task::spawn_local;

use super::ap_confirm_modal::AccessPointConfirmModal;
use super::apply_button::ApplyCameraSettingsButton;
use super::device_select::CaptureDeviceSelect;
use super::loading_state::CameraSettingsLoadingState;
use super::network_source_form::NetworkSourceForm;
use super::resolution_controls::ResolutionControls;
use super::source_select::CaptureSourceSelect;
use super::status_row::CameraStatusRow;
use super::stereo_toggle::StereoOverlayToggle;
use super::{
    is_stereo_camera, offers_access_point, parse_network_form, station_suggestion,
    AddressSuggestion, CameraFormat, NetworkFormError, Resolution,
};

/// How often the access point is re-read while the remote camera form is on
/// screen: a remote camera joining it raises no event.
const AP_POLL_MS: u64 = 3_000;

/// Push only the stereo combine settings. Applied immediately (no camera
/// restart, since stereo lives in the inference-service, not the SHM config).
fn push_stereo_enabled(enabled: bool) {
    spawn_local(async move {
        let _ = api::update_stereo_config(Some(enabled), None, None, None).await;
    });
}

#[component]
pub fn CameraSettings(
    refresh_camera: ReadSignal<u32>,
    refresh_network: ReadSignal<u32>,
    /// Live camera health from the event stream (see `main_view`).
    camera_health: ReadSignal<Option<api::CameraHealth>>,
    set_error_msg: WriteSignal<String>,
    set_success_msg: WriteSignal<String>,
) -> impl IntoView {
    let i18n = use_i18n();

    // Device list: (index, label)
    let (devices, set_devices) = signal(Vec::<(u32, String)>::new());
    let (selected_index, set_selected_index) = signal(0u32);
    let (selected_res, set_selected_res) = signal(Resolution { w: 640, h: 480 });
    // Supported (width, height, [fps...]) combinations reported by the camera.
    // When non-empty the UI shows real dropdowns instead of the static presets.
    let (formats, set_formats) = signal(Vec::<CameraFormat>::new());
    let (selected_framerate, set_selected_framerate) = signal(30u32);
    let (selected_stereo_enabled, set_selected_stereo_enabled) = signal(false);
    // Indices of the enumerated devices that are 3D cameras, and whether the
    // list has been read at all — an empty list before the first fetch means
    // "unknown", not "no 3D camera".
    let (stereo_devices, set_stereo_devices) = signal(Vec::<u32>::new());
    let (devices_loaded, set_devices_loaded) = signal(false);
    let (loading, set_loading) = signal(true);
    let (saving, set_saving) = signal(false);

    // Capture source. `source` follows the selector; `saved_source` is what the
    // device runs. The token field is write-only: blank on every load, with
    // `token_set` saying whether the device has one stored.
    let (source, set_source) = signal(api::SOURCE_LOCAL.to_string());
    let (saved_source, set_saved_source) = signal(api::SOURCE_LOCAL.to_string());
    let (network_host, set_network_host) = signal(String::new());
    let (saved_host, set_saved_host) = signal(String::new());
    let (network_port, set_network_port) = signal(String::new());
    let (network_token, set_network_token) = signal(String::new());
    let (token_set, set_token_set) = signal(false);
    let (camera_status, set_camera_status) = signal(String::new());
    let (camera_detail, set_camera_detail) = signal(String::new());
    // Wi-Fi gateway: on a remote camera's hotspot that is the remote camera itself.
    let (wifi_gateway, set_wifi_gateway) = signal(None::<String>);
    // The address the device's access point leased to the remote camera that joined.
    let (station_address, set_station_address) = signal(None::<String>);

    // The device's own access point, read while the remote camera form is on
    // screen: Apply offers to turn it on, and the remote camera that joins it
    // is where the address comes from.
    let ap_status = RwSignal::new(None::<api::ApStatus>);
    let (ap_confirm_open, set_ap_confirm_open) = signal(false);
    let (ap_busy, set_ap_busy) = signal(false);
    let reload_ap = move || {
        spawn_local(async move {
            if let Ok(st) = api::get_ap_status().await {
                ap_status.set(Some(st));
            }
        });
    };
    Effect::new(move |_| {
        let _ = refresh_network.get();
        reload_ap();
    });
    let ap_poll = StoredValue::new(None::<IntervalHandle>);
    let stop_ap_poll = move || {
        ap_poll.update_value(|slot| {
            if let Some(handle) = slot.take() {
                handle.clear();
            }
        });
    };
    Effect::new(move |_| {
        stop_ap_poll();
        if source.get() == api::SOURCE_NETWORK {
            reload_ap();
            let handle = set_interval_with_handle(
                reload_ap,
                std::time::Duration::from_millis(AP_POLL_MS),
            );
            ap_poll.set_value(handle.ok());
        }
    });
    on_cleanup(stop_ap_poll);

    // Shown while the address field still holds an unconfirmed suggestion.
    let suggestion = Signal::derive(move || {
        let host = network_host.get();
        if host.is_empty() || host == saved_host.get() {
            return None;
        }
        if station_address.get().as_deref() == Some(host.as_str()) {
            return Some(AddressSuggestion::Station(host));
        }
        if wifi_gateway.get().as_deref() == Some(host.as_str()) {
            return Some(AddressSuggestion::Gateway(host));
        }
        None
    });

    // The remote camera that joined the access point: fill its leased address
    // into the field while the operator has not typed or stored one. A hotspot
    // gateway suggestion gives way to it, a confirmed address never does.
    Effect::new(move |_| {
        let Some(address) = station_suggestion(ap_status.get().as_ref()) else {
            return;
        };
        let host = network_host.get_untracked();
        let unconfirmed = host.is_empty()
            || wifi_gateway.get_untracked().as_deref() == Some(host.as_str())
            || station_address.get_untracked().as_deref() == Some(host.as_str());
        if unconfirmed {
            if host != address {
                set_network_host.set(address.clone());
            }
            set_station_address.set(Some(address));
        }
    });

    // The stereo overlay only makes sense on a 3D camera: it splits one frame
    // into its left|right halves and blends them, which would tear an ordinary
    // picture in two. Resolution cannot tell a side-by-side frame from a merely
    // wide one, so the camera model decides. Follows the dropdown selection,
    // like the other settings, rather than the applied device.
    // A remote camera's frame is never side by side either, so a network source
    // counts as "not a 3D camera" once it is the applied source.
    let stereo_supported = Signal::derive(move || {
        devices_loaded.get()
            && saved_source.get() != api::SOURCE_NETWORK
            && stereo_devices.get().contains(&selected_index.get())
    });

    // Never leave the overlay enabled on a camera that is not a 3D camera.
    // Gated on `devices_loaded` so a pending or failed fetch counts as unknown
    // and cannot clobber a working 3D configuration.
    Effect::new(move |_| {
        if devices_loaded.get()
            && !stereo_supported.get()
            && selected_stereo_enabled.get_untracked()
        {
            set_selected_stereo_enabled.set(false);
            push_stereo_enabled(false);
        }
    });

    let reload_camera = move || {
        set_loading.set(true);
        let locale = i18n.get_locale_untracked();
        spawn_local(async move {
            match api::get_camera_devices().await {
                Ok(resp) => {
                    let list: Vec<(u32, String)> = resp
                        .devices
                        .iter()
                        .map(|d| {
                            let idx = if d.index >= 0 { d.index as u32 } else { 0 };
                            let label = if !d.name.is_empty() && d.name != d.path {
                                format!("{} ({})", d.name, d.path)
                            } else {
                                d.path.clone()
                            };
                            (idx, label)
                        })
                        .collect();
                    let stereo: Vec<u32> = resp
                        .devices
                        .iter()
                        .filter(|d| d.index >= 0 && is_stereo_camera(&d.name))
                        .map(|d| d.index as u32)
                        .collect();
                    set_devices.set(list);
                    set_stereo_devices.set(stereo);
                    set_selected_index.set(resp.current_index);
                    set_selected_res.set(Resolution {
                        w: resp.current_width,
                        h: resp.current_height,
                    });

                    // Prefer MJPG (the high-fps path); merge fps lists per resolution.
                    let mut map: std::collections::BTreeMap<(u32, u32), Vec<u32>> =
                        std::collections::BTreeMap::new();
                    if let Some(dev) = resp
                        .devices
                        .iter()
                        .find(|d| d.index == resp.current_index as i32)
                    {
                        let has_mjpg = dev
                            .supported_formats
                            .iter()
                            .any(|f| f.format.eq_ignore_ascii_case("MJPG"));
                        for f in &dev.supported_formats {
                            if has_mjpg && !f.format.eq_ignore_ascii_case("MJPG") {
                                continue;
                            }
                            let entry = map.entry((f.width, f.height)).or_default();
                            for &v in &f.fps {
                                if !entry.contains(&v) {
                                    entry.push(v);
                                }
                            }
                        }
                    }
                    let mut fmt_list: Vec<CameraFormat> = map
                        .into_iter()
                        .map(|((w, h), mut fps)| {
                            fps.sort_unstable_by(|a, b| b.cmp(a));
                            (w, h, fps)
                        })
                        .collect();
                    fmt_list.sort_by(|a, b| (b.0 * b.1).cmp(&(a.0 * a.1)));
                    set_formats.set(fmt_list);
                    set_selected_framerate.set(resp.current_framerate);
                    set_selected_stereo_enabled.set(resp.current_stereo_enabled);

                    set_source.set(resp.current_source.clone());
                    set_saved_source.set(resp.current_source);
                    set_saved_host.set(resp.current_network_host.clone());
                    set_network_port.set(match resp.current_network_port {
                        0 => "8080".to_string(),
                        port => port.to_string(),
                    });
                    set_network_token.set(String::new());
                    set_token_set.set(resp.network_token_set);
                    set_camera_status.set(resp.camera_status);
                    set_camera_detail.set(resp.camera_detail);

                    // With no address stored yet, suggest the Wi-Fi gateway: on
                    // a remote camera's hotspot that is the remote camera. It only
                    // fills the field — nothing is saved until the operator applies it.
                    if resp.current_network_host.is_empty() {
                        let gateway = api::get_network_config()
                            .await
                            .ok()
                            .and_then(|net| net.wifi.gateway)
                            .filter(|gw| !gw.is_empty());
                        set_network_host.set(gateway.clone().unwrap_or_default());
                        set_wifi_gateway.set(gateway);
                    } else {
                        set_network_host.set(resp.current_network_host);
                    }

                    // Last: everything the stereo guard reads is now in place.
                    set_devices_loaded.set(true);
                    set_loading.set(false);
                }
                Err(e) => {
                    set_error_msg.set(td_string!(locale, camera::failed_to_load_info, err = e));
                    set_loading.set(false);
                }
            }
        });
    };

    Effect::new(move |_| {
        let _ = refresh_camera.get();
        reload_camera();
    });

    // The status row follows the camera as it changes — only the status: the
    // form keeps whatever the operator is typing.
    Effect::new(move |_| {
        if let Some(health) = camera_health.get() {
            set_camera_status.set(health.status);
            set_camera_detail.set(health.detail);
            if !health.source.is_empty() {
                set_saved_source.set(health.source);
            }
        }
    });

    // Validates the remote camera form and makes it the device's source.
    let save_network_source = move || {
        let locale = i18n.get_locale_untracked();
        let form = parse_network_form(
            &network_host.get_untracked(),
            &network_port.get_untracked(),
            &network_token.get_untracked(),
            token_set.get_untracked(),
        );
        let (host, port, token) = match form {
            Ok(form) => form,
            Err(NetworkFormError::Address) => {
                return set_error_msg.set(td_string!(locale, camera::address_required).to_string())
            }
            Err(NetworkFormError::Port) => {
                return set_error_msg.set(td_string!(locale, camera::port_invalid).to_string())
            }
            Err(NetworkFormError::Token) => {
                return set_error_msg.set(td_string!(locale, camera::token_required).to_string())
            }
        };
        set_saving.set(true);
        spawn_local(async move {
            match api::update_camera_source(api::SOURCE_NETWORK, Some(host), Some(port), token)
                .await
            {
                Ok(_) => {
                    set_success_msg.set(td_string!(locale, camera::source_applied).to_string());
                    reload_camera();
                }
                Err(e) => set_error_msg.set(td_string!(locale, camera::failed_to_apply, err = e)),
            }
            set_saving.set(false);
        });
    };

    // The confirmation's choices. "Turn on" starts the access point first;
    // the source is then saved only with an address the operator entered or
    // confirmed — an empty field, or a suggestion still standing, means the
    // remote camera has not joined yet, and its address is filled in when it does.
    let ap_close = Callback::new(move |_| {
        if !ap_busy.get_untracked() {
            set_ap_confirm_open.set(false);
        }
    });
    let ap_apply_only = Callback::new(move |_| {
        set_ap_confirm_open.set(false);
        save_network_source();
    });
    let ap_turn_on = Callback::new(move |(passphrase, channel): (String, u32)| {
        if ap_busy.get_untracked() {
            return;
        }
        let locale = i18n.get_locale_untracked();
        set_ap_busy.set(true);
        spawn_local(async move {
            let outcome = api::start_ap(passphrase, channel).await;
            set_ap_busy.set(false);
            set_ap_confirm_open.set(false);
            reload_ap();
            match outcome {
                Ok(resp) if resp.success => {
                    let address_pending = network_host.get_untracked().trim().is_empty()
                        || suggestion.get_untracked().is_some();
                    if address_pending {
                        set_success_msg
                            .set(td_string!(locale, camera::ap_started_await_address).to_string());
                    } else {
                        save_network_source();
                    }
                }
                Ok(resp) => set_error_msg.set(resp.message),
                Err(e) => set_error_msg.set(td_string!(locale, settings::ap_start_failed, err = e)),
            }
        });
    });

    // Applies device / resolution / framerate (the fields that restart capture).
    // Exposure, RGB, gamma and gain are adjusted live from the live-video overlay.
    let apply = Callback::new(move |_| {
        let locale = i18n.get_locale_untracked();

        if source.get() == api::SOURCE_NETWORK {
            // With the device's access point off, ask first: the remote camera
            // may need it, and turning it on drops the Wi-Fi link.
            if offers_access_point(&source.get(), ap_status.get().as_ref()) {
                set_ap_confirm_open.set(true);
            } else {
                save_network_source();
            }
            return;
        }

        let idx = selected_index.get();
        let res = selected_res.get();
        let framerate = selected_framerate.get();
        let leaving_network = saved_source.get() == api::SOURCE_NETWORK;
        set_saving.set(true);
        spawn_local(async move {
            // Back to the local camera first; the stored remote camera address and
            // token stay on the device for the next switch.
            if leaving_network
                && let Err(e) =
                    api::update_camera_source(api::SOURCE_LOCAL, None, None, None).await
            {
                set_error_msg.set(td_string!(locale, camera::failed_to_apply, err = e));
                set_saving.set(false);
                return;
            }
            match api::update_camera_config(
                Some(idx),
                Some(res.w),
                Some(res.h),
                Some(framerate),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )
            .await
            {
                Ok(_) => {
                    set_success_msg
                        .set(td_string!(locale, camera::settings_applied).to_string());
                    reload_camera();
                }
                Err(e) => {
                    set_error_msg.set(td_string!(locale, camera::failed_to_apply, err = e))
                }
            }
            set_saving.set(false);
        });
    });

    let push_stereo = Callback::new(move |enabled: bool| push_stereo_enabled(enabled));

    view! {
        <div class="ui-card ui-card-pad ui-card-scroll h-full">
            <PanelHeader title=move || t_string!(i18n, camera::title)>
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                    d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
            </PanelHeader>

            {move || if loading.get() {
                view! { <CameraSettingsLoadingState /> }.into_any()
            } else {
                view! {
                    <div class="camera-settings">
                        <CameraStatusRow
                            status=camera_status
                            detail=camera_detail
                            applied_source=saved_source
                            selected_source=source
                        />
                        <CaptureSourceSelect source=source set_source=set_source />
                        // Device, resolution, framerate and stereo describe a
                        // V4L2 camera; a remote camera has none of them.
                        {move || if source.get() == api::SOURCE_NETWORK {
                            view! {
                                <NetworkSourceForm
                                    host=network_host
                                    set_host=set_network_host
                                    port=network_port
                                    set_port=set_network_port
                                    token=network_token
                                    set_token=set_network_token
                                    token_set=token_set
                                    suggestion=suggestion
                                    ap_status=ap_status
                                />
                            }.into_any()
                        } else {
                            view! {
                                <CaptureDeviceSelect
                                    devices=devices
                                    selected_index=selected_index
                                    set_selected_index=set_selected_index
                                />
                                <ResolutionControls
                                    formats=formats
                                    selected_res=selected_res
                                    set_selected_res=set_selected_res
                                    selected_framerate=selected_framerate
                                    set_selected_framerate=set_selected_framerate
                                />
                                <StereoOverlayToggle
                                    stereo_supported=stereo_supported
                                    selected_stereo_enabled=selected_stereo_enabled
                                    set_selected_stereo_enabled=set_selected_stereo_enabled
                                    on_push=push_stereo
                                />
                            }.into_any()
                        }}
                        <ApplyCameraSettingsButton saving=saving on_apply=apply />
                    </div>
                }.into_any()
            }}
            <AccessPointConfirmModal
                open=ap_confirm_open
                status=ap_status
                busy=ap_busy
                on_close=ap_close
                on_apply_only=ap_apply_only
                on_turn_on=ap_turn_on
            />
        </div>
    }
}
