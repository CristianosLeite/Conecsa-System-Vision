// SPDX-FileCopyrightText: 2026 Conecsa
//
// SPDX-License-Identifier: AGPL-3.0-only

use super::{IP_CREDENTIAL_MAX_BYTES, IP_NAME_MAX_BYTES, IP_URL_MAX_BYTES, IpCameraDraft};
use crate::components::common::toolbar_icon::ToolbarIcon;
use crate::i18n::*;
use leptos::prelude::*;

/// Why one IP camera is not capturing, in words about an RTSP camera (the
/// shared `detail_*` texts speak of the remote camera's token).
pub(super) fn ip_detail_text(
    i18n: leptos_i18n::I18nContext<Locale>,
    detail: &str,
) -> Option<&'static str> {
    match detail {
        "connecting" => Some(t_string!(i18n, camera::ip_detail_connecting)),
        "unauthorized" => Some(t_string!(i18n, camera::ip_detail_unauthorized)),
        "rate_limited" => Some(t_string!(i18n, camera::ip_detail_rate_limited)),
        "unreachable" => Some(t_string!(i18n, camera::ip_detail_unreachable)),
        "stalled" => Some(t_string!(i18n, camera::ip_detail_stalled)),
        "bad_stream" => Some(t_string!(i18n, camera::ip_detail_bad_stream)),
        _ => None,
    }
}

/// The IP (RTSP) cameras, in mosaic order.
///
/// Apply sends the whole list. Passwords are write-only: the device never
/// sends one back, so each password field is blank on load and only says
/// whether one is stored; leaving it blank keeps the stored password.
#[component]
pub(super) fn IpCamerasForm(
    rows: RwSignal<Vec<IpCameraDraft>>,
    /// Source of the page-local row keys for added cameras.
    next_key: StoredValue<u32>,
    max: ReadSignal<u32>,
    /// Whether the IP cameras are the applied source, so the rows' statuses
    /// describe a running camera.
    applied: Signal<bool>,
    /// Whether applying the list would rearrange the mosaic under the
    /// detection areas.
    layout_changes: Signal<bool>,
) -> impl IntoView {
    let i18n = use_i18n();

    let count = move || rows.with(Vec::len);
    let can_add = move || count() < max.get() as usize;
    let add = move |_| {
        if !can_add() {
            return;
        }
        let key = next_key.get_value();
        next_key.set_value(key.wrapping_add(1));
        rows.update(|list| {
            list.push(IpCameraDraft {
                key,
                ..Default::default()
            })
        });
    };

    view! {
        <div class="camera-network-form">
            <p class="ui-help">{move || t_string!(i18n, camera::ip_help, max = max.get())}</p>
            <details class="camera-guide">
                <summary>{t!(i18n, camera::ip_layout)}</summary>
                <p class="ui-help">{t!(i18n, camera::ip_mosaic_note)}</p>
            </details>
            <Show when=move || { count() > 0 }>
                <ol class="camera-mosaic" class:camera-mosaic-single=move || count() == 1
                    aria-label=move || t_string!(i18n, camera::ip_layout)>
                    <For each=move || rows.get() key=|row| row.key children=move |row| {
                        let key = row.key;
                        view! {
                            <li class="camera-mosaic-cell">
                                <span class="ui-label">{move || rows.with(|list| {
                                    let n = list.iter().position(|r| r.key == key).unwrap_or(0) + 1;
                                    t_string!(i18n, camera::ip_camera_heading, n = n).to_string()
                                })}</span>
                                <span class="ui-help">{move || rows.with(|list| {
                                    list.iter().find(|r| r.key == key).map(|r| r.name.clone()).unwrap_or_default()
                                })}</span>
                            </li>
                        }
                    } />
                </ol>
            </Show>

            <Show when=move || count() == 0>
                <p class="ui-empty">{t!(i18n, camera::ip_none)}</p>
            </Show>
            <For
                each=move || rows.get()
                key=|row| row.key
                children=move |row: IpCameraDraft| {
                    view! { <IpCameraRow row_key=row.key rows=rows applied=applied /> }
                }
            />

            <div class="camera-inline-row">
                <button
                    type="button"
                    class="ui-button ui-button-neutral ui-button-sm"
                    disabled=move || !can_add()
                    on:click=add
                >
                    {t!(i18n, camera::ip_add)}
                </button>
                <span class="ui-muted text-xs">
                    {move || t_string!(i18n, camera::ip_count, count = count(), max = max.get())}
                </span>
            </div>

            <Show when=move || layout_changes.get()>
                <p class="ui-alert-warning px-3 py-2.5 text-sm">{t!(i18n, camera::ip_mosaic_warning)}</p>
            </Show>
        </div>
    }
}

/// One camera of the list: its place in the mosaic, live status, the stream
/// and its credentials.
#[component]
fn IpCameraRow(
    row_key: u32,
    rows: RwSignal<Vec<IpCameraDraft>>,
    applied: Signal<bool>,
) -> impl IntoView {
    let i18n = use_i18n();

    // Everything is read through the list, so a move, a status refresh or a
    // reload never leaves this row showing a stale copy.
    let read = move |f: fn(&IpCameraDraft) -> String| {
        rows.with(|list| {
            list.iter()
                .find(|r| r.key == row_key)
                .map(f)
                .unwrap_or_default()
        })
    };
    let flag = move |f: fn(&IpCameraDraft) -> bool| {
        rows.with(|list| list.iter().find(|r| r.key == row_key).is_some_and(f))
    };
    let edit = move |f: Box<dyn FnOnce(&mut IpCameraDraft)>| {
        rows.update(|list| {
            if let Some(row) = list.iter_mut().find(|r| r.key == row_key) {
                f(row);
            }
        });
    };
    let position =
        move || rows.with(|list| list.iter().position(|r| r.key == row_key).unwrap_or(0));
    let is_last = move || rows.with(|list| position() + 1 >= list.len());
    let shift = move |up: bool| {
        rows.update(|list| {
            let Some(i) = list.iter().position(|r| r.key == row_key) else {
                return;
            };
            let j = if up { i.checked_sub(1) } else { Some(i + 1) };
            if let Some(j) = j.filter(|&j| j < list.len()) {
                list.swap(i, j);
            }
        });
    };
    let remove = move |_| rows.update(|list| list.retain(|r| r.key != row_key));

    let id = move |field: &str| format!("ip-camera-{row_key}-{field}");
    let heading =
        move || t_string!(i18n, camera::ip_camera_heading, n = position() + 1).to_string();

    // The badge: live status once the camera runs; otherwise why it has none.
    let is_new = move || !flag(|r| r.id.is_some());
    let badge_class = move || {
        if is_new() || !applied.get() {
            return "ui-badge ui-badge-muted";
        }
        match read(|r| r.status.clone()).as_str() {
            "capturing" => "ui-badge ui-badge-success",
            "starting" => "ui-badge ui-badge-warning",
            _ => "ui-badge ui-badge-danger",
        }
    };
    let badge_text = move || {
        if is_new() {
            return t_string!(i18n, camera::ip_status_new);
        }
        if !applied.get() {
            return t_string!(i18n, camera::ip_status_inactive);
        }
        match read(|r| r.status.clone()).as_str() {
            "capturing" => t_string!(i18n, camera::status_capturing),
            "starting" => t_string!(i18n, camera::status_starting),
            _ => t_string!(i18n, camera::status_no_camera),
        }
    };
    let detail = move || {
        if is_new() || !applied.get() || read(|r| r.status.clone()) == "capturing" {
            return None;
        }
        ip_detail_text(i18n, &read(|r| r.detail.clone()))
    };

    let password_set = move || flag(|r| r.password_set);
    let clearing = move || flag(|r| r.clear_password);

    view! {
        <div class="camera-ip-card">
            <div class="camera-token-head">
                <span class="camera-inline-row">
                    <span class="ui-label">{heading}</span>
                    <span class=badge_class>{badge_text}</span>
                </span>
                <span class="camera-token-actions">
                    <button
                        type="button"
                        class="ui-icon-button ui-icon-button-md"
                        title=move || t_string!(i18n, camera::ip_move_up)
                        aria-label=move || t_string!(i18n, camera::ip_move_up)
                        disabled=move || position() == 0
                        on:click=move |_| shift(true)
                    >
                        <ToolbarIcon>
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 19V5m0 0l-6 6m6-6l6 6" />
                        </ToolbarIcon>
                    </button>
                    <button
                        type="button"
                        class="ui-icon-button ui-icon-button-md"
                        title=move || t_string!(i18n, camera::ip_move_down)
                        aria-label=move || t_string!(i18n, camera::ip_move_down)
                        disabled=is_last
                        on:click=move |_| shift(false)
                    >
                        <ToolbarIcon>
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 5v14m0 0l-6-6m6 6l6-6" />
                        </ToolbarIcon>
                    </button>
                    <button
                        type="button"
                        class="ui-icon-button ui-icon-button-md ui-icon-button-danger"
                        title=move || t_string!(i18n, camera::ip_remove)
                        aria-label=move || t_string!(i18n, camera::ip_remove)
                        on:click=remove
                    >
                        <ToolbarIcon>
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                        </ToolbarIcon>
                    </button>
                </span>
            </div>
            {move || detail().map(|text| view! {
                <span class="ui-muted camera-status-detail">{text}</span>
            })}

            <div class="camera-field">
                <label class="ui-label" for=id("name")>{t!(i18n, camera::ip_name)}</label>
                <input
                    id=id("name")
                    type="text"
                    autocomplete="off"
                    maxlength=IP_NAME_MAX_BYTES.to_string()
                    class="ui-input"
                    placeholder=heading
                    prop:value=move || read(|r| r.name.clone())
                    on:input=move |ev| {
                        let value = event_target_value(&ev);
                        edit(Box::new(move |r| r.name = value));
                    }
                />
            </div>
            <div class="camera-field">
                <label class="ui-label" for=id("url")>{t!(i18n, camera::ip_url)}</label>
                <input
                    id=id("url")
                    type="text"
                    inputmode="url"
                    autocomplete="off"
                    autocapitalize="off"
                    spellcheck="false"
                    maxlength=IP_URL_MAX_BYTES.to_string()
                    class="ui-input ui-input-mono"
                    placeholder="rtsp://192.0.2.10:554/stream1"
                    prop:value=move || read(|r| r.url.clone())
                    on:input=move |ev| {
                        let value = event_target_value(&ev);
                        edit(Box::new(move |r| r.url = value));
                    }
                />
            </div>
            <div class="camera-credentials">
                <div class="camera-field">
                    <label class="ui-label" for=id("username")>{t!(i18n, camera::ip_username)}</label>
                    <input
                        id=id("username")
                        type="text"
                        autocomplete="off"
                        autocapitalize="off"
                        spellcheck="false"
                        maxlength=IP_CREDENTIAL_MAX_BYTES.to_string()
                        class="ui-input"
                        prop:value=move || read(|r| r.username.clone())
                        on:input=move |ev| {
                            let value = event_target_value(&ev);
                            edit(Box::new(move |r| r.username = value));
                        }
                    />
                </div>
                <div class="camera-field">
                    <label class="ui-label" for=id("password")>{t!(i18n, camera::ip_password)}</label>
                    <Show
                        when=move || !clearing()
                        fallback=move || view! {
                            <p class="ui-help">{t!(i18n, camera::ip_password_removing)}</p>
                        }
                    >
                        <input
                            id=id("password")
                            type="password"
                            autocomplete="new-password"
                            maxlength=IP_CREDENTIAL_MAX_BYTES.to_string()
                            class="ui-input"
                            placeholder=move || if password_set() {
                                t_string!(i18n, camera::ip_password_saved_placeholder)
                            } else {
                                t_string!(i18n, camera::ip_password_none_placeholder)
                            }
                            prop:value=move || read(|r| r.password.clone())
                            on:input=move |ev| {
                                let value = event_target_value(&ev);
                                edit(Box::new(move |r| r.password = value));
                            }
                        />
                    </Show>
                </div>
            </div>
            <Show when=password_set>
                <div class="camera-token-head">
                    <span class=move || if clearing() {
                        "ui-badge ui-badge-muted"
                    } else {
                        "ui-badge ui-badge-success"
                    }>
                        {move || if clearing() {
                            t_string!(i18n, camera::ip_password_will_clear)
                        } else {
                            t_string!(i18n, camera::ip_password_saved)
                        }}
                    </span>
                    {move || if clearing() {
                        view! {
                            <button
                                type="button"
                                class="ui-button ui-button-ghost ui-button-xs"
                                on:click=move |_| edit(Box::new(|r| r.clear_password = false))
                            >
                                {t!(i18n, camera::ip_password_keep)}
                            </button>
                        }.into_any()
                    } else {
                        view! {
                            <button
                                type="button"
                                class="ui-button ui-button-ghost ui-button-xs"
                                on:click=move |_| edit(Box::new(|r| {
                                    r.clear_password = true;
                                    r.password.clear();
                                }))
                            >
                                {t!(i18n, camera::ip_password_remove)}
                            </button>
                        }.into_any()
                    }}
                </div>
            </Show>
        </div>
    }
}
