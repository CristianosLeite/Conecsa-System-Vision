// SPDX-FileCopyrightText: 2026 Conecsa
//
// SPDX-License-Identifier: AGPL-3.0-only

use crate::i18n::*;
use leptos::prelude::*;

/// The camera health the webcam-server reports, and — for a remote camera —
/// why it is not capturing.
///
/// The health always describes the *applied* source, so the row names it; when
/// the selector points elsewhere, it says so, or a "Capturing" from the local
/// camera would read as a working remote camera (and the other way round).
#[component]
pub(super) fn CameraStatusRow(
    status: ReadSignal<String>,
    detail: ReadSignal<String>,
    applied_source: ReadSignal<String>,
    selected_source: ReadSignal<String>,
) -> impl IntoView {
    let i18n = use_i18n();

    let applied_label = move || if applied_source.get() == crate::api::SOURCE_NETWORK {
        t_string!(i18n, camera::source_network)
    } else {
        t_string!(i18n, camera::source_local)
    };
    let pending_switch = move || selected_source.get() != applied_source.get();

    let badge = move || match status.get().as_str() {
        "capturing" => "ui-badge ui-badge-success",
        "starting" => "ui-badge ui-badge-warning",
        _ => "ui-badge ui-badge-danger",
    };
    let status_text = move || match status.get().as_str() {
        "capturing" => t_string!(i18n, camera::status_capturing),
        "starting" => t_string!(i18n, camera::status_starting),
        _ => t_string!(i18n, camera::status_no_camera),
    };
    // An unknown or unspecified detail adds nothing to the status.
    let detail_text = move || match detail.get().as_str() {
        "connecting" => Some(t_string!(i18n, camera::detail_connecting)),
        "unauthorized" => Some(t_string!(i18n, camera::detail_unauthorized)),
        "rate_limited" => Some(t_string!(i18n, camera::detail_rate_limited)),
        "unreachable" => Some(t_string!(i18n, camera::detail_unreachable)),
        "stalled" => Some(t_string!(i18n, camera::detail_stalled)),
        "bad_stream" => Some(t_string!(i18n, camera::detail_bad_stream)),
        _ => None,
    };

    view! {
        <div class="camera-status">
            <div class="ui-row ui-row-wrap">
                <span class="ui-label-xs">{t!(i18n, camera::status)}</span>
                <span class=badge>{status_text}</span>
                <span class="ui-muted camera-status-detail">
                    {move || t_string!(i18n, camera::status_applied_source, source = applied_label())}
                </span>
                {move || detail_text().map(|text| view! {
                    <span class="ui-muted camera-status-detail">{text}</span>
                })}
            </div>
            <Show when=pending_switch>
                <p class="ui-help">{t!(i18n, camera::status_not_applied)}</p>
            </Show>
        </div>
    }
}
