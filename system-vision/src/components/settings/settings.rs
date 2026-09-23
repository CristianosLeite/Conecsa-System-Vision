// SPDX-FileCopyrightText: 2026 Conecsa
//
// SPDX-License-Identifier: AGPL-3.0-only

use crate::components::access;
use crate::components::{GpioSettings, NetworkSettings};
use leptos::prelude::*;

use super::ApplicationSettings;

#[component]
pub fn Settings(
    refresh_network: ReadSignal<u32>,
    refresh_gpio: ReadSignal<u32>,
    set_error_msg: WriteSignal<String>,
    set_success_msg: WriteSignal<String>,
) -> impl IntoView {
    // Changing the application type is an administrator's decision; the
    // gateway enforces it too (PUT /api/v1/application is admin-only).
    let privileged = access::privileged();
    view! {
        <div class="min-h-full grid grid-cols-1 md:grid-cols-2 gap-4 items-start">
            {privileged.then(|| view! {
                <ApplicationSettings
                    set_error_msg=set_error_msg
                    set_success_msg=set_success_msg
                />
            })}
            <GpioSettings
                refresh_gpio=refresh_gpio
                set_error_msg=set_error_msg
                set_success_msg=set_success_msg
            />
            <NetworkSettings
                refresh_network=refresh_network
                set_error_msg=set_error_msg
                set_success_msg=set_success_msg
            />
        </div>
    }
}
