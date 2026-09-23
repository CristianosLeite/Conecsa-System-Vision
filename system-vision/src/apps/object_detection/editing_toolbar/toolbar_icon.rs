// SPDX-FileCopyrightText: 2026 Conecsa
//
// SPDX-License-Identifier: AGPL-3.0-only

use leptos::prelude::*;

#[component]
pub(super) fn ToolbarIcon(children: Children) -> impl IntoView {
    view! {
        <svg class="w-4 h-4 stroke-current" viewBox="0 0 24 24" fill="none" aria-hidden="true">
            {children()}
        </svg>
    }
}
