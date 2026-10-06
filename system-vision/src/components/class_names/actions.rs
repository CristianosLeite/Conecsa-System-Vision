// SPDX-FileCopyrightText: 2026 Conecsa
//
// SPDX-License-Identifier: AGPL-3.0-only

use crate::components::common::toolbar_icon::ToolbarIcon;
use crate::i18n::*;
use leptos::prelude::*;

#[component]
pub(super) fn ClassActions(
    edit_mode: ReadSignal<bool>,
    on_start_edit: Callback<()>,
    on_cancel: Callback<()>,
    on_save: Callback<()>,
    on_upload: Callback<()>,
) -> impl IntoView {
    let i18n = use_i18n();
    // Role gating: class-list editing is admin-only; buttons stay visible.
    let privileged = crate::components::access::privileged();
    let restricted_title = move || {
        if privileged {
            ""
        } else {
            t_string!(i18n, common::restricted_to_admins)
        }
    };
    view! {
        <div class="flex flex-wrap gap-2">
            {move || if !edit_mode.get() {
                view! {
                    <>
                        <button
                            type="button"
                            class="ui-icon-button ui-icon-button-md"
                            disabled=!privileged
                            title=move || if privileged { t_string!(i18n, models::edit) } else { restricted_title() }
                            aria-label=move || t_string!(i18n, models::edit)
                            on:click=move |_| on_start_edit.run(())
                        >
                            <ToolbarIcon>
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
                            </ToolbarIcon>
                        </button>
                        <button
                            class="ui-button ui-button-primary ui-button-sm disabled:opacity-50 disabled:cursor-not-allowed"
                            disabled=!privileged
                            title=restricted_title
                            on:click=move |_| on_upload.run(())
                        >
                            <svg class="w-4 h-4 stroke-current" viewBox="0 0 24 24" fill="none">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                            </svg>
                            {t!(i18n, models::upload_file)}
                        </button>
                    </>
                }.into_any()
            } else {
                view! {
                    <>
                        <button
                            type="button"
                            class="ui-icon-button ui-icon-button-md ui-icon-button-danger"
                            disabled=!privileged
                            title=move || if privileged { t_string!(i18n, common::cancel) } else { restricted_title() }
                            aria-label=move || t_string!(i18n, common::cancel)
                            on:click=move |_| on_cancel.run(())
                        >
                            <ToolbarIcon>
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
                            </ToolbarIcon>
                        </button>
                        <button
                            type="button"
                            class="ui-icon-button ui-icon-button-md ui-icon-button-success"
                            disabled=!privileged
                            title=move || if privileged { t_string!(i18n, common::save) } else { restricted_title() }
                            aria-label=move || t_string!(i18n, common::save)
                            on:click=move |_| on_save.run(())
                        >
                            <ToolbarIcon>
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" />
                            </ToolbarIcon>
                        </button>
                    </>
                }.into_any()
            }}
        </div>
    }
}
