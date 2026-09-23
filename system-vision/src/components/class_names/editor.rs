// SPDX-FileCopyrightText: 2026 Conecsa
//
// SPDX-License-Identifier: AGPL-3.0-only

use crate::i18n::*;
use leptos::prelude::*;

#[component]
pub(super) fn ClassEditor(
    edited_classes_text: ReadSignal<String>,
    set_edited_classes_text: WriteSignal<String>,
    /// A face model: the lines are the enrolled people, fixed in count and order.
    face: Signal<bool>,
) -> impl IntoView {
    let i18n = use_i18n();
    view! {
        <div class="flex flex-col min-h-[100px]">
            <textarea
                class="ui-textarea ui-input-mono flex-1"
                placeholder=move || t_string!(i18n, models::class_names_placeholder)
                prop:value={move || edited_classes_text.get()}
                on:input=move |ev| {
                    set_edited_classes_text.set(event_target_value(&ev));
                }
            />
            <p class="ui-help mt-1 shrink-0">
                {t!(i18n, models::class_editor_help)}
            </p>
            <Show when=move || face.get()>
                <p class="ui-help mt-1 shrink-0">
                    {t!(i18n, models::face_class_editor_help)}
                </p>
            </Show>
        </div>
    }
}
