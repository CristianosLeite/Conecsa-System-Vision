// SPDX-FileCopyrightText: 2026 Conecsa
//
// SPDX-License-Identifier: AGPL-3.0-only

use leptos::prelude::*;

use crate::components::application_select::use_application;
use crate::i18n::*;
use crate::models::Task;

/// Upload modal: format notice + name + .zip picker. The gallery owns the
/// state, the file input ref and the upload call; this only renders + emits.
/// The notice follows the device's task: a new dataset is labeled for it, so
/// the archive must be a YOLO detection export, one folder per class, or — for
/// face recognition — one folder of photos per person.
#[component]
pub(super) fn DatasetUploadModal(
    visible: ReadSignal<bool>,
    name: ReadSignal<String>,
    set_name: WriteSignal<String>,
    busy: ReadSignal<bool>,
    file_input_ref: NodeRef<leptos::html::Input>,
    on_submit: Callback<()>,
    on_cancel: Callback<()>,
) -> impl IntoView {
    let i18n = use_i18n();
    let application = use_application();
    let notice = move || match application.task() {
        Some(Task::Face) => t_string!(i18n, training::upload_dataset_notice_face),
        Some(Task::Classify) => t_string!(i18n, training::upload_dataset_notice_classify),
        _ => t_string!(i18n, training::upload_dataset_notice),
    };
    view! {
        {move || if visible.get() {
            view! {
                <div class="ui-modal-backdrop">
                    <div class="ui-card ui-card-pad ui-modal">
                        <h3 class="ui-card-title mb-2">{t_string!(i18n, training::upload_dataset)}</h3>
                        <p class="ui-help text-sm mb-3">{notice}</p>
                        <div class="flex flex-col gap-2">
                            <input
                                type="text"
                                class="ui-input"
                                placeholder=t_string!(i18n, training::dataset_name_placeholder)
                                prop:value=move || name.get()
                                on:input=move |ev| set_name.set(event_target_value(&ev))
                            />
                            <input
                                type="file"
                                accept=".zip"
                                class="ui-input"
                                node_ref=file_input_ref
                            />
                        </div>
                        <div class="ui-modal-actions">
                            <button
                                class="ui-button ui-button-neutral ui-button-md"
                                disabled=move || busy.get()
                                on:click=move |_| on_cancel.run(())
                            >
                                {t_string!(i18n, common::cancel)}
                            </button>
                            <button
                                class="ui-button ui-button-success ui-button-md"
                                disabled=move || busy.get()
                                on:click=move |_| on_submit.run(())
                            >
                                {move || if busy.get() {
                                    t_string!(i18n, training::importing)
                                } else {
                                    t_string!(i18n, training::upload)
                                }}
                            </button>
                        </div>
                    </div>
                </div>
            }.into_any()
        } else {
            view! { <div/> }.into_any()
        }}
    }
}
