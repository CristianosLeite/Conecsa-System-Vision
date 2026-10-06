// SPDX-FileCopyrightText: 2026 Conecsa
//
// SPDX-License-Identifier: AGPL-3.0-only

//! A confirmation for a destructive action, on top of [`Modal`].
//!
//! Cancel comes first in the dialog, so it takes the initial focus and Enter
//! never confirms by accident; the danger button is disabled while `busy`.
//! An optional preview (the image about to be deleted) sits beside the text.

use leptos::prelude::*;

use super::modal::Modal;
use crate::i18n::*;

#[component]
pub fn DangerConfirmModal(
    /// Whether the dialog is shown.
    #[prop(into)]
    open: Signal<bool>,
    /// `id` given to the title, for `aria-labelledby`.
    #[prop(into)]
    labelled_by: String,
    #[prop(into)] title: Signal<String>,
    #[prop(into)] body: Signal<String>,
    /// The danger button's text ("Delete", or "Deleting…" while busy).
    #[prop(into)]
    confirm_label: Signal<String>,
    /// The action is running: both buttons are disabled and Esc is ignored.
    #[prop(into, optional)]
    busy: Signal<bool>,
    /// A picture to show beside the text (`None` = text only).
    #[prop(into, optional)]
    preview_src: Signal<Option<String>>,
    #[prop(into, optional)] preview_alt: Signal<String>,
    on_confirm: Callback<()>,
    on_cancel: Callback<()>,
) -> impl IntoView {
    let i18n = use_i18n();
    let title_id = labelled_by.clone();
    let on_close = Callback::new(move |_: ()| {
        if !busy.get_untracked() {
            on_cancel.run(());
        }
    });

    view! {
        <Modal open=open on_close=on_close labelled_by=labelled_by>
            <div class="ui-modal-lead">
                {move || preview_src.get().map(|src| view! {
                    <img
                        src=src
                        class="ui-media-bg ui-modal-preview"
                        alt=move || preview_alt.get()
                        loading="lazy"
                    />
                })}
                <div class="ui-modal-lead-text">
                    <h3 id=title_id.clone() class="ui-card-title mb-2">{move || title.get()}</h3>
                    <p class="ui-help text-sm">{move || body.get()}</p>
                </div>
            </div>
            <div class="ui-modal-actions">
                <button
                    type="button"
                    class="ui-button ui-button-neutral ui-button-md"
                    disabled=move || busy.get()
                    on:click=move |_| on_cancel.run(())
                >
                    {t!(i18n, common::cancel)}
                </button>
                <button
                    type="button"
                    class="ui-button ui-button-danger ui-button-md"
                    disabled=move || busy.get()
                    on:click=move |_| on_confirm.run(())
                >
                    {move || confirm_label.get()}
                </button>
            </div>
        </Modal>
    }
}
