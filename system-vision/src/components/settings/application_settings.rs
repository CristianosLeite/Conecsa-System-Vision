// SPDX-FileCopyrightText: 2026 Conecsa
//
// SPDX-License-Identifier: AGPL-3.0-only

//! Settings → Application type (admin only): shows the device's application
//! and changes it behind a confirmation.

use leptos::prelude::*;
use leptos::task::spawn_local;

use crate::api;
use crate::components::application_select::{task_label, use_application};
use crate::components::common::modal::Modal;
use crate::i18n::*;
use crate::models::Task;

#[component]
pub fn ApplicationSettings(
    set_error_msg: WriteSignal<String>,
    set_success_msg: WriteSignal<String>,
) -> impl IntoView {
    let i18n = use_i18n();
    let app = use_application();
    // The task awaiting confirmation; `Some` opens the dialog.
    let (pending, set_pending) = signal(None::<Task>);
    let (busy, set_busy) = signal(false);

    let confirm = move |_| {
        let Some(task) = pending.get_untracked() else {
            return;
        };
        if busy.get_untracked() {
            return;
        }
        set_busy.set(true);
        let locale = i18n.get_locale_untracked();
        spawn_local(async move {
            match api::set_application(task.id()).await {
                Ok(info) => {
                    app.apply(info);
                    let _ = set_success_msg.try_set(td_string!(
                        locale,
                        application::changed,
                        task = task_label(locale, task)
                    ));
                }
                Err(e) => {
                    let _ = set_error_msg
                        .try_set(td_string!(locale, application::failed_to_change, err = e));
                }
            }
            let _ = set_busy.try_set(false);
            let _ = set_pending.try_set(None);
        });
    };
    let close = Callback::new(move |_| {
        if !busy.get_untracked() {
            set_pending.set(None);
        }
    });

    let current = move || match app.task() {
        Some(task) => task_label(i18n.get_locale(), task),
        None => t_string!(i18n, application::none).to_string(),
    };

    view! {
        <section class="ui-card ui-card-pad md:col-span-2" aria-labelledby="application-settings-title">
            <h3 id="application-settings-title" class="ui-card-title mb-2">
                {t!(i18n, application::title)}
            </h3>
            <p class="ui-help">
                {t!(i18n, application::current)} ": " <strong>{current}</strong>
            </p>
            {move || {
                app.info
                    .get()
                    .filter(|info| info.migrated)
                    .map(|_| view! { <p class="ui-help">{t!(i18n, application::migrated_note)}</p> })
            }}
            <div class="ui-application-options" role="group" aria-labelledby="application-settings-title">
                {Task::ALL
                    .into_iter()
                    .map(|task| {
                        let supported = move || app.supports(task);
                        let is_current = move || app.task() == Some(task);
                        view! {
                            <div class="ui-application-option">
                                <button
                                    type="button"
                                    class="ui-button ui-button-neutral ui-button-sm"
                                    aria-pressed=move || is_current().to_string()
                                    disabled=move || !supported() || is_current() || busy.get()
                                    title=move || {
                                        if supported() {
                                            String::new()
                                        } else {
                                            t_string!(i18n, application::later_release).to_string()
                                        }
                                    }
                                    on:click=move |_| set_pending.set(Some(task))
                                >
                                    {move || task_label(i18n.get_locale(), task)}
                                </button>
                                {move || {
                                    (!supported())
                                        .then(|| view! {
                                            <span class="ui-application-card-note">
                                                {t_string!(i18n, application::later_release)}
                                            </span>
                                        })
                                }}
                            </div>
                        }
                    })
                    .collect_view()}
            </div>

            <Modal
                open=Signal::derive(move || pending.get().is_some())
                on_close=close
                labelled_by="application-confirm-title"
            >
                <h3 id="application-confirm-title" class="ui-card-title mb-2">
                    {t!(i18n, application::confirm_title)}
                </h3>
                <p class="ui-help text-sm">
                    {move || {
                        pending
                            .get()
                            .map(|task| {
                                td_string!(
                                    i18n.get_locale(),
                                    application::confirm_body,
                                    task = task_label(i18n.get_locale(), task)
                                )
                            })
                    }}
                </p>
                <div class="ui-modal-actions">
                    <button
                        type="button"
                        class="ui-button ui-button-neutral ui-button-md"
                        disabled=move || busy.get()
                        on:click=move |_| close.run(())
                    >
                        {t_string!(i18n, common::cancel)}
                    </button>
                    <button
                        type="button"
                        class="ui-button ui-button-warning ui-button-md"
                        disabled=move || busy.get()
                        on:click=confirm
                    >
                        {move || if busy.get() {
                            t_string!(i18n, application::choosing).to_string()
                        } else {
                            t_string!(i18n, application::confirm_action).to_string()
                        }}
                    </button>
                </div>
            </Modal>
        </section>
    }
}
