// SPDX-FileCopyrightText: 2026 Conecsa
//
// SPDX-License-Identifier: AGPL-3.0-only

//! The selection screen a blank device shows instead of the dashboard.

use leptos::prelude::*;
use leptos::task::spawn_local;

use crate::api;
use crate::components::access;
use crate::i18n::*;
use crate::models::{AppState, Task};

use super::state::{task_description, task_label, use_application};

/// Shown while no application type is chosen (or the device reports one this
/// UI does not know). An administrator picks one of the tasks the device's
/// build supports; the others are disabled with the reason visible. Anyone
/// else sees a waiting notice — the header and the device's other controls
/// stay in place, only the dashboard gives way.
#[component]
pub fn ApplicationSelect(
    set_error_msg: WriteSignal<String>,
    set_success_msg: WriteSignal<String>,
) -> impl IntoView {
    let i18n = use_i18n();
    let app = use_application();
    let privileged = access::privileged();
    let (busy, set_busy) = signal(None::<Task>);

    let choose = move |task: Task| {
        if busy.get_untracked().is_some() {
            return;
        }
        set_busy.set(Some(task));
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
            let _ = set_busy.try_set(None);
        });
    };

    let cards = move || {
        Task::ALL
            .into_iter()
            .map(|task| {
                let supported = move || app.supports(task);
                view! {
                    <button
                        type="button"
                        class="ui-application-card"
                        disabled=move || !supported() || busy.get().is_some()
                        title=move || {
                            if supported() {
                                String::new()
                            } else {
                                t_string!(i18n, application::later_release).to_string()
                            }
                        }
                        on:click=move |_| choose(task)
                    >
                        <span class="ui-application-card-title">
                            {move || task_label(i18n.get_locale(), task)}
                        </span>
                        <span class="ui-help">
                            {move || task_description(i18n.get_locale(), task)}
                        </span>
                        {move || {
                            if !supported() {
                                view! {
                                    <span class="ui-application-card-note">
                                        {t_string!(i18n, application::later_release)}
                                    </span>
                                }
                                    .into_any()
                            } else if busy.get() == Some(task) {
                                view! {
                                    <span class="ui-application-card-note">
                                        {t_string!(i18n, application::choosing)}
                                    </span>
                                }
                                    .into_any()
                            } else {
                                view! { <span></span> }.into_any()
                            }
                        }}
                    </button>
                }
            })
            .collect_view()
    };

    view! {
        <section class="ui-application-select" aria-labelledby="application-select-title">
            <div>
                <h2 id="application-select-title" class="ui-application-title">
                    {t!(i18n, application::select_title)}
                </h2>
                <p class="ui-help">{t!(i18n, application::select_intro)}</p>
            </div>

            {move || match app.state.get() {
                AppState::Unsupported(id) => view! {
                    <div class="ui-alert-warning" role="status">
                        <strong>{t_string!(i18n, application::unsupported_title)}</strong>
                        " "
                        {td_string!(i18n.get_locale(), application::unsupported_body, task = id)}
                    </div>
                }
                    .into_any(),
                _ => view! { <div></div> }.into_any(),
            }}

            {if privileged {
                view! { <div class="ui-application-cards">{cards}</div> }.into_any()
            } else {
                view! {
                    <div class="ui-card ui-card-pad" role="status">
                        <h3 class="ui-card-title mb-2">
                            {t!(i18n, application::waiting_title)}
                        </h3>
                        <p class="ui-help">{t!(i18n, application::waiting_body)}</p>
                    </div>
                }
                    .into_any()
            }}
        </section>
    }
}
