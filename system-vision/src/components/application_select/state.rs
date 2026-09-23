// SPDX-FileCopyrightText: 2026 Conecsa
//
// SPDX-License-Identifier: AGPL-3.0-only

//! The [`Application`] context: what the UI knows about the device's
//! application type, loaded once per mounted UI and re-read on events.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use gloo_timers::future::TimeoutFuture;
use leptos::prelude::*;
use leptos::task::spawn_local;

use crate::api;
use crate::i18n::*;
use crate::models::{AppState, ApplicationInfo, Task};

/// Delay before the next attempt to read the application type after a
/// failure: 2 s, 5 s, 10 s, then every 10 s.
pub(super) fn retry_delay_ms(attempt: usize) -> u32 {
    match attempt {
        0 => 2_000,
        1 => 5_000,
        _ => 10_000,
    }
}

/// The device's application type, shared by every component that depends on
/// it (the dashboard gate, the model list, the upload, Settings).
#[derive(Clone, Copy)]
pub struct Application {
    pub state: RwSignal<AppState>,
    pub info: RwSignal<Option<ApplicationInfo>>,
}

impl Application {
    fn new() -> Self {
        Self {
            state: RwSignal::new(AppState::Loading),
            info: RwSignal::new(None),
        }
    }

    /// Adopt a backend answer (GET or PUT).
    pub fn apply(&self, info: ApplicationInfo) {
        let _ = self.state.try_set(AppState::from_task(info.task.as_deref()));
        let _ = self.info.try_set(Some(info));
    }

    /// Whether the device's build can run `task` (tracked).
    pub fn supports(&self, task: Task) -> bool {
        self.info.get().is_some_and(|info| info.supports(task))
    }

    /// The chosen task, when known (tracked).
    pub fn task(&self) -> Option<Task> {
        self.state.get().task()
    }

    /// Re-read after an event or a status change. A failure keeps the last
    /// known state: a transient error must never bring the selector up.
    pub fn reload(self) {
        spawn_local(async move {
            match api::get_application().await {
                Ok(info) => self.apply(info),
                Err(e) => {
                    leptos::logging::warn!("Could not re-read the application type: {}", e);
                    if self.state.try_get_untracked() == Some(AppState::Loading) {
                        let _ = self.state.try_set(AppState::Error);
                    }
                }
            }
        });
    }
}

/// The [`Application`] context, provided (and loaded) on first use.
///
/// Called by `MainView` before any consumer renders, so the host page — the
/// app itself or the interactive manual — needs no setup of its own. The
/// first read retries with [`retry_delay_ms`] until it succeeds; until then
/// the state is `Loading`, then `Error`, and the dashboard stays up.
pub fn use_application() -> Application {
    if let Some(app) = use_context::<Application>() {
        return app;
    }
    let app = Application::new();
    provide_context(app);

    let alive = Arc::new(AtomicBool::new(true));
    {
        let alive = alive.clone();
        on_cleanup(move || alive.store(false, Ordering::Relaxed));
    }
    spawn_local(async move {
        let mut attempt = 0;
        loop {
            match api::get_application().await {
                Ok(info) => {
                    if alive.load(Ordering::Relaxed) {
                        app.apply(info);
                    }
                    break;
                }
                Err(e) => {
                    leptos::logging::warn!("Could not read the application type: {}", e);
                    // Stop when unmounted or when an event-driven re-read got
                    // there while this request was in flight: its answer
                    // stands, a late failure must not replace it with Error.
                    let known = app.state.try_get_untracked().is_none_or(|s| s.is_known());
                    if !alive.load(Ordering::Relaxed) || known {
                        break;
                    }
                    let _ = app.state.try_set(AppState::Error);
                }
            }
            TimeoutFuture::new(retry_delay_ms(attempt)).await;
            attempt += 1;
            // Stop when unmounted or when an event-driven re-read got there.
            let known = app.state.try_get_untracked().is_none_or(|s| s.is_known());
            if !alive.load(Ordering::Relaxed) || known {
                break;
            }
        }
    });
    app
}

/// The localized name of a task.
pub fn task_label(locale: Locale, task: Task) -> String {
    match task {
        Task::Detect => td_string!(locale, application::task_detect).to_string(),
        Task::Classify => td_string!(locale, application::task_classify).to_string(),
        Task::Segment => td_string!(locale, application::task_segment).to_string(),
        Task::Face => td_string!(locale, application::task_face).to_string(),
    }
}

/// One localized sentence on what a task does.
pub fn task_description(locale: Locale, task: Task) -> String {
    match task {
        Task::Detect => td_string!(locale, application::desc_detect).to_string(),
        Task::Classify => td_string!(locale, application::desc_classify).to_string(),
        Task::Segment => td_string!(locale, application::desc_segment).to_string(),
        Task::Face => td_string!(locale, application::desc_face).to_string(),
    }
}
