// SPDX-FileCopyrightText: 2026 Conecsa
//
// SPDX-License-Identifier: AGPL-3.0-only

//! Work spawned when the editor mounts: the initial loads, the SAM warm-up,
//! the active-class clamp and the training-job poll.
//!
//! Every spawner is called synchronously from the component body so effects
//! and cleanups bind to the editor's owner; the long-lived loops take the
//! editor's `alive` flag and stop touching signals once it is cleared.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use gloo_timers::future::TimeoutFuture;
use leptos::prelude::*;
use leptos::task::spawn_local;

use crate::api;
use crate::components::configuration::model_conversion::PendingConversion;
use crate::components::control_panel::ViewMode;
use crate::i18n::*;
use crate::models::Task;

use super::actions;
use super::logic::{self, JobTransition};
use super::state::{Assistant, EditorState, I18n};

/// Initial load: the dataset record, its images and classes, the device model
/// lists, plus a labeling engine still resident from a previous visit (it is
/// unloaded on exit, so this is usually empty).
pub(super) fn spawn_initial_loads(st: EditorState, i18n: I18n, alive: Arc<AtomicBool>) {
    spawn_local(async move {
        let ds = st.dataset_id.get_value();
        if let Ok(d) = api::get_training_dataset(&ds).await {
            let _ = st.training.min_images.try_set(d.min_images.max(1));
            let _ = st.images.cover_image_id.try_set(d.cover_image_id);
        }
    });
    actions::refresh_images(st, i18n);

    let locale = i18n.get_locale_untracked();
    spawn_local(async move {
        let ds = st.dataset_id.get_value();
        match api::get_training_classes(&ds).await {
            Ok(r) => {
                let _ = st.classes.list.try_set(r.classes);
            }
            Err(e) => st
                .notices
                .error(td_string!(locale, training::failed_load_classes, err = e)),
        }
    });

    // Face recognition has no labeling assistant and no fine-tune base: the
    // photos are labeled by name and the device builds a gallery from them.
    if st.task == Task::Face {
        return;
    }

    spawn_local(async move {
        // Only engines of the dataset's task can suggest its labels or serve
        // as its fine-tune base (the backend refuses the others).
        if let Ok(lists) = api::list_training_models_for(st.task.id()).await {
            if !alive.load(Ordering::Relaxed) {
                return;
            }
            let _ = st.ai.label_models.try_set(lists.labeling);
            let _ = st.training.base_models.try_set(lists.fine_tune);
        }
        if let Ok(s) = api::get_label_model_status().await {
            if !alive.load(Ordering::Relaxed) {
                return;
            }
            if s.loaded {
                let _ = st
                    .ai
                    .assistant
                    .try_set(Assistant::Model(s.model_name.clone()));
            }
            let _ = st.ai.model_status.try_set(Some(s));
        }
    });
}

/// Predictive SAM warm-up: start the cold load (~1min on the Orin) as soon
/// as the dataset is opened so it overlaps the user's first capture and
/// labeling actions instead of blocking the first SAM toggle. Silent — the
/// user has not asked for SAM yet, so on failure SAM loads when it is
/// toggled. Duplicate-safe: SamService.load() is
/// idempotent under its lock, so a toggle mid-warm-up simply joins it.
pub(super) fn spawn_sam_warmup(st: EditorState, alive: Arc<AtomicBool>) {
    // A face dataset never uses SAM; loading it would take the GPU for nothing.
    if st.task == Task::Face {
        return;
    }
    spawn_local(async move {
        let Ok(s) = api::get_sam_status().await else {
            return;
        };
        if !alive.load(Ordering::Relaxed) {
            return;
        }
        let _ = st.ai.sam_status.try_set(Some(s.clone()));
        if !s.available || s.loaded {
            return;
        }
        // A running job owns the GPU; the backend refuses LoadSam anyway.
        if let Ok(j) = api::get_training_status().await {
            if j.is_active() {
                return;
            }
        }
        if !alive.load(Ordering::Relaxed) {
            return;
        }
        let _ = api::load_sam().await;
        if !alive.load(Ordering::Relaxed) {
            return;
        }
        if let Ok(s) = api::get_sam_status().await {
            let _ = st.ai.sam_status.try_set(Some(s));
        }
    });
}

/// Keep active_class within bounds whenever the class list changes, so the
/// class id stamped on new/accepted boxes is always a real class (an
/// out-of-range or empty-list id is what the backend rejects as "Unknown
/// class id"). Defaults to the first class when the list becomes non-empty.
pub(super) fn install_active_class_clamp(st: EditorState) {
    Effect::new(move |_| {
        let len = st.classes.list.get().len();
        if len > 0 {
            st.classes
                .active
                .update(|c| *c = logic::clamp_active(*c, len));
        }
    });
}

/// Poll the training job while this page is mounted. Cheap when idle; on
/// completion it hands the conversion job to the dashboard and exits.
///
/// `exited` is TrainingView's exit-dedup flag: set when this loop exits
/// training mode itself (training-done handoff) so the parent's defensive
/// cleanup never fires a second exit that would resume detection
/// mid-conversion.
pub(super) fn spawn_job_poll(
    st: EditorState,
    i18n: I18n,
    alive: Arc<AtomicBool>,
    exited: Arc<AtomicBool>,
    set_pending_conversion: WriteSignal<Option<PendingConversion>>,
    set_current_view: WriteSignal<ViewMode>,
) {
    spawn_local(async move {
        // Seed the baseline: a job that finished in a PREVIOUS session is
        // still reported "done" by the backend. Without seeding, the loop's
        // first comparison (prev_status = "") would treat that stale "done"
        // as a completion on THIS page and instantly bounce back to the
        // dashboard, re-firing the conversion handoff.
        if let Ok(j) = api::get_training_status().await {
            if !alive.load(Ordering::Relaxed) {
                return;
            }
            let _ = st.training.job.try_set(Some(j));
        }
        loop {
            TimeoutFuture::new(2_000).await;
            if !alive.load(Ordering::Relaxed) {
                break;
            }
            let Ok(j) = api::get_training_status().await else {
                continue;
            };
            if !alive.load(Ordering::Relaxed) {
                break;
            }
            let prev = st.training.job.try_get_untracked().flatten();
            let prev_status = prev.as_ref().map(|w| w.status.clone()).unwrap_or_default();
            match logic::job_transition(&prev_status, &j, prev.is_some(), st.task) {
                JobTransition::Finished { pending } => {
                    let _ = st.training.job.try_set(Some(j));
                    // Claim the exit BEFORE the request, and never release
                    // it (even if the call fails): an unmount while it is
                    // in flight — or a stuck training mode afterwards —
                    // must never be answered by the parent's cleanup with
                    // a resume=true exit while the conversion may hold the
                    // GPU; genuinely stuck states are the orphan
                    // watchdog's job.
                    exited.store(true, Ordering::Relaxed);
                    // Training finished: leave detection stopped — the model
                    // is being converted/optimized and the auto-select will
                    // load the new engine without starting detection.
                    let _ = api::training_exit(false).await;
                    // Parent-owned signals (MainView): safe across our own
                    // unmount, but bail if the loop was cleaned up.
                    if !alive.load(Ordering::Relaxed) {
                        break;
                    }
                    set_pending_conversion.set(pending);
                    set_current_view.set(ViewMode::LiveStream);
                    break;
                }
                JobTransition::Failed => {
                    let locale = i18n.get_locale_untracked();
                    st.notices.error(td_string!(
                        locale,
                        training::training_failed,
                        err = j.error.clone()
                    ));
                    let _ = st.training.job.try_set(Some(j));
                }
                JobTransition::Canceled => {
                    let locale = i18n.get_locale_untracked();
                    st.notices
                        .success(td_string!(locale, training::training_canceled).to_string());
                    let _ = st.training.job.try_set(Some(j));
                }
                JobTransition::Update => {
                    let _ = st.training.job.try_set(Some(j));
                }
                JobTransition::Ignore => {}
            }
        }
    });
}
