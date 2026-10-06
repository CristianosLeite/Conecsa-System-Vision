// SPDX-FileCopyrightText: 2026 Conecsa
//
// SPDX-License-Identifier: AGPL-3.0-only

use leptos::prelude::*;

use crate::api::TrainingJobStatus;
use crate::i18n::*;
use crate::models::Task;

use super::metrics_chart::{ChartKind, ChartLabels, charts_for, has_data, metric_chart};

/// Full-screen training screen while a job runs: a status header (message,
/// epoch counter, progress bar, cancel / finish early) above a grid of
/// per-epoch charts (mAP@50:95 and the train/val losses of the task).
#[component]
pub(super) fn TrainingProgressOverlay(
    job: ReadSignal<Option<TrainingJobStatus>>,
    /// The dataset's task: picks which metrics are charted (none for a face
    /// gallery, which has no epochs).
    task: Task,
    on_cancel: Callback<()>,
    /// Gracefully finish the run early, keeping the best model so far.
    on_finish: Callback<()>,
) -> impl IntoView {
    let i18n = use_i18n();
    // Local confirm gate for "Finish early": the button row swaps to a
    // confirmation prompt before the request is actually sent.
    let confirm_finish = RwSignal::new(false);

    // Drop any pending confirmation once the job leaves "training" (it finished,
    // was canceled, or moved on to uploading) — in an effect, not the render, so
    // reading+writing the flag can't loop.
    Effect::new(move |_| {
        let active = job.get().map(|j| j.status == "training").unwrap_or(false);
        if !active {
            confirm_finish.set(false);
        }
    });

    let chart_title = move |kind: ChartKind| match kind {
        ChartKind::Map => t_string!(i18n, training::chart_map).to_string(),
        ChartKind::Accuracy => t_string!(i18n, training::chart_accuracy).to_string(),
        ChartKind::BoxLoss => t_string!(i18n, training::chart_box_loss).to_string(),
        ChartKind::ClassLoss => t_string!(i18n, training::chart_class_loss).to_string(),
        ChartKind::DflLoss => t_string!(i18n, training::chart_dfl_loss).to_string(),
        ChartKind::SegLoss => t_string!(i18n, training::chart_seg_loss).to_string(),
        ChartKind::Loss => t_string!(i18n, training::chart_loss).to_string(),
    };

    view! {
        {move || {
            let Some(j) = job.get() else {
                return view! { <div/> }.into_any();
            };
            if !matches!(j.status.as_str(), "preparing" | "training" | "uploading") {
                return view! { <div/> }.into_any();
            }
            let progress = j.progress;
            let epoch_label = if j.total_epochs > 0 && j.epoch > 0 {
                t_string!(i18n, training::epoch_progress, epoch = j.epoch, total = j.total_epochs)
            } else {
                String::new()
            };
            let title = t_string!(i18n, training::training_model_message, name = j.model_name.clone());
            // The backend's per-epoch message is the untranslated epoch
            // counter the badge already shows; other messages (preparing,
            // finishing early, uploading) are kept.
            let backend_epoch = format!("Epoch {}/{}", j.epoch, j.total_epochs);
            let message = if j.message == backend_epoch { String::new() } else { j.message.clone() };
            let cancellable = matches!(j.status.as_str(), "preparing" | "training");
            // Graceful early finish only makes sense once epochs are running.
            let finishable = j.status == "training";
            // Before the first epoch every chart shows its placeholder; after
            // it, a chart the backend never filled is hidden.
            let charts: Vec<_> = charts_for(task)
                .into_iter()
                .filter(|spec| j.history.is_empty() || has_data(spec, &j.history))
                .map(|spec| {
                    metric_chart(spec, &j.history, j.total_epochs, ChartLabels {
                        title: chart_title(spec.kind),
                        train: t_string!(i18n, training::chart_train).to_string(),
                        val: t_string!(i18n, training::chart_val).to_string(),
                        waiting: t_string!(i18n, training::chart_waiting).to_string(),
                        epoch: t_string!(i18n, training::chart_epoch).to_string(),
                    })
                })
                .collect();
            let has_charts = !charts.is_empty();
            view! {
                <div class="ui-overlay-panel ui-training-run fixed inset-0 z-50">
                    <div class="ui-training-run-body" class:ui-training-run-compact=!has_charts>
                        <section class="ui-card ui-card-pad-sm ui-training-run-status">
                            <div class="ui-training-run-heading">
                                <svg class="animate-spin w-5 h-5 ui-accent shrink-0" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                                    <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"/>
                                    <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/>
                                </svg>
                                <div class="min-w-0 flex-1">
                                    <h2 class="ui-training-run-title">{title}</h2>
                                    {(!message.is_empty()).then(|| view! {
                                        <p class="ui-help truncate">{message}</p>
                                    })}
                                </div>
                                {(!epoch_label.is_empty()).then(|| view! {
                                    <span class="ui-training-run-epoch">{epoch_label}</span>
                                })}
                            </div>
                            <div class="mt-3">
                                <div class="ui-help flex justify-between mb-1">
                                    <span>{t_string!(i18n, training::progress)}</span>
                                    <span>{format!("{}%", progress)}</span>
                                </div>
                                <div class="ui-progress-track">
                                    <div
                                        class="ui-progress-bar ui-progress-bar-primary"
                                        style=format!("width: {}%", progress)
                                    />
                                </div>
                            </div>
                            <div class="ui-training-run-actions">
                                <p class="ui-help italic">
                                    {t_string!(i18n, training::keep_tab_open)}
                                </p>
                                {if finishable && confirm_finish.get() {
                                    view! {
                                        <div class="ui-training-run-buttons">
                                            <p class="ui-help">
                                                {t_string!(i18n, training::finish_now_question)}
                                            </p>
                                            <button
                                                class="ui-button ui-button-neutral ui-button-md"
                                                on:click=move |_| confirm_finish.set(false)
                                            >
                                                {t_string!(i18n, training::keep_training)}
                                            </button>
                                            <button
                                                class="ui-button ui-button-success ui-button-md"
                                                on:click=move |_| {
                                                    confirm_finish.set(false);
                                                    on_finish.run(());
                                                }
                                            >
                                                {t_string!(i18n, training::finish_now)}
                                            </button>
                                        </div>
                                    }.into_any()
                                } else {
                                    view! {
                                        <div class="ui-training-run-buttons">
                                            {cancellable.then(|| view! {
                                                <button
                                                    class="ui-button ui-button-danger ui-button-md"
                                                    on:click=move |_| on_cancel.run(())
                                                >
                                                    {t_string!(i18n, training::cancel_training)}
                                                </button>
                                            })}
                                            {finishable.then(|| view! {
                                                <button
                                                    class="ui-button ui-button-primary ui-button-md"
                                                    on:click=move |_| confirm_finish.set(true)
                                                >
                                                    {t_string!(i18n, training::finish_early)}
                                                </button>
                                            })}
                                        </div>
                                    }.into_any()
                                }}
                            </div>
                        </section>
                        {has_charts.then(|| view! {
                            <div class="ui-training-run-charts">{charts}</div>
                        })}
                    </div>
                </div>
            }.into_any()
        }}
    }
}
