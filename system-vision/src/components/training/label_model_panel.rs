// SPDX-FileCopyrightText: 2026 Conecsa
//
// SPDX-License-Identifier: AGPL-3.0-only

use leptos::prelude::*;

use crate::api::model_stem;
use crate::components::configuration::threshold_slider::ThresholdSlider;
use crate::i18n::*;

use super::dataset_editor::AiState;

/// AI-assist bar for an existing device model: Detect/Accept/Clear plus the
/// confidence threshold. Rendered by the editor only while a model is the
/// active assistant. Suggestions come back tagged with the model's class
/// names; Accept resolves them against the dataset (creating missing ones).
#[component]
pub(super) fn LabelModelPanel(
    ai: AiState,
    on_detect: Callback<()>,
    on_accept: Callback<()>,
    on_clear: Callback<()>,
) -> impl IntoView {
    let i18n = use_i18n();
    let model_status = ai.model_status;
    let ai_busy = ai.busy;
    let suggestions = ai.suggestions;
    let threshold = ai.threshold.read_only();
    let set_threshold = ai.threshold.write_only();
    let summary = move || {
        model_status
            .get()
            .filter(|s| s.loaded)
            .map(|s| {
                format!(
                    "{} · {}",
                    model_stem(&s.model_name),
                    t_string!(i18n, training::model_classes, count = s.class_names.len())
                )
            })
            .unwrap_or_default()
    };
    view! {
        <div class="ui-list-box flex flex-col gap-2 p-2">
            <div class="flex items-center gap-2 flex-wrap">
                <span class="text-sm font-medium flex-1 min-w-32 truncate" title=summary>
                    {summary}
                </span>
                <button
                    class="ui-button ui-button-primary ui-button-xs"
                    disabled=move || ai_busy.get()
                    on:click=move |_| on_detect.run(())
                >
                    {move || if ai_busy.get() {
                        t_string!(i18n, training::detecting)
                    } else {
                        t_string!(i18n, training::detect)
                    }}
                </button>
                {move || {
                    let n = suggestions.get().len();
                    (n > 0).then(|| view! {
                        <button
                            class="ui-button ui-button-success ui-button-xs"
                            on:click=move |_| on_accept.run(())
                            title=t_string!(i18n, training::accept_model_suggestions_title)
                        >
                            {t_string!(i18n, training::accept_model_suggestions, count = n)}
                        </button>
                    })
                }}
                <button
                    class="ui-button ui-button-neutral ui-button-xs"
                    on:click=move |_| on_clear.run(())
                >
                    {t!(i18n, training::clear)}
                </button>
            </div>
            // Confidence threshold for the model's suggestions — reuses the
            // dashboard's slider; re-runs Detect on change.
            <ThresholdSlider
                label=move || t_string!(i18n, training::ai_confidence)
                description=move || t_string!(i18n, training::ai_confidence_desc)
                value=threshold
                set_value=set_threshold
                on_change=Callback::new(move |_: f32| on_detect.run(()))
            />
        </div>
    }
}
