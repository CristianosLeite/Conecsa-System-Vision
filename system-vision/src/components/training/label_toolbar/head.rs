// SPDX-FileCopyrightText: 2026 Conecsa
//
// SPDX-License-Identifier: AGPL-3.0-only

//! The toolbar's head row: the card title, the draw-mode select of a
//! segmentation dataset and the AI-assist selector. Nothing here depends on
//! the selection, the history or the zoom, so the row only re-wraps on a
//! viewport resize.

use leptos::prelude::*;

use crate::api::model_stem;
use crate::i18n::*;

use super::super::dataset_editor::{AiState, Assistant, DrawMode};

#[component]
pub(super) fn ToolbarHead(
    ai: AiState,
    on_assistant_change: Callback<Assistant>,
    /// How new rings are drawn (`Some` in a segmentation dataset).
    draw_mode: Option<RwSignal<DrawMode>>,
) -> impl IntoView {
    let i18n = use_i18n();
    let assistant = ai.assistant;
    let sam_status = ai.sam_status;
    // Built engines on the device (labeling candidates).
    let label_models = ai.label_models;
    let ai_busy = ai.busy;
    let sam_available = move || sam_status.get().map(|s| s.available).unwrap_or(false);
    let sam_unavailable_msg = move || {
        sam_status
            .get()
            .map(|s| s.message)
            .filter(|m| !m.is_empty())
            .unwrap_or_else(|| t_string!(i18n, training::ai_unavailable).to_string())
    };

    view! {
        <div class="ui-label-toolbar-head">
            <h2 class="ui-card-title">{t!(i18n, training::label_editor_title)}</h2>
            <div class="ui-label-toolbar-head-group">
                {draw_mode.map(|draw_mode| {
                    // The hint of the active draw mode, shown as the select's tooltip.
                    let mode_title = move || match draw_mode.get() {
                        DrawMode::Click => t_string!(i18n, training::draw_mode_click_title),
                        DrawMode::Freehand => t_string!(i18n, training::draw_mode_freehand_title),
                        DrawMode::Rect => t_string!(i18n, training::draw_mode_rect_title),
                    };
                    view! {
                        <label class="ui-label-toolbar-field">
                            <span class="ui-help text-xs">{t!(i18n, training::draw_mode_label)}</span>
                            <select
                                class="ui-select ui-select-sm w-auto cursor-pointer"
                                title=mode_title
                                on:change=move |ev| {
                                    let next = match event_target_value(&ev).as_str() {
                                        "freehand" => DrawMode::Freehand,
                                        "rect" => DrawMode::Rect,
                                        _ => DrawMode::Click,
                                    };
                                    draw_mode.set(next);
                                }
                            >
                                <option value="click" prop:selected=move || draw_mode.get() == DrawMode::Click>
                                    {t!(i18n, training::draw_mode_click)}
                                </option>
                                <option value="freehand" prop:selected=move || draw_mode.get() == DrawMode::Freehand>
                                    {t!(i18n, training::draw_mode_freehand)}
                                </option>
                                <option value="rect" prop:selected=move || draw_mode.get() == DrawMode::Rect>
                                    {t!(i18n, training::draw_mode_rect)}
                                </option>
                            </select>
                        </label>
                    }
                })}
                <label class="ui-label-toolbar-field">
                    <span class="ui-help text-xs">{t!(i18n, training::assistant_label)}</span>
                    // `prop:selected` (not the attribute): once the user has
                    // touched a select, the DOM ignores `selected` attributes,
                    // and a failed load must visibly snap back to Off.
                    <select
                        class="ui-select ui-select-sm w-auto cursor-pointer"
                        title=move || t_string!(i18n, training::assistant_title)
                        disabled=move || ai_busy.get()
                        on:change=move |ev| {
                            let v = event_target_value(&ev);
                            let next = if v == "sam" {
                                Assistant::Sam
                            } else if let Some(name) = v.strip_prefix("model:") {
                                Assistant::Model(name.to_string())
                            } else {
                                Assistant::Off
                            };
                            on_assistant_change.run(next);
                        }
                    >
                        <option
                            value="off"
                            prop:selected=move || assistant.get() == Assistant::Off
                        >
                            {t!(i18n, training::assistant_off)}
                        </option>
                        <option
                            value="sam"
                            disabled=move || !sam_available()
                            title=move || if sam_available() {
                                String::new()
                            } else {
                                sam_unavailable_msg()
                            }
                            prop:selected=move || assistant.get() == Assistant::Sam
                        >
                            {t!(i18n, training::assistant_sam)}
                        </option>
                        {move || {
                            let models = label_models.get();
                            (!models.is_empty()).then(|| view! {
                                <optgroup label=t_string!(i18n, training::assistant_model_group)>
                                    {models.into_iter().map(|name| {
                                        let value = format!("model:{name}");
                                        let this = name.clone();
                                        let is_selected = move || {
                                            assistant.get() == Assistant::Model(this.clone())
                                        };
                                        view! {
                                            <option value=value prop:selected=is_selected>
                                                {model_stem(&name).to_string()}
                                            </option>
                                        }
                                    }).collect::<Vec<_>>()}
                                </optgroup>
                            })
                        }}
                    </select>
                    // Always mounted: hidden while idle so its slot (and the
                    // row's wrapping) does not change when loading starts.
                    <span
                        class="ui-help text-xs"
                        class:ui-label-toolbar-hint-idle=move || !ai_busy.get()
                        aria-hidden=move || (!ai_busy.get()).to_string()
                    >
                        {t_string!(i18n, training::ai_loading)}
                    </span>
                </label>
            </div>
        </div>
    }
}
