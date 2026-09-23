// SPDX-FileCopyrightText: 2026 Conecsa
//
// SPDX-License-Identifier: AGPL-3.0-only

use leptos::prelude::*;

use crate::api::model_stem;
use crate::class_color::class_display_name;
use crate::i18n::*;

use super::dataset_editor::{AiState, Assistant, ClassesState, DrawMode};

/// Header row of the label editor: the card title, the selected-label
/// controls (reassign its class, delete it — and, for a polygon, delete the
/// selected vertex), the draw-mode switch of a segmentation dataset and the
/// AI-assist selector — off, SAM3 prompts, or one of the device's existing
/// engines.
#[allow(clippy::too_many_arguments)]
#[component]
pub(super) fn LabelToolbar(
    /// The class of the selected box or polygon (`None`: nothing selected).
    selected_class: Signal<Option<u32>>,
    classes: ClassesState,
    ai: AiState,
    /// Reassign the selected label's class.
    on_set_class: Callback<u32>,
    /// Delete the selected label.
    on_delete: Callback<()>,
    on_assistant_change: Callback<Assistant>,
    /// A segmentation dataset: the draw-mode select and vertex deletion.
    segment: bool,
    /// How new polygons are drawn (segmentation).
    draw_mode: RwSignal<DrawMode>,
    /// A vertex of the selected polygon is selected.
    vertex_selected: Signal<bool>,
    /// Delete the selected vertex.
    on_delete_vertex: Callback<()>,
) -> impl IntoView {
    let i18n = use_i18n();
    let classes = classes.list;
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
    // The hint of the active draw mode, shown as the select's tooltip.
    let mode_title = move || match draw_mode.get() {
        DrawMode::Click => t_string!(i18n, training::draw_mode_click_title),
        DrawMode::Freehand => t_string!(i18n, training::draw_mode_freehand_title),
        DrawMode::Rect => t_string!(i18n, training::draw_mode_rect_title),
    };

    view! {
        <div class="ui-label-toolbar">
            <h2 class="ui-card-title">{t!(i18n, training::label_editor_title)}</h2>
            <div class="ui-label-toolbar-group">
                // The selected-label controls are always present (disabled
                // with nothing selected) at a fixed width, so selecting a label
                // never re-wraps the toolbar and pushes the canvas down; only a
                // viewport resize changes its height.
                {move || {
                    let current = selected_class.get();
                    let has_selection = current.is_some();
                    let class_list = classes.get();
                    // A segmentation label is presented as a box like any
                    // other label; only its drawing tools differ.
                    let reassign_title = t_string!(i18n, training::reassign_box_class).to_string();
                    let delete_label = t_string!(i18n, training::delete_box).to_string();
                    let delete_title = t_string!(i18n, training::delete_box_title).to_string();
                    view! {
                        <select
                            class="ui-select ui-select-sm ui-label-toolbar-class cursor-pointer"
                            title=reassign_title
                            disabled=!has_selection
                            on:change=move |ev| {
                                if let Ok(v) = event_target_value(&ev).parse::<u32>() {
                                    on_set_class.run(v);
                                }
                            }
                        >
                            {(!has_selection).then(|| view! {
                                <option value="" selected=true>
                                    {t_string!(i18n, training::no_label_selected)}
                                </option>
                            })}
                            {class_list.iter().enumerate().map(|(ci, name)| {
                                let selected = current == Some(ci as u32);
                                view! {
                                    <option value={ci.to_string()} selected={selected}>
                                        {class_display_name(name)}
                                    </option>
                                }
                            }).collect::<Vec<_>>()}
                        </select>
                        {segment.then(|| view! {
                            <button
                                class="ui-button ui-button-neutral ui-button-xs"
                                disabled=move || !(has_selection && vertex_selected.get())
                                on:click=move |_| on_delete_vertex.run(())
                                title=t_string!(i18n, training::delete_vertex_title)
                            >
                                {t_string!(i18n, training::delete_vertex)}
                            </button>
                        })}
                        <button
                            class="ui-button ui-button-danger ui-button-xs"
                            disabled=!has_selection
                            on:click=move |_| on_delete.run(())
                            title=delete_title
                        >
                            {delete_label}
                        </button>
                    }
                }}
                {segment.then(|| view! {
                    <label class="flex items-center gap-2">
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
                })}
                <label class="flex items-center gap-2">
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
                    // toolbar's wrapping) does not change when loading starts.
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
