//! Leptos UI components for the web frontend.

use leptos::prelude::*;

use crate::api::{model_stem, LabelBox};
use crate::class_color::class_display_name;
use crate::i18n::*;

use super::dataset_editor::{AiState, Assistant, ClassesState};

/// Header row of the label editor: the card title, the selected-box controls
/// (reassign its class, delete it) and the AI-assist selector — off, SAM3
/// prompts, or one of the device's existing engines.
#[component]
pub(super) fn LabelToolbar(
    selected_box: ReadSignal<Option<usize>>,
    boxes: RwSignal<Vec<LabelBox>>,
    classes: ClassesState,
    ai: AiState,
    /// Reassign the selected box's class.
    on_set_class: Callback<u32>,
    /// Delete the selected box.
    on_delete: Callback<()>,
    on_assistant_change: Callback<Assistant>,
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

    view! {
        <div class="flex items-center justify-between gap-2 flex-wrap">
            <h2 class="ui-card-title">{t!(i18n, training::label_editor_title)}</h2>
            <div class="flex items-center gap-2 flex-wrap">
                {move || match selected_box.get() {
                    Some(idx) => {
                        let current = boxes.get().get(idx).map(|b| b.class_id).unwrap_or(0);
                        let class_list = classes.get();
                        view! {
                            <select
                                class="ui-select ui-select-sm w-auto cursor-pointer"
                                title=t_string!(i18n, training::reassign_box_class)
                                on:change=move |ev| {
                                    if let Ok(v) = event_target_value(&ev).parse::<u32>() {
                                        on_set_class.run(v);
                                    }
                                }
                            >
                                {class_list.iter().enumerate().map(|(ci, name)| {
                                    let selected = ci as u32 == current;
                                    view! {
                                        <option value={ci.to_string()} selected={selected}>
                                            {class_display_name(name)}
                                        </option>
                                    }
                                }).collect::<Vec<_>>()}
                            </select>
                            <button
                                class="ui-button ui-button-danger ui-button-xs"
                                on:click=move |_| on_delete.run(())
                                title=t_string!(i18n, training::delete_box_title)
                            >
                                {t_string!(i18n, training::delete_box)}
                            </button>
                        }.into_any()
                    }
                    None => view! { <span/> }.into_any(),
                }}
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
                    {move || ai_busy.get().then(|| view! {
                        <span class="ui-help text-xs">{t_string!(i18n, training::ai_loading)}</span>
                    })}
                </label>
            </div>
        </div>
    }
}
