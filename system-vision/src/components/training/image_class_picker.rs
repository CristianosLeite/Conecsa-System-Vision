// SPDX-FileCopyrightText: 2026 Conecsa
//
// SPDX-License-Identifier: AGPL-3.0-only

//! Image-class labeling (classification and face recognition).
//!
//! Such a dataset labels a whole image with one class, so there is
//! nothing to draw: the card shows the open image and one button per class.
//! Click a class — or press its number key — to label the image; pick it again
//! (or press 0) to clear. Each pick is saved at once (one gesture, one save).
//! A device classification engine can suggest the class; a face dataset has no
//! assistant (its classes are the enrolled people, and their photos and the
//! gallery stay on the device).

use leptos::prelude::*;

use crate::api::{model_stem, training_image_url};
use crate::class_color::{class_color_for, class_display_name};
use crate::i18n::*;

use super::dataset_editor::{AiState, Assistant, ClassesState, ImagesState, LabelActions};

/// The open image and its class buttons, plus the model assistant.
#[component]
pub(super) fn ImageClassPicker(
    dataset_id: String,
    images: ImagesState,
    classes: ClassesState,
    ai: AiState,
    actions: LabelActions,
    /// Set (or, with `None`, clear) the open image's class and save it.
    on_pick: Callback<Option<u32>>,
    /// Accept the model's suggested class.
    on_accept_class: Callback<()>,
    /// A face dataset: the classes are people and there is no AI assistant.
    #[prop(optional)]
    face: bool,
) -> impl IntoView {
    let i18n = use_i18n();
    let dataset_id = StoredValue::new(dataset_id);
    let selected = images.selected;
    let image_class = images.image_class;
    let class_list = classes.list;
    let assistant = ai.assistant;
    let busy = ai.busy;
    let suggestion = ai.class_suggestion;
    let label_models = ai.label_models;

    // Picking the image's current class again clears it.
    let toggle = move |index: u32| {
        let current = image_class.try_get_untracked().flatten();
        on_pick.run(if current == Some(index) { None } else { Some(index) });
    };

    // Number keys: 1–9 pick the first nine classes, 0 clears. Ignored while a
    // form field has focus (class names are typed on this page).
    let key_handle = window_event_listener(leptos::ev::keydown, move |ev: web_sys::KeyboardEvent| {
        if ev.ctrl_key() || ev.meta_key() || ev.alt_key() {
            return;
        }
        let focused_tag = web_sys::window()
            .and_then(|w| w.document())
            .and_then(|d| d.active_element())
            .map(|el| el.tag_name().to_uppercase());
        if matches!(focused_tag.as_deref(), Some("INPUT" | "TEXTAREA" | "SELECT")) {
            return;
        }
        // try_* throughout: a global listener can briefly outlive the signals.
        let Some(Some(_)) = selected.try_get_untracked() else {
            return;
        };
        let Some(len) = class_list.try_with_untracked(|c| c.len()) else {
            return;
        };
        let key = ev.key();
        if key == "0" {
            ev.prevent_default();
            on_pick.run(None);
        } else if let Ok(n @ 1..=9) = key.parse::<usize>() {
            if n <= len {
                ev.prevent_default();
                toggle((n - 1) as u32);
            }
        }
    });
    on_cleanup(move || key_handle.remove());

    let on_model_change = move |ev| {
        let value = event_target_value(&ev);
        actions.on_assistant_change.run(if value.is_empty() {
            Assistant::Off
        } else {
            Assistant::Model(value)
        });
    };
    let model_ready = move || matches!(assistant.get(), Assistant::Model(_)) && !busy.get();

    view! {
        <div class="ui-card ui-card-pad-sm flex flex-col gap-3">
            <div class="flex items-center justify-between gap-2 flex-wrap">
                <h2 class="ui-card-title">
                    {move || if face {
                        t_string!(i18n, training::class_picker_title_face)
                    } else {
                        t_string!(i18n, training::class_picker_title)
                    }}
                </h2>
                {(!face).then(|| view! {
                <div class="flex items-center gap-2 flex-wrap">
                    <select
                        class="ui-select ui-select-sm w-auto cursor-pointer"
                        aria-label=move || t_string!(i18n, training::assistant_label)
                        title=move || t_string!(i18n, training::assistant_title)
                        disabled=move || busy.get()
                        on:change=on_model_change
                    >
                        <option value="" selected=move || assistant.get() == Assistant::Off>
                            {t!(i18n, training::assistant_off)}
                        </option>
                        {move || label_models.get().into_iter().map(|name| {
                            let value = name.clone();
                            let stem = model_stem(&name).to_string();
                            let is_selected = move || assistant.get() == Assistant::Model(name.clone());
                            view! { <option value=value selected=is_selected>{stem}</option> }
                        }).collect_view()}
                    </select>
                    <button
                        class="ui-button ui-button-primary ui-button-xs"
                        disabled=move || !model_ready() || selected.get().is_none()
                        on:click=move |_| actions.on_model_detect.run(())
                    >
                        {move || if busy.get() {
                            t_string!(i18n, training::suggesting)
                        } else {
                            t_string!(i18n, training::suggest_class)
                        }}
                    </button>
                </div>
                })}
            </div>

            {(!face).then(|| view! {
            {move || suggestion.get().map(|s| view! {
                <div class="ui-class-picker-suggestion">
                    <span>
                        {t_string!(
                            i18n,
                            training::suggested_class,
                            name = s.class_name.clone(),
                            score = format!("{:.0}", (s.score.clamp(0.0, 1.0) * 100.0).round())
                        )}
                    </span>
                    <button
                        class="ui-button ui-button-success ui-button-xs"
                        on:click=move |_| on_accept_class.run(())
                    >
                        {t!(i18n, training::accept_class)}
                    </button>
                </div>
            })}
            })}

            <p class="ui-help">
                {move || if face {
                    t_string!(i18n, training::class_picker_hint_face)
                } else {
                    t_string!(i18n, training::class_picker_hint)
                }}
            </p>

            {move || match selected.get() {
                None => view! {
                    <p class="ui-help italic">{t!(i18n, training::select_image_hint)}</p>
                }.into_any(),
                Some(id) => view! {
                    <img
                        class="ui-class-picker-image"
                        src=training_image_url(&dataset_id.get_value(), &id)
                        alt=move || t_string!(i18n, training::labeling_image_alt)
                    />
                }.into_any(),
            }}

            <div
                class="ui-class-picker-classes"
                role="group"
                aria-label=move || t_string!(i18n, training::class_picker_title)
            >
                {move || {
                    let list = class_list.get();
                    if list.is_empty() {
                        return view! {
                            <p class="ui-help italic">
                                {move || if face {
                                    t_string!(i18n, training::create_person_before_picking)
                                } else {
                                    t_string!(i18n, training::create_class_before_picking)
                                }}
                            </p>
                        }.into_any();
                    }
                    list.iter().enumerate().map(|(i, raw)| {
                        let index = i as u32;
                        let name = class_display_name(raw);
                        let swatch = format!("background-color: {}", class_color_for(i, &list));
                        let picked = move || image_class.get() == Some(index);
                        view! {
                            <button
                                type="button"
                                class=move || if picked() {
                                    "ui-class-picker-class is-picked"
                                } else {
                                    "ui-class-picker-class"
                                }
                                aria-pressed=move || picked().to_string()
                                disabled=move || selected.get().is_none()
                                on:click=move |_| toggle(index)
                            >
                                <span class="ui-class-picker-swatch" style=swatch aria-hidden="true"></span>
                                {(i < 9).then(|| view! {
                                    <kbd class="ui-class-picker-key">{(i + 1).to_string()}</kbd>
                                })}
                                <span class="ui-class-picker-name">{name}</span>
                            </button>
                        }
                    }).collect_view().into_any()
                }}
            </div>

            <div class="ui-class-picker-status">
                {move || if image_class.get().is_none() && selected.get().is_some() {
                    view! {
                        <span class="ui-help">
                            {move || if face {
                                t_string!(i18n, training::no_person_selected)
                            } else {
                                t_string!(i18n, training::no_class_selected)
                            }}
                        </span>
                    }.into_any()
                } else {
                    view! { <span/> }.into_any()
                }}
                <button
                    class="ui-button ui-button-neutral ui-button-xs"
                    disabled=move || image_class.get().is_none()
                    on:click=move |_| on_pick.run(None)
                >
                    {move || if face {
                        t_string!(i18n, training::clear_person)
                    } else {
                        t_string!(i18n, training::clear_class)
                    }}
                </button>
            </div>
        </div>
    }
}
