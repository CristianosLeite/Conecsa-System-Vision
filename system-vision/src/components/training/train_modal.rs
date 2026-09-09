//! Leptos UI components for the web frontend.

use leptos::prelude::*;

use crate::api::model_stem;
use crate::i18n::*;

/// Training launch form: mandatory model name, the base model (stock weights
/// or an existing device model's last `best.pt`) and epochs/batch/patience.
/// `on_start` receives `(model_name, epochs, batch, patience, base_model)`
/// where `base_model` is a model-list name or empty for the stock weights.
#[component]
pub(super) fn TrainModal(
    visible: ReadSignal<bool>,
    set_visible: WriteSignal<bool>,
    /// Device models with a training checkpoint (fine-tune candidates).
    base_models: ReadSignal<Vec<String>>,
    /// Base preselected when the modal opens (the labeling model, if any).
    default_base_model: Signal<String>,
    on_start: Callback<(String, u32, u32, u32, String)>,
) -> impl IntoView {
    let i18n = use_i18n();
    let (name, set_name) = signal(String::new());
    let (epochs, set_epochs) = signal(50u32);
    let (batch, set_batch) = signal(4u32);
    let (patience, set_patience) = signal(50u32);
    let (base_model, set_base_model) = signal(String::new());

    // On open, preselect the labeling model as the base and, when the name is
    // still blank, seed it with that model's name — retraining under the same
    // name replaces the model's .pt/.engine, so "improve X" is two clicks.
    Effect::new(move |_| {
        if !visible.get() {
            return;
        }
        let base = default_base_model.get_untracked();
        set_base_model.set(base.clone());
        if !base.is_empty() && name.get_untracked().trim().is_empty() {
            set_name.set(model_stem(&base).to_string());
        }
    });

    let name_valid = move || {
        let n = name.get();
        let n = n.trim();
        !n.is_empty()
            && n.len() <= 64
            && n.chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
    };

    let start = move |_| {
        if !name_valid() {
            return;
        }
        on_start.run((
            name.get_untracked().trim().to_string(),
            epochs.get_untracked().clamp(1, 1000),
            batch.get_untracked().clamp(1, 32),
            patience.get_untracked().clamp(0, 1000),
            base_model.get_untracked(),
        ));
    };

    view! {
        {move || if visible.get() {
            view! {
                <div class="ui-modal-backdrop">
                    <div class="ui-card ui-card-pad ui-modal ui-modal-sm">
                        <h3 class="ui-card-title mb-4">
                            {t_string!(i18n, training::train_model_title)}
                        </h3>

                        <label class="ui-label-xs block mb-1">
                            {t_string!(i18n, training::model_name_required)}
                        </label>
                        <input
                            type="text"
                            class="ui-input mb-1"
                            placeholder=t_string!(i18n, training::model_name_placeholder)
                            prop:value=move || name.get()
                            on:input=move |ev| set_name.set(event_target_value(&ev))
                        />
                        <p class="ui-help-xs mb-4">
                            {t_string!(i18n, training::model_name_help)}
                        </p>

                        <label class="ui-label-xs block mb-1">
                            {t_string!(i18n, training::base_model)}
                        </label>
                        <select
                            class="ui-select mb-1 w-full cursor-pointer"
                            on:change=move |ev| {
                                let v = event_target_value(&ev);
                                if !v.is_empty() && name.get_untracked().trim().is_empty() {
                                    set_name.set(model_stem(&v).to_string());
                                }
                                set_base_model.set(v);
                            }
                        >
                            <option value="" prop:selected=move || base_model.get().is_empty()>
                                {t_string!(i18n, training::base_model_default)}
                            </option>
                            {move || base_models.get().into_iter().map(|m| {
                                let this = m.clone();
                                let is_selected = move || base_model.get() == this;
                                view! {
                                    <option value=m.clone() prop:selected=is_selected>
                                        {model_stem(&m).to_string()}
                                    </option>
                                }
                            }).collect::<Vec<_>>()}
                        </select>
                        <p class="ui-help-xs mb-4">
                            {t_string!(i18n, training::base_model_help)}
                        </p>

                        <div class="grid grid-cols-2 gap-3 mb-3">
                            <div>
                                <label class="ui-label-xs block mb-1">
                                    {t_string!(i18n, training::epochs)}
                                </label>
                                <input
                                    type="number"
                                    min="1" max="1000"
                                    class="ui-input"
                                    prop:value=move || epochs.get().to_string()
                                    on:input=move |ev| {
                                        if let Ok(v) = event_target_value(&ev).parse::<u32>() {
                                            set_epochs.set(v);
                                        }
                                    }
                                />
                            </div>
                            <div>
                                <label class="ui-label-xs block mb-1">
                                    {t_string!(i18n, training::batch_size)}
                                </label>
                                <input
                                    type="number"
                                    min="1" max="32"
                                    class="ui-input"
                                    prop:value=move || batch.get().to_string()
                                    on:input=move |ev| {
                                        if let Ok(v) = event_target_value(&ev).parse::<u32>() {
                                            set_batch.set(v);
                                        }
                                    }
                                />
                            </div>
                        </div>

                        <label class="ui-label-xs block mb-1">
                            {t_string!(i18n, training::patience)}
                        </label>
                        <input
                            type="number"
                            min="0" max="1000"
                            class="ui-input mb-1"
                            prop:value=move || patience.get().to_string()
                            on:input=move |ev| {
                                if let Ok(v) = event_target_value(&ev).parse::<u32>() {
                                    set_patience.set(v);
                                }
                            }
                        />
                        <p class="ui-help-xs mb-6">
                            {t_string!(i18n, training::patience_help)}
                        </p>

                        <div class="flex justify-end gap-2">
                            <button
                                class="ui-button ui-button-neutral ui-button-md"
                                on:click=move |_| set_visible.set(false)
                            >
                                {t_string!(i18n, common::cancel)}
                            </button>
                            <button
                                class="ui-button ui-button-success ui-button-md"
                                disabled=move || !name_valid()
                                on:click=start
                            >
                                {t_string!(i18n, training::start_training)}
                            </button>
                        </div>
                    </div>
                </div>
            }.into_any()
        } else {
            view! { <div/> }.into_any()
        }}
    }
}
