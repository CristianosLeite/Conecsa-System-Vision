//! AI-assisted labeling actions: the assistant selector (SAM3 prompts or an
//! existing device engine), Suggest / Detect, and accepting suggestions into
//! the open image's labels.

use leptos::prelude::*;
use leptos::task::spawn_local;

use crate::api;
use crate::i18n::*;

use super::actions::persist_labels;
use super::logic::{self, ClassSlot, SamTarget};
use super::state::{AiState, Assistant, EditorState, I18n};

// ── shared async steps ────────────────────────────────────────────────────────

/// Re-read both assistant statuses (the gateway unloads one when the other
/// loads, so both can change on any load).
pub(super) async fn refresh_assistants(ai: AiState) {
    if let Ok(s) = api::get_sam_status().await {
        let _ = ai.sam_status.try_set(Some(s));
    }
    if let Ok(s) = api::get_label_model_status().await {
        let _ = ai.model_status.try_set(Some(s));
    }
}

/// Tail of every load/unload: refresh both statuses and release `busy`.
async fn finish_assistant_switch(ai: AiState) {
    refresh_assistants(ai).await;
    let _ = ai.busy.try_set(false);
}

/// Resolve `name` against `class_list`, creating the class when missing (the
/// list and the state are updated from the backend's reply). On failure the
/// `failed_create_class` toast is shown and the caller must abort.
async fn ensure_class(
    st: EditorState,
    locale: Locale,
    ds: &str,
    class_list: &mut Vec<String>,
    name: &str,
) -> Result<ClassSlot, ()> {
    if let Some(idx) = logic::resolve_class(class_list, name) {
        return Ok(ClassSlot::Existing(idx));
    }
    match api::add_training_class(ds, name).await {
        Ok(r) => {
            *class_list = r.classes;
            let _ = st.classes.list.try_set(class_list.clone());
            Ok(ClassSlot::Created(
                logic::resolve_class(class_list, name)
                    .unwrap_or(class_list.len().saturating_sub(1)),
            ))
        }
        Err(e) => {
            st.notices.error(td_string!(
                locale,
                training::failed_create_class,
                name = name.to_string(),
                err = e
            ));
            Err(())
        }
    }
}

// ── assistant selector ────────────────────────────────────────────────────────

pub(super) fn assistant_change(st: EditorState, i18n: I18n) -> Callback<Assistant> {
    Callback::new(move |next: Assistant| {
        let ai = st.ai;
        if ai.busy.get_untracked() {
            return;
        }
        ai.clear_suggestions();
        let prev = ai.assistant.get_untracked();
        ai.assistant.set(next.clone());
        let locale = i18n.get_locale_untracked();
        match next {
            Assistant::Off => {
                // Free the private TensorRT worker right away; SAM keeps its
                // idle-unload timer (a re-toggle would otherwise pay the
                // ~1 min cold load again).
                if matches!(prev, Assistant::Model(_)) {
                    ai.busy.set(true);
                    spawn_local(async move {
                        if let Err(e) = api::unload_label_model().await {
                            leptos::logging::warn!("label-model unload failed: {e}");
                        }
                        finish_assistant_switch(ai).await;
                    });
                }
            }
            Assistant::Sam => {
                // Lazily load on first use.
                let loaded = ai
                    .sam_status
                    .get_untracked()
                    .map(|s| s.loaded)
                    .unwrap_or(false);
                if loaded {
                    return;
                }
                ai.busy.set(true);
                spawn_local(async move {
                    match api::load_sam().await {
                        Ok(_) => st
                            .notices
                            .success(td_string!(locale, training::sam_model_loaded).to_string()),
                        Err(e) => {
                            st.notices.error(td_string!(
                                locale,
                                training::failed_load_sam,
                                err = e
                            ));
                            let _ = ai.assistant.try_set(Assistant::Off);
                        }
                    }
                    finish_assistant_switch(ai).await;
                });
            }
            Assistant::Model(name) => {
                ai.busy.set(true);
                spawn_local(async move {
                    let stem = api::model_stem(&name).to_string();
                    match api::load_label_model(&name).await {
                        Ok(_) => st.notices.success(td_string!(
                            locale,
                            training::model_loaded,
                            name = stem
                        )),
                        Err(e) => {
                            st.notices.error(td_string!(
                                locale,
                                training::failed_load_model,
                                name = stem,
                                err = e
                            ));
                            let _ = ai.assistant.try_set(Assistant::Off);
                        }
                    }
                    finish_assistant_switch(ai).await;
                });
            }
        }
    })
}

// ── suggestions ───────────────────────────────────────────────────────────────

pub(super) fn sam_suggest(st: EditorState, i18n: I18n) -> Callback<()> {
    Callback::new(move |_: ()| {
        let ai = st.ai;
        let Some(id) = st.images.selected.get_untracked() else {
            st.notices
                .error(t_string!(i18n, training::select_image_first).to_string());
            return;
        };
        let text = ai.sam_text.get_untracked();
        let points = ai.sam_points.get_untracked();
        if text.trim().is_empty() && points.is_empty() {
            st.notices
                .error(t_string!(i18n, training::sam_prompt_needed).to_string());
            return;
        }
        if ai.busy.get_untracked() {
            return;
        }
        let threshold = ai.threshold.get_untracked();
        ai.busy.set(true);
        let locale = i18n.get_locale_untracked();
        spawn_local(async move {
            let ds = st.dataset_id.get_value();
            match api::sam_segment(&ds, &id, text.trim(), &points, threshold).await {
                Ok(r) if r.boxes.is_empty() => {
                    st.notices
                        .success(td_string!(locale, training::sam_no_objects).to_string());
                    let _ = ai.suggestions.try_set(Vec::new());
                }
                Ok(r) => {
                    let _ = ai.suggestions.try_set(r.boxes);
                }
                Err(e) => {
                    st.notices
                        .error(td_string!(locale, training::segmentation_failed, err = e))
                }
            }
            // SAM suggestions carry no class of their own.
            let _ = ai.suggestion_names.try_set(Vec::new());
            if let Ok(s) = api::get_sam_status().await {
                let _ = ai.sam_status.try_set(Some(s));
            }
            let _ = ai.busy.try_set(false);
        });
    })
}

pub(super) fn model_detect(st: EditorState, i18n: I18n) -> Callback<()> {
    Callback::new(move |_: ()| {
        let ai = st.ai;
        let Some(id) = st.images.selected.get_untracked() else {
            st.notices
                .error(t_string!(i18n, training::select_image_first).to_string());
            return;
        };
        let Assistant::Model(name) = ai.assistant.get_untracked() else {
            return;
        };
        if ai.busy.get_untracked() {
            return;
        }
        let threshold = ai.threshold.get_untracked();
        // Reload silently when the backend no longer reports this engine as
        // loaded (a runtime release unloads it).
        let needs_load = logic::model_needs_load(ai.model_status.get_untracked().as_ref(), &name);
        ai.busy.set(true);
        let locale = i18n.get_locale_untracked();
        spawn_local(async move {
            let ds = st.dataset_id.get_value();
            if needs_load && let Err(e) = api::load_label_model(&name).await {
                st.notices.error(td_string!(
                    locale,
                    training::failed_load_model,
                    name = api::model_stem(&name).to_string(),
                    err = e
                ));
                // No engine behind the selector: snap back to Off rather
                // than leaving a "Model" assistant that cannot detect.
                let _ = ai.assistant.try_set(Assistant::Off);
                finish_assistant_switch(ai).await;
                return;
            }
            match api::label_detect(&ds, &id, threshold).await {
                Ok(r) if r.boxes.is_empty() => {
                    st.notices
                        .success(td_string!(locale, training::model_no_objects).to_string());
                    let _ = ai.suggestions.try_set(Vec::new());
                    let _ = ai.suggestion_names.try_set(Vec::new());
                }
                Ok(r) => {
                    let _ = ai.suggestions.try_set(r.boxes);
                    let _ = ai.suggestion_names.try_set(r.class_names);
                }
                Err(e) => st
                    .notices
                    .error(td_string!(locale, training::detection_failed, err = e)),
            }
            if let Ok(s) = api::get_label_model_status().await {
                let _ = ai.model_status.try_set(Some(s));
            }
            let _ = ai.busy.try_set(false);
        });
    })
}

/// Accept the pending suggestions into the open image's boxes and save.
pub(super) fn accept(st: EditorState, i18n: I18n) -> Callback<()> {
    Callback::new(move |_: ()| {
        let ai = st.ai;
        let pending = ai.suggestions.get_untracked();
        if pending.is_empty() {
            return;
        }
        let Some(image_id) = st.images.selected.get_untracked() else {
            st.notices
                .error(t_string!(i18n, training::select_image_first).to_string());
            return;
        };
        let names = ai.suggestion_names.get_untracked();
        let from_model = matches!(ai.assistant.get_untracked(), Assistant::Model(_));
        // SAM: the text prompt IS the class label — accepting "bottle"
        // suggestions tags them as class "bottle" (creating it if needed),
        // NOT the manually-selected active class. Point-only suggestions (no
        // prompt) fall back to the active class.
        let prompt = ai.sam_text.get_untracked().trim().to_string();
        let active = st.classes.active.get_untracked();
        let locale = i18n.get_locale_untracked();
        spawn_local(async move {
            let ds = st.dataset_id.get_value();
            let mut class_list = st.classes.list.try_get_untracked().unwrap_or_default();
            let mut tagged = Vec::with_capacity(pending.len());
            if from_model {
                // Model suggestions carry the model's class names: resolve
                // each against the dataset, creating the missing ones.
                for (i, mut b) in pending.into_iter().enumerate() {
                    let Some(name) = logic::model_suggestion_name(&names, i) else {
                        continue;
                    };
                    let Ok(slot) = ensure_class(st, locale, &ds, &mut class_list, &name).await
                    else {
                        return;
                    };
                    b.class_id = slot.index() as u32;
                    tagged.push(b);
                }
            } else {
                let class_id = match logic::sam_accept_target(&prompt, &class_list, active) {
                    SamTarget::Existing(idx) | SamTarget::Active(idx) => idx as u32,
                    SamTarget::Create => {
                        let Ok(slot) =
                            ensure_class(st, locale, &ds, &mut class_list, &prompt).await
                        else {
                            return;
                        };
                        // A class created from the prompt becomes the active one.
                        if let ClassSlot::Created(idx) = slot {
                            let _ = st.classes.active.try_set(idx);
                        }
                        slot.index() as u32
                    }
                    SamTarget::NoClasses => {
                        st.notices.error(
                            td_string!(locale, training::create_class_before_accepting).to_string(),
                        );
                        return;
                    }
                };
                tagged.extend(pending.into_iter().map(|mut b| {
                    b.class_id = class_id;
                    b
                }));
            }

            let mut bs = st.images.boxes.try_get_untracked().unwrap_or_default();
            bs.extend(tagged);
            let _ = st.images.boxes.try_set(bs.clone());
            ai.clear_suggestions();
            persist_labels(st, locale, &ds, &image_id, &bs, false).await;
        });
    })
}

pub(super) fn clear(st: EditorState) -> Callback<()> {
    Callback::new(move |_: ()| st.ai.clear_suggestions())
}

/// Surfaced by the editor when the user tries to draw a box with no class.
pub(super) fn need_class(st: EditorState, i18n: I18n) -> Callback<()> {
    Callback::new(move |_: ()| {
        st.notices
            .error(t_string!(i18n, training::create_class_before_drawing).to_string());
    })
}

// ── label editor bundle ───────────────────────────────────────────────────────

/// The label editor's callbacks (toolbar, assistant panels and canvas).
#[derive(Clone, Copy)]
pub(crate) struct LabelActions {
    pub(crate) on_assistant_change: Callback<Assistant>,
    pub(crate) on_sam_suggest: Callback<()>,
    pub(crate) on_model_detect: Callback<()>,
    pub(crate) on_accept: Callback<()>,
    pub(crate) on_clear: Callback<()>,
    /// Persist the current boxes (true = show a toast).
    pub(crate) on_save: Callback<bool>,
    /// Fired when the user tries to draw a box with no class selected.
    pub(crate) on_need_class: Callback<()>,
}

impl LabelActions {
    pub(super) fn new(st: EditorState, i18n: I18n, on_save: Callback<bool>) -> Self {
        Self {
            on_assistant_change: assistant_change(st, i18n),
            on_sam_suggest: sam_suggest(st, i18n),
            on_model_detect: model_detect(st, i18n),
            on_accept: accept(st, i18n),
            on_clear: clear(st),
            on_save,
            on_need_class: need_class(st, i18n),
        }
    }
}
