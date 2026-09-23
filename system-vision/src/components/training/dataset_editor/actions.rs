// SPDX-FileCopyrightText: 2026 Conecsa
//
// SPDX-License-Identifier: AGPL-3.0-only

//! Non-AI editor actions: image selection and autosave, capture / delete /
//! cover, replication, classes, training controls and leaving the editor.
//!
//! Each `fn` builds one `Callback` over the `Copy` [`EditorState`]; explicit
//! dependencies (another callback, the parent's `on_back`) are parameters so
//! the wiring is visible in `DatasetEditor`.

use leptos::prelude::*;
use leptos::task::spawn_local;

use crate::api::{self, LabelBox, LabelPolygon};
use crate::i18n::*;
use crate::models::Task;

use super::logic;
use super::state::{EditorState, I18n};

// ── shared async steps ────────────────────────────────────────────────────────

/// Write an image's labels in the dataset's kind: `boxes` for detection,
/// `polygons` for segmentation, the `image_class` for classification and for
/// face recognition (the class is the person).
pub(super) async fn write_labels(
    st: EditorState,
    ds: &str,
    image_id: &str,
    boxes: &[LabelBox],
    polygons: &[LabelPolygon],
    image_class: Option<u32>,
) -> Result<(), String> {
    match st.task {
        Task::Classify | Task::Face => {
            api::set_training_image_class(ds, image_id, image_class).await?;
        }
        Task::Segment => {
            api::set_training_polygons(ds, image_id, polygons).await?;
        }
        Task::Detect => {
            api::set_training_labels(ds, image_id, boxes).await?;
        }
    }
    Ok(())
}

/// Re-read the image list (labeled flags, counts). Silent on failure; callers
/// that want the toast use [`reload_images_or_report`].
pub(super) async fn reload_images(st: EditorState, ds: &str) -> Result<(), String> {
    let r = api::list_training_images(ds).await?;
    let _ = st.images.list.try_set(r.images);
    Ok(())
}

/// [`reload_images`] with the `failed_load_images` toast on error.
pub(super) async fn reload_images_or_report(st: EditorState, locale: Locale, ds: &str) {
    if let Err(e) = reload_images(st, ds).await {
        st.notices
            .error(td_string!(locale, training::failed_load_images, err = e));
    }
}

/// Spawned image-list refresh for synchronous (mount-time) callers.
pub(super) fn refresh_images(st: EditorState, i18n: I18n) {
    let locale = i18n.get_locale_untracked();
    spawn_local(async move {
        let ds = st.dataset_id.get_value();
        reload_images_or_report(st, locale, &ds).await;
    });
}

/// Persist `boxes` (or, for a segmentation / classification dataset, the open
/// image's polygons / class) as `image_id`'s labels, then refresh the gallery
/// counts. `notify` shows the "labels saved" toast on success.
pub(super) async fn persist_labels(
    st: EditorState,
    locale: Locale,
    ds: &str,
    image_id: &str,
    boxes: &[LabelBox],
    notify: bool,
) {
    let image_class = st.images.image_class.try_get_untracked().flatten();
    let polygons = st.images.polygons.try_get_untracked().unwrap_or_default();
    match write_labels(st, ds, image_id, boxes, &polygons, image_class).await {
        Ok(_) => {
            if notify {
                st.notices
                    .success(td_string!(locale, training::labels_saved).to_string());
            }
            let _ = reload_images(st, ds).await;
        }
        Err(e) => st
            .notices
            .error(td_string!(locale, training::failed_save_labels, err = e)),
    }
}

// ── image selection (autosaves the previous image's labels) ──────────────────

pub(super) fn select_image(st: EditorState, i18n: I18n) -> Callback<String> {
    Callback::new(move |id: String| {
        // Also reached from `capture` after its await, so the editor may
        // already be unmounted: every access is disposal-safe and bails.
        let Some(prev) = st.images.selected.try_get_untracked() else {
            return;
        };
        if prev.as_deref() == Some(id.as_str()) {
            return;
        }
        let Some(prev_boxes) = st.images.boxes.try_get_untracked() else {
            return;
        };
        let Some(prev_polygons) = st.images.polygons.try_get_untracked() else {
            return;
        };
        let Some(prev_class) = st.images.image_class.try_get_untracked() else {
            return;
        };
        st.ai.clear_suggestions();
        let _ = st.images.selected.try_set(Some(id.clone()));
        let locale = i18n.get_locale_untracked();
        spawn_local(async move {
            let Some(ds) = st.dataset_id.try_get_value() else {
                return;
            };
            if let Some(prev_id) = prev {
                if let Err(e) =
                    write_labels(st, &ds, &prev_id, &prev_boxes, &prev_polygons, prev_class).await
                {
                    st.notices
                        .error(td_string!(locale, training::failed_save_labels, err = e));
                }
            }
            match api::get_training_labels(&ds, &id).await {
                Ok(r) => {
                    let _ = st.images.boxes.try_set(r.boxes);
                    let _ = st.images.polygons.try_set(r.polygons);
                    let _ = st.images.image_class.try_set(r.image_class);
                }
                Err(e) => {
                    st.notices
                        .error(td_string!(locale, training::failed_load_labels, err = e))
                }
            }
            let _ = reload_images(st, &ds).await;
        });
    })
}

/// Set (or, with `None`, clear) the open image's class and save it at once:
/// in a classification dataset one pick is one completed labeling gesture.
pub(super) fn pick_class(st: EditorState, i18n: I18n) -> Callback<Option<u32>> {
    Callback::new(move |class_id: Option<u32>| {
        // Also reached after an await (accepting a suggestion): disposal-safe.
        let Some(Some(id)) = st.images.selected.try_get_untracked() else {
            st.notices
                .error(t_string!(i18n, training::select_image_first).to_string());
            return;
        };
        let _ = st.images.image_class.try_set(class_id);
        st.ai.clear_suggestions();
        let locale = i18n.get_locale_untracked();
        spawn_local(async move {
            let Some(ds) = st.dataset_id.try_get_value() else {
                return;
            };
            persist_labels(st, locale, &ds, &id, &[], false).await;
        });
    })
}

/// Persist the open image's boxes (`true` = show a toast).
pub(super) fn save_labels(st: EditorState, i18n: I18n) -> Callback<bool> {
    Callback::new(move |notify: bool| {
        let Some(id) = st.images.selected.get_untracked() else {
            return;
        };
        let bs = st.images.boxes.get_untracked();
        let locale = i18n.get_locale_untracked();
        spawn_local(async move {
            let ds = st.dataset_id.get_value();
            persist_labels(st, locale, &ds, &id, &bs, notify).await;
        });
    })
}

// ── capture / delete / cover ──────────────────────────────────────────────────

pub(super) fn capture(st: EditorState, i18n: I18n, select_image: Callback<String>) -> Callback<()> {
    Callback::new(move |_: ()| {
        if st.images.capturing.get_untracked() {
            return;
        }
        st.images.capturing.set(true);
        let locale = i18n.get_locale_untracked();
        spawn_local(async move {
            let ds = st.dataset_id.get_value();
            match api::capture_training_image(&ds).await {
                // Open the fresh capture straight into the editor; `select_image`
                // autosaves the previous image and refreshes the gallery for us.
                Ok(info) => select_image.run(info.image_id),
                Err(e) => st
                    .notices
                    .error(td_string!(locale, training::capture_failed, err = e)),
            }
            let _ = st.images.capturing.try_set(false);
        });
    })
}

pub(super) fn delete_image(st: EditorState, i18n: I18n) -> Callback<String> {
    Callback::new(move |id: String| {
        let locale = i18n.get_locale_untracked();
        spawn_local(async move {
            let ds = st.dataset_id.get_value();
            match api::delete_training_image(&ds, &id).await {
                Ok(_) => {
                    let still_selected =
                        st.images.selected.try_get_untracked().flatten().as_deref()
                            == Some(id.as_str());
                    if still_selected {
                        let _ = st.images.selected.try_set(None);
                        let _ = st.images.boxes.try_set(Vec::new());
                        let _ = st.images.polygons.try_set(Vec::new());
                        st.ai.clear_suggestions();
                    }
                    reload_images_or_report(st, locale, &ds).await;
                }
                Err(e) => {
                    st.notices
                        .error(td_string!(locale, training::failed_delete_image, err = e))
                }
            }
        });
    })
}

pub(super) fn set_cover(st: EditorState, i18n: I18n) -> Callback<String> {
    Callback::new(move |id: String| {
        let locale = i18n.get_locale_untracked();
        spawn_local(async move {
            let ds = st.dataset_id.get_value();
            match api::set_dataset_cover(&ds, &id).await {
                Ok(_) => {
                    let _ = st.images.cover_image_id.try_set(id);
                    st.notices
                        .success(td_string!(locale, training::cover_image_set).to_string());
                }
                Err(e) => st
                    .notices
                    .error(td_string!(locale, training::failed_set_cover, err = e)),
            }
        });
    })
}

// ── replicate ─────────────────────────────────────────────────────────────────

/// Open the Replicate modal for `id`.
pub(super) fn replicate(st: EditorState) -> Callback<String> {
    Callback::new(move |id: String| {
        st.replicate.target.set(Some(id));
        st.replicate.count.set(5);
        st.replicate.show_modal.set(true);
    })
}

pub(super) fn replicate_confirm(st: EditorState, i18n: I18n) -> Callback<()> {
    Callback::new(move |_: ()| {
        let Some(id) = st.replicate.target.get_untracked() else {
            return;
        };
        if st.replicate.busy.get_untracked() {
            return;
        }
        let count = st.replicate.count.get_untracked().clamp(1, 50);
        st.replicate.busy.set(true);
        let locale = i18n.get_locale_untracked();
        spawn_local(async move {
            let ds = st.dataset_id.get_value();
            match api::replicate_training_image(&ds, &id, count).await {
                Ok(_) => {
                    let _ = st.replicate.show_modal.try_set(false);
                    st.notices.success(if count == 1 {
                        td_string!(locale, training::replicas_created_one).to_string()
                    } else {
                        td_string!(locale, training::replicas_created, count = count)
                    });
                    reload_images_or_report(st, locale, &ds).await;
                }
                Err(e) => st.notices.error(td_string!(
                    locale,
                    training::failed_replicate_image,
                    err = e
                )),
            }
            let _ = st.replicate.busy.try_set(false);
        });
    })
}

// ── classes ───────────────────────────────────────────────────────────────────

pub(super) fn class_add(st: EditorState, i18n: I18n) -> Callback<String> {
    Callback::new(move |name: String| {
        let locale = i18n.get_locale_untracked();
        spawn_local(async move {
            let ds = st.dataset_id.get_value();
            match api::add_training_class(&ds, &name).await {
                Ok(r) => {
                    let _ = st.classes.active.try_set(r.classes.len().saturating_sub(1));
                    let _ = st.classes.list.try_set(r.classes);
                }
                Err(e) => st
                    .notices
                    .error(td_string!(locale, training::failed_add_class, err = e)),
            }
        });
    })
}

pub(super) fn class_rename(st: EditorState, i18n: I18n) -> Callback<(usize, String)> {
    Callback::new(move |(index, name): (usize, String)| {
        let locale = i18n.get_locale_untracked();
        spawn_local(async move {
            let ds = st.dataset_id.get_value();
            match api::rename_training_class(&ds, index, &name).await {
                Ok(r) => {
                    let _ = st.classes.list.try_set(r.classes);
                }
                Err(e) => {
                    st.notices
                        .error(td_string!(locale, training::failed_rename_class, err = e))
                }
            }
        });
    })
}

pub(super) fn class_remove(st: EditorState, i18n: I18n) -> Callback<usize> {
    Callback::new(move |index: usize| {
        let locale = i18n.get_locale_untracked();
        spawn_local(async move {
            let ds = st.dataset_id.get_value();
            match api::remove_training_class(&ds, index).await {
                Ok(r) => {
                    let len = r.classes.len();
                    let _ = st.classes.list.try_set(r.classes);
                    let _ = st
                        .classes
                        .active
                        .try_update(|c| *c = logic::active_after_remove(*c, index, len));
                    // Labels were reindexed server-side; reload the open image.
                    if let Some(id) = st.images.selected.try_get_untracked().flatten() {
                        if let Ok(r) = api::get_training_labels(&ds, &id).await {
                            let _ = st.images.boxes.try_set(r.boxes);
                            let _ = st.images.polygons.try_set(r.polygons);
                            let _ = st.images.image_class.try_set(r.image_class);
                        }
                    }
                    reload_images_or_report(st, locale, &ds).await;
                }
                Err(e) => {
                    st.notices
                        .error(td_string!(locale, training::failed_remove_class, err = e))
                }
            }
        });
    })
}

// ── training ──────────────────────────────────────────────────────────────────

pub(super) fn train_request(st: EditorState, save_labels: Callback<bool>) -> Callback<()> {
    Callback::new(move |_: ()| {
        // Persist the open image's labels before the gate is evaluated.
        save_labels.run(false);
        st.training.show_modal.set(true);
    })
}

/// `(name, epochs, batch, patience, base_model)` from the Train modal. A face
/// dataset sends the defaults with them: the device builds a gallery, and the
/// training-service ignores the YOLO parameters and the base model.
pub(super) fn train_start(
    st: EditorState,
    i18n: I18n,
) -> Callback<(String, u32, u32, u32, String)> {
    Callback::new(
        move |(name, epochs, batch, patience, base_model): (String, u32, u32, u32, String)| {
            let locale = i18n.get_locale_untracked();
            spawn_local(async move {
                let ds = st.dataset_id.get_value();
                match api::start_training(&ds, &name, epochs, batch, patience, &base_model).await {
                    Ok(j) => {
                        let _ = st.training.job.try_set(Some(j));
                        let _ = st.training.show_modal.try_set(false);
                    }
                    Err(e) => st.notices.error(td_string!(
                        locale,
                        training::failed_start_training,
                        err = e
                    )),
                }
            });
        },
    )
}

pub(super) fn train_cancel(st: EditorState, i18n: I18n) -> Callback<()> {
    Callback::new(move |_: ()| {
        let locale = i18n.get_locale_untracked();
        spawn_local(async move {
            if let Err(e) = api::cancel_training().await {
                st.notices.error(td_string!(
                    locale,
                    training::failed_cancel_training,
                    err = e
                ));
            }
        });
    })
}

pub(super) fn train_finish(st: EditorState, i18n: I18n) -> Callback<()> {
    Callback::new(move |_: ()| {
        let locale = i18n.get_locale_untracked();
        spawn_local(async move {
            if let Err(e) = api::finish_training().await {
                st.notices.error(td_string!(
                    locale,
                    training::failed_finish_training,
                    err = e
                ));
            }
        });
    })
}

// ── back to the dataset gallery ───────────────────────────────────────────────

pub(super) fn back(st: EditorState, i18n: I18n, on_back: Callback<()>) -> Callback<()> {
    Callback::new(move |_: ()| {
        if st
            .training
            .job
            .get_untracked()
            .map(|j| j.is_active())
            .unwrap_or(false)
        {
            st.notices
                .error(t_string!(i18n, training::wait_training_finish).to_string());
            return;
        }
        let prev = st.images.selected.get_untracked();
        let prev_boxes = st.images.boxes.get_untracked();
        let prev_polygons = st.images.polygons.get_untracked();
        let prev_class = st.images.image_class.get_untracked();
        spawn_local(async move {
            let ds = st.dataset_id.get_value();
            // Autosave the open image's labels; stay in training mode (the
            // gallery is still part of the training page).
            if let Some(id) = prev {
                let _ = write_labels(st, &ds, &id, &prev_boxes, &prev_polygons, prev_class).await;
            }
            on_back.run(());
        });
    })
}

// ── gallery bundle ────────────────────────────────────────────────────────────

/// The gallery's per-thumbnail actions.
#[derive(Clone, Copy)]
pub(crate) struct GalleryActions {
    pub(crate) on_select: Callback<String>,
    pub(crate) on_delete: Callback<String>,
    pub(crate) on_set_cover: Callback<String>,
    pub(crate) on_replicate: Callback<String>,
}

impl GalleryActions {
    pub(super) fn new(st: EditorState, i18n: I18n) -> Self {
        Self {
            on_select: select_image(st, i18n),
            on_delete: delete_image(st, i18n),
            on_set_cover: set_cover(st, i18n),
            on_replicate: replicate(st),
        }
    }
}
