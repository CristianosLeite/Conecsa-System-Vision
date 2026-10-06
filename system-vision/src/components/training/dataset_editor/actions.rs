// SPDX-FileCopyrightText: 2026 Conecsa
//
// SPDX-License-Identifier: AGPL-3.0-only

//! Non-AI editor actions: image selection and autosave, capture / upload /
//! delete / cover, replication, classes, training controls and leaving the
//! editor.
//!
//! Each `fn` builds one `Callback` over the `Copy` [`EditorState`]; explicit
//! dependencies (another callback, the parent's `on_back`) are parameters so
//! the wiring is visible in `DatasetEditor`.

use leptos::prelude::*;
use leptos::task::spawn_local;

use crate::api::{self, LabelBox, LabelPolygon};
use crate::i18n::*;
use crate::models::Task;

use super::history::LabelSnapshot;
use super::label_queue::{LabelOp, LabelWrite};
use super::logic::{self, ImageStep};
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

/// Queue a label save or load (see `label_queue`) and, unless a task is
/// already draining the queue, start one: the requests then run one at a
/// time in the order they were issued, and the gallery counts are
/// refreshed once the queue is empty.
pub(super) fn queue_labels(st: EditorState, locale: Locale, op: LabelOp) {
    if !st.label_queue.enqueue(op) {
        return;
    }
    spawn_local(async move {
        let Some(ds) = st.dataset_id.try_get_value() else {
            st.label_queue.finish();
            return;
        };
        let mut refresh = false;
        loop {
            match st.label_queue.pop() {
                Some(op) => refresh |= run_label_op(st, locale, &ds, op).await,
                None if refresh => {
                    refresh = false;
                    let _ = reload_images(st, &ds).await;
                }
                None => break,
            }
        }
        st.label_queue.finish();
    });
}

/// Run one queued request; true when the gallery counts may have changed.
async fn run_label_op(st: EditorState, locale: Locale, ds: &str, op: LabelOp) -> bool {
    match op {
        LabelOp::Write(w) => {
            // Captured before a class removal: its ids move like the ones on
            // the server did.
            let w = st.label_queue.rebase_write(w);
            match write_labels(st, ds, &w.image_id, &w.boxes, &w.polygons, w.image_class).await {
                Ok(()) => {
                    if w.notify {
                        st.notices
                            .success(td_string!(locale, training::labels_saved).to_string());
                    }
                    true
                }
                Err(e) => {
                    st.notices
                        .error(td_string!(locale, training::failed_save_labels, err = e));
                    false
                }
            }
        }
        LabelOp::Load(id) => {
            match api::get_training_labels(ds, &id).await {
                // Another image was opened meanwhile: its own load shows its
                // labels; these would overwrite them with the wrong image's.
                Ok(_) if st.images.selected.try_get_untracked().flatten() != Some(id.clone()) => {}
                Ok(r) => st.images.load_labels(&id, r),
                Err(e) => st
                    .notices
                    .error(td_string!(locale, training::failed_load_labels, err = e)),
            }
            true
        }
        LabelOp::RemoveClass { index, epoch } => {
            // Confirmed against an older list: the class it named may have
            // moved, or already be gone (removed twice).
            let Some(index) = st.label_queue.rebase_index(index, epoch) else {
                return false;
            };
            match api::remove_training_class(ds, index).await {
                Ok(r) => {
                    st.label_queue.record_removal(index);
                    let len = r.classes.len();
                    let _ = st.classes.list.try_set(r.classes);
                    let _ = st
                        .classes
                        .active
                        .try_update(|c| *c = logic::active_after_remove(*c, index, len));
                    // The open image's labels move with the server's, edits
                    // made meanwhile included, in the same step as the list.
                    st.images.renumber_labels(index);
                    st.ai.clear_suggestions();
                    true
                }
                Err(e) => {
                    st.notices
                        .error(td_string!(locale, training::failed_remove_class, err = e));
                    false
                }
            }
        }
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
        // Autosave the previous image only from its own labels: while they
        // were still loading (a held arrow key steps faster than the server
        // answers) the signals are empty, and writing them would wipe it.
        let prev = prev.filter(|p| st.images.holds_labels_of(p));
        st.ai.clear_suggestions();
        let _ = st.images.selected.try_set(Some(id.clone()));
        // Nothing to show, edit or undo until this image's labels are in.
        st.images.unload_labels();
        let locale = i18n.get_locale_untracked();
        if let Some(prev_id) = prev {
            let save = LabelWrite {
                image_id: prev_id,
                boxes: prev_boxes,
                polygons: prev_polygons,
                image_class: prev_class,
                notify: false,
                epoch: st.label_queue.epoch(),
            };
            queue_labels(st, locale, LabelOp::Write(save));
        }
        queue_labels(st, locale, LabelOp::Load(id));
    })
}

/// Open the previous / next / first / last image of the gallery through
/// `select_image` (which autosaves the open one and resets the history).
pub(super) fn step_image(st: EditorState, select_image: Callback<String>) -> Callback<ImageStep> {
    Callback::new(move |step: ImageStep| {
        // Reached from global key listeners: disposal-safe.
        let Some(selected) = st.images.selected.try_get_untracked() else {
            return;
        };
        let Some(next) = st
            .images
            .list
            .try_with_untracked(|list| logic::neighbor_image(list, selected.as_deref(), step))
            .flatten()
        else {
            return;
        };
        select_image.run(next);
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
        // Not before the image's class is loaded: the load would put the old
        // one back on screen while the server kept this pick.
        if !st.images.labels_loaded() {
            return;
        }
        let _ = st.images.image_class.try_set(class_id);
        st.ai.clear_suggestions();
        let save = LabelWrite {
            image_id: id,
            boxes: Vec::new(),
            polygons: Vec::new(),
            image_class: class_id,
            notify: false,
            epoch: st.label_queue.epoch(),
        };
        queue_labels(st, i18n.get_locale_untracked(), LabelOp::Write(save));
    })
}

/// Persist the open image's boxes / polygons (`true` = show a toast). A
/// gesture that lands before the image's labels are loaded is dropped: the
/// load overwrites it anyway, and saving it would wipe the image's labels.
pub(super) fn save_labels(st: EditorState, i18n: I18n) -> Callback<bool> {
    Callback::new(move |notify: bool| {
        let Some(id) = st.images.selected.get_untracked() else {
            return;
        };
        if !st.images.labels_loaded() {
            return;
        }
        // The signals already hold the committed state: this is an undo step.
        st.images.record_history();
        let save = LabelWrite {
            image_id: id,
            boxes: st.images.boxes.get_untracked(),
            polygons: st.images.polygons.get_untracked(),
            image_class: st.images.image_class.get_untracked(),
            notify,
            epoch: st.label_queue.epoch(),
        };
        queue_labels(st, i18n.get_locale_untracked(), LabelOp::Write(save));
    })
}

// ── undo / redo / clear ───────────────────────────────────────────────────────

/// Show and persist `snapshot` as the open image's labels (an undo/redo
/// target: not recorded again).
fn restore_labels(st: EditorState, locale: Locale, id: String, snapshot: LabelSnapshot) {
    let LabelSnapshot { boxes, polygons } = snapshot;
    let _ = st.images.boxes.try_set(boxes.clone());
    let _ = st.images.polygons.try_set(polygons.clone());
    // Queued: a held Ctrl+Z issues many writes, and the last must win.
    let save = LabelWrite {
        image_id: id,
        boxes,
        polygons,
        image_class: st.images.image_class.try_get_untracked().flatten(),
        notify: false,
        epoch: st.label_queue.epoch(),
    };
    queue_labels(st, locale, LabelOp::Write(save));
}

/// Step the open image's labels back to the state before the last edit.
pub(super) fn undo_labels(st: EditorState, i18n: I18n) -> Callback<()> {
    Callback::new(move |_: ()| {
        let Some(Some(id)) = st.images.selected.try_get_untracked() else {
            return;
        };
        let Some(Some(snapshot)) = st.images.history.try_update(|h| h.undo()) else {
            return;
        };
        restore_labels(st, i18n.get_locale_untracked(), id, snapshot);
    })
}

/// Step the open image's labels forward again after an undo.
pub(super) fn redo_labels(st: EditorState, i18n: I18n) -> Callback<()> {
    Callback::new(move |_: ()| {
        let Some(Some(id)) = st.images.selected.try_get_untracked() else {
            return;
        };
        let Some(Some(snapshot)) = st.images.history.try_update(|h| h.redo()) else {
            return;
        };
        restore_labels(st, i18n.get_locale_untracked(), id, snapshot);
    })
}

/// Remove every label of the open image, as one (undoable) edit.
pub(super) fn clear_labels(st: EditorState, save_labels: Callback<bool>) -> Callback<()> {
    Callback::new(move |_: ()| {
        let empty = st.images.boxes.with_untracked(Vec::is_empty)
            && st.images.polygons.with_untracked(Vec::is_empty);
        if empty {
            return;
        }
        st.images.boxes.set(Vec::new());
        st.images.polygons.set(Vec::new());
        save_labels.run(false);
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

/// Upload image files from the operator's computer, one after another: a file
/// that fails is reported by name and the rest still go up. The last image
/// added opens in the editor, like a capture.
pub(super) fn upload_images(
    st: EditorState,
    i18n: I18n,
    select_image: Callback<String>,
) -> Callback<Vec<web_sys::File>> {
    Callback::new(move |files: Vec<web_sys::File>| {
        if files.is_empty() || st.images.uploading.get_untracked() {
            return;
        }
        st.images.uploading.set(true);
        let locale = i18n.get_locale_untracked();
        spawn_local(async move {
            let ds = st.dataset_id.get_value();
            let mut added = Vec::new();
            for file in files {
                let name = file.name();
                match api::upload_training_image(&ds, file).await {
                    Ok(info) => added.push(info.image_id),
                    Err(e) => st.notices.error(td_string!(
                        locale,
                        training::upload_image_failed,
                        name = name,
                        err = e
                    )),
                }
            }
            if added.len() > 1 {
                st.notices.success(td_string!(
                    locale,
                    training::images_uploaded,
                    count = added.len()
                ));
            }
            if let Some(last) = added.pop() {
                // `select_image` autosaves the previous image and refreshes
                // the gallery for us.
                select_image.run(last);
            }
            let _ = st.images.uploading.try_set(false);
        });
    })
}

/// Ask before deleting `id`: a thumbnail's trash sits a few pixels from the
/// picture itself, so a single tap never removes an image.
pub(super) fn delete_image(st: EditorState) -> Callback<String> {
    Callback::new(move |id: String| {
        st.delete_image.target.set(Some(id));
    })
}

pub(super) fn delete_image_confirm(st: EditorState, i18n: I18n) -> Callback<()> {
    Callback::new(move |_: ()| {
        let Some(id) = st.delete_image.target.get_untracked() else {
            return;
        };
        if st.delete_image.busy.get_untracked() {
            return;
        }
        st.delete_image.busy.set(true);
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
                        st.images.unload_labels();
                        st.ai.clear_suggestions();
                    }
                    reload_images_or_report(st, locale, &ds).await;
                }
                Err(e) => {
                    st.notices
                        .error(td_string!(locale, training::failed_delete_image, err = e))
                }
            }
            let _ = st.delete_image.target.try_set(None);
            let _ = st.delete_image.busy.try_set(false);
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

/// Remove a class: queued behind the pending label saves (see
/// `label_queue`), because it renumbers the classes after it.
pub(super) fn class_remove(st: EditorState, i18n: I18n) -> Callback<usize> {
    Callback::new(move |index: usize| {
        let epoch = st.label_queue.epoch();
        queue_labels(
            st,
            i18n.get_locale_untracked(),
            LabelOp::RemoveClass { index, epoch },
        );
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
        // Autosave the open image's labels (only once they are its own: see
        // `select_image`); stay in training mode (the gallery is still part
        // of the training page).
        let prev = st
            .images
            .selected
            .get_untracked()
            .filter(|p| st.images.holds_labels_of(p));
        if let Some(id) = prev {
            let save = LabelWrite {
                image_id: id,
                boxes: st.images.boxes.get_untracked(),
                polygons: st.images.polygons.get_untracked(),
                image_class: st.images.image_class.get_untracked(),
                notify: false,
                epoch: st.label_queue.epoch(),
            };
            queue_labels(st, i18n.get_locale_untracked(), LabelOp::Write(save));
        }
        // Leaving unmounts the editor and its queue: let every queued save
        // reach the server first.
        spawn_local(async move {
            st.label_queue.drained().await;
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
            on_delete: delete_image(st),
            on_set_cover: set_cover(st, i18n),
            on_replicate: replicate(st),
        }
    }
}
