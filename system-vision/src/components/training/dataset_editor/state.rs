// SPDX-FileCopyrightText: 2026 Conecsa
//
// SPDX-License-Identifier: AGPL-3.0-only

//! Signal bundles for the dataset editor.
//!
//! Everything here is `Copy` (arena signals and a `StoredValue`), so the task
//! and action helpers capture one [`EditorState`] by value in every closure
//! instead of threading thirty individual signals around, and the children
//! take a bundle instead of a prop list.

use leptos::prelude::*;

use crate::api::{
    DatasetSummary, LabelBox, LabelClassSuggestion, LabelModelStatusResponse, LabelPolygon,
    SamStatusResponse, TrainingImageInfo, TrainingJobStatus,
};
use crate::i18n::*;
use crate::models::Task;

use super::MIN_IMAGES_DEFAULT;

/// The i18n context handle, captured by value into callbacks.
pub(crate) type I18n = leptos_i18n::I18nContext<Locale>;

/// Which AI helper (if any) the label editor is using. One at a time (8 GB
/// GPU budget): the gateway's load routes drop the other assistant first, and
/// the editor unloads the engine when the selector goes back to Off.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Assistant {
    Off,
    /// SAM3 text/point prompts.
    Sam,
    /// An existing device model, by its model-list name (`X.engine`).
    Model(String),
}

/// How a new polygon is drawn in a segmentation dataset.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DrawMode {
    /// Click to place vertices; click the first vertex or double-click to close.
    Click,
    /// Press, trace the outline and release.
    Freehand,
    /// Drag a rectangle, stored as a four-vertex polygon.
    Rect,
}

/// The dataset's images, the open one and its labels.
#[derive(Clone, Copy)]
pub(crate) struct ImagesState {
    pub(crate) list: RwSignal<Vec<TrainingImageInfo>>,
    /// `image_id` of the image open in the label editor.
    pub(crate) selected: RwSignal<Option<String>>,
    /// Committed boxes of the open image (autosaved on image switch).
    pub(crate) boxes: RwSignal<Vec<LabelBox>>,
    /// Committed polygon rings of the open image in a segmentation dataset
    /// (saved after every completed gesture, and on image switch).
    pub(crate) polygons: RwSignal<Vec<LabelPolygon>>,
    /// The open image's class in a classification dataset (saved per pick).
    pub(crate) image_class: RwSignal<Option<u32>>,
    pub(crate) cover_image_id: RwSignal<String>,
    pub(crate) capturing: RwSignal<bool>,
}

/// The dataset's class list and the class new boxes are stamped with.
#[derive(Clone, Copy)]
pub(crate) struct ClassesState {
    pub(crate) list: RwSignal<Vec<String>>,
    pub(crate) active: RwSignal<usize>,
}

/// AI-assisted labeling: the selected assistant, its prompts, its pending
/// suggestions and the backend statuses.
#[derive(Clone, Copy)]
pub(crate) struct AiState {
    pub(crate) assistant: RwSignal<Assistant>,
    /// SAM point prompts as normalized `(x, y, positive)`.
    pub(crate) sam_points: RwSignal<Vec<(f32, f32, bool)>>,
    pub(crate) sam_text: RwSignal<String>,
    /// Confidence threshold shared by SAM and the model assistant.
    pub(crate) threshold: RwSignal<f32>,
    /// Pending AI suggestions; `suggestion_names[i]` is the model's class name
    /// for `suggestions[i]` (empty for SAM, whose prompt is the class).
    pub(crate) suggestions: RwSignal<Vec<LabelBox>>,
    pub(crate) suggestion_names: RwSignal<Vec<String>>,
    /// Each suggestion's mask as normalized rings, parallel to `suggestions`
    /// (SAM and segmentation engines); a suggestion without one is accepted
    /// in a segmentation dataset as its box promoted to a rectangle.
    pub(crate) suggestion_polygons: RwSignal<Vec<Vec<Vec<[f32; 2]>>>>,
    /// A classification engine's suggested class for the open image.
    pub(crate) class_suggestion: RwSignal<Option<LabelClassSuggestion>>,
    /// The image the pending suggestions were computed for: a response that
    /// settles after another image was opened is dropped, and an accepted
    /// suggestion is written to this image, not to whichever one is open.
    pub(crate) suggested_for: RwSignal<Option<String>>,
    /// A load/unload/segment/detect request is in flight.
    pub(crate) busy: RwSignal<bool>,
    pub(crate) sam_status: RwSignal<Option<SamStatusResponse>>,
    pub(crate) model_status: RwSignal<Option<LabelModelStatusResponse>>,
    /// Device engines usable for labeling (on the device's TensorRT runtime).
    pub(crate) label_models: RwSignal<Vec<String>>,
}

impl AiState {
    /// Drop the pending suggestions and SAM points. `try_*` because it is also
    /// called after awaits (see [`EditorState`]).
    pub(crate) fn clear_suggestions(self) {
        let _ = self.suggestions.try_set(Vec::new());
        let _ = self.suggestion_names.try_set(Vec::new());
        let _ = self.suggestion_polygons.try_set(Vec::new());
        let _ = self.class_suggestion.try_set(None);
        let _ = self.suggested_for.try_set(None);
        let _ = self.sam_points.try_set(Vec::new());
    }
}

/// The training job and the Train modal.
#[derive(Clone, Copy)]
pub(crate) struct TrainingState {
    pub(crate) job: RwSignal<Option<TrainingJobStatus>>,
    pub(crate) show_modal: RwSignal<bool>,
    /// Checkpoint-bearing device models usable as fine-tune bases.
    pub(crate) base_models: RwSignal<Vec<String>>,
    /// Images required before Train is enabled (from the dataset record).
    pub(crate) min_images: RwSignal<u32>,
}

/// The Replicate modal.
#[derive(Clone, Copy)]
pub(crate) struct ReplicateState {
    pub(crate) show_modal: RwSignal<bool>,
    /// `image_id` being replicated.
    pub(crate) target: RwSignal<Option<String>>,
    pub(crate) count: RwSignal<u32>,
    pub(crate) busy: RwSignal<bool>,
}

/// The editor's own popup messages.
#[derive(Clone, Copy)]
pub(crate) struct Notices {
    pub(crate) error_msg: RwSignal<String>,
    pub(crate) success_msg: RwSignal<String>,
    pub(crate) info_view: RwSignal<Option<String>>,
}

impl Notices {
    /// Show an error toast. Always `try_set`: safe after an await.
    pub(crate) fn error(self, msg: String) {
        let _ = self.error_msg.try_set(msg);
    }

    /// Show a success toast. Always `try_set`: safe after an await.
    pub(crate) fn success(self, msg: String) {
        let _ = self.success_msg.try_set(msg);
    }
}

/// Every signal of one mounted `DatasetEditor`, scoped by the dataset's id.
///
/// NOTE: every signal write that happens AFTER an `.await` in this module tree
/// uses the `try_*` variants — the page can unmount (back / training-done
/// handoff) while a request is in flight, and a plain `set()` on a disposed
/// signal panics the whole WASM app. Likewise `dataset_id.get_value()` must be
/// the first statement of an async block, never after an await.
#[derive(Clone, Copy)]
pub(crate) struct EditorState {
    /// Copy-able handle so every closure can grab the id without
    /// clone-per-closure boilerplate.
    pub(crate) dataset_id: StoredValue<String>,
    /// The dataset's task: boxes for detection, polygon rings for
    /// segmentation, one class per image for classification.
    pub(crate) task: Task,
    pub(crate) images: ImagesState,
    pub(crate) classes: ClassesState,
    pub(crate) ai: AiState,
    pub(crate) training: TrainingState,
    pub(crate) replicate: ReplicateState,
    pub(crate) notices: Notices,
}

impl EditorState {
    pub(crate) fn new(dataset: &DatasetSummary) -> Self {
        Self {
            dataset_id: StoredValue::new(dataset.dataset_id.clone()),
            task: Task::parse(&dataset.task).unwrap_or(Task::Detect),
            images: ImagesState {
                list: RwSignal::new(Vec::new()),
                selected: RwSignal::new(None),
                boxes: RwSignal::new(Vec::new()),
                polygons: RwSignal::new(Vec::new()),
                image_class: RwSignal::new(None),
                cover_image_id: RwSignal::new(dataset.cover_image_id.clone()),
                capturing: RwSignal::new(false),
            },
            classes: ClassesState {
                list: RwSignal::new(Vec::new()),
                active: RwSignal::new(0),
            },
            ai: AiState {
                assistant: RwSignal::new(Assistant::Off),
                sam_points: RwSignal::new(Vec::new()),
                sam_text: RwSignal::new(String::new()),
                threshold: RwSignal::new(0.5),
                suggestions: RwSignal::new(Vec::new()),
                suggestion_names: RwSignal::new(Vec::new()),
                suggestion_polygons: RwSignal::new(Vec::new()),
                class_suggestion: RwSignal::new(None),
                suggested_for: RwSignal::new(None),
                busy: RwSignal::new(false),
                sam_status: RwSignal::new(None),
                model_status: RwSignal::new(None),
                label_models: RwSignal::new(Vec::new()),
            },
            training: TrainingState {
                job: RwSignal::new(None),
                show_modal: RwSignal::new(false),
                base_models: RwSignal::new(Vec::new()),
                min_images: RwSignal::new(MIN_IMAGES_DEFAULT),
            },
            replicate: ReplicateState {
                show_modal: RwSignal::new(false),
                target: RwSignal::new(None),
                count: RwSignal::new(5),
                busy: RwSignal::new(false),
            },
            notices: Notices {
                error_msg: RwSignal::new(String::new()),
                success_msg: RwSignal::new(String::new()),
                info_view: RwSignal::new(None),
            },
        }
    }
}
