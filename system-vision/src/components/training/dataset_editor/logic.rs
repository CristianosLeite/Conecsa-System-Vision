// SPDX-FileCopyrightText: 2026 Conecsa
//
// SPDX-License-Identifier: AGPL-3.0-only

//! Pure decision helpers for the dataset editor.
//!
//! No signals, no i18n, no DOM: the async callbacks in `actions`/`ai`/`tasks`
//! call these for every non-trivial decision, so the rules are unit-tested in
//! `tests.rs` without mounting a component.

use crate::api::{
    LabelBox, LabelModelStatusResponse, LabelPolygon, TrainingImageInfo, TrainingJobStatus,
};
use crate::class_color::class_display_name;
use crate::components::configuration::model_conversion::PendingConversion;
use crate::models::Task;

use super::super::label_geometry::{box_ring, Pt};
use super::state::Assistant;

/// The instance id for a new object in a segmentation image: one past the
/// highest in use, so the rings of different objects never share one.
pub(crate) fn next_instance(polygons: &[LabelPolygon]) -> u32 {
    polygons.iter().map(|p| p.instance + 1).max().unwrap_or(0)
}

/// The rings an accepted suggestion becomes in a segmentation dataset: its
/// mask's rings, or its box promoted to a rectangle when it has no mask.
pub(crate) fn suggestion_rings(b: &LabelBox, rings: Option<&Vec<Vec<Pt>>>) -> Vec<Vec<Pt>> {
    match rings {
        Some(rings) if !rings.is_empty() => rings.clone(),
        _ => vec![box_ring(b.cx, b.cy, b.w, b.h)],
    }
}

/// Index of the dataset class whose display name matches `name`
/// (case-insensitive; class entries may carry a trailing color).
pub(crate) fn resolve_class(classes: &[String], name: &str) -> Option<usize> {
    classes
        .iter()
        .position(|c| class_display_name(c).eq_ignore_ascii_case(name))
}

/// What the job poll should do with a fresh status, given the previous one.
#[derive(Debug, PartialEq)]
pub(crate) enum JobTransition {
    /// A run finished on THIS page: hand the conversion to the dashboard and
    /// leave the training page.
    Finished { pending: Option<PendingConversion> },
    /// A run this page was watching failed.
    Failed,
    /// A run this page was watching was canceled.
    Canceled,
    /// Keep the overlay's snapshot fresh.
    Update,
    /// Nothing to show (idle, or a terminal state we never saw start).
    Ignore,
}

/// Classify a polled status against the previous one. `had_job` is whether
/// the page already holds a job snapshot (a stale terminal status from a
/// previous session counts, which is what the poll's seeding relies on).
pub(crate) fn job_transition(
    prev: &str,
    next: &TrainingJobStatus,
    had_job: bool,
    task: Task,
) -> JobTransition {
    match next.status.as_str() {
        "done" if prev != "done" => JobTransition::Finished {
            pending: pending_conversion(next, task),
        },
        "failed" if prev != "failed" && !prev.is_empty() => JobTransition::Failed,
        "canceled" if prev != "canceled" && !prev.is_empty() => JobTransition::Canceled,
        _ if next.is_active() || had_job => JobTransition::Update,
        _ => JobTransition::Ignore,
    }
}

/// The conversion job a finished run handed off, if it produced one. A face
/// run uploads a `.faces` package (the photos the gallery is built from), every
/// other task the trained `.pt` weights.
pub(crate) fn pending_conversion(
    job: &TrainingJobStatus,
    task: Task,
) -> Option<PendingConversion> {
    let ext = if task == Task::Face { "faces" } else { "pt" };
    (!job.conversion_job_id.is_empty()).then(|| PendingConversion {
        job_id: job.conversion_job_id.clone(),
        filename: format!("{}.{ext}", job.model_name),
        elapsed_secs: 0.0,
    })
}

/// Keep `active` inside a class list of `len` entries. An empty list leaves it
/// untouched (there is nothing valid to clamp to).
pub(crate) fn clamp_active(active: usize, len: usize) -> usize {
    if len == 0 {
        active
    } else {
        active.min(len - 1)
    }
}

/// The active class after the class at `removed` was deleted, leaving
/// `len_after` entries: entries above the removed one shift down, and the
/// result is clamped into the new list.
pub(crate) fn active_after_remove(active: usize, removed: usize, len_after: usize) -> usize {
    let mut c = active;
    if c >= removed && c > 0 {
        c -= 1;
    }
    c.min(len_after.saturating_sub(1))
}

/// Which class accepted SAM suggestions are tagged with.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum SamTarget {
    /// The text prompt names an existing class.
    Existing(usize),
    /// The text prompt names a class that must be created first.
    Create,
    /// No text prompt (point-only): use the active class.
    Active(usize),
    /// No text prompt and no classes at all.
    NoClasses,
}

/// SAM: the text prompt IS the class label — accepting "bottle" suggestions
/// tags them as class "bottle" (creating it if needed), NOT the
/// manually-selected active class. Point-only suggestions (no prompt) fall
/// back to the active class.
pub(crate) fn sam_accept_target(prompt: &str, classes: &[String], active: usize) -> SamTarget {
    let prompt = prompt.trim();
    if !prompt.is_empty() {
        match resolve_class(classes, prompt) {
            Some(idx) => SamTarget::Existing(idx),
            None => SamTarget::Create,
        }
    } else if classes.is_empty() {
        SamTarget::NoClasses
    } else {
        SamTarget::Active(active.min(classes.len() - 1))
    }
}

/// Outcome of resolving a class name against the dataset, creating it when
/// missing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ClassSlot {
    Existing(usize),
    Created(usize),
}

impl ClassSlot {
    pub(crate) fn index(self) -> usize {
        match self {
            ClassSlot::Existing(i) | ClassSlot::Created(i) => i,
        }
    }
}

/// The model's class name for suggestion `i`, trimmed; `None` when missing or
/// blank (such a suggestion is skipped on Accept).
pub(crate) fn model_suggestion_name(names: &[String], i: usize) -> Option<String> {
    names
        .get(i)
        .map(|n| n.trim().to_string())
        .filter(|n| !n.is_empty())
}

/// Whether `name` has to be (re)loaded before detecting: the backend no longer
/// reports it as the loaded engine (a runtime release unloads it).
pub(crate) fn model_needs_load(status: Option<&LabelModelStatusResponse>, name: &str) -> bool {
    !status
        .map(|s| s.loaded && s.model_name == name)
        .unwrap_or(false)
}

/// The labeling engine doubles as the suggested fine-tune base when it keeps
/// its checkpoint: "improve model X" is label with X → Train → Start.
pub(crate) fn default_base_model(assistant: &Assistant, base_models: &[String]) -> String {
    match assistant {
        Assistant::Model(name) if base_models.contains(name) => name.clone(),
        _ => String::new(),
    }
}

pub(crate) fn labeled_count(images: &[TrainingImageInfo]) -> u32 {
    images.iter().filter(|i| i.labeled).count() as u32
}

/// Train is enabled with enough images, at least one class and one label.
pub(crate) fn can_train(
    image_count: u32,
    labeled: u32,
    min_images: u32,
    has_classes: bool,
) -> bool {
    image_count >= min_images && has_classes && labeled > 0
}

/// Distinct classes that label at least one image (classification).
pub(crate) fn labeled_class_count(images: &[TrainingImageInfo]) -> u32 {
    let mut classes: Vec<u32> = images.iter().filter_map(|i| i.image_class).collect();
    classes.sort_unstable();
    classes.dedup();
    classes.len() as u32
}

/// A classifier trains with enough images and labeled images of at least two
/// classes — one class is nothing to tell apart (the training-service refuses
/// it too).
pub(crate) fn can_train_classify(image_count: u32, min_images: u32, labeled_classes: u32) -> bool {
    image_count >= min_images && labeled_classes >= 2
}

/// A face gallery is built, not trained: one person with one labeled photo is
/// already a gallery, so there is no image minimum to meet.
pub(crate) fn can_train_face(labeled_people: u32) -> bool {
    labeled_people >= 1
}

#[cfg(test)]
mod tests;
