// SPDX-FileCopyrightText: 2026 Conecsa
//
// SPDX-License-Identifier: AGPL-3.0-only

//! Header of the label editor card: the title row with the draw-mode and
//! AI-assist selects (`head`) over the tool strip (`strip`) — the selected
//! label's class and delete buttons, undo / redo / clear labels and the zoom
//! controls, drawn as icons (`icons`) and shown only while they apply.
//!
//! The strip never wraps and every control in it is a fixed-height box, so
//! a control appearing or disappearing changes the strip's width, never the
//! toolbar's height: the canvas below stays put (only a viewport resize
//! re-wraps the head row).

use leptos::prelude::*;

use super::dataset_editor::{AiState, Assistant, ClassesState, DrawMode};

mod head;
mod icons;
mod strip;

use head::ToolbarHead;
use strip::ToolStrip;

/// The selected label and the commands on it.
#[derive(Clone, Copy)]
pub(super) struct SelectionTools {
    /// The class of the selected box or ring (`None`: nothing selected).
    pub(super) class: Signal<Option<u32>>,
    /// A vertex of the selected ring is selected (never in a detection
    /// dataset).
    pub(super) vertex: Signal<bool>,
    /// Reassign the selected label's class.
    pub(super) on_set_class: Callback<u32>,
    /// Delete the selected label.
    pub(super) on_delete: Callback<()>,
    /// Delete the selected vertex.
    pub(super) on_delete_vertex: Callback<()>,
}

/// The edit history of the open image and the commands on it.
#[derive(Clone, Copy)]
pub(super) struct HistoryTools {
    pub(super) can_undo: Signal<bool>,
    pub(super) can_redo: Signal<bool>,
    /// The open image has at least one label.
    pub(super) has_labels: Signal<bool>,
    pub(super) on_undo: Callback<()>,
    pub(super) on_redo: Callback<()>,
    /// Remove every label of the open image (one undo step).
    pub(super) on_clear: Callback<()>,
}

/// The image zoom and the commands on it.
#[derive(Clone, Copy)]
pub(super) struct ZoomTools {
    /// The zoom in percent (100 = the whole image).
    pub(super) percent: Signal<u32>,
    /// The largest zoom, in percent (zoom in stops there).
    pub(super) max_percent: u32,
    pub(super) on_in: Callback<()>,
    pub(super) on_out: Callback<()>,
    /// Back to 100 %.
    pub(super) on_reset: Callback<()>,
}

/// Header of the label editor card: the title, the draw-mode select of a
/// segmentation dataset and the AI-assist selector — off, SAM3 prompts, or
/// one of the device's existing engines — over the tool strip.
#[component]
pub(super) fn LabelToolbar(
    classes: ClassesState,
    ai: AiState,
    on_assistant_change: Callback<Assistant>,
    /// How new rings are drawn (`Some` in a segmentation dataset).
    draw_mode: Option<RwSignal<DrawMode>>,
    selection: SelectionTools,
    history: HistoryTools,
    zoom: ZoomTools,
) -> impl IntoView {
    view! {
        <div class="ui-label-toolbar">
            <ToolbarHead ai=ai on_assistant_change=on_assistant_change draw_mode=draw_mode />
            <ToolStrip classes=classes selection=selection history=history zoom=zoom />
        </div>
    }
}
