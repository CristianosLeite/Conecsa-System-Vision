// SPDX-FileCopyrightText: 2026 Conecsa
//
// SPDX-License-Identifier: AGPL-3.0-only

//! Label editor card: the shared state types (`PolygonTool`, and `ViewState`
//! from the shared `conecsa_ui_view` crate), the commands on the selection
//! (`tools`), the keyboard map (`keys`) and
//! the component that wires them to the toolbar, the assistant panels, the
//! canvas and the global listeners.

use leptos::prelude::*;

use super::dataset_editor::{
    AiState, Assistant, ClassesState, DrawMode, ImagesState, LabelActions,
};
use super::label_canvas::LabelCanvas;
use super::label_geometry::{PolyDraft, PolyDrag};
use super::label_model_panel::LabelModelPanel;
use super::label_sam_panel::LabelSamPanel;
use super::label_toolbar::{HistoryTools, LabelToolbar, SelectionTools, ZoomTools};
use conecsa_ui_view::transform::ZOOM_MAX;
/// Zoom and pan of the image on the canvas, shared by the canvas (gestures),
/// the toolbar (buttons, readout) and the global key handler.
pub(super) use conecsa_ui_view::viewport::ViewState;

mod keys;
mod tools;

use keys::{activates_focused_control, command, typing, Command, Mods};
use tools::LabelTools;

/// The polygon tool of a segmentation dataset: the selected ring and vertex,
/// the edit in progress, the ring being drawn and how new rings are drawn.
/// `enabled` is false in a detection dataset.
#[derive(Clone, Copy)]
pub(super) struct PolygonTool {
    pub(super) enabled: bool,
    pub(super) selected: RwSignal<Option<usize>>,
    pub(super) vertex: RwSignal<Option<usize>>,
    pub(super) drag: RwSignal<Option<PolyDrag>>,
    pub(super) draft: RwSignal<Option<PolyDraft>>,
    pub(super) mode: RwSignal<DrawMode>,
}

impl PolygonTool {
    fn new(enabled: bool) -> Self {
        Self {
            enabled,
            selected: RwSignal::new(None),
            vertex: RwSignal::new(None),
            drag: RwSignal::new(None),
            draft: RwSignal::new(None),
            mode: RwSignal::new(DrawMode::Click),
        }
    }

    /// Forget the selection and any gesture. `try_*`: also reached from global
    /// listeners that can briefly outlive the editor.
    fn reset(self) {
        let _ = self.selected.try_set(None);
        let _ = self.vertex.try_set(None);
        let _ = self.drag.try_set(None);
        let _ = self.draft.try_set(None);
    }
}

/// Label editor card: bounding boxes for a detection dataset, polygons for a
/// segmentation one. Owns the shared selection and the in-progress edit
/// (`LabelTools`) plus the global keyboard / pointerup listeners, and
/// composes the toolbar, the (optional) assistant bar — SAM prompts or a
/// device model's Detect — and the canvas.
///
/// Boxes: drag the body to move, drag a corner handle to resize, drag on the
/// background to draw. Polygons: see `LabelCanvas`. In SAM mode clicks become
/// point prompts (Shift/Ctrl-click = negative) and suggestions render dashed
/// until accepted. The keyboard map (`keys`): Delete, `+` / `-` / `0` zoom,
/// Space+drag pans, Ctrl+Z / Ctrl+Y (or the toolbar's undo / redo) undo and redo,
/// Clear labels removes every label as one undoable step, the arrows nudge
/// the selected box, ring or vertex by one screen pixel (ten with Shift) and
/// are saved once on release, with nothing selected ← / → open the previous
/// / next image (Home / End the first / last, PageUp / PageDown always step),
/// and Esc abandons a ring being drawn, else clears the selection.
#[component]
pub(super) fn LabelEditor(
    dataset_id: String,
    images: ImagesState,
    classes: ClassesState,
    ai: AiState,
    actions: LabelActions,
    /// A segmentation dataset: polygons instead of boxes.
    #[prop(optional)]
    segment: bool,
) -> impl IntoView {
    let selected = images.selected;
    let boxes = images.boxes;
    let polygons = images.polygons;
    let assistant = ai.assistant;
    let tools = LabelTools::new(images, actions, segment);
    let selected_box = tools.selected_box;
    let poly = tools.poly;
    let view = tools.view;
    let sam_mode = Signal::derive(move || assistant.get() == Assistant::Sam);

    // A fresh image clears any selection/edit so a stale index can't carry over
    // to the next image's labels after they autoload, and shows the whole
    // picture again.
    Effect::new(move |_| {
        let _ = selected.get();
        tools.image_opened();
    });
    // Switching the draw mode abandons a ring half drawn in the other mode.
    Effect::new(move |_| {
        let _ = poly.mode.get();
        let _ = poly.draft.try_set(None);
    });

    // Keys are global (the canvas is not focusable). Esc comes first: it also
    // works from a text field. Every other key is ignored while a form field
    // is focused, and Alt-combinations are the browser's.
    let key_handle = window_event_listener(leptos::ev::keydown, move |ev: web_sys::KeyboardEvent| {
        let key = ev.key();
        if key == "Escape" {
            tools.escape();
            return;
        }
        if typing() || ev.alt_key() {
            return;
        }
        let mods = Mods {
            ctrl: ev.ctrl_key() || ev.meta_key(),
            shift: ev.shift_key(),
        };
        let Some(cmd) = command(&key, mods, tools.has_selection()) else {
            return;
        };
        // Space on a focused button/link activates it instead of panning.
        if cmd == Command::Pan && activates_focused_control() {
            return;
        }
        if cmd.cancels_default() {
            ev.prevent_default();
        }
        tools.run(cmd);
    });
    on_cleanup(move || key_handle.remove());

    // Space released disarms panning; an arrow released saves the nudge. The
    // window losing focus counts as every key released, so nothing sticks
    // after an alt-tab.
    let keyup_handle = window_event_listener(leptos::ev::keyup, move |ev: web_sys::KeyboardEvent| {
        let key = ev.key();
        if key == " " {
            tools.release_space();
        } else if key.starts_with("Arrow") {
            tools.flush_nudge();
        }
    });
    on_cleanup(move || keyup_handle.remove());
    let blur_handle = window_event_listener(leptos::ev::blur, move |_| {
        tools.release_space();
        tools.flush_nudge();
    });
    on_cleanup(move || blur_handle.remove());

    // Pointer (not mouse) so touch/pen drags commit too. Polygon gestures
    // capture the pointer, so their release always reaches the canvas.
    let up_handle = window_event_listener(leptos::ev::pointerup, move |_| tools.commit_box_drag());
    on_cleanup(move || up_handle.remove());

    // What the toolbar shows and does. A segmentation label is presented as
    // a box like any other label; only its drawing tools differ.
    let selected_class = Signal::derive(move || {
        if segment {
            poly.selected
                .get()
                .and_then(|i| polygons.with(|ps| ps.get(i).map(|p| p.class_id)))
        } else {
            selected_box
                .get()
                .and_then(|i| boxes.with(|bs| bs.get(i).map(|b| b.class_id)))
        }
    });
    let selection = SelectionTools {
        class: selected_class,
        vertex: Signal::derive(move || poly.vertex.get().is_some()),
        on_set_class: Callback::new(move |class_id: u32| tools.set_class(class_id)),
        on_delete: Callback::new(move |_: ()| tools.delete_selected()),
        on_delete_vertex: Callback::new(move |_: ()| tools.delete()),
    };
    let history = images.history;
    let history = HistoryTools {
        can_undo: Signal::derive(move || history.with(|h| h.can_undo())),
        can_redo: Signal::derive(move || history.with(|h| h.can_redo())),
        has_labels: Signal::derive(move || {
            !boxes.with(Vec::is_empty) || !polygons.with(Vec::is_empty)
        }),
        on_undo: Callback::new(move |_: ()| tools.undo()),
        on_redo: Callback::new(move |_: ()| tools.redo()),
        on_clear: Callback::new(move |_: ()| tools.clear_all()),
    };
    let zoom = ZoomTools {
        percent: Signal::derive(move || view.transform.get().percent()),
        max_percent: (ZOOM_MAX * 100.0).round() as u32,
        on_in: Callback::new(move |_: ()| tools.zoom_in()),
        on_out: Callback::new(move |_: ()| tools.zoom_out()),
        on_reset: Callback::new(move |_: ()| tools.zoom_reset()),
    };

    view! {
        <div class="ui-card ui-card-pad-sm ui-label-editor">
            <LabelToolbar
                classes=classes
                ai=ai
                on_assistant_change=actions.on_assistant_change
                draw_mode=segment.then_some(poly.mode)
                selection=selection
                history=history
                zoom=zoom
            />

            {move || match assistant.get() {
                Assistant::Sam => view! {
                    <LabelSamPanel
                        ai=ai
                        on_sam_suggest=actions.on_sam_suggest
                        on_sam_accept=actions.on_accept
                        on_sam_clear=actions.on_clear
                    />
                }.into_any(),
                Assistant::Model(_) => view! {
                    <LabelModelPanel
                        ai=ai
                        on_detect=actions.on_model_detect
                        on_accept=actions.on_accept
                        on_clear=actions.on_clear
                    />
                }.into_any(),
                Assistant::Off => view! { <span/> }.into_any(),
            }}

            <LabelCanvas
                dataset_id=dataset_id
                images=images
                classes=classes
                ai=ai
                selected_box=selected_box
                drag=tools.drag
                poly=poly
                view=view
                sam_mode=sam_mode
                on_save=actions.on_save
                on_need_class=actions.on_need_class
                on_refused=actions.on_refused
            />
        </div>
    }
}
