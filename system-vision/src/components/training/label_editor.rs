// SPDX-FileCopyrightText: 2026 Conecsa
//
// SPDX-License-Identifier: AGPL-3.0-only

use leptos::prelude::*;

use super::dataset_editor::{AiState, Assistant, ClassesState, DrawMode, ImagesState, LabelActions};
use super::label_canvas::LabelCanvas;
use super::label_geometry::{valid_ring, BoxDrag, PolyDraft, PolyDrag};
use super::label_model_panel::LabelModelPanel;
use super::label_sam_panel::LabelSamPanel;
use super::label_toolbar::LabelToolbar;

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
/// segmentation one. Owns the shared selection and the in-progress edit plus
/// the global keyboard / pointerup listeners, and composes the toolbar, the
/// (optional) assistant bar — SAM prompts or a device model's Detect — and
/// the canvas.
///
/// Boxes: drag the body to move, drag a corner handle to resize, drag on the
/// background to draw. Polygons: see `LabelCanvas`; Esc abandons a ring being
/// drawn, Delete removes the selected vertex (or, on a triangle, the ring).
/// In SAM mode clicks become point prompts (Shift/Ctrl-click = negative) and
/// suggestions render dashed until accepted.
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
    let on_save = actions.on_save;
    let on_refused = actions.on_refused;
    let selected_box = RwSignal::new(None::<usize>);
    // Active move/resize of an already-committed box (shared with the canvas and
    // the global pointerup listener).
    let drag = RwSignal::new(None::<BoxDrag>);
    let poly = PolygonTool::new(segment);
    let sam_mode = Signal::derive(move || assistant.get() == Assistant::Sam);
    let panel_open = Signal::derive(move || assistant.get() != Assistant::Off);

    // A fresh image clears any selection/edit so a stale index can't carry over
    // to the next image's labels after they autoload.
    Effect::new(move |_| {
        let _ = selected.get();
        selected_box.set(None);
        drag.set(None);
        poly.reset();
    });
    // Switching the draw mode abandons a ring half drawn in the other mode.
    Effect::new(move |_| {
        let _ = poly.mode.get();
        let _ = poly.draft.try_set(None);
    });

    // Remove the selected vertex. A triangle keeps its vertices (the caller
    // removes the ring instead) and a ring that would cross itself is refused.
    // Returns whether the key press was handled here.
    let delete_vertex = move || -> bool {
        let (Some(Some(idx)), Some(Some(v))) =
            (poly.selected.try_get_untracked(), poly.vertex.try_get_untracked())
        else {
            return false;
        };
        let Some(Some(mut ring)) =
            polygons.try_with_untracked(|ps| ps.get(idx).map(|p| p.points.clone()))
        else {
            return false;
        };
        if ring.len() <= 3 || v >= ring.len() {
            return false;
        }
        ring.remove(v);
        let _ = poly.vertex.try_set(None);
        if !valid_ring(&ring) {
            on_refused.run(());
            return true;
        }
        let _ = polygons.try_update(|ps| {
            if let Some(p) = ps.get_mut(idx) {
                p.points = ring;
            }
        });
        on_save.run(false);
        true
    };

    // Remove the selected ring or box.
    let delete_selected = move || {
        if segment {
            if let Some(Some(idx)) = poly.selected.try_get_untracked() {
                let _ = polygons.try_update(|ps| {
                    if idx < ps.len() {
                        ps.remove(idx);
                    }
                });
                poly.reset();
                on_save.run(false);
            }
            return;
        }
        if let Some(Some(idx)) = selected_box.try_get_untracked() {
            let _ = boxes.try_update(|bs| {
                if idx < bs.len() {
                    bs.remove(idx);
                }
            });
            let _ = selected_box.try_set(None);
            let _ = drag.try_set(None);
            on_save.run(false);
        }
    };

    // Delete removes the selected vertex, ring or box (Delete only — not
    // Backspace, which is used while typing class names / SAM prompts), and is
    // ignored when a form field is focused so typing never deletes a label. Esc
    // abandons a ring being drawn.
    let key_handle = window_event_listener(leptos::ev::keydown, move |ev: web_sys::KeyboardEvent| {
        let key = ev.key();
        if key == "Escape" {
            // try_* throughout: these are global listeners that can briefly
            // outlive the editor's signals during teardown — a disposed signal
            // must no-op, not panic.
            if poly.draft.try_with_untracked(Option::is_some).unwrap_or(false) {
                let _ = poly.draft.try_set(None);
            }
            return;
        }
        if key != "Delete" {
            return;
        }
        let focused_tag = web_sys::window()
            .and_then(|w| w.document())
            .and_then(|d| d.active_element())
            .map(|el| el.tag_name().to_uppercase());
        if let Some(tag) = focused_tag {
            if matches!(tag.as_str(), "INPUT" | "TEXTAREA" | "SELECT") {
                return;
            }
        }
        if segment && delete_vertex() {
            return;
        }
        delete_selected();
    });
    on_cleanup(move || key_handle.remove());

    // Commit an in-progress box move/resize on pointerup anywhere — so a drag
    // that runs past the (small) canvas edge still finishes instead of being
    // stranded. Pointer (not mouse) so touch/pen drags commit too. Polygon
    // gestures capture the pointer, so their release always reaches the canvas.
    let up_handle = window_event_listener(leptos::ev::pointerup, move |_| {
        if let Some(Some(d)) = drag.try_get_untracked() {
            let _ = drag.try_set(None);
            if d.moved {
                on_save.run(false);
            }
        }
    });
    on_cleanup(move || up_handle.remove());

    // Toolbar actions on the selected label.
    let on_delete = Callback::new(move |_: ()| delete_selected());
    let on_delete_vertex = Callback::new(move |_: ()| {
        if !delete_vertex() {
            delete_selected();
        }
    });
    let on_set_class = Callback::new(move |class_id: u32| {
        if segment {
            if let Some(idx) = poly.selected.get_untracked() {
                // Every ring of the object takes the class.
                polygons.update(|ps| {
                    let instance = ps.get(idx).map(|p| p.instance);
                    for p in ps.iter_mut().filter(|p| Some(p.instance) == instance) {
                        p.class_id = class_id;
                    }
                });
                on_save.run(false);
            }
            return;
        }
        if let Some(idx) = selected_box.get_untracked() {
            boxes.update(|bs| {
                if let Some(b) = bs.get_mut(idx) {
                    b.class_id = class_id;
                }
            });
            on_save.run(false);
        }
    });
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
    let vertex_selected = Signal::derive(move || poly.vertex.get().is_some());

    view! {
        <div class="ui-card ui-card-pad-sm flex flex-col gap-3">
            <LabelToolbar
                selected_class=selected_class
                classes=classes
                ai=ai
                on_set_class=on_set_class
                on_delete=on_delete
                on_assistant_change=actions.on_assistant_change
                segment=segment
                draw_mode=poly.mode
                vertex_selected=vertex_selected
                on_delete_vertex=on_delete_vertex
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
                drag=drag
                poly=poly
                sam_mode=sam_mode
                panel_open=panel_open
                on_save=on_save
                on_need_class=actions.on_need_class
                on_refused=on_refused
            />
        </div>
    }
}
