// SPDX-FileCopyrightText: 2026 Conecsa
//
// SPDX-License-Identifier: AGPL-3.0-only

//! The label editor's commands: everything the toolbar, the keyboard and the
//! global pointer / focus listeners do to the selected label, the edit
//! history and the view.
//!
//! [`LabelTools`] is `Copy` (arena signals plus the `Copy` action bundle), so
//! the component hands one value to every listener and callback instead of a
//! closure per command. Every method uses the `try_*` signal variants: the
//! global listeners can briefly outlive the editor's signals during teardown,
//! and a disposed signal must no-op, not panic the WASM app.

use leptos::prelude::*;

use crate::api::{LabelBox, LabelPolygon};

use super::super::dataset_editor::{ImagesState, LabelActions};
use super::super::label_geometry::{
    nudge_box, nudge_vertex, translate_ring, valid_ring, BoxDrag, VIEW,
};
use super::keys::Command;
use super::{PolygonTool, ViewState};

/// The editor's working state and the commands on it.
#[derive(Clone, Copy)]
pub(super) struct LabelTools {
    /// A segmentation dataset: rings instead of boxes.
    segment: bool,
    boxes: RwSignal<Vec<LabelBox>>,
    polygons: RwSignal<Vec<LabelPolygon>>,
    /// The selected box of a detection dataset.
    pub(super) selected_box: RwSignal<Option<usize>>,
    /// Active move / resize of an already-committed box (shared with the
    /// canvas and the global pointerup listener).
    pub(super) drag: RwSignal<Option<BoxDrag>>,
    /// The polygon tool of a segmentation dataset (disabled for detection).
    pub(super) poly: PolygonTool,
    /// Zoom / pan of the image, shared with the canvas and the toolbar.
    pub(super) view: ViewState,
    /// An arrow-key hold has moved the selected label: saved on release.
    nudged: RwSignal<bool>,
    actions: LabelActions,
}

impl LabelTools {
    pub(super) fn new(images: ImagesState, actions: LabelActions, segment: bool) -> Self {
        Self {
            segment,
            boxes: images.boxes,
            polygons: images.polygons,
            selected_box: RwSignal::new(None),
            drag: RwSignal::new(None),
            poly: PolygonTool::new(segment),
            view: ViewState::new((VIEW, VIEW)),
            nudged: RwSignal::new(false),
            actions,
        }
    }

    // ── state queries ────────────────────────────────────────────────────────

    /// A committed shape is being dragged with the pointer: the labels must
    /// not change under it.
    pub(super) fn editing(self) -> bool {
        self.drag.try_with_untracked(Option::is_some).unwrap_or(false)
            || self.poly.drag.try_with_untracked(Option::is_some).unwrap_or(false)
    }

    /// A ring is being drawn.
    fn drawing(self) -> bool {
        self.poly.draft.try_with_untracked(Option::is_some).unwrap_or(false)
    }

    /// A box or ring is selected (a ring's vertex counts as its ring).
    pub(super) fn has_selection(self) -> bool {
        if self.segment {
            self.poly.selected.try_get_untracked().flatten().is_some()
        } else {
            self.selected_box.try_get_untracked().flatten().is_some()
        }
    }

    // ── selection ────────────────────────────────────────────────────────────

    /// Forget the selection and any gesture on it.
    fn clear_selection(self) {
        let _ = self.selected_box.try_set(None);
        let _ = self.drag.try_set(None);
        self.poly.reset();
    }

    /// Another image was opened: any selection, gesture or nudge in progress
    /// belonged to the previous one (autosaved with it), and a stale index
    /// must not carry over to the new labels once they load. The whole
    /// picture shows again.
    pub(super) fn image_opened(self) {
        self.clear_selection();
        let _ = self.nudged.try_set(false);
        self.view.reset();
    }

    /// Esc: abandon a ring being drawn, else clear the selection (a drag in
    /// progress keeps it until the pointer is released).
    pub(super) fn escape(self) {
        if self.drawing() {
            let _ = self.poly.draft.try_set(None);
            return;
        }
        if !self.editing() {
            self.clear_selection();
        }
    }

    // ── edits ────────────────────────────────────────────────────────────────

    /// Stamp the selected label with `class_id`: every ring of the object in
    /// a segmentation dataset, the box otherwise.
    pub(super) fn set_class(self, class_id: u32) {
        if self.segment {
            let Some(Some(idx)) = self.poly.selected.try_get_untracked() else {
                return;
            };
            let _ = self.polygons.try_update(|ps| {
                let instance = ps.get(idx).map(|p| p.instance);
                for p in ps.iter_mut().filter(|p| Some(p.instance) == instance) {
                    p.class_id = class_id;
                }
            });
            self.actions.on_save.run(false);
            return;
        }
        let Some(Some(idx)) = self.selected_box.try_get_untracked() else {
            return;
        };
        let _ = self.boxes.try_update(|bs| {
            if let Some(b) = bs.get_mut(idx) {
                b.class_id = class_id;
            }
        });
        self.actions.on_save.run(false);
    }

    /// Remove the selected vertex. A triangle keeps its vertices (the caller
    /// removes the ring instead) and a ring that would cross itself is
    /// refused. Returns whether the request was handled here.
    pub(super) fn delete_vertex(self) -> bool {
        let (Some(Some(idx)), Some(Some(v))) = (
            self.poly.selected.try_get_untracked(),
            self.poly.vertex.try_get_untracked(),
        ) else {
            return false;
        };
        let Some(Some(mut ring)) = self
            .polygons
            .try_with_untracked(|ps| ps.get(idx).map(|p| p.points.clone()))
        else {
            return false;
        };
        if ring.len() <= 3 || v >= ring.len() {
            return false;
        }
        ring.remove(v);
        let _ = self.poly.vertex.try_set(None);
        if !valid_ring(&ring) {
            self.actions.on_refused.run(());
            return true;
        }
        let _ = self.polygons.try_update(|ps| {
            if let Some(p) = ps.get_mut(idx) {
                p.points = ring;
            }
        });
        self.actions.on_save.run(false);
        true
    }

    /// Remove the selected ring or box.
    pub(super) fn delete_selected(self) {
        if self.segment {
            if let Some(Some(idx)) = self.poly.selected.try_get_untracked() {
                let _ = self.polygons.try_update(|ps| {
                    if idx < ps.len() {
                        ps.remove(idx);
                    }
                });
                self.poly.reset();
                self.actions.on_save.run(false);
            }
            return;
        }
        if let Some(Some(idx)) = self.selected_box.try_get_untracked() {
            let _ = self.boxes.try_update(|bs| {
                if idx < bs.len() {
                    bs.remove(idx);
                }
            });
            let _ = self.selected_box.try_set(None);
            let _ = self.drag.try_set(None);
            self.actions.on_save.run(false);
        }
    }

    /// Remove the selected vertex if there is one, else the selected ring or
    /// box (the Delete key and the toolbar's vertex button).
    pub(super) fn delete(self) {
        if !self.delete_vertex() {
            self.delete_selected();
        }
    }

    /// Move the selected label by `(dx, dy)` screen px: the selected vertex
    /// alone, else the whole ring, else the box. Nothing moves during a
    /// pointer drag or while a ring is being drawn; a vertex nudge that would
    /// make the ring cross itself is refused like the equivalent drag. The
    /// save waits for the key release ([`Self::flush_nudge`]), so a held key
    /// is one undo step.
    pub(super) fn nudge(self, dx_px: f32, dy_px: f32) {
        if self.editing() || self.drawing() {
            return;
        }
        let Some((cw, ch)) = self.view.canvas_px() else {
            return;
        };
        if cw <= 0.0 || ch <= 0.0 {
            return;
        }
        let (dx, dy) = (dx_px / cw, dy_px / ch);
        if self.segment {
            let Some(Some(idx)) = self.poly.selected.try_get_untracked() else {
                return;
            };
            let Some(Some(ring)) = self
                .polygons
                .try_with_untracked(|ps| ps.get(idx).map(|p| p.points.clone()))
            else {
                return;
            };
            let moved = match self.poly.vertex.try_get_untracked().flatten() {
                Some(v) if v < ring.len() => nudge_vertex(&ring, v, dx, dy),
                _ => translate_ring(&ring, dx, dy),
            };
            if moved == ring {
                return;
            }
            if !valid_ring(&moved) {
                self.actions.on_refused.run(());
                return;
            }
            let _ = self.polygons.try_update(|ps| {
                if let Some(p) = ps.get_mut(idx) {
                    p.points = moved;
                }
            });
            let _ = self.nudged.try_set(true);
            return;
        }
        let Some(Some(idx)) = self.selected_box.try_get_untracked() else {
            return;
        };
        let mut changed = false;
        let _ = self.boxes.try_update(|bs| {
            if let Some(b) = bs.get_mut(idx) {
                let next = nudge_box((b.cx, b.cy, b.w, b.h), dx, dy);
                changed = next != (b.cx, b.cy, b.w, b.h);
                (b.cx, b.cy, b.w, b.h) = next;
            }
        });
        if changed {
            let _ = self.nudged.try_set(true);
        }
    }

    /// The arrow key is up (or the window lost focus): the nudged labels are
    /// saved once, as one undo step.
    pub(super) fn flush_nudge(self) {
        if self.nudged.try_get_untracked() == Some(true) {
            let _ = self.nudged.try_set(false);
            self.actions.on_save.run(false);
        }
    }

    /// The pointer went up anywhere: an in-progress box move / resize is
    /// committed, so a drag that runs past the (small) canvas edge still
    /// finishes instead of being stranded.
    pub(super) fn commit_box_drag(self) {
        if let Some(Some(d)) = self.drag.try_get_untracked() {
            let _ = self.drag.try_set(None);
            if d.moved {
                self.actions.on_save.run(false);
            }
        }
    }

    // ── history ──────────────────────────────────────────────────────────────

    /// Undo / redo replace the label lists wholesale, so no index survives
    /// them. While a ring is being placed point by point, undo takes back the
    /// last vertex (and the ring itself with its first one) instead.
    pub(super) fn undo(self) {
        if let Some(Some(d)) = self.poly.draft.try_get_untracked() {
            if d.points.len() <= 1 {
                let _ = self.poly.draft.try_set(None);
            } else {
                let _ = self.poly.draft.try_update(|slot| {
                    if let Some(d) = slot.as_mut() {
                        d.points.pop();
                    }
                });
            }
            return;
        }
        if self.editing() {
            return;
        }
        self.clear_selection();
        self.actions.on_undo.run(());
    }

    pub(super) fn redo(self) {
        if self.editing() {
            return;
        }
        self.clear_selection();
        self.actions.on_redo.run(());
    }

    /// Remove every label of the image as one undoable step.
    pub(super) fn clear_all(self) {
        self.clear_selection();
        self.actions.on_clear_labels.run(());
    }

    // ── view ─────────────────────────────────────────────────────────────────

    /// Zoom about the viewport centre (the toolbar buttons and keys).
    pub(super) fn zoom_in(self) {
        self.view.zoom_in();
    }

    pub(super) fn zoom_out(self) {
        self.view.zoom_out();
    }

    pub(super) fn zoom_reset(self) {
        self.view.reset();
    }

    /// Space is up (or the window lost focus with it held): panning is
    /// disarmed so it never sticks after an alt-tab.
    pub(super) fn release_space(self) {
        let _ = self.view.space_held.try_set(false);
    }

    // ── keyboard ─────────────────────────────────────────────────────────────

    /// Run a keyboard command.
    pub(super) fn run(self, cmd: Command) {
        match cmd {
            Command::Undo => self.undo(),
            Command::Redo => self.redo(),
            Command::Nudge(dx, dy) => self.nudge(dx, dy),
            Command::Step(step) => self.actions.on_step_image.run(step),
            Command::ZoomIn => self.zoom_in(),
            Command::ZoomOut => self.zoom_out(),
            Command::ZoomReset => self.zoom_reset(),
            Command::Pan => {
                let _ = self.view.space_held.try_set(true);
            }
            Command::Delete => self.delete(),
        }
    }
}
