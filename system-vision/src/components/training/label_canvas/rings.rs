// SPDX-FileCopyrightText: 2026 Conecsa
//
// SPDX-License-Identifier: AGPL-3.0-only

//! Polygon rings on the canvas of a segmentation dataset: drawing a new ring
//! (click, freehand or rectangle) and editing a committed one by vertex, edge
//! or whole-ring drag. Vertices are stored normalized, hit tests run in
//! rendered pixels, and each completed gesture is one save. A ring whose
//! edges cross is refused.

use leptos::prelude::*;
use wasm_bindgen::closure::Closure;
use wasm_bindgen::JsCast;

use crate::api::LabelPolygon;

use super::super::dataset_editor::{next_instance, ClassesState};
use super::super::label_editor::PolygonTool;
use super::super::label_geometry::{
    apply_poly_drag, clockwise, closes_ring, dedupe_ring, far_enough, hit_edge, hit_vertex,
    insert_vertex, simplify_rdp, valid_ring, PolyDraft, PolyDrag, PolyDragKind, PolyPress, Pt,
    DRAG_EPS, MIN_VERTEX_GAP_PX, RDP_EPS_PX,
};
use super::has_class;

/// The polygon tool's canvas side: the committed rings, the tool's selection
/// and edit signals, and the callbacks a finished gesture reaches.
#[derive(Clone, Copy)]
pub(super) struct Rings {
    polygons: RwSignal<Vec<LabelPolygon>>,
    poly: PolygonTool,
    classes: ClassesState,
    /// The on-screen size of the zoomed image, in CSS px.
    canvas: Memo<(f32, f32)>,
    /// Pointer moves of a drag, and the click-mode rubber band, are coalesced
    /// to one update per animation frame: only the latest position is applied.
    pending: StoredValue<Option<Pt>>,
    frame_queued: StoredValue<bool>,
    on_save: Callback<bool>,
    on_need_class: Callback<()>,
    on_refused: Callback<()>,
}

impl Rings {
    pub(super) fn new(
        polygons: RwSignal<Vec<LabelPolygon>>,
        poly: PolygonTool,
        classes: ClassesState,
        canvas: Memo<(f32, f32)>,
        on_save: Callback<bool>,
        on_need_class: Callback<()>,
        on_refused: Callback<()>,
    ) -> Self {
        Self {
            polygons,
            poly,
            classes,
            canvas,
            pending: StoredValue::new(None),
            frame_queued: StoredValue::new(false),
            on_save,
            on_need_class,
            on_refused,
        }
    }

    /// A drag or a draft is in progress.
    pub(super) fn active(self) -> bool {
        self.poly.drag.with_untracked(Option::is_some)
            || self.poly.draft.with_untracked(Option::is_some)
    }

    /// No ring and no vertex selected (a new ring is about to be drawn).
    pub(super) fn deselect(self) {
        self.poly.selected.set(None);
        self.poly.vertex.set(None);
    }

    /// A new object: one ring of the active class, selected, saved.
    pub(super) fn add(self, ring: Vec<Pt>) {
        if !has_class(self.classes) {
            self.on_need_class.run(());
            return;
        }
        let class_id = self.classes.active.get_untracked() as u32;
        let mut index = 0;
        self.polygons.update(|ps| {
            let instance = next_instance(ps);
            ps.push(LabelPolygon {
                class_id,
                instance,
                points: ring,
            });
            index = ps.len() - 1;
        });
        self.poly.selected.set(Some(index));
        self.poly.vertex.set(None);
        self.on_save.run(false);
    }

    /// Close a drawn ring: a freehand trace is simplified first; near-duplicate
    /// vertices merge; fewer than three is an accidental click, a crossing
    /// ring is refused.
    fn finish(self, points: Vec<Pt>, freehand: bool) {
        self.poly.draft.set(None);
        let px = self.canvas.get_untracked();
        let points = if freehand {
            simplify_rdp(&points, px, RDP_EPS_PX)
        } else {
            points
        };
        let points = dedupe_ring(&points, px, MIN_VERTEX_GAP_PX);
        if points.len() < 3 {
            return;
        }
        if !valid_ring(&points) {
            self.on_refused.run(());
            return;
        }
        self.add(clockwise(points));
    }

    /// Click mode: the first press starts a ring, a press on its first vertex
    /// (from the third vertex on) closes it, any other press adds a vertex.
    pub(super) fn place_point(self, p: Pt) {
        match self.poly.draft.get_untracked() {
            None => {
                if !has_class(self.classes) {
                    self.on_need_class.run(());
                    return;
                }
                self.deselect();
                self.poly.draft.set(Some(PolyDraft {
                    points: vec![p],
                    freehand: false,
                    hover: Some(p),
                }));
            }
            Some(d) if closes_ring(&d.points, p, self.canvas.get_untracked()) => {
                self.finish(d.points, false)
            }
            Some(_) => self.poly.draft.update(|slot| {
                if let Some(d) = slot.as_mut() {
                    d.points.push(p);
                }
            }),
        }
    }

    /// Freehand mode: start a trace at `p`. False (nothing started) when no
    /// class is selected.
    pub(super) fn start_trace(self, p: Pt) -> bool {
        if !has_class(self.classes) {
            self.on_need_class.run(());
            return false;
        }
        self.deselect();
        self.poly.draft.set(Some(PolyDraft {
            points: vec![p],
            freehand: true,
            hover: None,
        }));
        true
    }

    /// Apply the latest coalesced pointer position to the drag in progress,
    /// or to the click-mode rubber band.
    fn flush(self) {
        let Some(Some(p)) = self.pending.try_update_value(|slot| slot.take()) else {
            return;
        };
        if let Some(Some(d)) = self.poly.drag.try_get_untracked() {
            let ring = apply_poly_drag(&d, p);
            let idx = d.idx;
            let _ = self.polygons.try_update(|ps| {
                if let Some(pg) = ps.get_mut(idx) {
                    pg.points = ring;
                }
            });
            let past_eps =
                (p[0] - d.origin[0]).abs() > DRAG_EPS || (p[1] - d.origin[1]).abs() > DRAG_EPS;
            if !d.moved && past_eps {
                let _ = self.poly.drag.try_update(|slot| {
                    if let Some(slot) = slot.as_mut() {
                        slot.moved = true;
                    }
                });
            }
        } else {
            let _ = self.poly.draft.try_update(|slot| {
                if let Some(d) = slot.as_mut().filter(|d| !d.freehand) {
                    d.hover = Some(p);
                }
            });
        }
    }

    fn schedule(self, p: Pt) {
        self.pending.set_value(Some(p));
        if self.frame_queued.get_value() {
            return;
        }
        let Some(window) = web_sys::window() else {
            self.flush();
            return;
        };
        self.frame_queued.set_value(true);
        let tick = Closure::once_into_js(move || {
            let _ = self.frame_queued.try_update_value(|queued| *queued = false);
            self.flush();
        });
        if window.request_animation_frame(tick.unchecked_ref()).is_err() {
            self.frame_queued.set_value(false);
            self.flush();
        }
    }

    /// Reported by a committed ring (body, edge or vertex) with the normalized
    /// click. A press while placing points places one there instead.
    pub(super) fn press(self, idx: usize, what: PolyPress, p: Pt) {
        if self.poly.draft.get_untracked().is_some_and(|d| !d.freehand) {
            self.place_point(p);
            return;
        }
        let Some(ring) = self
            .polygons
            .with_untracked(|ps| ps.get(idx).map(|pg| pg.points.clone()))
        else {
            return;
        };
        let px = self.canvas.get_untracked();
        let was_selected = self.poly.selected.get_untracked() == Some(idx);
        self.poly.selected.set(Some(idx));
        let vertex = match what {
            PolyPress::Vertex(v) => Some(v),
            PolyPress::Body if was_selected => hit_vertex(&ring, p, px),
            _ => None,
        };
        if let Some(v) = vertex {
            self.poly.vertex.set(Some(v));
            self.poly.drag.set(Some(PolyDrag {
                idx,
                kind: PolyDragKind::Vertex(v),
                origin: p,
                start: ring,
                moved: false,
            }));
            return;
        }
        if let (PolyPress::Edge(_), Some((edge, q))) = (what, hit_edge(&ring, p, px)) {
            // Insert a vertex under the pointer and keep dragging it; the
            // insertion alone is an edit, so the release saves.
            let mut ring = ring;
            insert_vertex(&mut ring, edge, q);
            let inserted = ring.clone();
            self.polygons.update(|ps| {
                if let Some(pg) = ps.get_mut(idx) {
                    pg.points = inserted;
                }
            });
            self.poly.vertex.set(Some(edge + 1));
            self.poly.drag.set(Some(PolyDrag {
                idx,
                kind: PolyDragKind::Vertex(edge + 1),
                origin: q,
                start: ring,
                moved: true,
            }));
            return;
        }
        self.poly.vertex.set(None);
        self.poly.drag.set(Some(PolyDrag {
            idx,
            kind: PolyDragKind::Move,
            origin: p,
            start: ring,
            moved: false,
        }));
    }

    /// The pointer moved to `p` (normalized) while a drag or draft is in
    /// progress: a drag and the click-mode rubber band follow it at frame
    /// rate, a trace samples it once it is far enough from the last vertex.
    /// False when the polygon tool has nothing in progress.
    pub(super) fn motion(self, p: Pt) -> bool {
        if self.poly.drag.with_untracked(Option::is_some) {
            self.schedule(p);
            return true;
        }
        let Some(d) = self.poly.draft.get_untracked() else {
            return false;
        };
        if !d.freehand {
            self.schedule(p);
        } else if d
            .points
            .last()
            .is_none_or(|last| far_enough(*last, p, self.canvas.get_untracked()))
        {
            self.poly.draft.update(|slot| {
                if let Some(d) = slot.as_mut() {
                    d.points.push(p);
                }
            });
        }
        true
    }

    /// Finish an edit of a committed ring: a valid one is saved, one that now
    /// crosses itself is put back as it was and refused.
    fn commit_edit(self, d: PolyDrag) {
        let Some(ring) = self
            .polygons
            .with_untracked(|ps| ps.get(d.idx).map(|pg| pg.points.clone()))
        else {
            return;
        };
        if valid_ring(&ring) {
            self.on_save.run(false);
        } else {
            self.polygons.update(|ps| {
                if let Some(pg) = ps.get_mut(d.idx) {
                    pg.points = d.start.clone();
                }
            });
            self.on_refused.run(());
        }
    }

    /// The pointer went up: a drag is judged and saved or refused, a trace
    /// closes, a click-placed ring waits for its close. False when the
    /// polygon tool had nothing in progress.
    pub(super) fn release(self) -> bool {
        if self.poly.drag.with_untracked(Option::is_some) {
            // Apply the last coalesced position before judging the edit.
            self.flush();
            if let Some(d) = self.poly.drag.get_untracked() {
                self.poly.drag.set(None);
                if d.moved {
                    self.commit_edit(d);
                }
            }
            return true;
        }
        if let Some(d) = self.poly.draft.get_untracked() {
            if d.freehand {
                self.finish(d.points, true);
            }
            return true;
        }
        false
    }

    /// The gesture is abandoned: the draft dropped, an edit put back as it was.
    pub(super) fn abort(self) {
        self.poly.draft.set(None);
        self.pending.set_value(None);
        if let Some(d) = self.poly.drag.get_untracked() {
            self.poly.drag.set(None);
            self.polygons.update(|ps| {
                if let Some(pg) = ps.get_mut(d.idx) {
                    pg.points = d.start;
                }
            });
        }
    }

    /// The pointer left the canvas: the rubber band hides.
    pub(super) fn hide_hover(self) {
        if self
            .poly
            .draft
            .with_untracked(|d| d.as_ref().is_some_and(|d| d.hover.is_some()))
        {
            self.poly.draft.update(|slot| {
                if let Some(d) = slot.as_mut() {
                    d.hover = None;
                }
            });
        }
    }

    /// A double-click closes a click-placed ring of three or more vertices.
    pub(super) fn double_click(self) {
        if let Some(d) = self.poly.draft.get_untracked()
            && !d.freehand
            && d.points.len() >= 3
        {
            self.finish(d.points, false);
        }
    }
}
