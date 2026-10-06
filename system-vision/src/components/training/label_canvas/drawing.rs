// SPDX-FileCopyrightText: 2026 Conecsa
//
// SPDX-License-Identifier: AGPL-3.0-only

//! Pointer handling of the drawing surface: SAM point prompts, box drawing
//! and editing, and the dispatch to the polygon tool (`rings`).
//!
//! The svg-level handlers see only true background presses (on the svg
//! itself); presses on a shape are reported by that element's own handler
//! through [`Drawing::box_press`] or the polygon tool's `press`. In SAM mode
//! the shapes are pointer-transparent, so every press reaches the svg and
//! becomes a point prompt. Pointer (not mouse) events so touch and pen work
//! too: `PointerEvent` derefs to `MouseEvent`, so the geometry helpers and
//! the shift/ctrl modifiers are unchanged.

use leptos::prelude::*;

use crate::api::LabelBox;

use super::super::dataset_editor::{ClassesState, DrawMode};
use super::super::label_editor::PolygonTool;
use super::super::label_geometry::{
    apply_drag, box_ring, capture_pointer, is_background, norm_coords, BoxDrag, Corner, DragKind,
    DRAG_EPS, MIN_DRAW_PX, MIN_SIZE,
};
use super::has_class;
use super::rings::Rings;

/// The canvas's pointer state and everything its handlers touch.
#[derive(Clone, Copy)]
pub(super) struct Drawing {
    /// The open image; nothing draws without one.
    pub(super) selected: ReadSignal<Option<String>>,
    pub(super) boxes: RwSignal<Vec<LabelBox>>,
    pub(super) selected_box: RwSignal<Option<usize>>,
    pub(super) drag: RwSignal<Option<BoxDrag>>,
    /// Draft rectangle while drawing a new box (or a rectangle ring),
    /// normalized (x0, y0, x1, y1).
    pub(super) draft: RwSignal<Option<(f32, f32, f32, f32)>>,
    pub(super) poly: PolygonTool,
    pub(super) rings: Rings,
    pub(super) classes: ClassesState,
    /// The on-screen size of the zoomed image, in CSS px.
    pub(super) canvas: Memo<(f32, f32)>,
    /// SAM prompt mode: background clicks become point prompts.
    pub(super) sam_mode: Signal<bool>,
    pub(super) sam_points: RwSignal<Vec<(f32, f32, bool)>>,
    pub(super) on_save: Callback<bool>,
    pub(super) on_need_class: Callback<()>,
}

impl Drawing {
    fn arm_drag(self, idx: usize, kind: DragKind, origin: (f32, f32)) {
        if let Some(bx) = self.boxes.get_untracked().get(idx) {
            self.drag.set(Some(BoxDrag {
                idx,
                kind,
                origin,
                start: (bx.cx, bx.cy, bx.w, bx.h),
                moved: false,
            }));
        }
    }

    /// Reported by a box rect (corner = None: select + move) or one of its
    /// resize handles (Some(corner)) when pressed, with the normalized click.
    pub(super) fn box_press(self, idx: usize, corner: Option<Corner>, origin: (f32, f32)) {
        match corner {
            None => {
                self.selected_box.set(Some(idx));
                self.arm_drag(idx, DragKind::Move, origin);
            }
            Some(corner) => self.arm_drag(idx, DragKind::Resize(corner), origin),
        }
    }

    /// A press on the svg background.
    pub(super) fn press(self, ev: leptos::ev::PointerEvent) {
        if self.selected.get_untracked().is_none() || !is_background(&ev) {
            return;
        }
        let Some((x, y)) = norm_coords(&ev) else {
            return;
        };
        ev.prevent_default();
        if self.sam_mode.get_untracked() {
            let positive = !(ev.shift_key() || ev.ctrl_key());
            self.sam_points.update(|p| p.push((x, y, positive)));
            return;
        }
        if !self.poly.enabled {
            // Background: clear selection and start drawing a new box.
            self.selected_box.set(None);
            self.draft.set(Some((x, y, x, y)));
            return;
        }
        match self.poly.mode.get_untracked() {
            DrawMode::Click => self.rings.place_point([x, y]),
            DrawMode::Freehand => {
                if self.rings.start_trace([x, y]) {
                    capture_pointer(&ev);
                }
            }
            DrawMode::Rect => {
                self.rings.deselect();
                self.draft.set(Some((x, y, x, y)));
                capture_pointer(&ev);
            }
        }
    }

    pub(super) fn motion(self, ev: leptos::ev::PointerEvent) {
        let active_drag = self.drag.get_untracked();
        if active_drag.is_none() && self.draft.get_untracked().is_none() && !self.rings.active() {
            return;
        }
        let Some((x, y)) = norm_coords(&ev) else {
            return;
        };
        if let Some(mut d) = active_drag {
            let (cx, cy, w, h) = apply_drag(&d, x, y);
            let idx = d.idx;
            self.boxes.update(|bs| {
                if let Some(b) = bs.get_mut(idx) {
                    b.cx = cx;
                    b.cy = cy;
                    b.w = w;
                    b.h = h;
                }
            });
            if !d.moved
                && ((x - d.origin.0).abs() > DRAG_EPS || (y - d.origin.1).abs() > DRAG_EPS)
            {
                d.moved = true;
                self.drag.set(Some(d));
            }
            return;
        }
        if self.rings.motion([x, y]) {
            return;
        }
        self.draft.update(|d| {
            if let Some(d) = d.as_mut() {
                d.2 = x;
                d.3 = y;
            }
        });
    }

    pub(super) fn release(self) {
        // Finishing a move/resize: persist only if the box actually changed
        // (a plain click on a box just selects it).
        if let Some(d) = self.drag.get_untracked() {
            self.drag.set(None);
            if d.moved {
                self.on_save.run(false);
            }
            return;
        }
        if self.rings.release() {
            return;
        }
        let Some((x0, y0, x1, y1)) = self.draft.get_untracked() else {
            return;
        };
        self.draft.set(None);
        let (w, h) = ((x1 - x0).abs(), (y1 - y0).abs());
        // Ignore accidental clicks: the guard is a few *screen* pixels, not a
        // share of the image, so tiny objects on a native-resolution frame
        // can still be labeled.
        let (cw, ch) = self.canvas.get_untracked();
        if w * cw < MIN_DRAW_PX || h * ch < MIN_DRAW_PX || w < MIN_SIZE || h < MIN_SIZE {
            return;
        }
        let (cx, cy) = ((x0 + x1) / 2.0, (y0 + y1) / 2.0);
        if self.poly.enabled {
            // Rectangle mode: a box promoted to a four-vertex ring.
            self.rings.add(box_ring(cx, cy, w, h));
            return;
        }
        // A box needs a real class id, or set_labels rejects the save with
        // "Unknown class id". Block drawing until a class exists.
        if !has_class(self.classes) {
            self.on_need_class.run(());
            return;
        }
        self.boxes.update(|bs| {
            bs.push(LabelBox {
                class_id: self.classes.active.get_untracked() as u32,
                cx,
                cy,
                w,
                h,
            });
        });
        self.on_save.run(false);
    }

    /// The drawing gesture is abandoned: drafts dropped, an edit put back as
    /// it was. Reached when the browser took the pointer (a system gesture, a
    /// palm) and when a second finger turns the gesture into a pinch.
    pub(super) fn abort(self) {
        self.draft.set(None);
        self.rings.abort();
        if let Some(d) = self.drag.get_untracked() {
            self.drag.set(None);
            self.boxes.update(|bs| {
                if let Some(b) = bs.get_mut(d.idx) {
                    (b.cx, b.cy, b.w, b.h) = d.start;
                }
            });
        }
    }

    /// Leaving the canvas: an uncaptured box draft is dropped (captured
    /// gestures never leave), and the rubber band hides.
    pub(super) fn leave(self) {
        if !self.poly.enabled {
            self.draft.set(None);
        }
        self.rings.hide_hover();
    }
}
