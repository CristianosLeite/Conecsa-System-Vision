// SPDX-FileCopyrightText: 2026 Conecsa
//
// SPDX-License-Identifier: AGPL-3.0-only

//! Pure geometry + interaction primitives shared by the label-editor pieces.
//!
//! No Leptos/DOM state here beyond reading a `MouseEvent` — just the math for
//! YOLO boxes (normalized cx/cy/w/h in 0..1 relative to the stored image,
//! drawn on a viewBox sized in CSS pixels to the rendered canvas, so strokes,
//! handles and text keep a fixed on-screen size) and the move/resize drag
//! model. Nothing here assumes a square image.

use wasm_bindgen::JsCast;

/// Fallback canvas side (px) used for the overlay's viewBox until the rendered
/// canvas has been measured. The overlay draws in CSS pixels: the viewBox is
/// set to the canvas's own width/height (with `preserveAspectRatio="none"`, so
/// a lagging measurement still stretches over the whole image), and normalized
/// coordinates map onto it by multiplying with the canvas size — it is a
/// coordinate space, not the image size (native-resolution datasets are 16:9).
pub(super) const VIEW: f32 = 640.0;
/// Minimum normalized box side enforced by a resize. Deliberately tiny (≈2.5 px
/// of a 1280 px frame): small objects are the point of native-resolution
/// datasets, and a resize is an intentional drag, not an accidental click.
pub(super) const MIN_SIZE: f32 = 0.002;
/// Minimum on-screen side (px) for a drawn draft to become a box; anything
/// smaller is treated as an accidental click on the background.
pub(super) const MIN_DRAW_PX: f32 = 4.0;
/// Resize-handle side in CSS pixels — fixed on screen whatever the canvas size,
/// and small so it does not cover the object being fitted (see `handle_rects`
/// for the small-box placement).
pub(super) const HANDLE: f32 = 8.0;
/// Height reserved for a box's class-name text, in px (12px font plus its
/// stroke).
pub(super) const LABEL_TEXT_H: f32 = 14.0;
/// Gap between the class-name baseline and the box edge, in px.
pub(super) const LABEL_TEXT_GAP: f32 = 3.0;
/// Inset of the class name from the box's top-left corner (x, baseline y) in
/// px, used only when there is no room outside the box.
pub(super) const LABEL_TEXT_INSET: (f32, f32) = (4.0, 12.0);
/// Pointer travel (normalized) past which a box mousedown counts as a drag
/// rather than a plain select — below it, no save fires.
pub(super) const DRAG_EPS: f32 = 0.003;

/// The four corners, in handle render order.
pub(super) const CORNERS: [Corner; 4] = [Corner::Nw, Corner::Ne, Corner::Sw, Corner::Se];

#[derive(Clone, Copy, PartialEq)]
pub(super) enum Corner {
    Nw,
    Ne,
    Sw,
    Se,
}

impl Corner {
    /// CSS modifier class carrying the right resize cursor.
    pub(super) fn cursor_class(self) -> &'static str {
        match self {
            Corner::Nw | Corner::Se => "ui-label-handle-nwse",
            Corner::Ne | Corner::Sw => "ui-label-handle-nesw",
        }
    }
}

#[derive(Clone, Copy)]
pub(super) enum DragKind {
    Move,
    Resize(Corner),
}

/// An in-progress move/resize of an existing committed box.
#[derive(Clone, Copy)]
pub(super) struct BoxDrag {
    pub(super) idx: usize,
    pub(super) kind: DragKind,
    pub(super) origin: (f32, f32),          // pointer (normalized) at mousedown
    pub(super) start: (f32, f32, f32, f32), // box (cx, cy, w, h) at mousedown
    pub(super) moved: bool,                 // crossed DRAG_EPS → a real edit, will save
}

/// Mouse position normalized to 0..1 within the SVG canvas. Resolves the
/// enclosing `<svg>` from `event.target` (the real element under the cursor —
/// a child `<rect>`, handle, or the svg itself), so the scale is measured
/// against the full canvas. X and Y are normalized independently against the
/// svg's own width and height, so a non-square canvas needs no extra
/// correction. NB: `current_target` is unusable here —
/// Leptos delegates events on `window`, so it is the window, not the svg.
pub(super) fn norm_coords(ev: &leptos::ev::MouseEvent) -> Option<(f32, f32)> {
    let target: web_sys::Element = ev.target()?.dyn_into().ok()?;
    let svg = target.closest("svg").ok().flatten()?;
    let rect = svg.get_bounding_client_rect();
    if rect.width() <= 0.0 || rect.height() <= 0.0 {
        return None;
    }
    let x = ((ev.client_x() as f64 - rect.left()) / rect.width()).clamp(0.0, 1.0);
    let y = ((ev.client_y() as f64 - rect.top()) / rect.height()).clamp(0.0, 1.0);
    Some((x as f32, y as f32))
}

/// True when the click landed on the `<svg>` canvas itself (empty background),
/// not on a child `<rect>`. Box/handle rects carry their own on:mousedown that
/// captures their index in Rust — DOM `data-*` attributes are unreliable here
/// (Leptos mangles `attr:data-foo` to a literal `attr:data-foo` name on SVG).
pub(super) fn is_background(ev: &leptos::ev::MouseEvent) -> bool {
    ev.target()
        .and_then(|t| t.dyn_into::<web_sys::Element>().ok())
        .map(|e| e.tag_name().eq_ignore_ascii_case("svg"))
        .unwrap_or(false)
}

/// Baseline (x, y) for a box's class name, given its left/top/bottom edges and
/// the canvas height, all in px. The name sits *outside* the box so it never
/// covers what is being labeled: above it, flush with the left edge; below it
/// when the box touches the top of the canvas; and inside the top-left corner
/// only when neither side has room (the box spans the full height).
pub(super) fn label_anchor(x: f32, y: f32, bottom: f32, canvas_h: f32) -> (f32, f32) {
    if y - LABEL_TEXT_H >= 0.0 {
        (x, y - LABEL_TEXT_GAP)
    } else if bottom + LABEL_TEXT_H <= canvas_h {
        (x, bottom + LABEL_TEXT_H - LABEL_TEXT_GAP)
    } else {
        (x + LABEL_TEXT_INSET.0, y + LABEL_TEXT_INSET.1)
    }
}

/// Top-left corner of each resize handle (a `HANDLE`-sided square) for a box
/// with the given edges, in px, in `CORNERS` order. Handles are centered on the
/// corners of a box at least `3 * HANDLE` wide and tall; a smaller box would
/// vanish under them, so its handles step diagonally outward until their inner
/// edge touches the corner, leaving the body visible and grabbable. Every
/// handle is then clamped inside the canvas so a corner on the edge keeps a
/// reachable handle.
pub(super) fn handle_rects(
    x: f32,
    y: f32,
    right: f32,
    bottom: f32,
    canvas: (f32, f32),
) -> [(f32, f32, Corner); 4] {
    let small = right - x < 3.0 * HANDLE || bottom - y < 3.0 * HANDLE;
    // Offset from the corner to the handle's top-left, per side: centered
    // (half a handle back) or fully outside (a whole handle back / none).
    let (before, after) = if small {
        (HANDLE, 0.0)
    } else {
        (HANDLE / 2.0, HANDLE / 2.0)
    };
    let (max_x, max_y) = ((canvas.0 - HANDLE).max(0.0), (canvas.1 - HANDLE).max(0.0));
    let origins = [
        (x - before, y - before),
        (right - after, y - before),
        (x - before, bottom - after),
        (right - after, bottom - after),
    ];
    let mut handles = [(0.0, 0.0, Corner::Nw); 4];
    for (slot, (corner, (hx, hy))) in handles.iter_mut().zip(CORNERS.into_iter().zip(origins)) {
        *slot = (hx.clamp(0.0, max_x), hy.clamp(0.0, max_y), corner);
    }
    handles
}

/// (left, top, right, bottom) of a YOLO box.
fn edges(cx: f32, cy: f32, w: f32, h: f32) -> (f32, f32, f32, f32) {
    (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)
}

/// YOLO (cx, cy, w, h) from edges (order-agnostic).
fn from_edges(l: f32, t: f32, r: f32, b: f32) -> (f32, f32, f32, f32) {
    let (l, r) = (l.min(r), l.max(r));
    let (t, b) = (t.min(b), t.max(b));
    ((l + r) / 2.0, (t + b) / 2.0, r - l, b - t)
}

/// New box geometry for a drag at pointer `(mx, my)`. Move keeps the size and
/// clamps the box inside `[0,1]`; Resize anchors the opposite corner and enforces
/// a minimum side.
pub(super) fn apply_drag(d: &BoxDrag, mx: f32, my: f32) -> (f32, f32, f32, f32) {
    let (cx, cy, w, h) = d.start;
    let (l, t, r, b) = edges(cx, cy, w, h);
    match d.kind {
        DragKind::Move => {
            let nl = (l + (mx - d.origin.0)).clamp(0.0, 1.0 - w);
            let nt = (t + (my - d.origin.1)).clamp(0.0, 1.0 - h);
            (nl + w / 2.0, nt + h / 2.0, w, h)
        }
        DragKind::Resize(corner) => {
            let mx = mx.clamp(0.0, 1.0);
            let my = my.clamp(0.0, 1.0);
            let (nl, nt, nr, nb) = match corner {
                Corner::Nw => (mx.min(r - MIN_SIZE), my.min(b - MIN_SIZE), r, b),
                Corner::Ne => (l, my.min(b - MIN_SIZE), mx.max(l + MIN_SIZE), b),
                Corner::Sw => (mx.min(r - MIN_SIZE), t, r, my.max(t + MIN_SIZE)),
                Corner::Se => (l, t, mx.max(l + MIN_SIZE), my.max(t + MIN_SIZE)),
            };
            from_edges(nl, nt, nr, nb)
        }
    }
}

// ── polygons (segmentation) ──────────────────────────────────────────────────
//
// Vertices are stored normalized (0..1 of the stored image) like boxes, but
// every hit test and sampling distance is measured in rendered pixels, so a
// handle is as easy to grab on a phone as on a wide monitor and on a 16:9
// image as on a square one.

/// A normalized polygon vertex `[x, y]` (0..1 of the stored image).
pub(super) type Pt = [f32; 2];

/// Vertex hit radius, in rendered px.
pub(super) const VERTEX_HIT_PX: f32 = 8.0;
/// Vertex handle radius as drawn, in rendered px.
pub(super) const VERTEX_R: f32 = 4.0;
/// Edge hit distance, in rendered px: a click this close inserts a vertex.
pub(super) const EDGE_HIT_PX: f32 = 6.0;
/// A click this close (rendered px) to the first vertex closes a click-placed ring.
pub(super) const CLOSE_HIT_PX: f32 = 10.0;
/// Freehand tracing samples a vertex whenever the pointer moved this far (rendered px).
pub(super) const FREEHAND_STEP_PX: f32 = 4.0;
/// Ramer–Douglas–Peucker tolerance applied to a finished freehand trace (rendered px).
pub(super) const RDP_EPS_PX: f32 = 1.5;
/// Vertices closer than this (rendered px) are merged when a ring closes: the
/// two presses of a double-click, or a trace ending where it started.
pub(super) const MIN_VERTEX_GAP_PX: f32 = 2.0;

/// Route the rest of a pointer gesture to the enclosing `<svg>`, so a drag or
/// trace that leaves the canvas keeps reporting there until it is released.
pub(super) fn capture_pointer(ev: &leptos::ev::PointerEvent) {
    let svg = ev
        .target()
        .and_then(|t| t.dyn_into::<web_sys::Element>().ok())
        .and_then(|el| el.closest("svg").ok().flatten());
    if let Some(svg) = svg {
        let _ = svg.set_pointer_capture(ev.pointer_id());
    }
}

/// The ring with consecutive vertices closer than `min_px` merged, the
/// closing pair (last back to first) included.
pub(super) fn dedupe_ring(points: &[Pt], canvas: (f32, f32), min_px: f32) -> Vec<Pt> {
    let mut out: Vec<Pt> = Vec::with_capacity(points.len());
    for p in points {
        if out.last().is_none_or(|q| distance_px(*q, *p, canvas) >= min_px) {
            out.push(*p);
        }
    }
    while out.len() > 1 && distance_px(out[0], out[out.len() - 1], canvas) < min_px {
        out.pop();
    }
    out
}

fn to_px(p: Pt, canvas: (f32, f32)) -> (f32, f32) {
    (p[0] * canvas.0, p[1] * canvas.1)
}

/// Distance between two vertices, in rendered px.
pub(super) fn distance_px(a: Pt, b: Pt, canvas: (f32, f32)) -> f32 {
    let ((ax, ay), (bx, by)) = (to_px(a, canvas), to_px(b, canvas));
    ((ax - bx).powi(2) + (ay - by).powi(2)).sqrt()
}

/// Distance (rendered px) from `p` to the segment `a`–`b`, and where along the
/// segment (0..1) the closest point lies.
pub(super) fn segment_distance_px(p: Pt, a: Pt, b: Pt, canvas: (f32, f32)) -> (f32, f32) {
    let ((px, py), (ax, ay), (bx, by)) = (to_px(p, canvas), to_px(a, canvas), to_px(b, canvas));
    let (dx, dy) = (bx - ax, by - ay);
    let len2 = dx * dx + dy * dy;
    let t = if len2 > 0.0 {
        (((px - ax) * dx + (py - ay) * dy) / len2).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let (cx, cy) = (ax + t * dx, ay + t * dy);
    (((px - cx).powi(2) + (py - cy).powi(2)).sqrt(), t)
}

/// The vertex closest to `p` within [`VERTEX_HIT_PX`].
pub(super) fn hit_vertex(ring: &[Pt], p: Pt, canvas: (f32, f32)) -> Option<usize> {
    ring.iter()
        .enumerate()
        .map(|(i, v)| (i, distance_px(*v, p, canvas)))
        .filter(|(_, d)| *d <= VERTEX_HIT_PX)
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(i, _)| i)
}

/// The edge closest to `p` within [`EDGE_HIT_PX`]: the index of its first
/// vertex and the point on the edge under the pointer.
pub(super) fn hit_edge(ring: &[Pt], p: Pt, canvas: (f32, f32)) -> Option<(usize, Pt)> {
    let n = ring.len();
    if n < 2 {
        return None;
    }
    (0..n)
        .map(|i| {
            let (a, b) = (ring[i], ring[(i + 1) % n]);
            let (d, t) = segment_distance_px(p, a, b, canvas);
            (i, d, [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])])
        })
        .filter(|(_, d, _)| *d <= EDGE_HIT_PX)
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(i, _, q)| (i, q))
}

/// Whether a click at `p` closes a click-placed draft: it has at least three
/// vertices and the click lands on the first one.
pub(super) fn closes_ring(draft: &[Pt], p: Pt, canvas: (f32, f32)) -> bool {
    draft.len() >= 3 && distance_px(draft[0], p, canvas) <= CLOSE_HIT_PX
}

/// Insert `p` on the edge that starts at vertex `edge`.
pub(super) fn insert_vertex(ring: &mut Vec<Pt>, edge: usize, p: Pt) {
    let at = (edge + 1).min(ring.len());
    ring.insert(at, p);
}

/// Whether a freehand trace has moved far enough from its last sample to add one.
pub(super) fn far_enough(last: Pt, p: Pt, canvas: (f32, f32)) -> bool {
    distance_px(last, p, canvas) >= FREEHAND_STEP_PX
}

/// Ramer–Douglas–Peucker simplification with a tolerance in rendered px.
pub(super) fn simplify_rdp(points: &[Pt], canvas: (f32, f32), eps_px: f32) -> Vec<Pt> {
    if points.len() < 3 {
        return points.to_vec();
    }
    let last = points.len() - 1;
    let mut keep = vec![false; points.len()];
    keep[0] = true;
    keep[last] = true;
    let mut stack = vec![(0usize, last)];
    while let Some((start, end)) = stack.pop() {
        if end <= start + 1 {
            continue;
        }
        let (mut index, mut max) = (start, 0.0f32);
        for (i, p) in points.iter().enumerate().take(end).skip(start + 1) {
            let (d, _) = segment_distance_px(*p, points[start], points[end], canvas);
            if d > max {
                (index, max) = (i, d);
            }
        }
        if max > eps_px {
            keep[index] = true;
            stack.push((start, index));
            stack.push((index, end));
        }
    }
    points.iter().zip(keep).filter(|(_, k)| *k).map(|(p, _)| *p).collect()
}

fn orient(a: Pt, b: Pt, c: Pt) -> f32 {
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
}

/// Whether the segments `a`–`b` and `c`–`d` properly cross (touching ends do not).
pub(super) fn segments_cross(a: Pt, b: Pt, c: Pt, d: Pt) -> bool {
    orient(a, b, c) * orient(a, b, d) < 0.0 && orient(c, d, a) * orient(c, d, b) < 0.0
}

/// Whether two non-adjacent edges of the (implicitly closed) ring cross.
pub(super) fn self_intersects(ring: &[Pt]) -> bool {
    let n = ring.len();
    if n < 4 {
        return false;
    }
    (0..n).any(|i| {
        (i + 2..n).any(|j| {
            // Edges i and j share a vertex when adjacent (including last-first).
            j - i != n - 1 && segments_cross(ring[i], ring[(i + 1) % n], ring[j], ring[(j + 1) % n])
        })
    })
}

/// Shoelace area; positive for a clockwise ring in image coordinates (y down).
pub(super) fn signed_area(ring: &[Pt]) -> f32 {
    let n = ring.len();
    (0..n)
        .map(|i| {
            let (a, b) = (ring[i], ring[(i + 1) % n]);
            a[0] * b[1] - b[0] * a[1]
        })
        .sum::<f32>()
        / 2.0
}

/// The ring oriented clockwise in image coordinates, the service's convention.
pub(super) fn clockwise(mut ring: Vec<Pt>) -> Vec<Pt> {
    if signed_area(&ring) < 0.0 {
        ring.reverse();
    }
    ring
}

/// Whether a ring can be saved: at least 3 vertices, some area, no crossing edges
/// (the service would silently re-extract a self-intersecting ring).
pub(super) fn valid_ring(ring: &[Pt]) -> bool {
    ring.len() >= 3 && signed_area(ring).abs() > 1e-7 && !self_intersects(ring)
}

/// (left, top, right, bottom) of a ring; zeros for an empty one.
pub(super) fn ring_bbox(ring: &[Pt]) -> (f32, f32, f32, f32) {
    if ring.is_empty() {
        return (0.0, 0.0, 0.0, 0.0);
    }
    ring.iter().fold((1.0f32, 1.0f32, 0.0f32, 0.0f32), |(l, t, r, b), p| {
        (l.min(p[0]), t.min(p[1]), r.max(p[0]), b.max(p[1]))
    })
}

/// Every vertex moved by `(dx, dy)`, clamped so the whole ring stays in the image.
pub(super) fn translate_ring(ring: &[Pt], dx: f32, dy: f32) -> Vec<Pt> {
    let (l, t, r, b) = ring_bbox(ring);
    let dx = dx.max(-l).min(1.0 - r);
    let dy = dy.max(-t).min(1.0 - b);
    ring.iter().map(|p| [p[0] + dx, p[1] + dy]).collect()
}

/// Where a press on a committed polygon landed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum PolyPress {
    /// Inside the ring: select it, then drag moves it.
    Body,
    /// On the edge starting at this vertex: insert a vertex there.
    Edge(usize),
    /// On this vertex: select it, then drag moves it.
    Vertex(usize),
}

/// A polygon being drawn: the vertices placed (or sampled) so far.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct PolyDraft {
    pub(super) points: Vec<Pt>,
    /// Sampled from a pointer trace, which closes on release, rather than
    /// placed by clicks.
    pub(super) freehand: bool,
    /// The pointer, for the rubber band from the last placed vertex.
    pub(super) hover: Option<Pt>,
}

/// What a drag on a committed polygon changes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum PolyDragKind {
    /// One vertex follows the pointer.
    Vertex(usize),
    /// The whole ring moves.
    Move,
}

/// An in-progress edit of a committed polygon.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct PolyDrag {
    pub(super) idx: usize,
    pub(super) kind: PolyDragKind,
    pub(super) origin: Pt,
    /// The ring at pointer-down: a cancelled or self-crossing edit restores it.
    pub(super) start: Vec<Pt>,
    /// Crossed [`DRAG_EPS`]: a real edit, saved on release.
    pub(super) moved: bool,
}

/// The ring a polygon drag produces with the pointer at `p`: the dragged vertex
/// clamped into the image, or the whole ring translated (kept inside).
pub(super) fn apply_poly_drag(d: &PolyDrag, p: Pt) -> Vec<Pt> {
    match d.kind {
        PolyDragKind::Vertex(i) => {
            let mut ring = d.start.clone();
            if let Some(v) = ring.get_mut(i) {
                *v = [p[0].clamp(0.0, 1.0), p[1].clamp(0.0, 1.0)];
            }
            ring
        }
        PolyDragKind::Move => translate_ring(&d.start, p[0] - d.origin[0], p[1] - d.origin[1]),
    }
}

/// A YOLO box as a clockwise 4-vertex ring: the promotion of a box to a polygon.
pub(super) fn box_ring(cx: f32, cy: f32, w: f32, h: f32) -> Vec<Pt> {
    let (l, t, r, b) = edges(cx, cy, w, h);
    vec![[l, t], [r, t], [r, b], [l, b]]
}

#[cfg(test)]
mod tests;
