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

#[cfg(test)]
mod tests;
