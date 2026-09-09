//! Unit tests for the pure label-editor geometry (headless browser).
use super::*;
use wasm_bindgen_test::*;

fn approx(a: f32, b: f32) {
    assert!((a - b).abs() < 1e-5, "{a} != {b}");
}

#[wasm_bindgen_test]
fn corner_cursor_classes() {
    assert_eq!(Corner::Nw.cursor_class(), "ui-label-handle-nwse");
    assert_eq!(Corner::Se.cursor_class(), "ui-label-handle-nwse");
    assert_eq!(Corner::Ne.cursor_class(), "ui-label-handle-nesw");
    assert_eq!(Corner::Sw.cursor_class(), "ui-label-handle-nesw");
}

#[wasm_bindgen_test]
fn edges_round_trips_through_from_edges() {
    let (cx, cy, w, h) = (0.5, 0.5, 0.2, 0.4);
    let (l, t, r, b) = edges(cx, cy, w, h);
    approx(l, 0.4);
    approx(t, 0.3);
    approx(r, 0.6);
    approx(b, 0.7);
    let (cx2, cy2, w2, h2) = from_edges(l, t, r, b);
    approx(cx2, cx);
    approx(cy2, cy);
    approx(w2, w);
    approx(h2, h);
}

#[wasm_bindgen_test]
fn from_edges_is_order_agnostic() {
    // Passing corners swapped still yields a positive-size box.
    let (cx, cy, w, h) = from_edges(0.6, 0.7, 0.4, 0.3);
    approx(cx, 0.5);
    approx(cy, 0.5);
    approx(w, 0.2);
    approx(h, 0.4);
}

fn move_drag(origin: (f32, f32), start: (f32, f32, f32, f32)) -> BoxDrag {
    BoxDrag {
        idx: 0,
        kind: DragKind::Move,
        origin,
        start,
        moved: false,
    }
}

#[wasm_bindgen_test]
fn apply_drag_move_keeps_size_and_translates() {
    let d = move_drag((0.5, 0.5), (0.5, 0.5, 0.2, 0.2));
    let (cx, cy, w, h) = apply_drag(&d, 0.6, 0.55);
    approx(w, 0.2);
    approx(h, 0.2);
    approx(cx, 0.6);
    approx(cy, 0.55);
}

#[wasm_bindgen_test]
fn apply_drag_move_clamps_inside_canvas() {
    let d = move_drag((0.5, 0.5), (0.9, 0.9, 0.2, 0.2));
    // Drag far past the bottom-right edge; box stays fully inside [0,1].
    let (cx, cy, w, h) = apply_drag(&d, 2.0, 2.0);
    let (l, t, r, b) = edges(cx, cy, w, h);
    assert!(l >= -1e-6 && t >= -1e-6);
    assert!(r <= 1.0 + 1e-6 && b <= 1.0 + 1e-6);
    approx(w, 0.2);
    approx(h, 0.2);
}

#[wasm_bindgen_test]
fn apply_drag_resize_se_enforces_min_size() {
    let d = BoxDrag {
        idx: 0,
        kind: DragKind::Resize(Corner::Se),
        origin: (0.5, 0.5),
        start: (0.5, 0.5, 0.4, 0.4), // edges l=0.3, t=0.3, r=0.7, b=0.7
        moved: false,
    };
    // Drag the SE corner back onto the NW corner -> clamped to MIN_SIZE.
    let (_cx, _cy, w, h) = apply_drag(&d, 0.0, 0.0);
    assert!(w >= MIN_SIZE - 1e-6);
    assert!(h >= MIN_SIZE - 1e-6);
    approx(w, MIN_SIZE);
    approx(h, MIN_SIZE);
}

#[wasm_bindgen_test]
fn label_anchor_sits_above_the_box() {
    let (x, y) = label_anchor(100.0, 200.0, 300.0, VIEW);
    approx(x, 100.0);
    approx(y, 200.0 - LABEL_TEXT_GAP);
}

#[wasm_bindgen_test]
fn label_anchor_flips_below_a_box_at_the_top_edge() {
    let (x, y) = label_anchor(100.0, 0.0, 300.0, VIEW);
    approx(x, 100.0);
    approx(y, 300.0 + LABEL_TEXT_H - LABEL_TEXT_GAP);
    // Just short of the room needed above → still below.
    let (_, y) = label_anchor(100.0, LABEL_TEXT_H - 1.0, 300.0, VIEW);
    approx(y, 300.0 + LABEL_TEXT_H - LABEL_TEXT_GAP);
}

#[wasm_bindgen_test]
fn label_anchor_falls_back_inside_a_full_height_box() {
    let (x, y) = label_anchor(100.0, 0.0, VIEW, VIEW);
    approx(x, 100.0 + LABEL_TEXT_INSET.0);
    approx(y, LABEL_TEXT_INSET.1);
}

#[wasm_bindgen_test]
fn handle_rects_center_on_the_corners_of_a_large_box() {
    let (x, y, r, b) = (100.0, 100.0, 300.0, 250.0);
    let hs = handle_rects(x, y, r, b, (VIEW, VIEW));
    let half = HANDLE / 2.0;
    assert!(matches!(hs[0].2, Corner::Nw));
    approx(hs[0].0, x - half);
    approx(hs[0].1, y - half);
    assert!(matches!(hs[1].2, Corner::Ne));
    approx(hs[1].0, r - half);
    approx(hs[1].1, y - half);
    assert!(matches!(hs[2].2, Corner::Sw));
    approx(hs[2].0, x - half);
    approx(hs[2].1, b - half);
    assert!(matches!(hs[3].2, Corner::Se));
    approx(hs[3].0, r - half);
    approx(hs[3].1, b - half);
}

#[wasm_bindgen_test]
fn handle_rects_step_outside_a_small_box() {
    // A box narrower than three handles: every handle lies fully outside it,
    // touching the corner, so the body stays visible and grabbable.
    let (x, y, r, b) = (200.0, 200.0, 210.0, 260.0);
    let hs = handle_rects(x, y, r, b, (VIEW, VIEW));
    approx(hs[0].0 + HANDLE, x);
    approx(hs[0].1 + HANDLE, y);
    approx(hs[1].0, r);
    approx(hs[1].1 + HANDLE, y);
    approx(hs[2].0 + HANDLE, x);
    approx(hs[2].1, b);
    approx(hs[3].0, r);
    approx(hs[3].1, b);
}

#[wasm_bindgen_test]
fn handle_rects_stay_inside_the_canvas() {
    // Box flush with the canvas edges: handles are clamped, never clipped away.
    let hs = handle_rects(0.0, 0.0, VIEW, VIEW, (VIEW, VIEW));
    for (hx, hy, _) in hs {
        assert!(hx >= 0.0 && hx + HANDLE <= VIEW, "{hx}");
        assert!(hy >= 0.0 && hy + HANDLE <= VIEW, "{hy}");
    }
    approx(hs[0].0, 0.0);
    approx(hs[3].0, VIEW - HANDLE);
}
