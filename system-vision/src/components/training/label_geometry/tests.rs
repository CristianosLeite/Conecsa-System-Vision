// SPDX-FileCopyrightText: 2026 Conecsa
//
// SPDX-License-Identifier: AGPL-3.0-only

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

// ── polygons ──────────────────────────────────────────────────────────────────

const SQUARE: [Pt; 4] = [[0.1, 0.1], [0.5, 0.1], [0.5, 0.5], [0.1, 0.5]];
/// A "U" open at the bottom: concave, with a notch between x 0.3 and 0.7.
const U_SHAPE: [Pt; 8] = [
    [0.1, 0.1],
    [0.9, 0.1],
    [0.9, 0.9],
    [0.7, 0.9],
    [0.7, 0.3],
    [0.3, 0.3],
    [0.3, 0.9],
    [0.1, 0.9],
];

#[wasm_bindgen_test]
fn vertex_hits_are_measured_in_rendered_pixels() {
    let ring = [[0.5, 0.5], [0.9, 0.5], [0.9, 0.9]];
    let canvas = (1000.0, 500.0);
    assert_eq!(hit_vertex(&ring, [0.505, 0.5], canvas), Some(0)); // 5 px
    assert_eq!(hit_vertex(&ring, [0.51, 0.5], canvas), None); // 10 px
    // The same normalized offset is fewer pixels on the shorter axis.
    assert_eq!(hit_vertex(&ring, [0.5, 0.515], canvas), Some(0)); // 7.5 px
}

#[wasm_bindgen_test]
fn edge_hits_return_the_edge_and_the_point_on_it() {
    let canvas = (100.0, 100.0);
    let (edge, q) = hit_edge(&SQUARE, [0.3, 0.13], canvas).expect("3 px from the top edge");
    assert_eq!(edge, 0);
    approx(q[0], 0.3);
    approx(q[1], 0.1);
    assert!(hit_edge(&SQUARE, [0.3, 0.3], canvas).is_none());
    // The closing edge (last vertex back to the first) counts too.
    assert_eq!(hit_edge(&SQUARE, [0.12, 0.3], canvas).map(|(e, _)| e), Some(3));
}

#[wasm_bindgen_test]
fn a_ring_closes_on_its_first_vertex_after_three_points() {
    let canvas = (200.0, 200.0);
    let draft = [[0.1, 0.1], [0.5, 0.1], [0.5, 0.5]];
    assert!(closes_ring(&draft, [0.12, 0.1], canvas)); // 4 px
    assert!(!closes_ring(&draft, [0.2, 0.1], canvas));
    assert!(!closes_ring(&draft[..2], [0.1, 0.1], canvas));
}

#[wasm_bindgen_test]
fn vertices_are_inserted_after_the_edge_start() {
    let mut ring = SQUARE.to_vec();
    insert_vertex(&mut ring, 0, [0.3, 0.1]);
    assert_eq!(ring[1], [0.3, 0.1]);
    insert_vertex(&mut ring, 4, [0.1, 0.3]);
    assert_eq!(ring.last(), Some(&[0.1, 0.3]));
    assert_eq!(ring.len(), 6);
}

#[wasm_bindgen_test]
fn freehand_samples_every_four_pixels() {
    let canvas = (100.0, 100.0);
    assert!(!far_enough([0.0, 0.0], [0.03, 0.0], canvas));
    assert!(far_enough([0.0, 0.0], [0.04, 0.0], canvas));
}

#[wasm_bindgen_test]
fn rdp_drops_collinear_samples_and_keeps_corners() {
    let canvas = (100.0, 100.0);
    let trace = [[0.0, 0.0], [0.25, 0.001], [0.5, 0.0], [0.5, 0.5], [0.5, 1.0]];
    assert_eq!(
        simplify_rdp(&trace, canvas, RDP_EPS_PX),
        vec![[0.0, 0.0], [0.5, 0.0], [0.5, 1.0]]
    );
    assert_eq!(simplify_rdp(&trace[..2], canvas, RDP_EPS_PX).len(), 2);
}

#[wasm_bindgen_test]
fn crossing_edges_are_detected() {
    assert!(segments_cross([0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]));
    assert!(!segments_cross([0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 1.0]), "touching ends");
    assert!(!segments_cross([0.0, 0.0], [1.0, 0.0], [0.0, 0.5], [1.0, 0.5]), "parallel");
    let bowtie = [[0.1, 0.1], [0.5, 0.5], [0.5, 0.1], [0.1, 0.5]];
    assert!(self_intersects(&bowtie));
    assert!(!self_intersects(&SQUARE));
    assert!(!self_intersects(&U_SHAPE));
}

#[wasm_bindgen_test]
fn only_simple_rings_with_area_are_valid() {
    assert!(valid_ring(&SQUARE));
    assert!(valid_ring(&U_SHAPE));
    assert!(!valid_ring(&SQUARE[..2]));
    assert!(!valid_ring(&[[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]]), "collinear");
    assert!(!valid_ring(&[[0.1, 0.1], [0.5, 0.5], [0.5, 0.1], [0.1, 0.5]]), "bowtie");
}

#[wasm_bindgen_test]
fn rings_are_oriented_clockwise_with_y_down() {
    assert!(signed_area(&SQUARE) > 0.0);
    let reversed: Vec<Pt> = SQUARE.iter().rev().copied().collect();
    assert!(signed_area(&reversed) < 0.0);
    assert!(signed_area(&clockwise(reversed)) > 0.0);
}

#[wasm_bindgen_test]
fn a_moved_ring_stays_inside_the_image() {
    let moved = translate_ring(&SQUARE, 0.8, -0.3);
    let (l, t, r, b) = ring_bbox(&moved);
    approx(r, 1.0);
    approx(t, 0.0);
    approx(r - l, 0.4);
    approx(b - t, 0.4);
}

#[wasm_bindgen_test]
fn a_box_promotes_to_a_clockwise_rectangle() {
    let ring = box_ring(0.5, 0.5, 0.2, 0.4);
    assert_eq!(ring.len(), 4);
    assert!(signed_area(&ring) > 0.0);
    let (l, t, r, b) = ring_bbox(&ring);
    approx(l, 0.4);
    approx(t, 0.3);
    approx(r, 0.6);
    approx(b, 0.7);
}

#[wasm_bindgen_test]
fn a_vertex_drag_is_clamped_and_a_body_drag_moves_the_ring() {
    let mut drag = PolyDrag {
        idx: 0,
        kind: PolyDragKind::Vertex(2),
        origin: [0.5, 0.5],
        start: SQUARE.to_vec(),
        moved: false,
    };
    let ring = apply_poly_drag(&drag, [1.4, 0.7]);
    assert_eq!(ring[2], [1.0, 0.7]);
    assert_eq!(ring[0], SQUARE[0]);
    drag.kind = PolyDragKind::Move;
    let moved = apply_poly_drag(&drag, [0.6, 0.45]);
    approx(moved[0][0], 0.2);
    approx(moved[0][1], 0.05);
    assert_eq!(moved.len(), SQUARE.len());
}

#[wasm_bindgen_test]
fn closing_merges_near_duplicate_vertices() {
    let canvas = (100.0, 100.0);
    // A double-click places the last vertex twice; a trace ends on its start.
    let ring = [[0.1, 0.1], [0.5, 0.1], [0.5, 0.5], [0.5, 0.51], [0.105, 0.1]];
    assert_eq!(
        dedupe_ring(&ring, canvas, MIN_VERTEX_GAP_PX),
        vec![[0.1, 0.1], [0.5, 0.1], [0.5, 0.5]]
    );
    assert_eq!(dedupe_ring(&SQUARE, canvas, MIN_VERTEX_GAP_PX), SQUARE.to_vec());
}
