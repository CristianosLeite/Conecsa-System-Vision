// SPDX-FileCopyrightText: 2026 Conecsa
//
// SPDX-License-Identifier: AGPL-3.0-only

//! Unit tests for the label request queue's coalescing rule.
use super::*;
use wasm_bindgen_test::*;

/// A write of `image` whose single box's width encodes `n`.
fn write(image: &str, n: u32, notify: bool) -> LabelOp {
    LabelOp::Write(LabelWrite {
        image_id: image.to_string(),
        boxes: vec![LabelBox {
            class_id: 0,
            cx: 0.5,
            cy: 0.5,
            w: 0.1 + n as f32 * 0.01,
            h: 0.1,
        }],
        polygons: Vec::new(),
        image_class: None,
        notify,
        epoch: 0,
    })
}

fn b(class_id: u32) -> LabelBox {
    LabelBox { class_id, cx: 0.5, cy: 0.5, w: 0.1, h: 0.1 }
}

fn ring(class_id: u32, instance: u32) -> LabelPolygon {
    LabelPolygon { class_id, instance, points: vec![[0.1, 0.1], [0.2, 0.1], [0.2, 0.2]] }
}

fn load(image: &str) -> LabelOp {
    LabelOp::Load(image.to_string())
}

fn queued(ops: Vec<LabelOp>) -> Vec<LabelOp> {
    let mut q = VecDeque::new();
    for op in ops {
        push(&mut q, op);
    }
    q.into_iter().collect()
}

#[wasm_bindgen_test]
fn a_write_behind_a_write_of_the_same_image_replaces_it() {
    assert_eq!(queued(vec![write("a", 1, false), write("a", 2, false)]), vec![write("a", 2, false)]);
}

#[wasm_bindgen_test]
fn a_replaced_write_keeps_its_toast() {
    assert_eq!(queued(vec![write("a", 1, true), write("a", 2, false)]), vec![write("a", 2, true)]);
}

#[wasm_bindgen_test]
fn writes_of_different_images_keep_their_order() {
    let ops = vec![write("a", 1, false), write("b", 2, false), write("a", 3, false)];
    assert_eq!(queued(ops.clone()), ops);
}

#[wasm_bindgen_test]
fn a_load_is_never_merged_and_orders_the_writes_around_it() {
    // A save of `a`, the load of `b`, then `a` again (back to it): the second
    // save must not jump ahead of the load, nor the load ahead of the save.
    let ops = vec![write("a", 1, false), load("b"), write("a", 2, false), load("a")];
    assert_eq!(queued(ops.clone()), ops);
}

#[wasm_bindgen_test]
fn a_class_removal_is_a_barrier_no_write_merges_across() {
    // The save after the removal carries the new numbering (or is dropped
    // when it does not): it must stay a separate request behind it.
    let ops = vec![
        write("a", 1, false),
        LabelOp::RemoveClass { index: 0, epoch: 0 },
        write("a", 2, false),
    ];
    assert_eq!(queued(ops.clone()), ops);
}

#[wasm_bindgen_test]
fn loads_are_queued_as_issued() {
    assert_eq!(queued(vec![load("a"), load("a")]), vec![load("a"), load("a")]);
}

#[wasm_bindgen_test]
fn a_removal_drops_its_class_and_moves_the_ids_above_it_down() {
    assert_eq!(shift_class(0, 1), Some(0));
    assert_eq!(shift_class(1, 1), None);
    assert_eq!(shift_class(2, 1), Some(1));
}

#[wasm_bindgen_test]
fn renumber_applies_the_rule_to_boxes_rings_and_the_image_class() {
    let mut boxes = vec![b(0), b(1), b(2)];
    let mut polygons = vec![ring(1, 7), ring(2, 8)];
    let mut class = Some(2);
    renumber(&mut boxes, &mut polygons, &mut class, 1);
    assert_eq!(boxes, vec![b(0), b(1)]);
    assert_eq!(polygons, vec![ring(1, 8)]);
    assert_eq!(class, Some(1));
    let mut gone = Some(1);
    renumber(&mut Vec::new(), &mut Vec::new(), &mut gone, 1);
    assert_eq!(gone, None);
}

#[wasm_bindgen_test]
fn a_write_is_rebased_through_every_removal_since_it_was_captured() {
    // Classes a b c d; boxes of b and d; then c (2) and a (0) are removed:
    // b ends at 0, d at 1, and the numbering is two removals newer.
    let w = LabelWrite {
        image_id: "i".to_string(),
        boxes: vec![b(1), b(3)],
        polygons: Vec::new(),
        image_class: None,
        notify: false,
        epoch: 0,
    };
    let w = rebase_write(w, &[2, 0]);
    assert_eq!(w.boxes, vec![b(0), b(1)]);
    assert_eq!(w.epoch, 2);
}

#[wasm_bindgen_test]
fn a_second_removal_confirmed_on_the_old_list_removes_the_class_picked() {
    // Classes a b c: "remove a" (0) then "remove b" (1), both confirmed on
    // the original list. After a is gone, b is at 0 — not c at 1.
    assert_eq!(rebase_index(1, &[0]), Some(0));
    // "remove a" twice: the second finds it already gone.
    assert_eq!(rebase_index(0, &[0]), None);
    // A class below the removed one keeps its index.
    assert_eq!(rebase_index(0, &[2]), Some(0));
}
