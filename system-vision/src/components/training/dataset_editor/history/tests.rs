// SPDX-FileCopyrightText: 2026 Conecsa
//
// SPDX-License-Identifier: AGPL-3.0-only

//! Unit tests for the label undo/redo history (headless browser).
use super::*;
use wasm_bindgen_test::*;

/// A snapshot with `n` boxes (their size encodes `n`, so no two are equal).
fn boxes(n: usize) -> LabelSnapshot {
    LabelSnapshot {
        boxes: (0..n)
            .map(|i| LabelBox {
                class_id: 0,
                cx: 0.5,
                cy: 0.5,
                w: 0.1 + i as f32 * 0.01,
                h: 0.1,
            })
            .collect(),
        polygons: Vec::new(),
    }
}

#[wasm_bindgen_test]
fn default_has_nothing_to_undo_or_redo() {
    let mut h = LabelHistory::default();
    assert!(!h.can_undo());
    assert!(!h.can_redo());
    assert_eq!(h.undo(), None);
    assert_eq!(h.redo(), None);
}

#[wasm_bindgen_test]
fn reset_then_record_enables_undo_only() {
    let mut h = LabelHistory::default();
    h.reset(boxes(0));
    assert!(!h.can_undo());
    h.record(boxes(1));
    assert!(h.can_undo());
    assert!(!h.can_redo());
}

#[wasm_bindgen_test]
fn undo_returns_the_baseline_and_redo_the_edit() {
    let mut h = LabelHistory::default();
    h.reset(boxes(0));
    h.record(boxes(1));
    assert_eq!(h.undo(), Some(boxes(0)));
    assert!(!h.can_undo());
    assert!(h.can_redo());
    assert_eq!(h.redo(), Some(boxes(1)));
    assert!(!h.can_redo());
}

#[wasm_bindgen_test]
fn record_after_undo_drops_the_redo_tail() {
    let mut h = LabelHistory::default();
    h.reset(boxes(0));
    h.record(boxes(1));
    h.record(boxes(2));
    h.undo();
    h.record(boxes(3));
    assert!(!h.can_redo());
    assert_eq!(h.undo(), Some(boxes(1)));
    assert_eq!(h.redo(), Some(boxes(3)));
}

#[wasm_bindgen_test]
fn an_unchanged_state_is_not_recorded() {
    let mut h = LabelHistory::default();
    h.reset(boxes(2));
    h.record(boxes(2));
    assert!(!h.can_undo());
    h.record(boxes(3));
    h.record(boxes(3));
    assert_eq!(h.undo(), Some(boxes(2)));
    assert!(!h.can_undo());
}

#[wasm_bindgen_test]
fn record_without_a_reset_is_the_baseline() {
    let mut h = LabelHistory::default();
    h.record(boxes(4));
    assert!(!h.can_undo());
    assert!(!h.can_redo());
    h.record(boxes(5));
    assert_eq!(h.undo(), Some(boxes(4)));
}

#[wasm_bindgen_test]
fn the_cap_keeps_the_newest_states() {
    let mut h = LabelHistory::default();
    h.reset(boxes(0));
    for n in 1..=(HISTORY_CAP + 10) {
        h.record(boxes(n));
    }
    assert!(!h.can_redo());
    let mut steps = 0;
    let mut last = None;
    while let Some(s) = h.undo() {
        steps += 1;
        last = Some(s);
    }
    assert_eq!(steps, HISTORY_CAP - 1);
    assert_eq!(last, Some(boxes(11)));
}

#[wasm_bindgen_test]
fn a_full_cycle_replays_the_states_in_order() {
    let mut h = LabelHistory::default();
    h.reset(boxes(0));
    h.record(boxes(1));
    h.record(boxes(2));
    h.record(boxes(3));
    assert_eq!(h.undo(), Some(boxes(2)));
    assert_eq!(h.undo(), Some(boxes(1)));
    assert_eq!(h.undo(), Some(boxes(0)));
    assert_eq!(h.undo(), None);
    assert_eq!(h.redo(), Some(boxes(1)));
    assert_eq!(h.redo(), Some(boxes(2)));
    assert_eq!(h.redo(), Some(boxes(3)));
    assert_eq!(h.redo(), None);
}
