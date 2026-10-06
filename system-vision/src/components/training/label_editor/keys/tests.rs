// SPDX-FileCopyrightText: 2026 Conecsa
//
// SPDX-License-Identifier: AGPL-3.0-only

//! Unit tests for the label editor's keyboard map (headless browser).
use super::*;
use wasm_bindgen_test::*;

const NONE: Mods = Mods { ctrl: false, shift: false };
const CTRL: Mods = Mods { ctrl: true, shift: false };
const CTRL_SHIFT: Mods = Mods { ctrl: true, shift: true };
const SHIFT: Mods = Mods { ctrl: false, shift: true };

#[wasm_bindgen_test]
fn ctrl_z_undoes_and_ctrl_shift_z_or_ctrl_y_redoes() {
    assert_eq!(command("z", CTRL, false), Some(Command::Undo));
    assert_eq!(command("Z", CTRL_SHIFT, true), Some(Command::Redo));
    assert_eq!(command("y", CTRL, false), Some(Command::Redo));
    // Without Ctrl the letters mean nothing.
    assert_eq!(command("z", NONE, false), None);
    assert_eq!(command("y", SHIFT, false), None);
}

#[wasm_bindgen_test]
fn arrows_nudge_a_selected_label_by_one_px_or_ten_with_shift() {
    assert_eq!(command("ArrowLeft", NONE, true), Some(Command::Nudge(-1.0, 0.0)));
    assert_eq!(command("ArrowRight", NONE, true), Some(Command::Nudge(1.0, 0.0)));
    assert_eq!(command("ArrowUp", NONE, true), Some(Command::Nudge(0.0, -1.0)));
    assert_eq!(command("ArrowDown", SHIFT, true), Some(Command::Nudge(0.0, 10.0)));
}

#[wasm_bindgen_test]
fn arrows_step_the_images_when_nothing_is_selected() {
    assert_eq!(
        command("ArrowLeft", NONE, false),
        Some(Command::Step(ImageStep::Previous))
    );
    assert_eq!(command("ArrowRight", NONE, false), Some(Command::Step(ImageStep::Next)));
    // ↑ / ↓ keep their browser meaning (page scroll).
    assert_eq!(command("ArrowUp", NONE, false), None);
    assert_eq!(command("ArrowDown", SHIFT, false), None);
}

#[wasm_bindgen_test]
fn home_and_end_jump_only_without_a_selection_but_page_keys_always_step() {
    assert_eq!(command("Home", NONE, false), Some(Command::Step(ImageStep::First)));
    assert_eq!(command("End", NONE, false), Some(Command::Step(ImageStep::Last)));
    assert_eq!(command("Home", NONE, true), None);
    assert_eq!(command("End", NONE, true), None);
    assert_eq!(command("PageUp", NONE, true), Some(Command::Step(ImageStep::Previous)));
    assert_eq!(command("PageDown", NONE, false), Some(Command::Step(ImageStep::Next)));
}

#[wasm_bindgen_test]
fn zoom_space_and_delete_keys() {
    assert_eq!(command("+", NONE, false), Some(Command::ZoomIn));
    assert_eq!(command("=", NONE, false), Some(Command::ZoomIn));
    assert_eq!(command("-", CTRL, false), Some(Command::ZoomOut));
    assert_eq!(command("0", NONE, true), Some(Command::ZoomReset));
    assert_eq!(command(" ", NONE, false), Some(Command::Pan));
    assert_eq!(command("Delete", NONE, true), Some(Command::Delete));
    // Backspace is left to text fields.
    assert_eq!(command("Backspace", NONE, true), None);
    assert_eq!(command("a", NONE, true), None);
    assert_eq!(command("Enter", NONE, false), None);
}

#[wasm_bindgen_test]
fn navigation_and_undo_cancel_the_browser_default_zoom_and_delete_do_not() {
    assert!(Command::Undo.cancels_default());
    assert!(Command::Redo.cancels_default());
    assert!(Command::Nudge(1.0, 0.0).cancels_default());
    assert!(Command::Step(ImageStep::Next).cancels_default());
    assert!(Command::Pan.cancels_default());
    assert!(!Command::ZoomIn.cancels_default());
    assert!(!Command::ZoomOut.cancels_default());
    assert!(!Command::ZoomReset.cancels_default());
    assert!(!Command::Delete.cancels_default());
}
