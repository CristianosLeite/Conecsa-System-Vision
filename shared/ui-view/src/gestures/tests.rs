// SPDX-FileCopyrightText: 2026 Conecsa
//
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for the viewport gesture decisions.
use super::*;

fn mouse(id: i32) -> ActivePointer {
    ActivePointer { id, client: (0.0, 0.0), touch: false }
}

fn finger(id: i32, client: Client) -> ActivePointer {
    ActivePointer { id, client, touch: true }
}

#[test]
fn a_new_mouse_press_replaces_the_mouse_entry_but_never_a_finger() {
    let mut ps = vec![mouse(1)];
    track(&mut ps, mouse(2));
    assert_eq!(ps, vec![mouse(2)]);

    let mut ps = vec![finger(7, (1.0, 1.0))];
    track(&mut ps, mouse(2));
    assert_eq!(ps, vec![finger(7, (1.0, 1.0)), mouse(2)]);

    // A finger reported down again is tracked once, at its new position.
    let mut ps = vec![finger(7, (1.0, 1.0)), finger(8, (2.0, 2.0))];
    track(&mut ps, finger(7, (3.0, 3.0)));
    assert_eq!(ps, vec![finger(8, (2.0, 2.0)), finger(7, (3.0, 3.0))]);
}

#[test]
fn at_most_two_pointers_are_kept() {
    let mut ps = vec![finger(1, (0.0, 0.0)), finger(2, (9.0, 9.0))];
    track(&mut ps, finger(3, (5.0, 5.0)));
    assert_eq!(ps.len(), 2);
    assert!(ps.iter().all(|p| p.id != 3));
}

#[test]
fn a_plain_press_passes_through_and_the_middle_button_or_modifier_pans() {
    let ps = [mouse(1)];
    assert_eq!(classify_press(&ps, None, false, true, false, false), Press::Through);
    assert_eq!(classify_press(&ps, None, false, true, true, false), Press::Pan);
    assert_eq!(classify_press(&ps, None, false, true, false, true), Press::Pan);
    // A second mouse button is swallowed, whatever else is held.
    assert_eq!(classify_press(&ps, None, false, false, false, true), Press::Ignore);
}

#[test]
fn the_second_finger_pinches_and_a_gesture_in_progress_claims_every_press() {
    let two = [finger(1, (0.0, 0.0)), finger(2, (9.0, 9.0))];
    // The second finger is never the primary pointer; the pinch wins anyway.
    assert_eq!(classify_press(&two, None, true, false, false, false), Press::Pinch);
    assert_eq!(classify_press(&two[..1], None, true, true, false, false), Press::Through);
    // A mouse press next to a finger is not a pinch.
    let mixed = [finger(1, (0.0, 0.0)), mouse(2)];
    assert_eq!(classify_press(&mixed, None, false, true, false, false), Press::Through);
    assert_eq!(classify_press(&two, Some(ViewGesture::Pinch), true, true, false, false), Press::Claimed);
    assert_eq!(
        classify_press(&[mouse(1)], Some(ViewGesture::Pan { id: 1 }), false, true, false, false),
        Press::Claimed
    );
}

#[test]
fn a_second_finger_during_a_one_finger_pan_starts_a_pinch() {
    let two = [finger(1, (0.0, 0.0)), finger(2, (9.0, 9.0))];
    assert_eq!(
        classify_press(&two, Some(ViewGesture::Pan { id: 1 }), true, false, false, true),
        Press::Pinch
    );
    // A mouse press during a pan, or a finger after a pinch ended, is claimed.
    let mixed = [finger(1, (0.0, 0.0)), mouse(2)];
    assert_eq!(
        classify_press(&mixed, Some(ViewGesture::Pan { id: 1 }), false, true, false, true),
        Press::Claimed
    );
    assert_eq!(classify_press(&two, Some(ViewGesture::Suppress), true, false, false, false), Press::Claimed);
}

#[test]
fn releasing_the_panning_pointer_ends_the_pan_and_a_leftover_finger_is_suppressed() {
    assert_eq!(after_release(ViewGesture::Pan { id: 1 }, 1, 0), None);
    assert_eq!(after_release(ViewGesture::Pan { id: 1 }, 2, 1), Some(ViewGesture::Pan { id: 1 }));
    assert_eq!(after_release(ViewGesture::Pinch, 1, 1), Some(ViewGesture::Suppress));
    assert_eq!(after_release(ViewGesture::Pinch, 1, 0), None);
    assert_eq!(after_release(ViewGesture::Suppress, 2, 1), Some(ViewGesture::Suppress));
    assert_eq!(after_release(ViewGesture::Suppress, 2, 0), None);
}

#[test]
fn pinch_pairs_keep_the_still_finger_and_move_the_other() {
    let before = [finger(1, (10.0, 10.0)), finger(2, (50.0, 50.0))];
    assert_eq!(
        pinch_pairs(&before, 1, (0.0, 0.0)),
        Some((((10.0, 10.0), (50.0, 50.0)), ((0.0, 0.0), (50.0, 50.0))))
    );
    assert_eq!(
        pinch_pairs(&before, 2, (60.0, 60.0)),
        Some((((10.0, 10.0), (50.0, 50.0)), ((10.0, 10.0), (60.0, 60.0))))
    );
    // An unknown pointer, or a single one, is not a pinch.
    assert_eq!(pinch_pairs(&before, 3, (0.0, 0.0)), None);
    assert_eq!(pinch_pairs(&before[..1], 1, (0.0, 0.0)), None);
}

#[test]
fn ctrl_wheel_zooms_and_the_plain_wheel_pans_only_a_zoomed_image() {
    assert_eq!(wheel_action(true, false, 0.0, 5.0, true), WheelAction::Zoom);
    assert_eq!(wheel_action(true, true, 0.0, 5.0, false), WheelAction::Zoom);
    assert_eq!(wheel_action(false, false, 0.0, 5.0, true), WheelAction::Ignore);
    assert_eq!(wheel_action(false, false, 3.0, 5.0, false), WheelAction::Pan(3.0, 5.0));
    // Shift turns a mouse wheel's vertical delta sideways; a trackpad's own
    // horizontal delta is kept as it is.
    assert_eq!(wheel_action(false, true, 0.0, 5.0, false), WheelAction::Pan(5.0, 0.0));
    assert_eq!(wheel_action(false, true, 2.0, 5.0, false), WheelAction::Pan(2.0, 5.0));
}
