//! Unit tests for the nokhwa fallback loops' frame-error tracker.
use super::{FrameErrorTracker, MAX_CONSECUTIVE_FRAME_ERRORS};

#[test]
fn the_first_failure_is_logged_but_not_fatal() {
    let mut tracker = FrameErrorTracker::new();
    let action = tracker.on_error();
    assert!(action.log);
    assert!(!action.give_up);
}

#[test]
fn intermediate_failures_are_quiet() {
    let mut tracker = FrameErrorTracker::new();
    tracker.on_error();
    for _ in 1..(MAX_CONSECUTIVE_FRAME_ERRORS - 1) {
        let action = tracker.on_error();
        assert!(!action.log, "only the first and every 100th failure are logged");
        assert!(!action.give_up);
    }
}

#[test]
fn the_threshold_gives_the_camera_up() {
    // A disconnected camera used to spin the loop forever; after the
    // threshold the loop must return to the outer open/backoff loop.
    let mut tracker = FrameErrorTracker::new();
    for _ in 0..(MAX_CONSECUTIVE_FRAME_ERRORS - 1) {
        assert!(!tracker.on_error().give_up);
    }
    assert!(tracker.on_error().give_up);
    assert_eq!(tracker.consecutive(), MAX_CONSECUTIVE_FRAME_ERRORS);
}

#[test]
fn a_good_frame_resets_the_streak() {
    let mut tracker = FrameErrorTracker::new();
    for _ in 0..(MAX_CONSECUTIVE_FRAME_ERRORS - 1) {
        tracker.on_error();
    }
    tracker.on_ok();
    assert_eq!(tracker.consecutive(), 0);
    let action = tracker.on_error();
    assert!(action.log, "a new streak logs its first failure again");
    assert!(!action.give_up);
}

#[test]
fn every_hundredth_failure_of_a_long_streak_is_logged() {
    // Reachable only if the threshold is raised; pins the cadence anyway.
    let mut tracker = FrameErrorTracker::new();
    let mut logged = Vec::new();
    for n in 1..=200u32 {
        if tracker.on_error().log {
            logged.push(n);
        }
    }
    assert_eq!(logged, vec![1, 100, 200]);
}
