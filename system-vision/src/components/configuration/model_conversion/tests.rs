//! Unit tests for the conversion-overlay pure helpers (headless browser).
use super::*;
use wasm_bindgen_test::*;

#[wasm_bindgen_test]
fn terminal_statuses_end_the_job() {
    assert!(is_terminal("done"));
    assert!(is_terminal("failed"));
    assert!(!is_terminal("pending"));
    assert!(!is_terminal("converting_to_onnx"));
    assert!(!is_terminal("converting_to_engine"));
}

#[wasm_bindgen_test]
fn ramp_rises_with_elapsed_time() {
    assert_eq!(ramp_progress(0.0, 0, 90.0), 0);
    assert_eq!(ramp_progress(210.0, 0, 90.0), 50);
    assert_eq!(ramp_progress(420.0, 0, 95.0), 95);
}

#[wasm_bindgen_test]
fn ramp_is_capped_but_server_progress_is_not() {
    // The ramp never exceeds the cap on its own …
    assert_eq!(ramp_progress(10_000.0, 0, 90.0), 90);
    // … but a real backend milestone always wins.
    assert_eq!(ramp_progress(0.0, 40, 90.0), 40);
    assert_eq!(ramp_progress(10_000.0, 100, 90.0), 100);
}

#[wasm_bindgen_test]
fn negative_elapsed_cannot_wrap_the_bar() {
    // A clock that ran backwards must floor at 0, not saturate or panic.
    assert_eq!(ramp_progress(-1.8e9, 0, 90.0), 0);
    assert_eq!(ramp_progress(-1.8e9, 45, 90.0), 45);
}

#[wasm_bindgen_test]
fn overlay_message_falls_back_while_the_job_is_queued() {
    assert_eq!(
        overlay_message("Building TensorRT engine…", "best.pt", Locale::en),
        "Building TensorRT engine…"
    );
    // A queued job has no step message yet; the overlay must still say something.
    assert!(overlay_message("", "best.pt", Locale::en).contains("best.pt"));
}

#[wasm_bindgen_test]
fn only_one_driver_may_finish_a_job() {
    let (active, set_active) = signal(Some("j1".to_string()));

    // The poll and the event handler both see "done" for j1 …
    assert!(claim_finished("j1", active, set_active));
    // … but only the first claim wins, so the result is toasted once.
    assert!(!claim_finished("j1", active, set_active));
    assert_eq!(active.get_untracked(), None);
}

#[wasm_bindgen_test]
fn a_stale_job_cannot_close_someone_elses_overlay() {
    let (active, set_active) = signal(Some("j2".to_string()));
    assert!(!claim_finished("j1", active, set_active));
    assert_eq!(active.get_untracked().as_deref(), Some("j2"));
}
