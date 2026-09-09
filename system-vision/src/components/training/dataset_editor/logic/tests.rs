//! Unit tests for the dataset editor's pure decision helpers (headless browser).
use super::*;
use wasm_bindgen_test::*;

fn job(v: serde_json::Value) -> TrainingJobStatus {
    serde_json::from_value(v).unwrap()
}

fn model_status(v: serde_json::Value) -> LabelModelStatusResponse {
    serde_json::from_value(v).unwrap()
}

fn classes(names: &[&str]) -> Vec<String> {
    names.iter().map(|n| n.to_string()).collect()
}

// ── job_transition ───────────────────────────────────────────────────────────

#[wasm_bindgen_test]
fn done_after_active_finishes_with_conversion_handoff() {
    let j = job(serde_json::json!({
        "status": "done", "conversion_job_id": "c1", "model_name": "m"
    }));
    assert_eq!(
        job_transition("training", &j, true),
        JobTransition::Finished {
            pending: Some(PendingConversion {
                job_id: "c1".into(),
                filename: "m.pt".into(),
                elapsed_secs: 0.0,
            })
        }
    );
}

#[wasm_bindgen_test]
fn done_without_conversion_id_finishes_with_no_pending() {
    let j = job(serde_json::json!({ "status": "done", "model_name": "m" }));
    assert_eq!(
        job_transition("training", &j, true),
        JobTransition::Finished { pending: None }
    );
}

#[wasm_bindgen_test]
fn done_with_unseeded_baseline_still_finishes() {
    // This is why the poll seeds its baseline before looping: an empty
    // previous status cannot tell a stale "done" from a fresh one.
    let j = job(serde_json::json!({ "status": "done" }));
    assert!(matches!(
        job_transition("", &j, false),
        JobTransition::Finished { .. }
    ));
}

#[wasm_bindgen_test]
fn stale_done_from_previous_session_is_not_a_completion() {
    let j = job(serde_json::json!({ "status": "done" }));
    let t = job_transition("done", &j, true);
    assert!(!matches!(t, JobTransition::Finished { .. }));
    assert_eq!(t, JobTransition::Update);
}

#[wasm_bindgen_test]
fn failed_with_empty_prev_is_ignored() {
    let j = job(serde_json::json!({ "status": "failed" }));
    assert_eq!(job_transition("", &j, false), JobTransition::Ignore);
}

#[wasm_bindgen_test]
fn failed_after_active_prev_reports_failure() {
    let j = job(serde_json::json!({ "status": "failed" }));
    assert_eq!(job_transition("training", &j, true), JobTransition::Failed);
}

#[wasm_bindgen_test]
fn repeated_failed_is_a_plain_update() {
    let j = job(serde_json::json!({ "status": "failed" }));
    assert_eq!(job_transition("failed", &j, true), JobTransition::Update);
}

#[wasm_bindgen_test]
fn canceled_after_active_prev_reports_cancel() {
    let j = job(serde_json::json!({ "status": "canceled" }));
    assert_eq!(
        job_transition("training", &j, true),
        JobTransition::Canceled
    );
}

#[wasm_bindgen_test]
fn canceled_with_empty_prev_is_ignored() {
    let j = job(serde_json::json!({ "status": "canceled" }));
    assert_eq!(job_transition("", &j, false), JobTransition::Ignore);
}

#[wasm_bindgen_test]
fn active_job_updates_even_without_prior_job() {
    let j = job(serde_json::json!({ "status": "training" }));
    assert_eq!(job_transition("", &j, false), JobTransition::Update);
}

#[wasm_bindgen_test]
fn idle_without_prior_job_is_ignored() {
    let j = job(serde_json::json!({ "status": "idle" }));
    assert_eq!(job_transition("", &j, false), JobTransition::Ignore);
}

#[wasm_bindgen_test]
fn idle_with_prior_job_updates() {
    let j = job(serde_json::json!({ "status": "idle" }));
    assert_eq!(job_transition("training", &j, true), JobTransition::Update);
}

// ── active class arithmetic ──────────────────────────────────────────────────

#[wasm_bindgen_test]
fn clamp_active_keeps_in_range_and_ignores_empty_list() {
    assert_eq!(clamp_active(0, 0), 0);
    assert_eq!(clamp_active(5, 0), 5);
    assert_eq!(clamp_active(3, 3), 2);
    assert_eq!(clamp_active(1, 3), 1);
    assert_eq!(clamp_active(0, 1), 0);
}

#[wasm_bindgen_test]
fn active_after_remove_shifts_and_clamps() {
    assert_eq!(active_after_remove(2, 1, 3), 1);
    assert_eq!(active_after_remove(1, 1, 2), 0);
    assert_eq!(active_after_remove(0, 0, 2), 0);
    assert_eq!(active_after_remove(0, 2, 3), 0);
    assert_eq!(active_after_remove(3, 0, 3), 2);
    assert_eq!(active_after_remove(0, 0, 0), 0);
}

// ── SAM accept target ────────────────────────────────────────────────────────

#[wasm_bindgen_test]
fn sam_prompt_resolves_existing_class_case_insensitively_ignoring_color() {
    let cls = classes(&["Bottle #ff0000", "cup"]);
    assert_eq!(sam_accept_target("bottle", &cls, 1), SamTarget::Existing(0));
    assert_eq!(sam_accept_target("  CUP ", &cls, 0), SamTarget::Existing(1));
}

#[wasm_bindgen_test]
fn sam_prompt_for_unknown_class_requests_creation() {
    assert_eq!(
        sam_accept_target("can", &classes(&["bottle"]), 0),
        SamTarget::Create
    );
    assert_eq!(sam_accept_target("can", &[], 0), SamTarget::Create);
}

#[wasm_bindgen_test]
fn sam_empty_prompt_uses_clamped_active_class() {
    assert_eq!(
        sam_accept_target("", &classes(&["bottle", "cup"]), 1),
        SamTarget::Active(1)
    );
    assert_eq!(
        sam_accept_target("   ", &classes(&["bottle"]), 5),
        SamTarget::Active(0)
    );
}

#[wasm_bindgen_test]
fn sam_empty_prompt_with_no_classes_is_an_error() {
    assert_eq!(sam_accept_target("", &[], 0), SamTarget::NoClasses);
}

// ── class resolution ─────────────────────────────────────────────────────────

#[wasm_bindgen_test]
fn resolve_class_matches_display_name() {
    let cls = classes(&["Bottle #ff0000", "cup"]);
    assert_eq!(resolve_class(&cls, "cup"), Some(1));
    assert_eq!(resolve_class(&cls, "CUP"), Some(1));
    assert_eq!(resolve_class(&cls, "bottle"), Some(0));
    assert_eq!(resolve_class(&cls, "can"), None);
}

#[wasm_bindgen_test]
fn class_slot_index_is_the_same_for_both_outcomes() {
    assert_eq!(ClassSlot::Existing(3).index(), 3);
    assert_eq!(ClassSlot::Created(3).index(), 3);
}

#[wasm_bindgen_test]
fn model_suggestion_name_trims_and_skips_blank() {
    let names = classes(&["a", " b ", ""]);
    assert_eq!(model_suggestion_name(&names, 0), Some("a".to_string()));
    assert_eq!(model_suggestion_name(&names, 1), Some("b".to_string()));
    assert_eq!(model_suggestion_name(&names, 2), None);
    assert_eq!(model_suggestion_name(&names, 3), None);
}

// ── model assistant ──────────────────────────────────────────────────────────

#[wasm_bindgen_test]
fn model_needs_load_unless_that_engine_is_loaded() {
    assert!(model_needs_load(None, "a.engine"));
    let unloaded = model_status(serde_json::json!({ "loaded": false, "model_name": "a.engine" }));
    assert!(model_needs_load(Some(&unloaded), "a.engine"));
    let loaded = model_status(serde_json::json!({ "loaded": true, "model_name": "a.engine" }));
    assert!(!model_needs_load(Some(&loaded), "a.engine"));
    assert!(model_needs_load(Some(&loaded), "b.engine"));
}

#[wasm_bindgen_test]
fn default_base_model_only_when_engine_keeps_a_checkpoint() {
    let bases = classes(&["x.engine"]);
    assert_eq!(
        default_base_model(&Assistant::Model("x.engine".into()), &bases),
        "x.engine"
    );
    assert_eq!(
        default_base_model(&Assistant::Model("y.engine".into()), &bases),
        ""
    );
    assert_eq!(default_base_model(&Assistant::Sam, &bases), "");
    assert_eq!(default_base_model(&Assistant::Off, &bases), "");
}

// ── train gate ───────────────────────────────────────────────────────────────

#[wasm_bindgen_test]
fn can_train_requires_min_images_a_class_and_a_label() {
    assert!(can_train(20, 1, 20, true));
    assert!(!can_train(19, 1, 20, true));
    assert!(!can_train(20, 0, 20, true));
    assert!(!can_train(20, 1, 20, false));
}

#[wasm_bindgen_test]
fn labeled_count_counts_only_labeled_images() {
    let images: Vec<TrainingImageInfo> = serde_json::from_value(serde_json::json!([
        { "image_id": "a", "created_at": 0, "labeled": true, "box_count": 2 },
        { "image_id": "b", "created_at": 0, "labeled": false, "box_count": 0 },
        { "image_id": "c", "created_at": 0, "labeled": true, "box_count": 1 }
    ]))
    .unwrap();
    assert_eq!(labeled_count(&images), 2);
    assert_eq!(labeled_count(&[]), 0);
}
