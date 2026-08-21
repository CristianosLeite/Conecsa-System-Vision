//! Conversion-overlay state: the app-event handler's terminal-status helper and
//! the fallback poll that keeps the progress ramp moving.
//!
//! The overlay is driven primarily by `conversion_changed` app events (see
//! `MainView`), which carry the backend's own status/message/progress. This
//! module's poll is the fallback for when that stream is unavailable — it must
//! never be the reason the overlay disappears, so a transient failure is
//! tolerated and the hard timeout only rewords the overlay instead of tearing
//! it down.

use crate::api::{self, ConversionStatusResponse};
use crate::app::{load_models, ModelInfo};
use crate::i18n::*;
use leptos::prelude::*;

use gloo_timers::future::TimeoutFuture;
use js_sys::Date;

/// A conversion job started outside the Configuration panel (e.g. the
/// training-service uploading its trained model). MainView hands it to
/// Configuration, which attaches the standard overlay + poll to it.
#[derive(Debug, Clone, PartialEq)]
pub struct PendingConversion {
    pub job_id: String,
    /// Display name for the success/error messages (the model filename).
    pub filename: String,
    /// Seconds the job had already been running when the UI picked it up.
    pub elapsed_secs: f64,
}

/// Configuration for a conversion polling session.
pub struct ConversionPollConfig {
    pub job_id: String,
    /// Age of the job when this poll attached, as reported by the *device*.
    pub initial_elapsed_secs: f64,
    /// Display name used in the success/timeout messages (original filename).
    pub original_filename: String,
    /// Seconds after which the overlay stops promising an imminent finish.
    pub timeout_secs: f64,
    /// Maximum value (0–100) the time-based progress bar can reach while polling.
    pub progress_cap: f64,
    /// Locale captured by the caller (untracked) for the user-facing messages.
    pub locale: Locale,
}

/// Consecutive poll failures tolerated before the overlay gives up. A single
/// failed request used to destroy the overlay, which on a flaky link looked
/// exactly like "the conversion screen never showed up".
const MAX_POLL_FAILURES: u32 = 5;

/// Multiple of `timeout_secs` after which the overlay finally gives up. The
/// overlay must not hide a conversion that is still running (that was the whole
/// bug), but it must not pin the panel — and Start Detection — forever either.
const GIVE_UP_MULTIPLIER: f64 = 3.0;

/// Seconds the ramp takes to travel the full bar. The backend reports only
/// coarse milestones (5 / 40 / 45 / 100) and emits nothing at all during the
/// TensorRT build, so the bar interpolates between them.
const RAMP_FULL_SCALE_SECS: f64 = 420.0;

/// Whether a backend status string means the job is over.
pub(crate) fn is_terminal(status: &str) -> bool {
    matches!(status, "done" | "failed")
}

/// Bar value: the backend's real progress, smoothed by a time ramp so the long
/// opaque TensorRT build still looks alive. `elapsed` must be a single-clock
/// duration — never `Date::now() - started_at`, which mixes the browser's clock
/// with the device's.
pub(crate) fn ramp_progress(elapsed: f64, server_progress: u8, cap: f64) -> u8 {
    let ramp = ((elapsed / RAMP_FULL_SCALE_SECS) * 100.0).clamp(0.0, cap.clamp(0.0, 100.0));
    (ramp as u8).max(server_progress.min(100))
}

/// Claim the right to finish `job_id`, clearing the overlay. Returns true for
/// exactly one caller: the poll and the app-event handler both watch the same
/// job, and without this they would each toast the result and re-select the
/// engine. Synchronous on purpose — the claim must not straddle an await.
pub(crate) fn claim_finished(
    job_id: &str,
    active_job_id: ReadSignal<Option<String>>,
    set_active_job_id: WriteSignal<Option<String>>,
) -> bool {
    if active_job_id.get_untracked().as_deref() != Some(job_id) {
        return false;
    }
    set_active_job_id.set(None);
    true
}

/// Overlay text for a job: the backend's own step message, falling back to a
/// generic line while the job is still queued and has not produced one.
pub(crate) fn overlay_message(message: &str, filename: &str, locale: Locale) -> String {
    if message.is_empty() {
        td_string!(locale, models::converting_to_engine, name = filename.to_string())
    } else {
        message.to_string()
    }
}

/// Apply a finished job: select the new engine and toast the outcome. Call only
/// after [`claim_finished`] returned true, which is what closes the overlay.
/// Shared by the poll and the app-event handler so both end a job identically.
pub async fn apply_terminal_status(
    status: &ConversionStatusResponse,
    original_filename: &str,
    locale: Locale,
    set_success_msg: WriteSignal<String>,
    set_error_msg: WriteSignal<String>,
    set_models: WriteSignal<Vec<ModelInfo>>,
    set_model_refresh: WriteSignal<u32>,
) {
    if status.status != "done" {
        set_error_msg.set(td_string!(
            locale,
            models::conversion_failed,
            err = status.error.clone().unwrap_or_default()
        ));
        return;
    }

    let engine_name = status
        .auto_select_hint
        .clone()
        .or_else(|| status.engine_filename.clone())
        .unwrap_or_default();
    if !engine_name.is_empty() {
        let _ = api::select_model(&engine_name).await;
    }
    set_success_msg.set(td_string!(
        locale,
        models::conversion_success,
        name = original_filename.to_string()
    ));
    load_models(set_models, set_error_msg, locale).await;
    set_model_refresh.update(|n| *n = n.wrapping_add(1));
}

/// Polls a running conversion job, updating overlay signals on each tick, and
/// resolves the result (success / failure) when the job finishes.
///
/// The overlay signals are intentionally owned by the parent (`Configuration`)
/// so the overlay covers the full panel — this function only *writes* to them.
/// It stops as soon as `active_job_id` no longer names this job, so the app
/// event stream finishing the job first cannot produce a duplicate toast.
#[allow(clippy::too_many_arguments)]
pub async fn poll_conversion_job(
    cfg: ConversionPollConfig,
    active_job_id: ReadSignal<Option<String>>,
    set_active_job_id: WriteSignal<Option<String>>,
    set_overlay_message: WriteSignal<String>,
    set_overlay_progress: WriteSignal<u8>,
    set_success_msg: WriteSignal<String>,
    set_error_msg: WriteSignal<String>,
    set_models: WriteSignal<Vec<ModelInfo>>,
    set_model_refresh: WriteSignal<u32>,
) {
    let timeout_label = format!("{}", (cfg.timeout_secs / 60.0).round() as u64);
    // Single-clock elapsed: the device's own age for the job at attach time,
    // plus how long this tab has been watching. Subtracting the device's
    // `started_at` from the browser's `Date::now()` compares two unrelated
    // clocks — the device has no RTC battery and the hub steps its clock.
    let attached_at = Date::now() / 1000.0;
    let mut failures = 0u32;
    let mut warned_slow = false;

    loop {
        TimeoutFuture::new(2_000).await;

        // Someone else (the app event stream) already finished or replaced it.
        if active_job_id.get_untracked().as_deref() != Some(cfg.job_id.as_str()) {
            return;
        }

        let elapsed = cfg.initial_elapsed_secs + (Date::now() / 1000.0 - attached_at);

        match api::get_conversion_status(&cfg.job_id).await {
            Ok(status) => {
                failures = 0;

                if is_terminal(&status.status) {
                    if claim_finished(&cfg.job_id, active_job_id, set_active_job_id) {
                        apply_terminal_status(
                            &status,
                            &cfg.original_filename,
                            cfg.locale,
                            set_success_msg,
                            set_error_msg,
                            set_models,
                            set_model_refresh,
                        )
                        .await;
                    }
                    return;
                }

                // Prefer the device's own age whenever it sends one.
                let elapsed = status.elapsed_secs.unwrap_or(elapsed);
                set_overlay_progress.set(ramp_progress(elapsed, status.progress, cfg.progress_cap));

                if elapsed > cfg.timeout_secs * GIVE_UP_MULTIPLIER {
                    if claim_finished(&cfg.job_id, active_job_id, set_active_job_id) {
                        set_error_msg.set(td_string!(
                            cfg.locale,
                            models::conversion_gave_up,
                            minutes = format!(
                                "{}",
                                (cfg.timeout_secs * GIVE_UP_MULTIPLIER / 60.0).round() as u64
                            )
                        ));
                    }
                    return;
                }

                if elapsed > cfg.timeout_secs {
                    // Overrunning is not a reason to hide a conversion that is
                    // demonstrably still running — only to stop promising it is
                    // nearly done. Keep polling; the job still ends the loop.
                    if !warned_slow {
                        warned_slow = true;
                        set_overlay_message.set(td_string!(
                            cfg.locale,
                            models::conversion_taking_longer,
                            minutes = timeout_label.clone()
                        ));
                    }
                } else {
                    set_overlay_message.set(overlay_message(
                        &status.message,
                        &cfg.original_filename,
                        cfg.locale,
                    ));
                }
            }
            Err(e) => {
                failures += 1;
                leptos::logging::warn!(
                    "conversion poll failed ({}/{}): {}",
                    failures,
                    MAX_POLL_FAILURES,
                    e
                );
                if failures >= MAX_POLL_FAILURES {
                    set_active_job_id.set(None);
                    set_error_msg.set(td_string!(
                        cfg.locale,
                        models::failed_to_poll_conversion,
                        err = e
                    ));
                    return;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests;
