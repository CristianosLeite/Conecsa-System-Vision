// SPDX-FileCopyrightText: 2026 Conecsa
//
// SPDX-License-Identifier: AGPL-3.0-only

//! The live classification result under the video.
//!
//! A classifier draws nothing on the stream: this panel names the class above
//! the threshold with the top-k candidates. It polls the snapshot — without
//! frames, as a passive reader that never counts as the hub's heartbeat — only
//! while it is mounted, which is only while detection runs.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use gloo_timers::future::TimeoutFuture;
use leptos::prelude::*;
use leptos::task::spawn_local;

use crate::api;
use crate::i18n::*;
use crate::models::Snapshot;

/// Poll period: a glance-level panel, at the rate the hub records at.
const POLL_MS: u32 = 1_000;

/// A probability as a whole percentage (0..=100).
fn percent(confidence: f32) -> f32 {
    (confidence.clamp(0.0, 1.0) * 100.0).round()
}

/// The class above the threshold and the top-k candidates, refreshed live.
#[component]
pub fn ClassificationPanel() -> impl IntoView {
    let i18n = use_i18n();
    let snapshot = RwSignal::new(None::<Snapshot>);

    let alive = Arc::new(AtomicBool::new(true));
    {
        let alive = alive.clone();
        on_cleanup(move || alive.store(false, Ordering::Relaxed));
    }
    spawn_local(async move {
        let mut warned = false;
        while alive.load(Ordering::Relaxed) {
            match api::get_snapshot().await {
                Ok(s) => {
                    let _ = snapshot.try_set(Some(s));
                    warned = false;
                }
                Err(e) => {
                    // Keep the last result on screen; one warning per outage.
                    if !warned {
                        leptos::logging::warn!("Could not read the classification result: {}", e);
                        warned = true;
                    }
                }
            }
            TimeoutFuture::new(POLL_MS).await;
        }
    });

    let result = move || {
        let Some(s) = snapshot.get() else {
            return view! {
                <span class="ui-classify-muted">{t!(i18n, stream::classification_waiting)}</span>
            }
            .into_any();
        };
        match s.top_class().cloned() {
            Some(item) => {
                let swatch = item
                    .color
                    .map(|c| format!("background-color: {c}"))
                    .unwrap_or_default();
                view! {
                    <span class="ui-classify-class">
                        <span class="ui-classify-swatch" style=swatch aria-hidden="true"></span>
                        <span class="ui-classify-name">{item.class_name}</span>
                        <span class="ui-classify-score">
                            {format!("{:.0}%", percent(item.confidence))}
                        </span>
                    </span>
                }
                .into_any()
            }
            None => view! {
                <span class="ui-classify-muted">{t!(i18n, stream::no_class_above_threshold)}</span>
            }
            .into_any(),
        }
    };

    let candidates = move || {
        let s = snapshot.get().unwrap_or_default();
        let has_class = s.top_class().is_some();
        s.candidates
            .unwrap_or_default()
            .into_iter()
            .enumerate()
            .map(|(index, c)| {
                let pct = percent(c.confidence);
                // The first candidate is the reported class when it passed the gate.
                let class = if index == 0 && has_class {
                    "ui-classify-candidate is-top"
                } else {
                    "ui-classify-candidate"
                };
                view! {
                    <li class=class>
                        <span class="ui-classify-candidate-name">{c.class_name}</span>
                        <span class="ui-classify-bar" aria-hidden="true">
                            <span class="ui-classify-bar-fill" style=format!("width: {pct}%")></span>
                        </span>
                        <span class="ui-classify-candidate-score">{format!("{pct:.0}%")}</span>
                    </li>
                }
            })
            .collect_view()
    };

    view! {
        <section class="ui-classify-panel">
            <div class="ui-classify-result" aria-live="polite">
                <span class="ui-label-xs">{t!(i18n, stream::classification_result)}</span>
                {result}
            </div>
            <div class="ui-classify-top">
                <span class="ui-label-xs">{t!(i18n, stream::top_candidates)}</span>
                <ol class="ui-classify-candidates">{candidates}</ol>
            </div>
        </section>
    }
}
