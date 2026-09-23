// SPDX-FileCopyrightText: 2026 Conecsa
//
// SPDX-License-Identifier: AGPL-3.0-only

//! The objects of the latest segmented frame, by class.
//!
//! The device draws the masks on the stream; this legend names every class in
//! text beside its color with the number of instances the frame holds, and
//! says so when the device left outlines out of the snapshot to keep it small.
//! It polls the snapshot — without frames, as a passive reader that
//! never counts as the hub's heartbeat — only while mounted, which is only
//! while detection runs.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use gloo_timers::future::TimeoutFuture;
use leptos::prelude::*;
use leptos::task::spawn_local;

use crate::api;
use crate::i18n::*;
use crate::models::{Snapshot, SnapshotItem};

/// Poll period: a glance-level panel, at the rate the hub records at.
const POLL_MS: u32 = 1_000;

/// One legend chip: a class, its color and how many instances the frame holds.
#[derive(Clone, Debug, PartialEq)]
pub struct LegendEntry {
    pub class_name: String,
    pub color: Option<String>,
    pub count: usize,
}

/// The frame's instances grouped by class, in the order the classes first
/// appear (the snapshot lists instances by descending confidence).
pub fn legend_entries(items: &[SnapshotItem]) -> Vec<LegendEntry> {
    let mut entries: Vec<LegendEntry> = Vec::new();
    for item in items {
        match entries.iter_mut().find(|e| e.class_name == item.class_name) {
            Some(entry) => entry.count += 1,
            None => entries.push(LegendEntry {
                class_name: item.class_name.clone(),
                color: item.color.clone(),
                count: 1,
            }),
        }
    }
    entries
}

/// The live legend of segmented objects.
#[component]
pub fn SegmentationLegend() -> impl IntoView {
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
                        leptos::logging::warn!("Could not read the segmentation result: {}", e);
                        warned = true;
                    }
                }
            }
            TimeoutFuture::new(POLL_MS).await;
        }
    });

    let body = move || match snapshot.get() {
        None => view! {
            <span class="ui-segment-muted">{t!(i18n, stream::segmentation_waiting)}</span>
        }
        .into_any(),
        Some(s) if s.detections.is_empty() => view! {
            <span class="ui-segment-muted">{t!(i18n, stream::no_object_above_threshold)}</span>
        }
        .into_any(),
        Some(s) => {
            let chips = legend_entries(&s.detections)
                .into_iter()
                .map(|entry| {
                    let swatch = entry
                        .color
                        .map(|c| format!("background-color: {c}"))
                        .unwrap_or_default();
                    view! {
                        <li class="ui-segment-chip">
                            <span class="ui-segment-swatch" style=swatch aria-hidden="true"></span>
                            <span>{entry.class_name}</span>
                            <span class="ui-segment-count">{format!("×{}", entry.count)}</span>
                        </li>
                    }
                })
                .collect_view();
            let truncated = s.polygons_truncated.then(|| {
                view! {
                    <span class="ui-help ui-segment-muted">{t!(i18n, stream::polygons_truncated)}</span>
                }
            });
            view! {
                <ul class="ui-segment-chips">{chips}</ul>
                {truncated}
            }
            .into_any()
        }
    };

    view! {
        <section class="ui-segment-panel" aria-live="polite">
            <span class="ui-label-xs">{t!(i18n, stream::segmentation_result)}</span>
            {body}
        </section>
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    fn item(class_name: &str, color: &str) -> SnapshotItem {
        SnapshotItem {
            class_name: class_name.into(),
            confidence: 0.9,
            color: Some(color.into()),
            bbox: Some([0.1, 0.1, 0.2, 0.2]),
            polygons: Some(vec![vec![[0.1, 0.1], [0.2, 0.1], [0.2, 0.2]]]),
        }
    }

    #[wasm_bindgen_test]
    fn instances_are_counted_per_class_in_first_seen_order() {
        let items = [item("nut", "#00ff00"), item("bolt", "#ff0000"), item("nut", "#00ff00")];
        let entries = legend_entries(&items);
        assert_eq!(
            entries.iter().map(|e| (e.class_name.as_str(), e.count)).collect::<Vec<_>>(),
            vec![("nut", 2), ("bolt", 1)]
        );
        assert_eq!(entries[0].color.as_deref(), Some("#00ff00"));
        assert!(legend_entries(&[]).is_empty());
    }
}
