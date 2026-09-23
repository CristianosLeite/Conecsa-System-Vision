// SPDX-FileCopyrightText: 2026 Conecsa
//
// SPDX-License-Identifier: AGPL-3.0-only

//! The people recognized in the latest frame.
//!
//! The device draws the face boxes and names on the stream; this panel names
//! every recognized person in text with their best similarity, and counts the
//! faces that matched nobody in the gallery. It polls the snapshot — without
//! frames, as a passive reader that never counts as the hub's heartbeat — only
//! while mounted, which is only while detection runs.

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

/// The class name the device gives a face that matched nobody.
pub const UNKNOWN: &str = "unknown";

/// One recognized person in the frame.
#[derive(Clone, Debug, PartialEq)]
pub struct Identity {
    pub name: String,
    pub color: Option<String>,
    /// How many of the frame's faces took this name.
    pub count: usize,
    /// The best cosine similarity among those faces (0..1).
    pub similarity: f32,
}

/// The frame's faces: the recognized people, in the order they first appear
/// (the snapshot lists faces largest first), and the unknown faces.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Identities {
    pub people: Vec<Identity>,
    pub unknown: usize,
}

/// Group a face snapshot's items by person.
pub fn identities(items: &[SnapshotItem]) -> Identities {
    let mut out = Identities::default();
    for item in items {
        if item.class_name == UNKNOWN {
            out.unknown += 1;
            continue;
        }
        match out.people.iter_mut().find(|p| p.name == item.class_name) {
            Some(person) => {
                person.count += 1;
                person.similarity = person.similarity.max(item.confidence);
            }
            None => out.people.push(Identity {
                name: item.class_name.clone(),
                color: item.color.clone(),
                count: 1,
                similarity: item.confidence,
            }),
        }
    }
    out
}

/// A similarity as a whole percentage (0..=100).
fn percent(similarity: f32) -> f32 {
    (similarity.clamp(0.0, 1.0) * 100.0).round()
}

/// The live list of the people in the frame.
#[component]
pub fn IdentityPanel() -> impl IntoView {
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
                        leptos::logging::warn!("Could not read the face recognition result: {}", e);
                        warned = true;
                    }
                }
            }
            TimeoutFuture::new(POLL_MS).await;
        }
    });

    let body = move || match snapshot.get() {
        None => view! {
            <span class="ui-segment-muted">{t!(i18n, stream::face_waiting)}</span>
        }
        .into_any(),
        Some(s) if s.detections.is_empty() => view! {
            <span class="ui-segment-muted">{t!(i18n, stream::no_face_in_frame)}</span>
        }
        .into_any(),
        Some(s) => {
            let found = identities(&s.detections);
            let people = found
                .people
                .into_iter()
                .map(|person| {
                    let swatch = person
                        .color
                        .map(|c| format!("background-color: {c}"))
                        .unwrap_or_default();
                    let count = (person.count > 1).then(|| {
                        view! { <span class="ui-segment-count">{format!("×{}", person.count)}</span> }
                    });
                    view! {
                        <li class="ui-segment-chip">
                            <span class="ui-segment-swatch" style=swatch aria-hidden="true"></span>
                            <span>{person.name}</span>
                            <span
                                class="ui-segment-count"
                                title=move || t_string!(i18n, stream::face_similarity)
                            >
                                {format!("{:.0}%", percent(person.similarity))}
                            </span>
                            {count}
                        </li>
                    }
                })
                .collect_view();
            let unknown = found.unknown;
            let unknown_chip = (unknown > 0).then(|| {
                view! {
                    <li class="ui-segment-chip">
                        <span class="ui-segment-muted">{t!(i18n, stream::face_unknown)}</span>
                        <span class="ui-segment-count">{format!("×{unknown}")}</span>
                    </li>
                }
            });
            view! {
                <ul class="ui-segment-chips">{people}{unknown_chip}</ul>
            }
            .into_any()
        }
    };

    view! {
        <section class="ui-segment-panel" aria-live="polite">
            <span class="ui-label-xs">{t!(i18n, stream::face_result)}</span>
            {body}
            <span class="ui-help ui-segment-muted">{t!(i18n, stream::face_privacy_note)}</span>
        </section>
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    fn face(name: &str, similarity: f32) -> SnapshotItem {
        SnapshotItem {
            class_name: name.into(),
            confidence: similarity,
            color: Some("#00ff00".into()),
            bbox: Some([0.1, 0.1, 0.2, 0.2]),
            polygons: None,
        }
    }

    #[wasm_bindgen_test]
    fn faces_are_grouped_by_person_and_unknown_faces_are_counted() {
        let items = [
            face("Ana", 0.52),
            face(UNKNOWN, 0.10),
            face("Bruno", 0.61),
            face("Ana", 0.74),
            face(UNKNOWN, 0.20),
        ];
        let found = identities(&items);
        assert_eq!(
            found
                .people
                .iter()
                .map(|p| (p.name.as_str(), p.count, p.similarity))
                .collect::<Vec<_>>(),
            vec![("Ana", 2, 0.74), ("Bruno", 1, 0.61)]
        );
        assert_eq!(found.unknown, 2);
        assert_eq!(identities(&[]), Identities::default());
    }
}
