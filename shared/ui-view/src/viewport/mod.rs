// SPDX-FileCopyrightText: 2026 Conecsa
//
// SPDX-License-Identifier: Apache-2.0

//! The wasm32 side of the zoomable view: the shared signals ([`ViewState`]),
//! the capture-phase pointer and wheel handlers of the viewport element
//! ([`Gestures`]) and the element watcher ([`watch_viewport`]).
//!
//! The component owns the markup: a viewport `<div>` (`overflow: hidden`,
//! `touch-action: none`) with the three CSS custom properties `--label-zoom`,
//! `--label-ox`, `--label-oy` set from [`ViewState::transform`] and the four
//! `on:pointer*:capture` handlers wired to [`Gestures`]; inside it a stage
//! laid out from those properties holds the `<img>` and its overlay.

use leptos::prelude::*;
use wasm_bindgen::closure::Closure;
use wasm_bindgen::JsCast;

use crate::gestures::{
    after_release, classify_press, pinch_pairs, track, wheel_action, ActivePointer, Client, Press,
    ViewGesture, WheelAction,
};
use crate::transform::{wheel_delta_px, wheel_zoom_factor, Frac, ViewTransform, ZOOM_STEP};

/// Zoom and pan of the image, shared by the viewport (gestures), a toolbar
/// (buttons, readout) and a key handler.
#[derive(Clone, Copy)]
pub struct ViewState {
    pub transform: RwSignal<ViewTransform>,
    /// A primary-button drag pans instead of passing through: the label
    /// editor sets it while Space is held, a plain viewer whenever the image
    /// is zoomed.
    pub space_held: RwSignal<bool>,
    /// The viewport's rendered size in CSS px (measured by `watch_viewport`);
    /// the zoomed image is `zoom` times as large, which is what one screen
    /// pixel of an arrow-key nudge is measured against.
    pub viewport: RwSignal<(f32, f32)>,
}

impl ViewState {
    /// Identity transform, nothing held, the viewport seeded with `fallback`
    /// (used until the element is measured).
    pub fn new(fallback: (f32, f32)) -> Self {
        Self {
            transform: RwSignal::new(ViewTransform::IDENTITY),
            space_held: RwSignal::new(false),
            viewport: RwSignal::new(fallback),
        }
    }

    /// The on-screen size of the zoomed image, in CSS px (`None` once disposed).
    pub fn canvas_px(self) -> Option<(f32, f32)> {
        let (w, h) = self.viewport.try_get_untracked()?;
        let zoom = self.transform.try_get_untracked()?.zoom;
        Some((w * zoom, h * zoom))
    }

    /// Zoom about the viewport centre (toolbar buttons and keys). `try_*`:
    /// global listeners can briefly outlive the signals during teardown.
    fn zoom_by(self, factor: f32) {
        let _ = self
            .transform
            .try_update(|v| *v = v.zoom_by(factor, (0.5, 0.5)));
    }

    pub fn zoom_in(self) {
        self.zoom_by(ZOOM_STEP);
    }

    pub fn zoom_out(self) {
        self.zoom_by(1.0 / ZOOM_STEP);
    }

    pub fn reset(self) {
        let _ = self.transform.try_set(ViewTransform::IDENTITY);
    }
}

/// The event is ours: nothing below reacts, no window-level pointerup saves.
fn claim(ev: &leptos::ev::PointerEvent) {
    ev.stop_propagation();
    ev.prevent_default();
}

/// The view gestures of the viewport element and the state they run on.
#[derive(Clone, Copy)]
pub struct Gestures {
    view: ViewState,
    /// The pointers down on the viewport, the gesture in progress and the
    /// viewport rectangle it measures against. Plain values: the handlers run
    /// at pointer rate and nothing renders from them but the transform.
    pointers: StoredValue<Vec<ActivePointer>>,
    gesture: StoredValue<Option<ViewGesture>>,
    gesture_rect: StoredValue<(f32, f32, f32, f32)>,
    /// A pan is in progress (only drives the cursor).
    pub panning: RwSignal<bool>,
    /// Whatever the first finger was doing below the viewport (drawing, a
    /// swipe) is abandoned when a second finger starts a pinch.
    on_interrupt: Callback<()>,
}

impl Gestures {
    pub fn new(view: ViewState, on_interrupt: Callback<()>) -> Self {
        Self {
            view,
            pointers: StoredValue::new(Vec::new()),
            gesture: StoredValue::new(None),
            gesture_rect: StoredValue::new((0.0, 0.0, 1.0, 1.0)),
            panning: RwSignal::new(false),
            on_interrupt,
        }
    }

    /// A new image is a new viewport element: a gesture from the previous one
    /// can never finish here.
    pub fn reset(self) {
        self.pointers.set_value(Vec::new());
        self.gesture.set_value(None);
        self.panning.set(false);
    }

    /// Client px → viewport fraction, against the rectangle cached when the
    /// gesture started (it cannot change mid-gesture).
    fn frac(self, client: Client) -> Frac {
        let (left, top, w, h) = self.gesture_rect.get_value();
        ((client.0 - left) / w, (client.1 - top) / h)
    }

    fn start(self, container: NodeRef<leptos::html::Div>, g: ViewGesture, ids: &[i32]) {
        let Some(el) = container.get_untracked() else {
            return;
        };
        let rect = el.get_bounding_client_rect();
        if rect.width() <= 0.0 || rect.height() <= 0.0 {
            return;
        }
        self.gesture_rect.set_value((
            rect.left() as f32,
            rect.top() as f32,
            rect.width() as f32,
            rect.height() as f32,
        ));
        // The viewport takes the pointers over (a later capture wins over the
        // content's), so their moves and releases come back here even outside
        // the viewport.
        for id in ids {
            let _ = el.set_pointer_capture(*id);
        }
        self.gesture.set_value(Some(g));
        self.panning.set(matches!(g, ViewGesture::Pan { .. }));
    }

    /// Capture-phase pointerdown on the viewport.
    pub fn press(self, container: NodeRef<leptos::html::Div>, ev: leptos::ev::PointerEvent) {
        let id = ev.pointer_id();
        let touch = ev.pointer_type() == "touch";
        let client = (ev.client_x() as f32, ev.client_y() as f32);
        self.pointers
            .update_value(|ps| track(ps, ActivePointer { id, client, touch }));
        let press = self.pointers.with_value(|ps| {
            classify_press(
                ps,
                self.gesture.get_value(),
                touch,
                ev.is_primary(),
                ev.button() == 1,
                self.view.space_held.get_untracked(),
            )
        });
        match press {
            // A content press: let it through.
            Press::Through => {}
            Press::Ignore => ev.stop_propagation(),
            Press::Claimed => claim(&ev),
            Press::Pinch => {
                // Whatever the first finger was doing is abandoned and both
                // now zoom and drag the image.
                self.on_interrupt.run(());
                let ids = self
                    .pointers
                    .with_value(|ps| ps.iter().map(|p| p.id).collect::<Vec<_>>());
                self.start(container, ViewGesture::Pinch, &ids);
                claim(&ev);
            }
            Press::Pan => {
                self.start(container, ViewGesture::Pan { id }, &[id]);
                claim(&ev);
            }
        }
    }

    /// Capture-phase pointermove on the viewport.
    pub fn motion(self, ev: leptos::ev::PointerEvent) {
        let id = ev.pointer_id();
        let client = (ev.client_x() as f32, ev.client_y() as f32);
        // Tracked even outside a gesture: a pass-through touch that moved and
        // is then joined by a second finger pinches from where it is now, not
        // from where it went down (the pinch's first pair is this position).
        let before = self.pointers.get_value();
        self.pointers.update_value(|ps| {
            if let Some(p) = ps.iter_mut().find(|p| p.id == id) {
                p.client = client;
            }
        });
        let Some(g) = self.gesture.get_value() else {
            return;
        };
        match g {
            ViewGesture::Pan { id: pan_id } if pan_id == id => {
                if let Some(prev) = before.iter().find(|p| p.id == id).map(|p| p.client) {
                    let (_, _, w, h) = self.gesture_rect.get_value();
                    let moved = ((client.0 - prev.0) / w, (client.1 - prev.1) / h);
                    self.view.transform.update(|v| *v = v.pan_by(moved));
                }
                claim(&ev);
            }
            ViewGesture::Pinch => {
                if let Some(((p0, p1), (n0, n1))) = pinch_pairs(&before, id, client) {
                    let prev_pair = (self.frac(p0), self.frac(p1));
                    let next_pair = (self.frac(n0), self.frac(n1));
                    self.view.transform.update(|v| *v = v.pinch(prev_pair, next_pair));
                }
                claim(&ev);
            }
            ViewGesture::Pan { .. } | ViewGesture::Suppress => ev.stop_propagation(),
        }
    }

    /// Capture-phase pointerup / pointercancel on the viewport.
    pub fn release(self, ev: leptos::ev::PointerEvent) {
        let id = ev.pointer_id();
        self.pointers.update_value(|ps| ps.retain(|p| p.id != id));
        let Some(g) = self.gesture.get_value() else {
            return;
        };
        let next = after_release(g, id, self.pointers.with_value(Vec::len));
        self.gesture.set_value(next);
        self.panning.set(matches!(next, Some(ViewGesture::Pan { .. })));
        ev.stop_propagation();
    }

    /// The viewport's native wheel listener (see `watch_viewport`).
    pub fn wheel(self, container: NodeRef<leptos::html::Div>, ev: web_sys::WheelEvent) {
        let Some(el) = container.get_untracked() else {
            return;
        };
        let rect = el.get_bounding_client_rect();
        let (w, h) = (rect.width() as f32, rect.height() as f32);
        if w <= 0.0 || h <= 0.0 {
            return;
        }
        let mode = ev.delta_mode();
        let (dx, dy) = (wheel_delta_px(ev.delta_x(), mode), wheel_delta_px(ev.delta_y(), mode));
        let at_identity = self.view.transform.get_untracked().is_identity();
        match wheel_action(ev.ctrl_key() || ev.meta_key(), ev.shift_key(), dx, dy, at_identity) {
            WheelAction::Zoom => {
                let anchor = (
                    (ev.client_x() as f32 - rect.left() as f32) / w,
                    (ev.client_y() as f32 - rect.top() as f32) / h,
                );
                self.view
                    .transform
                    .update(|v| *v = v.zoom_by(wheel_zoom_factor(dy), anchor));
                ev.prevent_default();
                // The page-zoom guard on `window` need not see it.
                ev.stop_propagation();
            }
            WheelAction::Pan(dx, dy) => {
                self.view.transform.update(|v| *v = v.pan_by((-dx / w, -dy / h)));
                ev.prevent_default();
            }
            WheelAction::Ignore => {}
        }
    }
}

/// Watch the viewport element once it is mounted: measure it into `viewport`
/// whenever its layout size changes (image decoded, window resized, assistant
/// panel opened…; ResizeObserver reports once on `observe`, so the first
/// measurement needs no extra trigger) and take its wheel events natively
/// with `passive: false`. Leptos delegates `on:wheel` to `window`, where
/// browsers make wheel listeners passive, and a passive listener cannot
/// prevent the page zoom/scroll. The observer and the JS closures are not
/// Send, so they live in owner-scoped local slots that the cleanup drains;
/// `try_set` because a late callback must not touch a disposed signal.
pub fn watch_viewport(
    container: NodeRef<leptos::html::Div>,
    viewport: RwSignal<(f32, f32)>,
    on_wheel: impl Fn(web_sys::WheelEvent) + Copy + Send + Sync + 'static,
) {
    let observer = StoredValue::new_local(None::<(web_sys::ResizeObserver, Closure<dyn FnMut()>)>);
    // The native wheel listener and the element it is on.
    let wheel_listener = StoredValue::new_local(
        None::<(web_sys::HtmlDivElement, Closure<dyn FnMut(web_sys::WheelEvent)>)>,
    );
    let disconnect = move || {
        observer.update_value(|slot| {
            if let Some((obs, _)) = slot.take() {
                obs.disconnect();
            }
        });
        wheel_listener.update_value(|slot| {
            if let Some((el, cb)) = slot.take() {
                let _ = el.remove_event_listener_with_callback("wheel", cb.as_ref().unchecked_ref());
            }
        });
    };
    Effect::new(move |_| {
        let Some(el) = container.get() else {
            return;
        };
        disconnect();
        let wheel = Closure::<dyn FnMut(web_sys::WheelEvent)>::new(on_wheel);
        let active = web_sys::AddEventListenerOptions::new();
        active.set_passive(false);
        let _ = el.add_event_listener_with_callback_and_add_event_listener_options(
            "wheel",
            wheel.as_ref().unchecked_ref(),
            &active,
        );
        wheel_listener.set_value(Some((el.clone(), wheel)));
        let measured = el.clone();
        let measure = Closure::<dyn FnMut()>::new(move || {
            let rect = measured.get_bounding_client_rect();
            let size = (rect.width() as f32, rect.height() as f32);
            if size.0 > 0.0 && size.1 > 0.0 && viewport.try_get_untracked() != Some(size) {
                let _ = viewport.try_set(size);
            }
        });
        let Ok(obs) = web_sys::ResizeObserver::new(measure.as_ref().unchecked_ref()) else {
            return;
        };
        obs.observe(&el);
        observer.set_value(Some((obs, measure)));
    });
    on_cleanup(disconnect);
}
