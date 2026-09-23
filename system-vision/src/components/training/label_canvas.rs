// SPDX-FileCopyrightText: 2026 Conecsa
//
// SPDX-License-Identifier: AGPL-3.0-only

use leptos::prelude::*;
use wasm_bindgen::closure::Closure;
use wasm_bindgen::JsCast;

use crate::api::{training_image_url, LabelBox, LabelPolygon};
use crate::i18n::*;

use super::dataset_editor::{next_instance, AiState, ClassesState, DrawMode, ImagesState};
use super::label_editor::PolygonTool;
use super::label_geometry::{
    apply_drag, apply_poly_drag, box_ring, capture_pointer, clockwise, closes_ring, dedupe_ring,
    far_enough, hit_edge, hit_vertex, insert_vertex, is_background, norm_coords, simplify_rdp,
    valid_ring, BoxDrag, Corner, DragKind, PolyDraft, PolyDrag, PolyDragKind, PolyPress, Pt,
    DRAG_EPS, MIN_DRAW_PX, MIN_SIZE, MIN_VERTEX_GAP_PX, RDP_EPS_PX, VIEW,
};
use super::label_shapes::{
    committed_boxes_layer, committed_polygons_layer, draft_layer, points_layer,
    polygon_draft_layer, polygon_status_bar, polygon_suggestions_layer, status_bar,
    suggestions_layer,
};

/// The drawing surface: the dataset image with an SVG overlay for the boxes
/// or polygons, their handles, AI suggestions, SAM points and the shape being
/// drawn.
///
/// Owns the box `draft` rectangle and all pointer handling; the rendering of
/// each overlay layer lives in `label_shapes`. Selection and the active edit
/// (`selected_box`/`drag`, or the polygon tool's signals) are owned by the
/// parent so the toolbar and the global keyboard listener can share them.
/// Uses Pointer Events so mouse, touch and pen all draw.
///
/// A segmentation dataset (`poly.enabled`) draws polygon rings (click, freehand
/// or rectangle) and edits them by vertex, edge or whole-ring drags; vertices
/// are stored normalized, hit tests run in rendered pixels, and each completed
/// gesture is one save. A ring whose edges cross is refused.
#[component]
pub(super) fn LabelCanvas(
    dataset_id: String,
    images: ImagesState,
    classes: ClassesState,
    ai: AiState,
    selected_box: RwSignal<Option<usize>>,
    drag: RwSignal<Option<BoxDrag>>,
    /// The polygon tool of a segmentation dataset (disabled for detection).
    poly: PolygonTool,
    /// SAM prompt mode: background clicks become point prompts.
    sam_mode: Signal<bool>,
    /// Any assistant panel open above the canvas (it eats vertical room).
    panel_open: Signal<bool>,
    /// Persist the current labels (true = show a toast).
    on_save: Callback<bool>,
    /// Fired when the user tries to draw with no class selected.
    on_need_class: Callback<()>,
    /// Fired when a drawn or edited ring crosses itself (nothing is saved).
    on_refused: Callback<()>,
) -> impl IntoView {
    let i18n = use_i18n();
    let selected = images.selected.read_only();
    let boxes = images.boxes;
    let polygons = images.polygons;
    let active_class = classes.active.read_only();
    let classes = classes.list.read_only();
    let sam_points = ai.sam_points;
    let suggestions = ai.suggestions.read_only();
    let suggestion_names = ai.suggestion_names.read_only();
    let suggestion_polygons = ai.suggestion_polygons.read_only();
    // Copy-able handle so the canvas closure can build per-image URLs.
    let dataset_id = StoredValue::new(dataset_id);
    // Draft rectangle while drawing a new box (or a rectangle ring),
    // normalized (x0, y0, x1, y1).
    let draft = RwSignal::new(None::<(f32, f32, f32, f32)>);
    // Rendered canvas size in CSS px — the overlay's coordinate space, so
    // strokes, handles and text keep a fixed on-screen size whatever the
    // monitor or image aspect. Seeded with the fallback and kept current by a
    // ResizeObserver on the canvas container (see the image branch below).
    let canvas = RwSignal::new((VIEW, VIEW));

    // The canvas is width-driven in a fixed, non-scrolling viewport. Cap its
    // width by the available height so it always fits vertically; when an
    // assistant panel is open it eats ~11rem more, so shrink the canvas to
    // match (16rem ≈ top bar + toolbar + status; +11rem for the AI controls).
    // The image branch scales this cap by the image's aspect ratio (see
    // `canvas_box`).
    let canvas_height_avail = move || {
        if panel_open.get() {
            "100dvh - 27rem"
        } else {
            "100dvh - 16rem"
        }
    };
    let canvas_cap = move || {
        if panel_open.get() {
            "max-w-[calc(100dvh-27rem)]"
        } else {
            "max-w-[calc(100dvh-16rem)]"
        }
    };

    let has_class = move || active_class.get_untracked() < classes.get_untracked().len();

    // ── boxes ───────────────────────────────────────────────────────────────

    let arm_drag = move |idx: usize, kind: DragKind, origin: (f32, f32)| {
        if let Some(bx) = boxes.get_untracked().get(idx) {
            drag.set(Some(BoxDrag {
                idx,
                kind,
                origin,
                start: (bx.cx, bx.cy, bx.w, bx.h),
                moved: false,
            }));
        }
    };

    // Reported by a box rect (corner = None: select + move) or one of its
    // resize handles (Some(corner)) when pressed, with the normalized click.
    let on_press = Callback::new(move |(idx, corner, origin): (usize, Option<Corner>, (f32, f32))| {
        match corner {
            None => {
                selected_box.set(Some(idx));
                arm_drag(idx, DragKind::Move, origin);
            }
            Some(corner) => arm_drag(idx, DragKind::Resize(corner), origin),
        }
    });

    // ── polygons ────────────────────────────────────────────────────────────

    // A new object: one ring of the active class, selected, saved.
    let add_ring = move |ring: Vec<Pt>| {
        if !has_class() {
            on_need_class.run(());
            return;
        }
        let class_id = active_class.get_untracked() as u32;
        let mut index = 0;
        polygons.update(|ps| {
            let instance = next_instance(ps);
            ps.push(LabelPolygon {
                class_id,
                instance,
                points: ring,
            });
            index = ps.len() - 1;
        });
        poly.selected.set(Some(index));
        poly.vertex.set(None);
        on_save.run(false);
    };

    // Close a drawn ring: a freehand trace is simplified first; near-duplicate
    // vertices merge; fewer than three is an accidental click, a crossing ring
    // is refused.
    let finish_ring = move |points: Vec<Pt>, freehand: bool| {
        poly.draft.set(None);
        let px = canvas.get_untracked();
        let points = if freehand {
            simplify_rdp(&points, px, RDP_EPS_PX)
        } else {
            points
        };
        let points = dedupe_ring(&points, px, MIN_VERTEX_GAP_PX);
        if points.len() < 3 {
            return;
        }
        if !valid_ring(&points) {
            on_refused.run(());
            return;
        }
        add_ring(clockwise(points));
    };

    // Click mode: the first press starts a ring, a press on its first vertex
    // (from the third vertex on) closes it, any other press adds a vertex.
    let place_point = move |p: Pt| match poly.draft.get_untracked() {
        None => {
            if !has_class() {
                on_need_class.run(());
                return;
            }
            poly.selected.set(None);
            poly.vertex.set(None);
            poly.draft.set(Some(PolyDraft {
                points: vec![p],
                freehand: false,
                hover: Some(p),
            }));
        }
        Some(d) if closes_ring(&d.points, p, canvas.get_untracked()) => finish_ring(d.points, false),
        Some(_) => poly.draft.update(|slot| {
            if let Some(d) = slot.as_mut() {
                d.points.push(p);
            }
        }),
    };

    // Pointer moves of a polygon drag, and the click-mode rubber band, are
    // coalesced to one update per animation frame: only the latest position
    // is applied.
    let pending = StoredValue::new(None::<Pt>);
    let frame_queued = StoredValue::new(false);
    let flush = move || {
        let Some(Some(p)) = pending.try_update_value(|slot| slot.take()) else {
            return;
        };
        if let Some(Some(d)) = poly.drag.try_get_untracked() {
            let ring = apply_poly_drag(&d, p);
            let idx = d.idx;
            let _ = polygons.try_update(|ps| {
                if let Some(pg) = ps.get_mut(idx) {
                    pg.points = ring;
                }
            });
            let past_eps =
                (p[0] - d.origin[0]).abs() > DRAG_EPS || (p[1] - d.origin[1]).abs() > DRAG_EPS;
            if !d.moved && past_eps {
                let _ = poly.drag.try_update(|slot| {
                    if let Some(slot) = slot.as_mut() {
                        slot.moved = true;
                    }
                });
            }
        } else {
            let _ = poly.draft.try_update(|slot| {
                if let Some(d) = slot.as_mut().filter(|d| !d.freehand) {
                    d.hover = Some(p);
                }
            });
        }
    };
    let schedule = move |p: Pt| {
        pending.set_value(Some(p));
        if frame_queued.get_value() {
            return;
        }
        let Some(window) = web_sys::window() else {
            flush();
            return;
        };
        frame_queued.set_value(true);
        let tick = Closure::once_into_js(move || {
            let _ = frame_queued.try_update_value(|queued| *queued = false);
            flush();
        });
        if window.request_animation_frame(tick.unchecked_ref()).is_err() {
            frame_queued.set_value(false);
            flush();
        }
    };

    // Reported by a committed ring (body, edge or vertex) with the normalized
    // click. A press while placing points places one there instead.
    let on_poly_press = Callback::new(move |(idx, what, p): (usize, PolyPress, Pt)| {
        if poly.draft.get_untracked().is_some_and(|d| !d.freehand) {
            place_point(p);
            return;
        }
        let Some(ring) = polygons.with_untracked(|ps| ps.get(idx).map(|pg| pg.points.clone()))
        else {
            return;
        };
        let px = canvas.get_untracked();
        let was_selected = poly.selected.get_untracked() == Some(idx);
        poly.selected.set(Some(idx));
        let vertex = match what {
            PolyPress::Vertex(v) => Some(v),
            PolyPress::Body if was_selected => hit_vertex(&ring, p, px),
            _ => None,
        };
        if let Some(v) = vertex {
            poly.vertex.set(Some(v));
            poly.drag.set(Some(PolyDrag {
                idx,
                kind: PolyDragKind::Vertex(v),
                origin: p,
                start: ring,
                moved: false,
            }));
            return;
        }
        if let (PolyPress::Edge(_), Some((edge, q))) = (what, hit_edge(&ring, p, px)) {
            // Insert a vertex under the pointer and keep dragging it; the
            // insertion alone is an edit, so the release saves.
            let mut ring = ring;
            insert_vertex(&mut ring, edge, q);
            let inserted = ring.clone();
            polygons.update(|ps| {
                if let Some(pg) = ps.get_mut(idx) {
                    pg.points = inserted;
                }
            });
            poly.vertex.set(Some(edge + 1));
            poly.drag.set(Some(PolyDrag {
                idx,
                kind: PolyDragKind::Vertex(edge + 1),
                origin: q,
                start: ring,
                moved: true,
            }));
            return;
        }
        poly.vertex.set(None);
        poly.drag.set(Some(PolyDrag {
            idx,
            kind: PolyDragKind::Move,
            origin: p,
            start: ring,
            moved: false,
        }));
    });

    // Finish an edit of a committed ring: a valid one is saved, one that now
    // crosses itself is put back as it was and refused.
    let commit_poly_edit = move |d: PolyDrag| {
        let Some(ring) = polygons.with_untracked(|ps| ps.get(d.idx).map(|pg| pg.points.clone()))
        else {
            return;
        };
        if valid_ring(&ring) {
            on_save.run(false);
        } else {
            polygons.update(|ps| {
                if let Some(pg) = ps.get_mut(d.idx) {
                    pg.points = d.start.clone();
                }
            });
            on_refused.run(());
        }
    };

    // ── pointer handlers ────────────────────────────────────────────────────

    // SVG-level handler: only true background presses (on the svg itself).
    // Presses on a shape are handled by that element's own on:pointerdown.
    // In SAM mode the shapes are pointer-transparent, so every press reaches
    // here and becomes a point prompt. Pointer (not mouse) events so touch and
    // pen work too — `PointerEvent` derefs to `MouseEvent`, so the geometry
    // helpers and shift/ctrl modifiers are unchanged.
    let on_pointer_down = move |ev: leptos::ev::PointerEvent| {
        if selected.get_untracked().is_none() || !is_background(&ev) {
            return;
        }
        let Some((x, y)) = norm_coords(&ev) else {
            return;
        };
        ev.prevent_default();
        if sam_mode.get_untracked() {
            let positive = !(ev.shift_key() || ev.ctrl_key());
            sam_points.update(|p| p.push((x, y, positive)));
            return;
        }
        if !poly.enabled {
            // Background: clear selection and start drawing a new box.
            selected_box.set(None);
            draft.set(Some((x, y, x, y)));
            return;
        }
        match poly.mode.get_untracked() {
            DrawMode::Click => place_point([x, y]),
            DrawMode::Freehand => {
                if !has_class() {
                    on_need_class.run(());
                    return;
                }
                poly.selected.set(None);
                poly.vertex.set(None);
                poly.draft.set(Some(PolyDraft {
                    points: vec![[x, y]],
                    freehand: true,
                    hover: None,
                }));
                capture_pointer(&ev);
            }
            DrawMode::Rect => {
                poly.selected.set(None);
                poly.vertex.set(None);
                draft.set(Some((x, y, x, y)));
                capture_pointer(&ev);
            }
        }
    };

    let on_pointer_move = move |ev: leptos::ev::PointerEvent| {
        let active_drag = drag.get_untracked();
        let poly_active = poly.drag.with_untracked(Option::is_some)
            || poly.draft.with_untracked(Option::is_some);
        if active_drag.is_none() && draft.get_untracked().is_none() && !poly_active {
            return;
        }
        let Some((x, y)) = norm_coords(&ev) else {
            return;
        };
        if let Some(mut d) = active_drag {
            let (cx, cy, w, h) = apply_drag(&d, x, y);
            let idx = d.idx;
            boxes.update(|bs| {
                if let Some(b) = bs.get_mut(idx) {
                    b.cx = cx;
                    b.cy = cy;
                    b.w = w;
                    b.h = h;
                }
            });
            if !d.moved
                && ((x - d.origin.0).abs() > DRAG_EPS || (y - d.origin.1).abs() > DRAG_EPS)
            {
                d.moved = true;
                drag.set(Some(d));
            }
            return;
        }
        if poly.drag.with_untracked(Option::is_some) {
            schedule([x, y]);
            return;
        }
        if let Some(d) = poly.draft.get_untracked() {
            if !d.freehand {
                schedule([x, y]);
            } else if d
                .points
                .last()
                .is_none_or(|last| far_enough(*last, [x, y], canvas.get_untracked()))
            {
                poly.draft.update(|slot| {
                    if let Some(d) = slot.as_mut() {
                        d.points.push([x, y]);
                    }
                });
            }
            return;
        }
        draft.update(|d| {
            if let Some(d) = d.as_mut() {
                d.2 = x;
                d.3 = y;
            }
        });
    };

    let on_pointer_up = move |_: leptos::ev::PointerEvent| {
        // Finishing a move/resize: persist only if the box actually changed
        // (a plain click on a box just selects it).
        if let Some(d) = drag.get_untracked() {
            drag.set(None);
            if d.moved {
                on_save.run(false);
            }
            return;
        }
        if poly.drag.with_untracked(Option::is_some) {
            // Apply the last coalesced position before judging the edit.
            flush();
            if let Some(d) = poly.drag.get_untracked() {
                poly.drag.set(None);
                if d.moved {
                    commit_poly_edit(d);
                }
            }
            return;
        }
        if let Some(d) = poly.draft.get_untracked() {
            // A trace closes on release; a click-placed ring waits for its close.
            if d.freehand {
                finish_ring(d.points, true);
            }
            return;
        }
        let Some((x0, y0, x1, y1)) = draft.get_untracked() else {
            return;
        };
        draft.set(None);
        let (w, h) = ((x1 - x0).abs(), (y1 - y0).abs());
        // Ignore accidental clicks: the guard is a few *screen* pixels, not a
        // share of the image, so tiny objects on a native-resolution frame
        // can still be labeled.
        let (cw, ch) = canvas.get_untracked();
        if w * cw < MIN_DRAW_PX || h * ch < MIN_DRAW_PX || w < MIN_SIZE || h < MIN_SIZE {
            return;
        }
        let (cx, cy) = ((x0 + x1) / 2.0, (y0 + y1) / 2.0);
        if poly.enabled {
            // Rectangle mode: a box promoted to a four-vertex ring.
            add_ring(box_ring(cx, cy, w, h));
            return;
        }
        // A box needs a real class id, or set_labels rejects the save with
        // "Unknown class id". Block drawing until a class exists.
        if !has_class() {
            on_need_class.run(());
            return;
        }
        boxes.update(|bs| {
            bs.push(LabelBox {
                class_id: active_class.get_untracked() as u32,
                cx,
                cy,
                w,
                h,
            });
        });
        on_save.run(false);
    };

    // The browser took the pointer (a system gesture, a palm): the gesture is
    // abandoned — drafts dropped, an edit put back as it was.
    let on_pointer_cancel = move |_: leptos::ev::PointerEvent| {
        draft.set(None);
        poly.draft.set(None);
        pending.set_value(None);
        if let Some(d) = poly.drag.get_untracked() {
            poly.drag.set(None);
            polygons.update(|ps| {
                if let Some(pg) = ps.get_mut(d.idx) {
                    pg.points = d.start;
                }
            });
        }
        if let Some(d) = drag.get_untracked() {
            drag.set(None);
            boxes.update(|bs| {
                if let Some(b) = bs.get_mut(d.idx) {
                    (b.cx, b.cy, b.w, b.h) = d.start;
                }
            });
        }
    };

    // Leaving the canvas: an uncaptured box draft is dropped (captured
    // gestures never leave), and the rubber band hides.
    let on_pointer_leave = move |_: leptos::ev::PointerEvent| {
        if !poly.enabled {
            draft.set(None);
        }
        if poly.draft.with_untracked(|d| d.as_ref().is_some_and(|d| d.hover.is_some())) {
            poly.draft.update(|slot| {
                if let Some(d) = slot.as_mut() {
                    d.hover = None;
                }
            });
        }
    };

    // A double-click closes a click-placed ring of three or more vertices.
    let on_double_click = move |_: leptos::ev::MouseEvent| {
        if let Some(d) = poly.draft.get_untracked()
            && !d.freehand
            && d.points.len() >= 3
        {
            finish_ring(d.points, false);
        }
    };

    // ── view ────────────────────────────────────────────────────────────────

    view! {
        {move || match selected.get() {
            // No image yet: a square placeholder, width-driven (aspect-square)
            // and capped by the viewport height so it always fits vertically in
            // the fixed, non-scrolling app shell.
            None => view! {
                <div class=move || format!(
                    "ui-list-box aspect-square w-full {} mx-auto flex items-center justify-center text-sm ui-muted",
                    canvas_cap()
                )>
                    {t_string!(i18n, training::select_image_hint)}
                </div>
            }.into_any(),
            // Dataset images are not necessarily square (native-resolution
            // datasets store the 16:9 stereo-combined frame), and the image is
            // an `<img src=…>` — its size is only known once the browser has
            // decoded it. Read natural_width/height on `load` and size the
            // container by that ratio (square until then, so the layout does
            // not jump to zero height); the `<img>` and `<svg>` both fill the
            // container, so the picture is never distorted. The signal lives in
            // this branch so switching images starts from the fallback again.
            Some(image_id) => {
                let img_size = RwSignal::new(None::<(u32, u32)>);
                let on_img_load = move |ev: leptos::ev::Event| {
                    let Some(img) = ev
                        .target()
                        .and_then(|t| t.dyn_into::<web_sys::HtmlImageElement>().ok())
                    else {
                        return;
                    };
                    let (w, h) = (img.natural_width(), img.natural_height());
                    if w > 0 && h > 0 {
                        img_size.set(Some((w, h)));
                    }
                };
                let aspect = move || {
                    img_size
                        .get()
                        .map(|(w, h)| format!("{w} / {h}"))
                        .unwrap_or_else(|| "1 / 1".to_string())
                };
                // Height-fit cap scaled by the aspect ratio: a 16:9 image may be
                // wider than the square cap and still fit the available height.
                let canvas_box = move || {
                    let (w, h) = img_size.get().unwrap_or((1, 1));
                    format!("calc(({}) * {w} / {h})", canvas_height_avail())
                };
                // Measure the container whenever its layout size changes (image
                // decoded, window resized, assistant panel opened…) and feed
                // the overlay's px coordinate space. ResizeObserver reports once
                // on `observe`, so the first measurement needs no extra trigger.
                // The observer and its JS closure are not Send, so they live in
                // an owner-scoped local slot that the cleanup drains; `try_set`
                // because a late callback must not touch a disposed signal.
                let container = NodeRef::<leptos::html::Div>::new();
                let observer =
                    StoredValue::new_local(None::<(web_sys::ResizeObserver, Closure<dyn FnMut()>)>);
                let disconnect = move || {
                    observer.update_value(|slot| {
                        if let Some((obs, _)) = slot.take() {
                            obs.disconnect();
                        }
                    });
                };
                Effect::new(move |_| {
                    let Some(el) = container.get() else {
                        return;
                    };
                    disconnect();
                    let measured = el.clone();
                    let measure = Closure::<dyn FnMut()>::new(move || {
                        let rect = measured.get_bounding_client_rect();
                        let size = (rect.width() as f32, rect.height() as f32);
                        if size.0 > 0.0 && size.1 > 0.0 && canvas.try_get_untracked() != Some(size) {
                            let _ = canvas.try_set(size);
                        }
                    });
                    let Ok(obs) = web_sys::ResizeObserver::new(measure.as_ref().unchecked_ref())
                    else {
                        return;
                    };
                    obs.observe(&el);
                    observer.set_value(Some((obs, measure)));
                });
                on_cleanup(disconnect);
                let shapes = if poly.enabled {
                    view! {
                        {committed_polygons_layer(i18n, canvas, polygons, poly.selected, poly.vertex, sam_mode, classes, on_poly_press)}
                        {polygon_suggestions_layer(canvas, suggestions, suggestion_polygons, suggestion_names)}
                    }.into_any()
                } else {
                    view! {
                        {committed_boxes_layer(i18n, canvas, boxes, selected_box, sam_mode, classes, on_press)}
                        {suggestions_layer(canvas, suggestions, suggestion_names)}
                    }.into_any()
                };
                let status = if poly.enabled {
                    polygon_status_bar(i18n, polygons, classes, active_class).into_any()
                } else {
                    status_bar(i18n, boxes, classes, active_class).into_any()
                };
                view! {
                    <div
                        node_ref=container
                        class="ui-media-bg relative w-full mx-auto rounded overflow-hidden select-none"
                        style=("aspect-ratio", aspect)
                        style=("max-width", canvas_box)
                    >
                        <img
                            src=training_image_url(&dataset_id.get_value(), &image_id)
                            class="absolute inset-0 w-full h-full pointer-events-none"
                            alt=t_string!(i18n, training::labeling_image_alt)
                            draggable="false"
                            on:load=on_img_load
                        />
                        // touch-none: claim the gesture so a finger-drag draws
                        // instead of scrolling/zooming the page (browsers cancel
                        // pointermove mid-pan otherwise).
                        //
                        // The viewBox is the measured canvas size in CSS px, so one
                        // overlay unit is one screen pixel (fixed-size handles,
                        // strokes and text, undistorted glyphs on a 16:9 image) and
                        // normalized coordinates map onto the picture by multiplying
                        // with that size. preserveAspectRatio="none" keeps the
                        // overlay stretched over the whole image in the frame between
                        // a resize and the next measurement; the pointer math
                        // (`norm_coords`) measures against the svg's own box anyway.
                        <svg
                            class="absolute inset-0 w-full h-full cursor-crosshair touch-none"
                            viewBox=move || {
                                let (w, h) = canvas.get();
                                format!("0 0 {w} {h}")
                            }
                            preserveAspectRatio="none"
                            on:pointerdown=on_pointer_down
                            on:pointermove=on_pointer_move
                            on:pointerup=on_pointer_up
                            on:pointercancel=on_pointer_cancel
                            on:pointerleave=on_pointer_leave
                            on:dblclick=on_double_click
                        >
                            {shapes}
                            {points_layer(canvas, sam_points)}
                            {draft_layer(canvas, draft, classes, active_class)}
                            {polygon_draft_layer(canvas, poly.draft, classes, active_class)}
                        </svg>
                    </div>
                    {status}
                }.into_any()
            }
        }}
    }
}
