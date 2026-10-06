// SPDX-FileCopyrightText: 2026 Conecsa
//
// SPDX-License-Identifier: AGPL-3.0-only

//! The label editor's drawing surface. The component here owns the layout
//! and the overlay's coordinate space; `drawing` handles the pointer on the
//! svg (boxes, SAM points, dispatch), `rings` the polygon tool, and the
//! shared `conecsa_ui_view::viewport` the pan / pinch / wheel gestures of
//! the viewport and its DOM plumbing.

use conecsa_ui_view::viewport::{watch_viewport, Gestures};
use leptos::prelude::*;
use wasm_bindgen::JsCast;

use crate::api::training_image_url;
use crate::i18n::*;

use super::dataset_editor::{AiState, ClassesState, ImagesState};
use super::label_editor::{PolygonTool, ViewState};
use super::label_geometry::BoxDrag;
use super::label_shapes::{
    committed_boxes_layer, committed_polygons_layer, draft_layer, points_layer,
    polygon_draft_layer, polygon_status_bar, polygon_suggestions_layer, status_bar,
    suggestions_layer,
};

mod drawing;
mod rings;

use drawing::Drawing;
use rings::Rings;

/// A class is selected to label with.
fn has_class(classes: ClassesState) -> bool {
    classes.active.get_untracked() < classes.list.get_untracked().len()
}

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
///
/// The image can be zoomed and panned (`view`): Ctrl+wheel or a trackpad
/// pinch zooms about the cursor, the plain wheel pans once zoomed, two fingers
/// pinch and drag, and a middle-button or Space+drag pans. Those view gestures
/// are claimed by capture-phase handlers on the viewport before any drawing
/// handler sees them; a pinch that starts mid-gesture abandons that gesture
/// first.
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
    /// Zoom/pan of the image, shared with the toolbar and the key handler.
    view: ViewState,
    /// SAM prompt mode: background clicks become point prompts.
    sam_mode: Signal<bool>,
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
    let class_names = classes.list.read_only();
    let sam_points = ai.sam_points;
    let suggestions = ai.suggestions.read_only();
    let suggestion_names = ai.suggestion_names.read_only();
    let suggestion_polygons = ai.suggestion_polygons.read_only();
    // Copy-able handle so the canvas closure can build per-image URLs.
    let dataset_id = StoredValue::new(dataset_id);
    // Rendered viewport size in CSS px, seeded with the fallback and kept
    // current by a ResizeObserver on the viewport (see the image branch
    // below); shared through `view` so the editor's arrow-key nudge knows
    // how big a screen pixel is. The image stage inside it is `zoom` times
    // as large by construction, so the overlay's coordinate space — the
    // on-screen size of the zoomed image, in CSS px, so strokes, handles,
    // text and every hit test keep a fixed on-screen size whatever the
    // monitor, image aspect or zoom — is derived, not measured.
    let viewport = view.viewport;
    let canvas = Memo::new(move |_| {
        let (w, h) = viewport.get();
        let zoom = view.transform.get().zoom;
        (w * zoom, h * zoom)
    });

    let rings = Rings::new(polygons, poly, classes, canvas, on_save, on_need_class, on_refused);
    let drawing = Drawing {
        selected,
        boxes,
        selected_box,
        drag,
        draft: RwSignal::new(None),
        poly,
        rings,
        classes,
        canvas,
        sam_mode,
        sam_points,
        on_save,
        on_need_class,
    };
    let draft = drawing.draft;
    let gestures = Gestures::new(view, Callback::new(move |()| drawing.abort()));

    // Reported by the shapes of the overlay layers when pressed.
    let on_press = Callback::new(move |(idx, corner, origin)| drawing.box_press(idx, corner, origin));
    let on_poly_press = Callback::new(move |(idx, what, p)| rings.press(idx, what, p));

    view! {
        {move || match selected.get() {
            // No image yet: a square placeholder sized like the canvas (the
            // slot fits it to the height left in the card, see
            // `.ui-label-canvas` in label-editor.css).
            None => view! {
                <div class="ui-label-canvas">
                    <div class="ui-label-canvas-fit">
                        <div class="ui-list-box ui-label-placeholder ui-muted">
                            {t_string!(i18n, training::select_image_hint)}
                        </div>
                    </div>
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
                let container = NodeRef::<leptos::html::Div>::new();
                gestures.reset();
                watch_viewport(container, viewport, move |ev| gestures.wheel(container, ev));

                let shapes = if poly.enabled {
                    view! {
                        {committed_polygons_layer(i18n, canvas.into(), polygons, poly.selected, poly.vertex, sam_mode, class_names, on_poly_press)}
                        {polygon_suggestions_layer(canvas.into(), suggestions, suggestion_polygons, suggestion_names)}
                    }.into_any()
                } else {
                    view! {
                        {committed_boxes_layer(i18n, canvas.into(), boxes, selected_box, sam_mode, class_names, on_press)}
                        {suggestions_layer(canvas.into(), suggestions, suggestion_names)}
                    }.into_any()
                };
                let status = if poly.enabled {
                    polygon_status_bar(i18n, polygons, class_names, active_class).into_any()
                } else {
                    status_bar(i18n, boxes, class_names, active_class).into_any()
                };
                // The viewport clips the stage — the image and its overlay,
                // laid out `zoom` times as large by the CSS custom properties
                // (`.ui-label-stage` in label-editor.css) and offset so the
                // normalized point (ox, oy) sits at the top-left corner.
                // `touch-action: none` (in the CSS) claims every touch so a
                // finger draws and two zoom, instead of scrolling the page.
                // The capture-phase handlers run before the svg and shape
                // handlers (which stop propagation among themselves) and,
                // when they claim the event, stop it there.
                // The slot (`.ui-label-canvas`) fits the viewport to the
                // height left in the card through the aspect ratio.
                view! {
                    <div class="ui-label-canvas">
                        <div class="ui-label-canvas-fit">
                            <div
                                node_ref=container
                                class="ui-media-bg ui-label-viewport"
                                class:ui-label-viewport-pan=move || view.space_held.get()
                                class:ui-label-viewport-panning=move || gestures.panning.get()
                                style=("aspect-ratio", aspect)
                                style=("--label-zoom", move || view.transform.get().zoom.to_string())
                                style=("--label-ox", move || view.transform.get().ox.to_string())
                                style=("--label-oy", move || view.transform.get().oy.to_string())
                                on:pointerdown:capture=move |ev| gestures.press(container, ev)
                                on:pointermove:capture=move |ev| gestures.motion(ev)
                                on:pointerup:capture=move |ev| gestures.release(ev)
                                on:pointercancel:capture=move |ev| gestures.release(ev)
                            >
                                <div class="ui-label-stage">
                                    <img
                                        src=training_image_url(&dataset_id.get_value(), &image_id)
                                        alt=t_string!(i18n, training::labeling_image_alt)
                                        draggable="false"
                                        on:load=on_img_load
                                    />
                                    // The viewBox is the on-screen size of the zoomed image
                                    // in CSS px, so one overlay unit is one screen pixel
                                    // (fixed-size handles, strokes and text, undistorted
                                    // glyphs on a 16:9 image) and normalized coordinates map
                                    // onto the picture by multiplying with that size.
                                    // preserveAspectRatio="none" keeps the overlay stretched
                                    // over the whole image in the frame between a resize and
                                    // the next measurement; the pointer math (`norm_coords`)
                                    // measures against the svg's own box anyway.
                                    <svg
                                        viewBox=move || {
                                            let (w, h) = canvas.get();
                                            format!("0 0 {w} {h}")
                                        }
                                        preserveAspectRatio="none"
                                        on:pointerdown=move |ev| drawing.press(ev)
                                        on:pointermove=move |ev| drawing.motion(ev)
                                        on:pointerup=move |_| drawing.release()
                                        on:pointercancel=move |_| drawing.abort()
                                        on:pointerleave=move |_| drawing.leave()
                                        on:dblclick=move |_| rings.double_click()
                                    >
                                        {shapes}
                                        {points_layer(canvas.into(), sam_points)}
                                        {draft_layer(canvas.into(), draft, class_names, active_class)}
                                        {polygon_draft_layer(canvas.into(), poly.draft, class_names, active_class)}
                                    </svg>
                                </div>
                            </div>
                        </div>
                        {status}
                    </div>
                }.into_any()
            }
        }}
    }
}
