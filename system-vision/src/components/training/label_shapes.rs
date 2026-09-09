//! SVG overlay layers for the label canvas.
//!
//! These are plain view-builder functions (no component owners), so the
//! committed-boxes layer stays a single reactive closure that re-renders the
//! whole list on change — a per-box `<For>`/component would dispose+recreate
//! items mid-drag and panic on the disposed signal.

use leptos::prelude::*;

use crate::api::LabelBox;
use crate::i18n::*;

use super::label_geometry::{handle_rects, label_anchor, norm_coords, Corner, HANDLE};
use crate::class_color::{class_color_for, class_display_name};

/// Committed boxes: each as a class-colored rect with its class name drawn
/// *outside* the box (see `label_anchor`), plus hollow corner resize handles
/// on the selected one (outside SAM mode; placed by `handle_rects`, which
/// steps them outside a small box). While a box is in that editing state its
/// name is hidden so nothing covers the object being fitted — the tinted fill
/// and thicker stroke already mark it. Pointer events on a rect/handle report
/// up via the callbacks with the normalized click position; in SAM mode the
/// boxes are pointer-transparent so clicks become point prompts.
///
/// `canvas` is the rendered canvas size in px — the overlay's coordinate
/// space — so every stroke/handle/text keeps its on-screen size.
pub(super) fn committed_boxes_layer(
    i18n: leptos_i18n::I18nContext<Locale>,
    canvas: RwSignal<(f32, f32)>,
    boxes: RwSignal<Vec<LabelBox>>,
    selected_box: RwSignal<Option<usize>>,
    sam_mode: Signal<bool>,
    classes: ReadSignal<Vec<String>>,
    // on_press: (box index, pressed corner, normalized click). `None` is a
    // press on the body → select + start a move; `Some` a handle → resize.
    on_press: Callback<(usize, Option<Corner>, (f32, f32))>,
) -> impl IntoView {
    move || {
        let sel = selected_box.get();
        let in_sam = sam_mode.get();
        let cls = classes.get();
        let (cw, ch) = canvas.get();
        boxes.get().into_iter().enumerate().map(|(i, b)| {
            let color = class_color_for(b.class_id as usize, &cls);
            let label = cls.get(b.class_id as usize)
                .map(|e| class_display_name(e))
                .unwrap_or_else(|| t_string!(i18n, training::class_fallback, id = b.class_id));
            let x = (b.cx - b.w / 2.0) * cw;
            let y = (b.cy - b.h / 2.0) * ch;
            let right = (b.cx + b.w / 2.0) * cw;
            let bottom = (b.cy + b.h / 2.0) * ch;
            let is_sel = sel == Some(i);
            // Editing = selected with handles (never in SAM mode, where boxes
            // are locked).
            let editing = is_sel && !in_sam;
            let handles = editing.then(|| {
                let color = color.clone();
                handle_rects(x, y, right, bottom, (cw, ch)).into_iter().map(move |(hx, hy, corner)| view! {
                    <rect
                        class=format!("ui-label-handle {}", corner.cursor_class())
                        x=hx y=hy
                        width=HANDLE height=HANDLE
                        stroke=color.clone()
                        on:pointerdown=move |ev: leptos::ev::PointerEvent| {
                            ev.stop_propagation();
                            ev.prevent_default();
                            if let Some(o) = norm_coords(&ev) {
                                on_press.run((i, Some(corner), o));
                            }
                        }
                    />
                }).collect::<Vec<_>>()
            });
            let name = (!editing).then(|| {
                let (tx, ty) = label_anchor(x, y, bottom, ch);
                view! {
                    <text class="ui-label-text" x=tx y=ty fill=color.clone()>{label}</text>
                }
            });
            view! {
                <rect
                    // SAM mode: locked = pointer-transparent so clicks become
                    // point prompts, not selection.
                    class=if in_sam { "ui-label-box-locked" } else { "ui-label-box" }
                    x=x y=y
                    width=b.w * cw height=b.h * ch
                    fill=if is_sel { format!("{}33", color) } else { "none".to_string() }
                    stroke=color.clone()
                    stroke-width=if is_sel { 2.0 } else { 1.25 }
                    on:pointerdown=move |ev: leptos::ev::PointerEvent| {
                        ev.stop_propagation();
                        ev.prevent_default();
                        if let Some(o) = norm_coords(&ev) {
                            on_press.run((i, None, o));
                        }
                    }
                />
                {name}
                {handles}
            }
        }).collect::<Vec<_>>()
    }
}

/// AI suggestions — dashed boxes, not yet part of the labels. A model's
/// suggestions carry its class name (`names[i]`, drawn outside the box like a
/// committed one); SAM's carry none. One
/// reactive closure over the whole list (not a keyed `<For>`): a re-run with
/// the same number of boxes must redraw them, and index keys would not.
pub(super) fn suggestions_layer(
    canvas: RwSignal<(f32, f32)>,
    suggestions: ReadSignal<Vec<LabelBox>>,
    names: ReadSignal<Vec<String>>,
) -> impl IntoView {
    move || {
        let names = names.get();
        let (cw, ch) = canvas.get();
        suggestions.get().into_iter().enumerate().map(|(i, b)| {
            let x = (b.cx - b.w / 2.0) * cw;
            let y = (b.cy - b.h / 2.0) * ch;
            let bottom = (b.cy + b.h / 2.0) * ch;
            let label = names.get(i).filter(|n| !n.is_empty()).cloned();
            let (tx, ty) = label_anchor(x, y, bottom, ch);
            view! {
                <rect
                    class="ui-label-suggestion"
                    x=x y=y
                    width=b.w * cw
                    height=b.h * ch
                />
                {label.map(|n| view! {
                    <text class="ui-label-text" x=tx y=ty>{n}</text>
                })}
            }
        }).collect::<Vec<_>>()
    }
}

/// SAM point prompts — green (positive) / red (negative) dots.
pub(super) fn points_layer(
    canvas: RwSignal<(f32, f32)>,
    sam_points: RwSignal<Vec<(f32, f32, bool)>>,
) -> impl IntoView {
    view! {
        <For
            each={move || sam_points.get().into_iter().enumerate().collect::<Vec<_>>()}
            key=|(i, _)| *i
            children=move |(_, (x, y, positive)): (usize, (f32, f32, bool))| view! {
                <circle
                    class=if positive {
                        "ui-label-point ui-label-point-positive"
                    } else {
                        "ui-label-point ui-label-point-negative"
                    }
                    cx=move || x * canvas.get().0
                    cy=move || y * canvas.get().1
                    r="5"
                />
            }
        />
    }
}

/// The dashed rectangle drawn while dragging a new box on the background.
pub(super) fn draft_layer(
    canvas: RwSignal<(f32, f32)>,
    draft: RwSignal<Option<(f32, f32, f32, f32)>>,
    classes: ReadSignal<Vec<String>>,
    active_class: ReadSignal<usize>,
) -> impl IntoView {
    move || draft.get().map(|(x0, y0, x1, y1)| {
        let (cw, ch) = canvas.get();
        view! {
        <rect
            class="ui-label-draft"
            x=x0.min(x1) * cw
            y=y0.min(y1) * ch
            width=(x1 - x0).abs() * cw
            height=(y1 - y0).abs() * ch
            stroke=class_color_for(active_class.get_untracked(), &classes.get_untracked())
        />
    }})
}

/// Footer status: box count + the class new boxes will be drawn as. Labels
/// autosave, so there is no Save button — just this line.
pub(super) fn status_bar(
    i18n: leptos_i18n::I18nContext<Locale>,
    boxes: RwSignal<Vec<LabelBox>>,
    classes: ReadSignal<Vec<String>>,
    active_class: ReadSignal<usize>,
) -> impl IntoView {
    view! {
        <div class="flex items-center">
            <span class="ui-help">
                {move || t_string!(i18n, training::boxes_drawing_as, count = boxes.get().len())}
                <span class="font-semibold" style=move || classes.with(|c| format!(
                    "color: {}", class_color_for(active_class.get(), c)
                ))>
                    {move || classes.with(|c| c
                        .get(active_class.get())
                        .map(|e| class_display_name(e))
                        .unwrap_or_else(|| t_string!(i18n, training::no_class_yet).to_string()))}
                </span>
            </span>
        </div>
    }
}
