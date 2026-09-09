//! Leptos UI components for the web frontend.

use leptos::prelude::*;

use super::dataset_editor::{AiState, Assistant, ClassesState, ImagesState, LabelActions};
use super::label_canvas::LabelCanvas;
use super::label_geometry::BoxDrag;
use super::label_model_panel::LabelModelPanel;
use super::label_sam_panel::LabelSamPanel;
use super::label_toolbar::LabelToolbar;

/// Bounding-box editor card. Owns the shared selection (`selected_box`) and the
/// in-progress move/resize (`drag`) plus the global Delete / pointerup listeners,
/// and composes the toolbar, the (optional) assistant bar — SAM prompts or a
/// device model's Detect — and the canvas.
///
/// Canvas interaction: drag the body to move, drag a corner handle to resize,
/// drag on the background to draw a new box; in SAM mode clicks become point
/// prompts (Shift/Ctrl-click = negative) and suggestions render dashed until
/// accepted.
#[component]
pub(super) fn LabelEditor(
    dataset_id: String,
    images: ImagesState,
    classes: ClassesState,
    ai: AiState,
    actions: LabelActions,
) -> impl IntoView {
    let selected = images.selected;
    let boxes = images.boxes;
    let assistant = ai.assistant;
    let on_save = actions.on_save;
    let selected_box = RwSignal::new(None::<usize>);
    // Active move/resize of an already-committed box (shared with the canvas and
    // the global mouseup listener).
    let drag = RwSignal::new(None::<BoxDrag>);
    let sam_mode = Signal::derive(move || assistant.get() == Assistant::Sam);
    let panel_open = Signal::derive(move || assistant.get() != Assistant::Off);

    // A fresh image clears any selection/drag so a stale index can't carry over
    // to the next image's boxes after they autoload.
    Effect::new(move |_| {
        let _ = selected.get();
        selected_box.set(None);
        drag.set(None);
    });

    // Delete key removes the selected box (Delete only — not Backspace, which is
    // used while typing class names / SAM prompts). Ignored when a form field is
    // focused so typing never deletes a box.
    let key_handle = window_event_listener(leptos::ev::keydown, move |ev: web_sys::KeyboardEvent| {
        if ev.key() != "Delete" {
            return;
        }
        let focused_tag = web_sys::window()
            .and_then(|w| w.document())
            .and_then(|d| d.active_element())
            .map(|el| el.tag_name().to_uppercase());
        if let Some(tag) = focused_tag {
            if matches!(tag.as_str(), "INPUT" | "TEXTAREA" | "SELECT") {
                return;
            }
        }
        // try_* throughout: these are global listeners that can briefly outlive
        // the editor's signals during teardown — a disposed signal must no-op,
        // not panic.
        if let Some(Some(idx)) = selected_box.try_get_untracked() {
            let _ = boxes.try_update(|bs| {
                if idx < bs.len() {
                    bs.remove(idx);
                }
            });
            let _ = selected_box.try_set(None);
            let _ = drag.try_set(None);
            on_save.run(false);
        }
    });
    on_cleanup(move || key_handle.remove());

    // Commit an in-progress move/resize on pointerup anywhere — so a drag that
    // runs past the (small) canvas edge still finishes instead of being
    // stranded. Pointer (not mouse) so touch/pen drags commit too.
    let up_handle = window_event_listener(leptos::ev::pointerup, move |_| {
        if let Some(Some(d)) = drag.try_get_untracked() {
            let _ = drag.try_set(None);
            if d.moved {
                on_save.run(false);
            }
        }
    });
    on_cleanup(move || up_handle.remove());

    // Toolbar actions on the selected box.
    let on_delete = Callback::new(move |_: ()| {
        if let Some(idx) = selected_box.get_untracked() {
            boxes.update(|bs| {
                if idx < bs.len() {
                    bs.remove(idx);
                }
            });
            selected_box.set(None);
            on_save.run(false);
        }
    });
    let on_set_class = Callback::new(move |class_id: u32| {
        if let Some(idx) = selected_box.get_untracked() {
            boxes.update(|bs| {
                if let Some(b) = bs.get_mut(idx) {
                    b.class_id = class_id;
                }
            });
            on_save.run(false);
        }
    });

    view! {
        <div class="ui-card ui-card-pad-sm flex flex-col gap-3">
            <LabelToolbar
                selected_box=selected_box.read_only()
                boxes=boxes
                classes=classes
                ai=ai
                on_set_class=on_set_class
                on_delete=on_delete
                on_assistant_change=actions.on_assistant_change
            />

            {move || match assistant.get() {
                Assistant::Sam => view! {
                    <LabelSamPanel
                        ai=ai
                        on_sam_suggest=actions.on_sam_suggest
                        on_sam_accept=actions.on_accept
                        on_sam_clear=actions.on_clear
                    />
                }.into_any(),
                Assistant::Model(_) => view! {
                    <LabelModelPanel
                        ai=ai
                        on_detect=actions.on_model_detect
                        on_accept=actions.on_accept
                        on_clear=actions.on_clear
                    />
                }.into_any(),
                Assistant::Off => view! { <span/> }.into_any(),
            }}

            <LabelCanvas
                dataset_id=dataset_id
                images=images
                classes=classes
                ai=ai
                selected_box=selected_box
                drag=drag
                sam_mode=sam_mode
                panel_open=panel_open
                on_save=on_save
                on_need_class=actions.on_need_class
            />
        </div>
    }
}
