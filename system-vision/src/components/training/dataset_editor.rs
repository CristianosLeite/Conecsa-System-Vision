//! Capture → label → train editor for ONE dataset.
//!
//! `DatasetEditor` owns the editor's signals ([`state::EditorState`]), spawns
//! the mount-time work (`tasks`), wires the callbacks (`actions`, `ai`) and
//! renders the three-column layout. Decisions with no UI in them live in
//! `logic`, where they are unit-tested.

mod actions;
mod ai;
mod logic;
mod state;
mod tasks;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use leptos::prelude::*;

use crate::api::DatasetSummary;
use crate::components::PopupMessages;
use crate::components::configuration::model_conversion::PendingConversion;
use crate::components::control_panel::ViewMode;
use crate::i18n::*;

use super::capture_panel::CapturePanel;
use super::classes_panel::ClassesPanel;
use super::gallery::Gallery;
use super::label_editor::LabelEditor;
use super::progress_overlay::TrainingProgressOverlay;
use super::replicate_modal::ReplicateModal;
use super::train_modal::TrainModal;

pub(super) use actions::GalleryActions;
pub(super) use ai::LabelActions;
pub(super) use state::{AiState, Assistant, ClassesState, ImagesState};

use state::EditorState;

const MIN_IMAGES_DEFAULT: u32 = 20;

/// Capture → label → train flow for ONE dataset. Mounted by `TrainingView`
/// when a dataset is opened from the gallery; everything here is scoped by
/// the dataset's id.
#[component]
pub(super) fn DatasetEditor(
    dataset: DatasetSummary,
    on_back: Callback<()>,
    set_current_view: WriteSignal<ViewMode>,
    set_pending_conversion: WriteSignal<Option<PendingConversion>>,
    /// TrainingView's exit-dedup flag: set when this editor exits training
    /// mode itself (training-done handoff) so the parent's defensive cleanup
    /// never fires a second exit that would resume detection mid-conversion.
    exited: Arc<AtomicBool>,
) -> impl IntoView {
    let i18n = use_i18n();
    let dataset_name = dataset.name.clone();
    let st = EditorState::new(&dataset);

    // Guards the long-lived polling loop: signals owned by this component are
    // disposed on unmount, so the loop must stop touching them once we leave.
    let alive = Arc::new(AtomicBool::new(true));
    {
        let alive = alive.clone();
        on_cleanup(move || alive.store(false, Ordering::Relaxed));
    }

    tasks::spawn_initial_loads(st, i18n, alive.clone());
    tasks::spawn_sam_warmup(st, alive.clone());
    tasks::install_active_class_clamp(st);
    tasks::spawn_job_poll(
        st,
        i18n,
        alive,
        exited,
        set_pending_conversion,
        set_current_view,
    );

    // ── actions ───────────────────────────────────────────────────────────────

    let gallery = GalleryActions::new(st, i18n);
    let save_labels = actions::save_labels(st, i18n);
    let label = LabelActions::new(st, i18n, save_labels);
    let on_capture = actions::capture(st, i18n, gallery.on_select);
    let on_class_add = actions::class_add(st, i18n);
    let on_class_rename = actions::class_rename(st, i18n);
    let on_class_remove = actions::class_remove(st, i18n);
    let on_replicate_confirm = actions::replicate_confirm(st, i18n);
    let on_train_request = actions::train_request(st, save_labels);
    let on_train_start = actions::train_start(st, i18n);
    let on_train_cancel = actions::train_cancel(st, i18n);
    let on_train_finish = actions::train_finish(st, i18n);
    let on_back_click = actions::back(st, i18n, on_back);

    // ── derived ───────────────────────────────────────────────────────────────

    let labeled_count = Signal::derive(move || logic::labeled_count(&st.images.list.get()));
    let image_count = Signal::derive(move || st.images.list.get().len() as u32);
    let min_images = st.training.min_images;
    let can_train = Signal::derive(move || {
        logic::can_train(
            image_count.get(),
            labeled_count.get(),
            min_images.get(),
            !st.classes.list.get().is_empty(),
        )
    });
    let default_base_model = Signal::derive(move || {
        logic::default_base_model(&st.ai.assistant.get(), &st.training.base_models.get())
    });

    view! {
        <div class="flex flex-col h-full min-h-0">
            // ── editor bar ──────────────────────────────────────────────────
            <div class="ui-topbar flex items-center justify-between px-4 py-3">
                <div class="flex items-center gap-3">
                    <button
                        class="ui-button ui-button-neutral ui-button-md"
                        on:click=move |_| on_back_click.run(())
                    >
                        "← "
                        {t!(i18n, training::back_to_datasets)}
                    </button>
                    <h1 class="text-lg font-semibold">
                        {dataset_name}
                    </h1>
                </div>
                <div class="flex items-center gap-4">
                    <span class="ui-help text-sm">
                        {move || t_string!(
                            i18n,
                            training::images_counter,
                            count = image_count.get(),
                            labeled = labeled_count.get(),
                            min = min_images.get()
                        )}
                    </span>
                    <button
                        class="ui-button ui-button-success ui-button-md"
                        disabled=move || !can_train.get()
                        title=move || if can_train.get() {
                            t_string!(i18n, training::train_tooltip_ready).to_string()
                        } else {
                            t_string!(
                                i18n,
                                training::train_tooltip_requirements,
                                min = min_images.get()
                            )
                        }
                        on:click=move |_| on_train_request.run(())
                    >
                        {t!(i18n, training::train_model)}
                    </button>
                </div>
            </div>

            <div class="app-alert-slot">
                <PopupMessages
                    error_msg=st.notices.error_msg.read_only()
                    success_msg=st.notices.success_msg.read_only()
                    info_view=st.notices.info_view.read_only()
                    set_error_msg=st.notices.error_msg.write_only()
                    set_success_msg=st.notices.success_msg.write_only()
                    _set_info_view=st.notices.info_view.write_only()
                />
            </div>

            // ── body ────────────────────────────────────────────────────────
            <main class="app-main">
                // h-full + min-h-0 per column: the app shell is a fixed
                // viewport (overflow hidden), so each column scrolls on
                // its own instead of pushing past the window bottom.
                <div class="grid grid-cols-12 gap-4 w-full h-full min-h-0">
                    <div class="col-span-3 flex flex-col gap-4 min-h-0 max-h-full overflow-y-auto">
                        <CapturePanel
                            on_capture=on_capture
                            capturing=st.images.capturing.read_only()
                        />
                        <ClassesPanel
                            classes=st.classes.list.read_only()
                            active_class=st.classes.active.read_only()
                            set_active_class=st.classes.active.write_only()
                            on_add=on_class_add
                            on_rename=on_class_rename
                            on_remove=on_class_remove
                        />
                    </div>

                    <div class="col-span-6 min-h-0 max-h-full overflow-y-auto">
                        <LabelEditor
                            dataset_id=dataset.dataset_id.clone()
                            images=st.images
                            classes=st.classes
                            ai=st.ai
                            actions=label
                        />
                    </div>

                    <div class="col-span-3 min-h-0 max-h-full overflow-y-auto">
                        <Gallery
                            dataset_id=dataset.dataset_id.clone()
                            images=st.images
                            actions=gallery
                        />
                    </div>
                </div>
            </main>

            <TrainModal
                visible=st.training.show_modal.read_only()
                set_visible=st.training.show_modal.write_only()
                base_models=st.training.base_models.read_only()
                default_base_model=default_base_model
                on_start=on_train_start
            />

            <ReplicateModal
                visible=st.replicate.show_modal.read_only()
                set_visible=st.replicate.show_modal.write_only()
                count=st.replicate.count.read_only()
                set_count=st.replicate.count.write_only()
                busy=st.replicate.busy.read_only()
                on_confirm=on_replicate_confirm
            />

            <TrainingProgressOverlay
                job=st.training.job.read_only()
                on_cancel=on_train_cancel
                on_finish=on_train_finish
            />
        </div>
    }
}
