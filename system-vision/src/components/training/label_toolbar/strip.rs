// SPDX-FileCopyrightText: 2026 Conecsa
//
// SPDX-License-Identifier: AGPL-3.0-only

//! The tool strip: the selected label's class and delete buttons, undo /
//! redo / clear labels, and the zoom controls.
//!
//! Every control is mounted once and shown or hidden with the `hidden`
//! attribute, never re-created: the class select keeps its focus across a
//! reassignment, and the strip (a fixed-height, non-wrapping row) only
//! changes width as controls come and go. The zoom group is always shown as
//! the view's state, disabled at its bounds.

use leptos::prelude::*;

use crate::class_color::class_display_name;
use crate::i18n::*;

use super::super::dataset_editor::ClassesState;
use super::icons::ToolIcon;
use super::{HistoryTools, SelectionTools, ZoomTools};

/// One 2rem icon button of the strip: a tooltip and an accessible name (the
/// icon itself is decorative), hidden when it does not apply.
#[component]
fn ToolButton(
    icon: ToolIcon,
    /// The tooltip.
    #[prop(into)]
    title: Signal<String>,
    /// The accessible name.
    #[prop(into)]
    label: Signal<String>,
    /// A destructive action: red on hover.
    #[prop(optional)]
    danger: bool,
    #[prop(optional, into)]
    disabled: Option<Signal<bool>>,
    #[prop(optional, into)]
    hidden: Option<Signal<bool>>,
    on_click: Callback<()>,
) -> impl IntoView {
    view! {
        <button
            class="ui-icon-button ui-icon-button-md"
            class:ui-icon-button-danger=danger
            hidden=move || hidden.is_some_and(|h| h.get())
            disabled=move || disabled.is_some_and(|d| d.get())
            title=move || title.get()
            aria-label=move || label.get()
            on:click=move |_| on_click.run(())
        >
            {icon.view()}
        </button>
    }
}

#[component]
pub(super) fn ToolStrip(
    classes: ClassesState,
    selection: SelectionTools,
    history: HistoryTools,
    zoom: ZoomTools,
) -> impl IntoView {
    let i18n = use_i18n();
    let class_list = classes.list;
    let has_selection = Memo::new(move |_| selection.class.get().is_some());
    let no_vertex = Memo::new(move |_| !selection.vertex.get());
    let no_history = Memo::new(move |_| {
        !(history.can_undo.get() || history.can_redo.get() || history.has_labels.get())
    });
    let no_undo = Memo::new(move |_| !history.can_undo.get());
    let no_redo = Memo::new(move |_| !history.can_redo.get());
    let no_labels = Memo::new(move |_| !history.has_labels.get());
    // The comparisons live outside the markup: a bare `>=` in an attribute
    // closes the tag for the view macro.
    let max_percent = zoom.max_percent;
    let at_min = Memo::new(move |_| zoom.percent.get() <= 100);
    let at_max = Memo::new(move |_| zoom.percent.get() >= max_percent);

    view! {
        <div class="ui-label-tools" role="toolbar" aria-label=move || t_string!(i18n, training::label_tools)>
            // The selected label: reassign its class, delete its vertex or
            // delete it.
            <div class="ui-label-tools-group" hidden=move || !has_selection.get()>
                <select
                    class="ui-select ui-select-sm ui-label-tools-class cursor-pointer"
                    title=move || t_string!(i18n, training::reassign_box_class)
                    on:change=move |ev| {
                        if let Ok(v) = event_target_value(&ev).parse::<u32>() {
                            selection.on_set_class.run(v);
                        }
                    }
                >
                    // `prop:selected`: the select is long-lived, and the DOM
                    // ignores `selected` attributes once it has been touched.
                    {move || class_list.get().iter().enumerate().map(|(ci, name)| {
                        let ci = ci as u32;
                        view! {
                            <option
                                value=ci.to_string()
                                prop:selected=move || selection.class.get() == Some(ci)
                            >
                                {class_display_name(name)}
                            </option>
                        }
                    }).collect::<Vec<_>>()}
                </select>
                <ToolButton
                    icon=ToolIcon::DeleteVertex
                    title=Signal::derive(move || t_string!(i18n, training::delete_vertex_title).to_string())
                    label=Signal::derive(move || t_string!(i18n, training::delete_vertex).to_string())
                    hidden=no_vertex
                    on_click=selection.on_delete_vertex
                />
                <ToolButton
                    icon=ToolIcon::DeleteLabel
                    title=Signal::derive(move || t_string!(i18n, training::delete_box_title).to_string())
                    label=Signal::derive(move || t_string!(i18n, training::delete_box).to_string())
                    danger=true
                    on_click=selection.on_delete
                />
            </div>
            // Edit history: undo, redo, and clearing every label of the
            // image (itself one undo step).
            <div class="ui-label-tools-group" hidden=move || no_history.get()>
                <ToolButton
                    icon=ToolIcon::Undo
                    title=Signal::derive(move || t_string!(i18n, training::undo_title).to_string())
                    label=Signal::derive(move || t_string!(i18n, training::undo).to_string())
                    hidden=no_undo
                    on_click=history.on_undo
                />
                <ToolButton
                    icon=ToolIcon::Redo
                    title=Signal::derive(move || t_string!(i18n, training::redo_title).to_string())
                    label=Signal::derive(move || t_string!(i18n, training::redo).to_string())
                    hidden=no_redo
                    on_click=history.on_redo
                />
                <ToolButton
                    icon=ToolIcon::ClearLabels
                    title=Signal::derive(move || t_string!(i18n, training::clear_labels_title).to_string())
                    label=Signal::derive(move || t_string!(i18n, training::clear_labels).to_string())
                    danger=true
                    hidden=no_labels
                    on_click=history.on_clear
                />
            </div>
            // Zoom: out, the readout (click = back to 100 %), in.
            <div class="ui-label-tools-group">
                <ToolButton
                    icon=ToolIcon::ZoomOut
                    title=Signal::derive(move || t_string!(i18n, training::zoom_out_title).to_string())
                    label=Signal::derive(move || t_string!(i18n, training::zoom_out).to_string())
                    disabled=at_min
                    on_click=zoom.on_out
                />
                <button
                    class="ui-icon-button ui-icon-button-md ui-label-tools-readout"
                    disabled=move || at_min.get()
                    title=move || t_string!(i18n, training::zoom_reset_title)
                    aria-label=move || t_string!(i18n, training::zoom_reset)
                    on:click=move |_| zoom.on_reset.run(())
                >
                    {move || t_string!(i18n, training::zoom_percent, percent = zoom.percent.get())}
                </button>
                <ToolButton
                    icon=ToolIcon::ZoomIn
                    title=Signal::derive(move || t_string!(i18n, training::zoom_in_title).to_string())
                    label=Signal::derive(move || t_string!(i18n, training::zoom_in).to_string())
                    disabled=at_max
                    on_click=zoom.on_in
                />
            </div>
        </div>
    }
}
