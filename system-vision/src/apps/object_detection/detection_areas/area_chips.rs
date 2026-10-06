// SPDX-FileCopyrightText: 2026 Conecsa
//
// SPDX-License-Identifier: AGPL-3.0-only

use crate::components::common::toolbar_icon::ToolbarIcon;
use crate::i18n::*;
use leptos::prelude::*;

/// Lightweight projection of `api::DetectionArea` used by the
/// live-stream UI. Lives here because `AreaChips` is its primary consumer;
/// `LiveVideoStream` imports it from this module.
#[derive(Clone, Debug)]
pub struct AreaView {
    pub id: String,
    pub is_editing: bool,
    pub shape: String,
}

/// Chip strip in the top-left corner of the live stream. One chip per area:
/// clicking the label promotes that area into editing mode; clicking the trash icon
/// deletes it.
#[component]
pub fn AreaChips(
    areas: ReadSignal<Vec<AreaView>>,
    on_edit: Callback<String>,
    on_delete: Callback<String>,
) -> impl IntoView {
    let i18n = use_i18n();
    view! {
        <div class="absolute top-2 left-2 flex flex-wrap gap-1 max-w-[60%]">
            {move || {
                areas.get().into_iter().enumerate().map(|(idx, a)| {
                    let id_edit = a.id.clone();
                    let id_del = a.id.clone();
                    let label = format!("#{}", idx + 1);
                    let shape_glyph = if a.shape == "circle" { "○" } else { "□" };
                    let highlight = if a.is_editing {
                        "ui-area-chip-active"
                    } else {
                        ""
                    };
                    let chip_class = format!("ui-area-chip {}", highlight);
                    view! {
                        <div class=chip_class>
                            <button
                                type="button"
                                class="hover:underline"
                                title=t_string!(i18n, stream::edit_area)
                                on:click=move |_| on_edit.run(id_edit.clone())
                            >
                                {shape_glyph} " " {label}
                            </button>
                            <button
                                type="button"
                                class="ui-area-chip-delete"
                                title=t_string!(i18n, stream::delete_area)
                                aria-label=move || t_string!(i18n, stream::delete_area)
                                on:click=move |_| on_delete.run(id_del.clone())
                            >
                                <ToolbarIcon>
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                                </ToolbarIcon>
                            </button>
                        </div>
                    }
                }).collect_view()
            }}
        </div>
    }
}
