// SPDX-FileCopyrightText: 2026 Conecsa
//
// SPDX-License-Identifier: AGPL-3.0-only

//! The tool strip's icons.

use leptos::prelude::*;

use crate::components::common::toolbar_icon::ToolbarIcon;

/// One icon of the strip. The paths are Heroicons v1 outline shapes on the
/// 24-unit grid (stroke 2, round caps), the style of every icon in the app;
/// the eraser is Lucide's (ISC), the only metaphor that reads as "wipe the
/// whole image" next to the trash of "delete the selected one".
#[derive(Clone, Copy)]
pub(super) enum ToolIcon {
    /// x-circle: remove the selected point (not minus-circle, which next
    /// to the zoom-out magnifier reads as a second zoom out).
    DeleteVertex,
    /// Trash, as everywhere else an item is deleted.
    DeleteLabel,
    /// U-turn arrow to the left.
    Undo,
    /// Its mirror image.
    Redo,
    /// Eraser.
    ClearLabels,
    /// Magnifier with a minus.
    ZoomOut,
    /// Magnifier with a plus.
    ZoomIn,
}

impl ToolIcon {
    fn path(self) -> &'static str {
        match self {
            Self::DeleteVertex => {
                "M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z"
            }
            Self::DeleteLabel => {
                "M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
            }
            Self::Undo => "M3 10h10a8 8 0 018 8v2M3 10l6 6m-6-6l6-6",
            Self::Redo => "M21 10H11a8 8 0 00-8 8v2m18-10l-6 6m6-6l-6-6",
            Self::ClearLabels => {
                "M7 21l-4.3-4.3c-1-1-1-2.5 0-3.4l9.6-9.6c1-1 2.5-1 3.4 0l5.6 5.6c1 1 1 2.5 0 3.4L13 21M22 21H7M5 11l9 9"
            }
            Self::ZoomOut => "M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0zM13 10H7",
            Self::ZoomIn => "M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0zM10 7v3m0 0v3m0-3h3m-3 0H7",
        }
    }

    /// The icon at the strip's size (20 px in a 2rem button).
    pub(super) fn view(self) -> impl IntoView {
        view! {
            <ToolbarIcon size="w-5 h-5">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d=self.path() />
            </ToolbarIcon>
        }
    }
}
