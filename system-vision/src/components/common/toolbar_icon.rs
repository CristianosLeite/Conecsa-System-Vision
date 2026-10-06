// SPDX-FileCopyrightText: 2026 Conecsa
//
// SPDX-License-Identifier: AGPL-3.0-only

use leptos::prelude::*;

/// The outline icon of a tool button: a 24-unit grid drawn in the button's
/// text colour (`stroke-current`) and hidden from assistive technology, so
/// the button's own `title` / `aria-label` carry the meaning. The children
/// are the `<path>` (or shape) elements, stroke width 2 with round caps —
/// the Heroicons outline style used across both frontends.
#[component]
pub fn ToolbarIcon(
    /// Tailwind size classes of the SVG.
    #[prop(default = "w-4 h-4")]
    size: &'static str,
    children: Children,
) -> impl IntoView {
    view! {
        <svg class=format!("{size} stroke-current") viewBox="0 0 24 24" fill="none" aria-hidden="true">
            {children()}
        </svg>
    }
}
