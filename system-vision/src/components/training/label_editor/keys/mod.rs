// SPDX-FileCopyrightText: 2026 Conecsa
//
// SPDX-License-Identifier: AGPL-3.0-only

//! The label editor's keyboard map.
//!
//! [`command`] is pure — a key, its modifiers and whether a label is selected
//! give a [`Command`] or nothing — so the map is unit tested. The component
//! reads the `KeyboardEvent`, consults the focus helpers here and runs the
//! command on its `LabelTools`.
//!
//! Delete removes the selected vertex, ring or box (Delete only — not
//! Backspace, which is used while typing class names / SAM prompts); `+`,
//! `-` and `0` zoom; Space arms panning; Ctrl/Cmd+Z, Ctrl/Cmd+Shift+Z and
//! Ctrl/Cmd+Y undo and redo; the arrows nudge the selected label or, with
//! nothing selected, ← / → open the previous / next image (Home / End the
//! first / last; PageUp / PageDown step whatever is selected). Ctrl/Cmd+`+`
//! / `-` / `0` are accepted too: the page-zoom guard has already cancelled
//! their browser default. The component skips the map while a form field is
//! focused so typing never deletes a label or zooms, and text fields keep
//! their own undo and caret keys; Esc is handled before the map (it abandons
//! a ring being drawn, else clears the selection).

use super::super::dataset_editor::ImageStep;
use super::super::label_geometry::arrow_delta_px;

/// What a key press asks the editor to do.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum Command {
    Undo,
    Redo,
    /// Move the selected label by this many screen px.
    Nudge(f32, f32),
    /// Open another image of the gallery.
    Step(ImageStep),
    ZoomIn,
    ZoomOut,
    ZoomReset,
    /// Space: arm panning while it is held.
    Pan,
    /// Remove the selected vertex, ring or box.
    Delete,
}

impl Command {
    /// Whether the key's browser default is cancelled: the caret / scroll
    /// keys and the undo shortcuts would otherwise scroll the page or reach
    /// the browser. The zoom keys and Delete have no page default worth
    /// stopping.
    pub(super) fn cancels_default(self) -> bool {
        matches!(
            self,
            Command::Undo | Command::Redo | Command::Nudge(..) | Command::Step(_) | Command::Pan
        )
    }
}

/// The modifiers held with a key (`ctrl` covers Cmd on macOS).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct Mods {
    pub(super) ctrl: bool,
    pub(super) shift: bool,
}

/// The command `key` asks for, `None` when the key means nothing here or
/// keeps its browser meaning (↑ / ↓ and Home / End with nothing selected).
pub(super) fn command(key: &str, mods: Mods, has_selection: bool) -> Option<Command> {
    if mods.ctrl {
        match key.to_ascii_lowercase().as_str() {
            "z" if mods.shift => return Some(Command::Redo),
            "z" => return Some(Command::Undo),
            "y" => return Some(Command::Redo),
            _ => {}
        }
    }
    Some(match key {
        "ArrowLeft" | "ArrowRight" | "ArrowUp" | "ArrowDown" if has_selection => {
            let (dx, dy) = arrow_delta_px(key, mods.shift)?;
            Command::Nudge(dx, dy)
        }
        "ArrowLeft" | "PageUp" => Command::Step(ImageStep::Previous),
        "ArrowRight" | "PageDown" => Command::Step(ImageStep::Next),
        "Home" if !has_selection => Command::Step(ImageStep::First),
        "End" if !has_selection => Command::Step(ImageStep::Last),
        "+" | "=" => Command::ZoomIn,
        "-" => Command::ZoomOut,
        "0" => Command::ZoomReset,
        " " => Command::Pan,
        "Delete" => Command::Delete,
        _ => return None,
    })
}

/// The tag of the focused element, uppercase (`None` when nothing has focus).
fn focused_tag() -> Option<String> {
    web_sys::window()
        .and_then(|w| w.document())
        .and_then(|d| d.active_element())
        .map(|el| el.tag_name().to_uppercase())
}

/// A form field has focus: keys are text, not editor commands.
pub(super) fn typing() -> bool {
    focused_tag().is_some_and(|tag| matches!(tag.as_str(), "INPUT" | "TEXTAREA" | "SELECT"))
}

/// A button, link or disclosure has focus: Space activates it, so it must
/// not arm panning.
pub(super) fn activates_focused_control() -> bool {
    focused_tag().is_some_and(|t| matches!(t.as_str(), "BUTTON" | "A" | "SUMMARY"))
}

#[cfg(test)]
mod tests;
