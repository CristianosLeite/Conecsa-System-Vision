// SPDX-FileCopyrightText: 2026 Conecsa
//
// SPDX-License-Identifier: AGPL-3.0-only

//! Undo / redo history of the open image's labels.
//!
//! A linear list of snapshots with a cursor: `states[0]` is the baseline the
//! image was opened with and every completed, saved edit records the state it
//! produced. Undo and redo move the cursor and hand back the snapshot to show
//! (and persist). The history is per open image: the editor resets it when
//! another image is opened, and after a class removal (the server renumbers
//! the class ids, so older snapshots would be wrong).

use crate::api::{LabelBox, LabelPolygon};

/// The most recent states kept; the oldest is dropped past this.
pub(crate) const HISTORY_CAP: usize = 100;

/// The labels of an image at one point in time (boxes for detection, rings
/// for segmentation; the other list is empty).
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct LabelSnapshot {
    pub(crate) boxes: Vec<LabelBox>,
    pub(crate) polygons: Vec<LabelPolygon>,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct LabelHistory {
    states: Vec<LabelSnapshot>,
    /// Index of the state currently shown.
    cursor: usize,
}

impl LabelHistory {
    /// Start over from `base`: nothing to undo or redo.
    pub(crate) fn reset(&mut self, base: LabelSnapshot) {
        self.states.clear();
        self.states.push(base);
        self.cursor = 0;
    }

    /// The state currently shown (`None` only before the first reset).
    fn current(&self) -> Option<&LabelSnapshot> {
        self.states.get(self.cursor)
    }

    /// A completed edit produced `state`. Ignored when nothing changed, so a
    /// precautionary re-save adds no step; otherwise the redo tail is dropped
    /// and the oldest state past [`HISTORY_CAP`] forgotten. On a history that
    /// was never reset this is the baseline.
    pub(crate) fn record(&mut self, state: LabelSnapshot) {
        if self.current() == Some(&state) {
            return;
        }
        if self.states.is_empty() {
            self.reset(state);
            return;
        }
        self.states.truncate(self.cursor + 1);
        self.states.push(state);
        if self.states.len() > HISTORY_CAP {
            let extra = self.states.len() - HISTORY_CAP;
            self.states.drain(..extra);
        }
        self.cursor = self.states.len() - 1;
    }

    pub(crate) fn can_undo(&self) -> bool {
        self.cursor > 0
    }

    pub(crate) fn can_redo(&self) -> bool {
        self.cursor + 1 < self.states.len()
    }

    /// Step back; the state to show now.
    pub(crate) fn undo(&mut self) -> Option<LabelSnapshot> {
        if !self.can_undo() {
            return None;
        }
        self.cursor -= 1;
        self.current().cloned()
    }

    /// Step forward again; the state to show now.
    pub(crate) fn redo(&mut self) -> Option<LabelSnapshot> {
        if !self.can_redo() {
            return None;
        }
        self.cursor += 1;
        self.current().cloned()
    }
}

#[cfg(test)]
mod tests;
