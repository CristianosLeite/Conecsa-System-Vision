// SPDX-FileCopyrightText: 2026 Conecsa
//
// SPDX-License-Identifier: AGPL-3.0-only

//! The dataset editor's label requests — saves, loads and class removals —
//! run one at a time, in the order they were issued.
//!
//! Every edit, undo and redo used to send its own request, so two of them
//! in flight (a held Ctrl+Z, a save racing the autosave of an image switch)
//! could complete out of order and leave the server at an older state than
//! the editor shows. Queued, the last state issued is the last one written,
//! and a load queued behind a save of the same image reads what was saved.
//! A save right behind a pending save of the same image replaces it: only
//! the newest state has to reach the server.
//!
//! Removing a class renumbers the classes after it in every label file on
//! the server (its own labels are dropped, the ids above it go down by one).
//! A request captured before a removal ran still names classes by the old
//! numbering, so each one records the numbering it was captured with (its
//! `epoch`: how many removals had succeeded) and is rebased through the
//! removals since — by the server's own rule, [`shift_class`] — right
//! before it is sent. Nothing is dropped or blocked: an edit made while a
//! removal is pending is migrated like the labels on the server, and a
//! second removal confirmed against the old list still removes the class
//! the user picked.
//!
//! The async runner that drains the queue lives in `actions`
//! (`queue_labels`); this module is the queue, its coalescing rule and the
//! renumbering.

use std::collections::VecDeque;

use gloo_timers::future::TimeoutFuture;
use leptos::prelude::*;

use crate::api::{LabelBox, LabelPolygon};

/// One image's labels to persist, captured when the edit was committed
/// (never re-read from the editor's signals when the request is sent).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct LabelWrite {
    pub(crate) image_id: String,
    pub(crate) boxes: Vec<LabelBox>,
    pub(crate) polygons: Vec<LabelPolygon>,
    pub(crate) image_class: Option<u32>,
    /// Show the "labels saved" toast on success.
    pub(crate) notify: bool,
    /// The class numbering the ids refer to ([`LabelQueue::epoch`] when the
    /// labels were captured).
    pub(crate) epoch: usize,
}

/// A queued label request.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum LabelOp {
    /// Persist an image's labels.
    Write(LabelWrite),
    /// Read an image's labels into the editor (dropped if another image is
    /// open by the time they arrive).
    Load(String),
    /// Remove the class at `index` of the list as it was at `epoch` (the
    /// one the user confirmed against).
    RemoveClass { index: usize, epoch: usize },
}

/// Queue `op` behind the pending requests. A write right behind a pending
/// write of the same image takes its place (keeping a requested toast).
pub(crate) fn push(queue: &mut VecDeque<LabelOp>, op: LabelOp) {
    if let LabelOp::Write(next) = &op
        && let Some(LabelOp::Write(last)) = queue.back_mut()
        && last.image_id == next.image_id
    {
        let notify = last.notify || next.notify;
        if let LabelOp::Write(next) = op {
            *last = LabelWrite { notify, ..next };
        }
        return;
    }
    queue.push_back(op);
}

/// A class id after the class at `removed` was removed: `None` for that
/// class itself (its labels go), one less above it — the server's rule
/// (`dataset_service.remove_class`).
pub(crate) fn shift_class(id: u32, removed: usize) -> Option<u32> {
    match (id as usize).cmp(&removed) {
        std::cmp::Ordering::Less => Some(id),
        std::cmp::Ordering::Equal => None,
        std::cmp::Ordering::Greater => Some(id - 1),
    }
}

/// `boxes`, `polygons` and `image_class` after the removal of the class at
/// `removed`.
pub(crate) fn renumber(
    boxes: &mut Vec<LabelBox>,
    polygons: &mut Vec<LabelPolygon>,
    image_class: &mut Option<u32>,
    removed: usize,
) {
    boxes.retain_mut(|b| shift_class(b.class_id, removed).map(|id| b.class_id = id).is_some());
    polygons.retain_mut(|p| shift_class(p.class_id, removed).map(|id| p.class_id = id).is_some());
    *image_class = image_class.and_then(|c| shift_class(c, removed));
}

/// `write` moved from its numbering onto the current one through
/// `removals`, the classes removed since it was captured, in order.
pub(crate) fn rebase_write(mut write: LabelWrite, removals: &[usize]) -> LabelWrite {
    for &removed in removals {
        renumber(&mut write.boxes, &mut write.polygons, &mut write.image_class, removed);
    }
    write.epoch += removals.len();
    write
}

/// A class index of an older list on the current one, through the classes
/// removed since; `None` once that class itself was removed.
pub(crate) fn rebase_index(index: usize, removals: &[usize]) -> Option<usize> {
    removals.iter().try_fold(index, |i, &removed| {
        shift_class(i as u32, removed).map(|i| i as usize)
    })
}

/// The pending label requests, whether a task is draining them and the
/// classes removed so far. `try_*` throughout: requests are queued and
/// drained after awaits, when the editor may already be gone.
#[derive(Clone, Copy)]
pub(crate) struct LabelQueue {
    ops: StoredValue<VecDeque<LabelOp>>,
    busy: StoredValue<bool>,
    /// The index of every successful class removal, in order; its length is
    /// the current numbering's epoch.
    removed: StoredValue<Vec<usize>>,
}

impl LabelQueue {
    pub(crate) fn new() -> Self {
        Self {
            ops: StoredValue::new(VecDeque::new()),
            busy: StoredValue::new(false),
            removed: StoredValue::new(Vec::new()),
        }
    }

    /// The current class numbering (captured into every request that names
    /// classes).
    pub(crate) fn epoch(self) -> usize {
        self.removed.try_with_value(Vec::len).unwrap_or(0)
    }

    /// The class at `index` was removed on the server.
    pub(crate) fn record_removal(self, index: usize) {
        let _ = self.removed.try_update_value(|r| r.push(index));
    }

    /// The removals since `epoch`, in order.
    fn removals_since(self, epoch: usize) -> Vec<usize> {
        self.removed
            .try_with_value(|r| r.get(epoch..).map(<[usize]>::to_vec).unwrap_or_default())
            .unwrap_or_default()
    }

    /// `write` on the current class numbering.
    pub(crate) fn rebase_write(self, write: LabelWrite) -> LabelWrite {
        let removals = self.removals_since(write.epoch);
        rebase_write(write, &removals)
    }

    /// A class index confirmed at `epoch` on the current list (`None`: that
    /// class is already gone).
    pub(crate) fn rebase_index(self, index: usize, epoch: usize) -> Option<usize> {
        rebase_index(index, &self.removals_since(epoch))
    }

    /// Queue `op`. True when no task is draining the queue yet: the caller
    /// must start one (and it is then marked busy).
    pub(crate) fn enqueue(self, op: LabelOp) -> bool {
        if self.ops.try_update_value(|q| push(q, op)).is_none() {
            return false;
        }
        self.busy
            .try_update_value(|busy| !std::mem::replace(busy, true))
            .unwrap_or(false)
    }

    /// The next request, `None` when the queue is empty (or gone).
    pub(crate) fn pop(self) -> Option<LabelOp> {
        self.ops.try_update_value(VecDeque::pop_front).flatten()
    }

    /// The draining task found the queue empty and stops.
    pub(crate) fn finish(self) {
        let _ = self.busy.try_update_value(|busy| *busy = false);
    }

    /// Wait until every queued request has run (leaving the editor must not
    /// drop a save still in the queue).
    pub(crate) async fn drained(self) {
        while self.busy.try_get_value().unwrap_or(false) {
            TimeoutFuture::new(25).await;
        }
    }
}

#[cfg(test)]
mod tests;
