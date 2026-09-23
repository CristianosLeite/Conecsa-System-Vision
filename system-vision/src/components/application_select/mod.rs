// SPDX-FileCopyrightText: 2026 Conecsa
//
// SPDX-License-Identifier: AGPL-3.0-only

//! The device's application type in the UI: the shared [`Application`]
//! context and the selection screen a blank device shows instead of the
//! dashboard.

mod application_select;
mod state;

pub use application_select::ApplicationSelect;
pub use state::{task_description, task_label, use_application, Application};
