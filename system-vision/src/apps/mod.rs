// SPDX-FileCopyrightText: 2026 Conecsa
//
// SPDX-License-Identifier: AGPL-3.0-only

//! Per-application UI.
//!
//! A device runs one application — object detection, classification,
//! segmentation or face recognition. What only one application needs lives in its module here;
//! components every application shares stay in `crate::components` and branch
//! on the application's task (`components::application_select::Application`).

pub mod classification;
pub mod face_recognition;
pub mod object_detection;
pub mod segmentation;
