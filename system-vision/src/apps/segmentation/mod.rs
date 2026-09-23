// SPDX-FileCopyrightText: 2026 Conecsa
//
// SPDX-License-Identifier: AGPL-3.0-only

//! Segmentation: the live object legend under the video. The device burns the
//! masks into the stream; detection areas and the overlay threshold apply
//! exactly as for object detection.

mod segmentation_legend;

pub use segmentation_legend::SegmentationLegend;
