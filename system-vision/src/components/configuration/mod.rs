// SPDX-FileCopyrightText: 2026 Conecsa
//
// SPDX-License-Identifier: AGPL-3.0-only

mod configuration;
mod conversion_overlay;
pub mod model_conversion;
// Public so other pages (e.g. the training label editor) can reuse the slider.
pub mod threshold_slider;

pub use configuration::Configuration;
