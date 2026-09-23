// SPDX-FileCopyrightText: 2026 Conecsa
//
// SPDX-License-Identifier: AGPL-3.0-only

mod application_settings;
pub mod gpio_settings;
pub mod network_settings;
mod settings;

pub use application_settings::ApplicationSettings;
pub use gpio_settings::GpioSettings;
pub use network_settings::NetworkSettings;
pub use settings::Settings;
