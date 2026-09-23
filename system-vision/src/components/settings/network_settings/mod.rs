// SPDX-FileCopyrightText: 2026 Conecsa
//
// SPDX-License-Identifier: AGPL-3.0-only

mod ap_panel;
mod ip_config_form;
mod network_settings;
mod tabs;
mod wifi_network_row;
mod wifi_panel;
mod wired_panel;

#[cfg(test)]
mod tests;

pub use network_settings::NetworkSettings;
