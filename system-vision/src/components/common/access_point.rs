// SPDX-FileCopyrightText: 2026 Conecsa
//
// SPDX-License-Identifier: AGPL-3.0-only

//! Rules shared by every place that can turn the device's access point on.

/// WPA2 accepts 8 to 63 printable ASCII characters (space to `~`); the device
/// refuses anything else before the radio, so the form refuses it first.
pub fn passphrase_is_valid(passphrase: &str) -> bool {
    (8..=63).contains(&passphrase.len()) && passphrase.bytes().all(|b| (0x20..=0x7E).contains(&b))
}
