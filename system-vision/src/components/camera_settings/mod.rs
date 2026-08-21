//! Leptos UI components for the web frontend.

mod apply_button;
mod component;
mod device_select;
mod loading_state;
mod resolution_controls;
mod stereo_toggle;

#[cfg(test)]
mod tests;

pub use component::CameraSettings;

/// Preset resolutions shown in the UI.
/// Tuples: (width, height, label, aspect-ratio badge)
const RESOLUTIONS: &[(u32, u32, &str, &str)] = &[
    (640, 640, "640 × 640", "1:1"),
    (1280, 720, "1280 × 720", "16:9"),
    (1440, 1080, "1440 × 1080", "4:3"),
    (1920, 1080, "1920 × 1080", "16:9"),
];

/// A `Resolution` struct.
#[derive(Clone, Copy, PartialEq)]
struct Resolution {
    w: u32,
    h: u32,
}

type CameraFormat = (u32, u32, Vec<u32>);

/// V4L2 name marker for a side-by-side 3D camera ("3D USB Camera" & friends).
const STEREO_NAME_MARKER: &str = "3d";

/// Whether a V4L2 device name identifies a 3D (side-by-side stereo) camera.
///
/// The overlay blends the left|right halves of one frame into a single image,
/// so offering it for an ordinary camera would split a normal picture in two.
/// The camera model is the only reliable signal: resolution alone cannot tell a
/// side-by-side frame from a wide one.
fn is_stereo_camera(name: &str) -> bool {
    name.to_ascii_lowercase().contains(STEREO_NAME_MARKER)
}
