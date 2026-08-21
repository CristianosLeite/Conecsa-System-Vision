//! Unit tests for the 3D-camera detection used to gate the stereo overlay
//! (headless browser).
use super::*;
use wasm_bindgen_test::*;

#[wasm_bindgen_test]
fn recognises_a_3d_camera_by_name() {
    assert!(is_stereo_camera("3D USB Camera"));
    assert!(is_stereo_camera("3d usb camera"));
    assert!(is_stereo_camera("USB 3D Webcam: USB 3D Webcam"));
}

#[wasm_bindgen_test]
fn ordinary_cameras_are_not_3d() {
    assert!(!is_stereo_camera("C270 HD WEBCAM"));
    assert!(!is_stereo_camera("video0"));
    assert!(!is_stereo_camera(""));
}
