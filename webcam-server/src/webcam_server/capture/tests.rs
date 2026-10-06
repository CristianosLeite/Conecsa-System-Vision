// SPDX-FileCopyrightText: 2026 Conecsa
//
// SPDX-License-Identifier: AGPL-3.0-only

//! Unit tests for the nokhwa fallback loops' frame-error tracker.
use super::{FrameErrorTracker, MAX_CONSECUTIVE_FRAME_ERRORS};

#[test]
fn the_first_failure_is_logged_but_not_fatal() {
    let mut tracker = FrameErrorTracker::new();
    let action = tracker.on_error();
    assert!(action.log);
    assert!(!action.give_up);
}

#[test]
fn intermediate_failures_are_quiet() {
    let mut tracker = FrameErrorTracker::new();
    tracker.on_error();
    for _ in 1..(MAX_CONSECUTIVE_FRAME_ERRORS - 1) {
        let action = tracker.on_error();
        assert!(!action.log, "only the first and every 100th failure are logged");
        assert!(!action.give_up);
    }
}

#[test]
fn the_threshold_gives_the_camera_up() {
    // After the threshold a disconnected camera must return to the outer
    // open/backoff loop instead of spinning forever.
    let mut tracker = FrameErrorTracker::new();
    for _ in 0..(MAX_CONSECUTIVE_FRAME_ERRORS - 1) {
        assert!(!tracker.on_error().give_up);
    }
    assert!(tracker.on_error().give_up);
    assert_eq!(tracker.consecutive(), MAX_CONSECUTIVE_FRAME_ERRORS);
}

#[test]
fn a_good_frame_resets_the_streak() {
    let mut tracker = FrameErrorTracker::new();
    for _ in 0..(MAX_CONSECUTIVE_FRAME_ERRORS - 1) {
        tracker.on_error();
    }
    tracker.on_ok();
    assert_eq!(tracker.consecutive(), 0);
    let action = tracker.on_error();
    assert!(action.log, "a new streak logs its first failure again");
    assert!(!action.give_up);
}

#[test]
fn every_hundredth_failure_of_a_long_streak_is_logged() {
    // Reachable only if the threshold is raised; pins the cadence anyway.
    let mut tracker = FrameErrorTracker::new();
    let mut logged = Vec::new();
    for n in 1..=200u32 {
        if tracker.on_error().log {
            logged.push(n);
        }
    }
    assert_eq!(logged, vec![1, 100, 200]);
}

// ── source-aware config merge ────────────────────────────────────────────────

use super::super::shm::proto;
use super::super::{CameraConfig, CameraSource};
use super::{merge_shm_config, ConfigOutcome};

/// Visibly fake stream tokens.
const TOKEN: &str = "TESTT0KEN123";
const OTHER_TOKEN: &str = "0THERT0KEN99";

/// A message mirroring `cfg`'s local fields and carrying no `source` — what a
/// writer that predates the network source sends on every model-settings apply.
fn tuning_message(cfg: &CameraConfig) -> proto::CameraConfig {
    proto::CameraConfig {
        camera_index: cfg.camera_index,
        width: cfg.width,
        height: cfg.height,
        framerate: cfg.framerate,
        auto_exposure: cfg.auto_exposure,
        exposure_time: cfg.exposure_time,
        rgb_red: cfg.rgb_red as u32,
        rgb_green: cfg.rgb_green as u32,
        rgb_blue: cfg.rgb_blue as u32,
        gamma: cfg.gamma,
        gain: cfg.gain,
        ..Default::default()
    }
}

fn network_message(cfg: &CameraConfig, host: &str, port: u32, token: &str) -> proto::CameraConfig {
    proto::CameraConfig {
        source: Some(proto::CameraSource::Network as i32),
        network_host: host.to_string(),
        network_port: port,
        network_token: token.to_string(),
        ..tuning_message(cfg)
    }
}

fn network_config() -> CameraConfig {
    CameraConfig {
        source: CameraSource::Network,
        network_host: "192.0.2.1".to_string(),
        network_port: 8080,
        network_token: TOKEN.to_string(),
        ..Default::default()
    }
}

#[test]
fn a_message_without_source_leaves_an_active_network_source_untouched() {
    // The model-switch case: an unmodified inference-service rewrites the
    // local tuning and must not knock the device back to the local camera.
    let mut cfg = network_config();
    let mut msg = tuning_message(&cfg);
    msg.width = 640;
    msg.height = 480;
    msg.gain = 12;

    assert_eq!(merge_shm_config(&msg, &mut cfg), ConfigOutcome::Stored);
    assert_eq!(cfg.source, CameraSource::Network);
    assert_eq!(cfg.network_host, "192.0.2.1");
    assert_eq!(cfg.network_port, 8080);
    assert_eq!(cfg.network_token, TOKEN);
    assert_eq!((cfg.width, cfg.gain), (640, 12), "the tuning is still stored");
}

#[test]
fn a_local_source_restarts_on_format_changes_and_tunes_live_otherwise() {
    let mut cfg = CameraConfig::default();
    let mut msg = tuning_message(&cfg);
    msg.gain = 40;
    assert_eq!(merge_shm_config(&msg, &mut cfg), ConfigOutcome::ApplyControls);

    msg.framerate = cfg.framerate + 1;
    assert_eq!(merge_shm_config(&msg, &mut cfg), ConfigOutcome::Restart);
}

#[test]
fn a_network_source_never_restarts_or_runs_controls_for_local_fields() {
    let mut cfg = network_config();
    let mut msg = network_message(&cfg, "192.0.2.1", 8080, TOKEN);
    msg.camera_index = 3;
    msg.width = 320;
    msg.framerate = 5;
    msg.exposure_time = 77;
    assert_eq!(merge_shm_config(&msg, &mut cfg), ConfigOutcome::Stored);
}

#[test]
fn a_source_change_always_restarts() {
    let mut cfg = CameraConfig::default();
    let to_network = network_message(&cfg, "192.0.2.1", 8080, TOKEN);
    assert_eq!(merge_shm_config(&to_network, &mut cfg), ConfigOutcome::Restart);
    assert_eq!(cfg.source, CameraSource::Network);

    let to_local = proto::CameraConfig {
        source: Some(proto::CameraSource::Local as i32),
        ..to_network.clone()
    };
    assert_eq!(merge_shm_config(&to_local, &mut cfg), ConfigOutcome::Restart);
    assert_eq!(cfg.source, CameraSource::Local);
    assert_eq!(cfg.network_token, TOKEN, "dormant credentials survive the switch");
}

#[test]
fn a_network_source_restarts_only_on_host_port_or_token_changes() {
    let mut cfg = network_config();
    let same = network_message(&cfg, "192.0.2.1", 8080, TOKEN);
    assert_eq!(merge_shm_config(&same, &mut cfg), ConfigOutcome::Stored);

    for msg in [
        network_message(&cfg, "192.0.2.2", 8080, TOKEN),
        network_message(&cfg, "192.0.2.2", 8081, TOKEN),
        network_message(&cfg, "192.0.2.2", 8081, OTHER_TOKEN),
    ] {
        assert_eq!(merge_shm_config(&msg, &mut cfg), ConfigOutcome::Restart);
    }
    assert_eq!(cfg.network_token, OTHER_TOKEN);
}

#[test]
fn a_hyphenated_lower_case_token_is_the_same_token() {
    let mut cfg = network_config();
    let msg = network_message(&cfg, "192.0.2.1", 8080, "test-t0ke-n123");
    assert_eq!(merge_shm_config(&msg, &mut cfg), ConfigOutcome::Stored);
    assert_eq!(cfg.network_token, TOKEN);
}

#[test]
fn an_unknown_source_value_is_ignored_not_treated_as_local() {
    let mut cfg = network_config();
    let msg = proto::CameraConfig { source: Some(7), ..tuning_message(&cfg) };
    assert_eq!(merge_shm_config(&msg, &mut cfg), ConfigOutcome::Stored);
    assert_eq!(cfg.source, CameraSource::Network);
    assert_eq!(cfg.network_token, TOKEN);
}

#[test]
fn an_invalid_network_state_is_rejected_whole() {
    let mut cfg = CameraConfig::default();
    for msg in [
        network_message(&cfg, "camera.local", 8080, TOKEN),
        network_message(&cfg, "192.0.2.300", 8080, TOKEN),
        network_message(&cfg, "192.0.2.1", 0, TOKEN),
        network_message(&cfg, "192.0.2.1", 65_536, TOKEN),
        network_message(&cfg, "192.0.2.1", 8080, "SHORT"),
        network_message(&cfg, "192.0.2.1", 8080, "HASLETTERILOU"),
        network_message(&cfg, "192.0.2.1", 8080, ""),
    ] {
        assert_eq!(merge_shm_config(&msg, &mut cfg), ConfigOutcome::ApplyControls);
        assert_eq!(cfg.source, CameraSource::Local, "the source did not switch");
        assert!(cfg.network_host.is_empty() && cfg.network_token.is_empty());
    }
}

// ── IP camera source ─────────────────────────────────────────────────────────

use super::super::config::IpCameraConfig;

/// Visibly fake camera password.
const IP_PASSWORD: &str = "FAKE-PASSWORD-123";

fn ip_camera(n: u8) -> proto::IpCamera {
    proto::IpCamera {
        url: format!("rtsp://192.0.2.{n}:554/stream"),
        username: "fake-user".to_string(),
        password: IP_PASSWORD.to_string(),
    }
}

fn ip_message(cfg: &CameraConfig, cameras: Vec<proto::IpCamera>) -> proto::CameraConfig {
    proto::CameraConfig {
        source: Some(proto::CameraSource::Ip as i32),
        ip_cameras: cameras,
        ..tuning_message(cfg)
    }
}

fn ip_config(cameras: &[u8]) -> CameraConfig {
    CameraConfig {
        source: CameraSource::Ip,
        ip_cameras: cameras
            .iter()
            .map(|&n| {
                let c = ip_camera(n);
                IpCameraConfig { url: c.url, username: c.username, password: c.password }
            })
            .collect(),
        ..Default::default()
    }
}

#[test]
fn switching_from_local_to_ip_restarts_with_the_whole_list() {
    let mut cfg = CameraConfig::default();
    let msg = ip_message(&cfg, vec![ip_camera(1), ip_camera(2)]);
    assert_eq!(merge_shm_config(&msg, &mut cfg), ConfigOutcome::Restart);
    assert_eq!(cfg.source, CameraSource::Ip);
    assert_eq!(cfg.ip_cameras, ip_config(&[1, 2]).ip_cameras);
}

#[test]
fn an_unchanged_ip_list_is_just_stored() {
    let mut cfg = ip_config(&[1, 2]);
    let mut msg = ip_message(&cfg, vec![ip_camera(1), ip_camera(2)]);
    msg.width = 320;
    msg.camera_index = 4;
    assert_eq!(merge_shm_config(&msg, &mut cfg), ConfigOutcome::Stored);
    assert_eq!(cfg.width, 320, "the local tuning is still stored");
}

#[test]
fn editing_the_ip_list_reconciles_instead_of_restarting() {
    let mut cfg = ip_config(&[1, 2]);
    // A changed password at position 1, then a third camera, then one fewer.
    let mut changed = ip_camera(2);
    changed.password = "0THER-FAKE-PASSWORD".to_string();
    for cameras in [
        vec![ip_camera(1), changed.clone()],
        vec![ip_camera(1), changed.clone(), ip_camera(3)],
        vec![ip_camera(1)],
    ] {
        let msg = ip_message(&cfg, cameras.clone());
        assert_eq!(merge_shm_config(&msg, &mut cfg), ConfigOutcome::IpCamerasChanged);
        assert_eq!(cfg.ip_cameras.len(), cameras.len());
    }
    assert_eq!(cfg.source, CameraSource::Ip);
}

#[test]
fn switching_from_ip_to_local_restarts_and_keeps_the_list_dormant() {
    let mut cfg = ip_config(&[1]);
    let msg = proto::CameraConfig {
        source: Some(proto::CameraSource::Local as i32),
        ..ip_message(&cfg, vec![ip_camera(1)])
    };
    assert_eq!(merge_shm_config(&msg, &mut cfg), ConfigOutcome::Restart);
    assert_eq!(cfg.source, CameraSource::Local);
    assert_eq!(cfg.ip_cameras.len(), 1);
}

#[test]
fn switching_from_ip_to_network_restarts() {
    let mut cfg = ip_config(&[1]);
    let msg = network_message(&cfg, "192.0.2.1", 8080, TOKEN);
    assert_eq!(merge_shm_config(&msg, &mut cfg), ConfigOutcome::Restart);
    assert_eq!(cfg.source, CameraSource::Network);
    assert!(cfg.ip_cameras.is_empty(), "the message carried no cameras");
}

#[test]
fn an_invalid_ip_list_is_ignored_whole_and_never_becomes_local() {
    let mut credentials_in_url = ip_camera(2);
    credentials_in_url.url = format!("rtsp://fake-user:{IP_PASSWORD}@192.0.2.2/stream");
    let mut loopback = ip_camera(2);
    loopback.url = "rtsp://127.0.0.1/stream".to_string();
    let mut long_password = ip_camera(2);
    long_password.password = "p".repeat(65);
    let invalid_lists = [
        vec![],
        (1..=5).map(ip_camera).collect(),
        vec![ip_camera(1), credentials_in_url],
        vec![ip_camera(1), loopback],
        vec![ip_camera(1), long_password],
    ];

    // From local: stays local.
    let mut cfg = CameraConfig::default();
    for cameras in invalid_lists.clone() {
        let msg = ip_message(&cfg, cameras);
        assert_eq!(merge_shm_config(&msg, &mut cfg), ConfigOutcome::ApplyControls);
        assert_eq!(cfg.source, CameraSource::Local);
        assert!(cfg.ip_cameras.is_empty());
    }
    // From IP: the running list stays as it was.
    let mut cfg = ip_config(&[1, 2]);
    for cameras in invalid_lists {
        let msg = ip_message(&cfg, cameras);
        assert_eq!(merge_shm_config(&msg, &mut cfg), ConfigOutcome::Stored);
        assert_eq!(cfg.source, CameraSource::Ip);
        assert_eq!(cfg.ip_cameras, ip_config(&[1, 2]).ip_cameras);
    }
}

#[test]
fn a_message_without_source_leaves_an_active_ip_source_untouched() {
    let mut cfg = ip_config(&[1, 2]);
    let mut msg = tuning_message(&cfg);
    msg.gain = 9;
    assert_eq!(merge_shm_config(&msg, &mut cfg), ConfigOutcome::Stored);
    assert_eq!(cfg.source, CameraSource::Ip);
    assert_eq!(cfg.ip_cameras, ip_config(&[1, 2]).ip_cameras);
    assert_eq!(cfg.gain, 9);
}
