// SPDX-FileCopyrightText: 2026 Conecsa
//
// SPDX-License-Identifier: AGPL-3.0-only

//! Capture orchestration: nokhwa camera open/decode paths and the main
//! acquire→process→publish loop, with camera re-attach when the device is
//! absent. The direct V4L2 paths live in `v4l2`, the `v4l2-ctl` hardware
//! controls in `controls`, the format enumeration in `formats` and the remote
//! MJPEG source in `network`.

mod controls;
mod formats;
mod network;
mod v4l2;

#[cfg(test)]
mod tests;

use nokhwa::{
    pixel_format::RgbFormat,
    utils::{
        CameraFormat, CameraIndex, FrameFormat, RequestedFormat, RequestedFormatType, Resolution,
    },
    Camera,
};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use super::config::{normalize_token, validate_network};
use super::shm::proto::CameraHealthDetail;
use super::shm::{self, ShmProducer};
use super::{CameraConfig, CameraSource, WebcamServer};

/// Consecutive `cam.frame()` failures after which the nokhwa fallback loops
/// give the camera up and return to the outer open/backoff loop. At the
/// 10 ms pause per failure this is ~300 ms of a dead stream — long enough to
/// ride out a single dropped frame, short enough that a disconnected camera
/// is noticed at once.
pub(crate) const MAX_CONSECUTIVE_FRAME_ERRORS: u32 = 30;
/// After the first failure, log every Nth so a dead camera cannot flood the
/// journal at frame rate.
const FRAME_ERROR_LOG_EVERY: u32 = 100;
/// Pause after a failed `cam.frame()` so a failing camera does not spin a core.
const FRAME_ERROR_PAUSE: std::time::Duration = std::time::Duration::from_millis(10);

/// What the capture loop should do after a `cam.frame()` failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FrameErrorAction {
    /// Log this failure (the first one, then every `FRAME_ERROR_LOG_EVERY`th).
    pub log: bool,
    /// Give the camera up: publish `no_camera` and leave the loop.
    pub give_up: bool,
}

/// Counts consecutive frame failures for the nokhwa fallback loops, so a camera
/// that disconnects mid-stream returns to the outer loop that re-opens it
/// instead of spinning a core. The direct V4L2 loops propagate the error with `?`.
#[derive(Debug, Default)]
pub(crate) struct FrameErrorTracker {
    consecutive: u32,
}

impl FrameErrorTracker {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// A frame arrived: the streak is over.
    pub(crate) fn on_ok(&mut self) {
        self.consecutive = 0;
    }

    /// A frame failed: say whether to log it and whether to give up.
    pub(crate) fn on_error(&mut self) -> FrameErrorAction {
        self.consecutive = self.consecutive.saturating_add(1);
        FrameErrorAction {
            log: self.consecutive == 1 || self.consecutive.is_multiple_of(FRAME_ERROR_LOG_EVERY),
            give_up: self.consecutive >= MAX_CONSECUTIVE_FRAME_ERRORS,
        }
    }

    pub(crate) fn consecutive(&self) -> u32 {
        self.consecutive
    }
}

/// What the capture loop must do after a config message was merged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ConfigOutcome {
    /// Leave the running loop and start over with the new config.
    Restart,
    /// Local camera, same stream: re-apply the hardware controls live.
    ApplyControls,
    /// Nothing to act on (tuning stored while a network source is active).
    Stored,
}

/// Merge a protobuf `CameraConfig` into the in-memory config and say what the
/// change requires. Pure, so every source rule is unit-tested.
///
/// A message without `source` comes from a writer that only tunes the local
/// camera: it never changes the source or the network fields. With `source`
/// present, fields 13-15 are the complete network state.
pub(crate) fn merge_shm_config(
    proto_cfg: &shm::proto::CameraConfig,
    cfg: &mut CameraConfig,
) -> ConfigOutcome {
    let local_restart = proto_cfg.camera_index != cfg.camera_index
        || proto_cfg.width != cfg.width
        || proto_cfg.height != cfg.height
        || proto_cfg.framerate != cfg.framerate;

    cfg.camera_index = proto_cfg.camera_index;
    cfg.width = proto_cfg.width;
    cfg.height = proto_cfg.height;
    cfg.framerate = proto_cfg.framerate;
    cfg.auto_exposure = proto_cfg.auto_exposure;
    cfg.exposure_time = proto_cfg.exposure_time;
    cfg.rgb_red = proto_cfg.rgb_red as u16;
    cfg.rgb_green = proto_cfg.rgb_green as u16;
    cfg.rgb_blue = proto_cfg.rgb_blue as u16;
    cfg.gamma = proto_cfg.gamma;
    cfg.gain = proto_cfg.gain;

    let mut source_restart = false;
    match proto_cfg.source.map(|raw| (raw, CameraSource::from_proto(raw))) {
        None => {}
        // Never coerce an unknown value to local: keep the source as it is.
        Some((raw, None)) => eprintln!("[webcam] Ignoring unknown camera source {raw}"),
        Some((_, Some(CameraSource::Local))) => {
            source_restart = cfg.source != CameraSource::Local;
            cfg.source = CameraSource::Local;
            // Dormant while local; kept so the full state mirrors the writer's.
            cfg.network_host = proto_cfg.network_host.clone();
            cfg.network_port = proto_cfg.network_port;
            cfg.network_token = normalize_token(&proto_cfg.network_token);
        }
        Some((_, Some(CameraSource::Network))) => {
            let token = normalize_token(&proto_cfg.network_token);
            match validate_network(&proto_cfg.network_host, proto_cfg.network_port, &token) {
                Err(why) => eprintln!("[webcam] Ignoring invalid network source: {why}"),
                Ok(()) => {
                    source_restart = cfg.source != CameraSource::Network
                        || cfg.network_host != proto_cfg.network_host
                        || cfg.network_port != proto_cfg.network_port
                        || cfg.network_token != token;
                    cfg.source = CameraSource::Network;
                    cfg.network_host = proto_cfg.network_host.clone();
                    cfg.network_port = proto_cfg.network_port;
                    cfg.network_token = token;
                }
            }
        }
    }

    match cfg.source {
        _ if source_restart => ConfigOutcome::Restart,
        CameraSource::Local if local_restart => ConfigOutcome::Restart,
        CameraSource::Local => ConfigOutcome::ApplyControls,
        // Resolution, framerate and the V4L2 controls mean nothing to a remote camera.
        CameraSource::Network => ConfigOutcome::Stored,
    }
}

impl WebcamServer {
    /// Handle one `cam.frame()` failure in a nokhwa loop: pause, log at the
    /// tracker's cadence, and after `MAX_CONSECUTIVE_FRAME_ERRORS` publish
    /// `no_camera` and return `true` so the caller leaves the loop (the outer
    /// loop then re-tries V4L2, nokhwa, and finally the 5 s `no_camera` backoff).
    fn frame_failed(
        tracker: &mut FrameErrorTracker,
        err: &dyn std::fmt::Display,
        shm: &ShmProducer,
        cfg: &CameraConfig,
    ) -> bool {
        let action = tracker.on_error();
        if action.log {
            eprintln!(
                "[webcam] Camera {} frame error #{}: {err}",
                cfg.camera_index,
                tracker.consecutive()
            );
        }
        if action.give_up {
            eprintln!(
                "[webcam] Camera {} gave no frame {} times in a row; re-opening",
                cfg.camera_index,
                tracker.consecutive()
            );
            Self::publish_health(shm, cfg, "no_camera");
            return true;
        }
        std::thread::sleep(FRAME_ERROR_PAUSE);
        false
    }

    pub(crate) fn try_open_camera(cfg: &CameraConfig) -> Result<Camera, Box<dyn std::error::Error>> {
        let index = CameraIndex::Index(cfg.camera_index);
        let resolution = Resolution::new(cfg.width, cfg.height);

        let fmt_exact = CameraFormat::new(resolution, FrameFormat::MJPEG, cfg.framerate);
        let req_exact = RequestedFormat::new::<RgbFormat>(RequestedFormatType::Exact(fmt_exact));
        if let Ok(mut cam) = Camera::new(CameraIndex::Index(cfg.camera_index), req_exact)
            && cam.open_stream().is_ok() {
                eprintln!(
                    "[webcam] Opened with Exact MJPEG {}x{} @ {}fps",
                    cfg.width, cfg.height, cfg.framerate
                );
                return Ok(cam);
            }

        let req_hfr =
            RequestedFormat::new::<RgbFormat>(RequestedFormatType::HighestFrameRate(cfg.framerate));
        if let Ok(mut cam) = Camera::new(CameraIndex::Index(cfg.camera_index), req_hfr)
            && cam.open_stream().is_ok() {
                eprintln!("[webcam] Opened with HighestFrameRate");
                return Ok(cam);
            }

        let f_yuyv = CameraFormat::new(resolution, FrameFormat::YUYV, cfg.framerate);
        let r_yuyv = RequestedFormat::new::<RgbFormat>(RequestedFormatType::Closest(f_yuyv));
        if let Ok(mut cam) = Camera::new(CameraIndex::Index(cfg.camera_index), r_yuyv)
            && cam.open_stream().is_ok() {
                eprintln!("[webcam] Opened with YUYV fallback");
                return Ok(cam);
            }

        let fmt_close = CameraFormat::new(resolution, FrameFormat::MJPEG, cfg.framerate);
        let req_close = RequestedFormat::new::<RgbFormat>(RequestedFormatType::Closest(fmt_close));
        let mut camera = Camera::new(index, req_close)?;
        camera.open_stream()?;
        eprintln!("[webcam] Opened with Closest MJPEG fallback");
        Ok(camera)
    }

    /// Apply a protobuf CameraConfig received from the consumer to the
    /// in-memory config, optionally triggering a restart.
    fn apply_shm_config(
        proto_cfg: &shm::proto::CameraConfig,
        config: &std::sync::Mutex<CameraConfig>,
        shm: &ShmProducer,
        restart_flag: &AtomicBool,
        rgb_hardware_supported: &AtomicBool,
    ) {
        let mut cfg = config.lock().unwrap();
        match merge_shm_config(proto_cfg, &mut cfg) {
            ConfigOutcome::Restart => {
                eprintln!("[webcam] SHM config change requires restart");
                restart_flag.store(true, Ordering::Relaxed);
            }
            ConfigOutcome::ApplyControls => {
                let rgb_hw = Self::apply_exposure(cfg.camera_index, &cfg);
                rgb_hardware_supported.store(rgb_hw, Ordering::Relaxed);
                Self::publish_health(shm, &cfg, "capturing");
            }
            // A network source reports `capturing` from its own frames only.
            ConfigOutcome::Stored => {}
        }
    }

    fn publish_health(shm: &ShmProducer, _cfg: &CameraConfig, status: &str) {
        Self::publish_health_detail(shm, status, CameraHealthDetail::Unspecified);
    }

    /// Health with the reason a network source is not capturing.
    fn publish_health_detail(shm: &ShmProducer, status: &str, detail: CameraHealthDetail) {
        shm.publish_health(&shm::proto::HealthStatus {
            status: status.to_string(),
            detail: detail as i32,
        });
    }

    /// Main capture loop — runs on the calling thread (blocking).
    pub fn start_capture_loop(self: Arc<Self>) {
        let shm = self.shm.clone();
        let config_arc = self.config.clone();
        let restart_flag = self.needs_restart.clone();
        let rgb_hardware_supported = self.rgb_hardware_supported.clone();

        // Track the last error logged by the retry loop so a persistently
        // unavailable camera doesn't spam the same message every retry cycle.
        let mut last_error: Option<String> = None;
        let mut network_retry = network::RetryState::default();

        loop {
            let cfg = config_arc.lock().unwrap().clone();
            restart_flag.store(false, Ordering::Relaxed);

            // Dispatch on the source before anything touches /dev/video*: a
            // network source runs no V4L2 open and no `v4l2-ctl`, and never
            // falls back to a local camera — a silent fallback would make the
            // inspection source ambiguous.
            if cfg.source == CameraSource::Network {
                last_error = None;
                // Only a fresh start reports "connecting"; while retrying, the
                // last failure stays visible until a frame arrives.
                if network_retry.is_fresh() {
                    eprintln!(
                        "[webcam] Connecting to network camera {}:{}",
                        cfg.network_host, cfg.network_port
                    );
                    Self::publish_health_detail(&shm, "starting", CameraHealthDetail::Connecting);
                }
                match Self::try_network_mjpeg_loop(
                    &cfg,
                    &shm,
                    &config_arc,
                    &restart_flag,
                    &rgb_hardware_supported,
                    &mut network_retry,
                ) {
                    Ok(()) => network_retry = network::RetryState::default(),
                    Err(e) => {
                        let (delay, log) = network_retry.on_failure(&e);
                        if log {
                            eprintln!(
                                "{}",
                                network::transition_line(&cfg.network_host, cfg.network_port, &e)
                            );
                        }
                        // No frames and "no_camera": detection stays gated.
                        Self::publish_health_detail(&shm, "no_camera", e.detail());
                        Self::network_retry_wait(
                            delay,
                            &shm,
                            &config_arc,
                            &restart_flag,
                            &rgb_hardware_supported,
                        );
                    }
                }
                continue;
            }
            network_retry = network::RetryState::default();

            eprintln!(
                "[webcam] Opening camera {} @ {}x{} {} fps",
                cfg.camera_index, cfg.width, cfg.height, cfg.framerate
            );

            Self::publish_health(&shm, &cfg, "starting");

            // Primary path: V4L2 MJPEG passthrough — correct fps + real JPEG
            // size. Returns Ok(()) only when a restart was requested (re-loop);
            // on Err we fall back to the nokhwa / raw-sensor paths below.
            let dev_path = format!("/dev/video{}", cfg.camera_index);
            match Self::try_v4l2_mjpeg_loop(
                &dev_path,
                &cfg,
                &shm,
                &config_arc,
                &restart_flag,
                &rgb_hardware_supported,
            ) {
                Ok(()) => {
                    last_error = None;
                    continue;
                }
                Err(e) => {
                    let msg = format!("[webcam] V4L2 MJPEG path unavailable ({e}); falling back");
                    if last_error.as_deref() != Some(msg.as_str()) {
                        eprintln!("{msg}");
                        last_error = Some(msg);
                    }
                }
            }

            let camera_result = Self::try_open_camera(&cfg);

            match camera_result {
                Ok(mut cam) => {
                    last_error = None;
                    let rgb_hw = Self::apply_exposure(cfg.camera_index, &cfg);
                    rgb_hardware_supported.store(rgb_hw, Ordering::Relaxed);
                    let use_software_rgb = cfg.has_non_neutral_rgb_levels() && !rgb_hw;
                    let is_mjpeg = cam.camera_format().format() == FrameFormat::MJPEG;
                    eprintln!(
                        "[webcam] Camera {} opened - {} mode",
                        cfg.camera_index,
                        if is_mjpeg && !use_software_rgb {
                            "MJPEG passthrough"
                        } else {
                            "RGB decode"
                        }
                    );

                    Self::publish_health(&shm, &cfg, "capturing");

                    let mut errors = FrameErrorTracker::new();
                    if is_mjpeg && !use_software_rgb {
                        // MJPEG passthrough — publish JPEG directly to SHM.
                        loop {
                            if restart_flag.load(Ordering::Relaxed) {
                                break;
                            }
                            // Poll for config changes from consumer.
                            if let Some(proto_cfg) = shm.poll_config() {
                                Self::apply_shm_config(
                                    &proto_cfg,
                                    &config_arc,
                                    &shm,
                                    &restart_flag,
                                    &rgb_hardware_supported,
                                );
                            }
                            match cam.frame() {
                                Ok(frame) => {
                                    errors.on_ok();
                                    let buf = frame.buffer();
                                    shm.publish_frame_jpeg(&buf[..Self::jpeg_payload_len(buf)]);
                                }
                                Err(e) => {
                                    if Self::frame_failed(&mut errors, &e, &shm, &cfg) {
                                        break;
                                    }
                                }
                            }
                        }
                    } else {
                        // RGB decode path — no JPEG encoding needed.
                        loop {
                            if restart_flag.load(Ordering::Relaxed) {
                                break;
                            }
                            // Poll for config changes from consumer.
                            if let Some(proto_cfg) = shm.poll_config() {
                                Self::apply_shm_config(
                                    &proto_cfg,
                                    &config_arc,
                                    &shm,
                                    &restart_flag,
                                    &rgb_hardware_supported,
                                );
                            }
                            // Re-read config to pick up any changes.
                            let current_cfg = config_arc.lock().unwrap().clone();
                            let use_software_rgb = current_cfg.has_non_neutral_rgb_levels()
                                && !rgb_hardware_supported.load(Ordering::Relaxed);
                            let frame = match cam.frame() {
                                Ok(frame) => frame,
                                Err(e) => {
                                    if Self::frame_failed(&mut errors, &e, &shm, &cfg) {
                                        break;
                                    }
                                    continue;
                                }
                            };
                            // `cam.frame()` succeeded, so the camera is alive and the
                            // failure streak ends here. A frame that will not decode
                            // is a bad frame, not a dead camera: skip it.
                            errors.on_ok();
                            if let Ok(img) = frame.decode_image::<RgbFormat>() {
                                let w = img.width();
                                let h = img.height();
                                let mut raw = img.into_raw();
                                if use_software_rgb {
                                    Self::apply_software_rgb_levels(
                                        &mut raw,
                                        current_cfg.rgb_red,
                                        current_cfg.rgb_green,
                                        current_cfg.rgb_blue,
                                    );
                                }
                                shm.publish_frame_rgb(&raw, w, h);
                            }
                        }
                    }
                }
                Err(_e) => {
                    let dev_path = format!("/dev/video{}", cfg.camera_index);
                    match Self::try_v4l2_raw_loop(
                        &dev_path,
                        &cfg,
                        &shm,
                        &config_arc,
                        &restart_flag,
                        &rgb_hardware_supported,
                    ) {
                        Ok(()) => {
                            last_error = None;
                        }
                        Err(e2) => {
                            let msg = format!(
                                "[webcam] Camera {} V4L2 raw also failed ({e2}). No camera.",
                                cfg.camera_index
                            );
                            if last_error.as_deref() != Some(msg.as_str()) {
                                eprintln!("{msg}");
                                last_error = Some(msg);
                            }
                            // Publish NO frames — consumers must see the absence
                            // of a camera, not a synthetic image the model could
                            // detect objects in. "no_camera" gates detection
                            // downstream.
                            Self::publish_health(&shm, &cfg, "no_camera");

                            // Idle briefly, then break so the outer capture loop
                            // re-attempts the real camera. Lets a reconnected /
                            // re-powered camera self-heal without restarting
                            // webcam-server (and therefore without restarting
                            // inference, which shares the SHM).
                            let retry_at =
                                std::time::Instant::now() + std::time::Duration::from_secs(5);
                            loop {
                                if restart_flag.load(Ordering::Relaxed)
                                    || std::time::Instant::now() >= retry_at
                                {
                                    break;
                                }
                                // Poll for config changes.
                                if let Some(proto_cfg) = shm.poll_config() {
                                    Self::apply_shm_config(
                                        &proto_cfg,
                                        &config_arc,
                                        &shm,
                                        &restart_flag,
                                        &rgb_hardware_supported,
                                    );
                                    // apply_shm_config reports "capturing" when the
                                    // change needs no restart; there is still no
                                    // camera here, so restore the real status.
                                    if !restart_flag.load(Ordering::Relaxed) {
                                        Self::publish_health(&shm, &cfg, "no_camera");
                                    }
                                }
                                std::thread::sleep(std::time::Duration::from_millis(100));
                            }
                        }
                    }
                }
            }
        }
    }
}
