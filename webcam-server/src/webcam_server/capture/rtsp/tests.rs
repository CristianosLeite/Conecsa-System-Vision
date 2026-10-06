// SPDX-FileCopyrightText: 2026 Conecsa
//
// SPDX-License-Identifier: AGPL-3.0-only

//! Unit tests for the RTSP IP camera source. Nothing here needs a network or
//! an ffmpeg binary: the decoder is faked behind `FrameDecoder`.
use std::cell::Cell;
use std::io::Read;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use super::super::network::StreamError;
use super::*;

/// Visibly fake credentials; the password needs percent-encoding.
const USER: &str = "fake user";
const PASSWORD: &str = "FAKE:p@ss/w0rd%";
const PASSWORD_ENCODED: &str = "FAKE%3Ap%40ss%2Fw0rd%25";

fn camera() -> IpCameraConfig {
    IpCameraConfig {
        url: "rtsp://192.0.2.10:554/cam/realmonitor?channel=1".to_string(),
        username: USER.to_string(),
        password: PASSWORD.to_string(),
    }
}

// ── names, limits, sizes ─────────────────────────────────────────────────────

#[test]
fn camera_rings_are_named_after_the_main_ring_and_the_position() {
    assert_eq!(ring_name("conecsa_frame_shm", 0), "conecsa_frame_shm_ip0");
    assert_eq!(ring_name("conecsa_frame_shm", 3), "conecsa_frame_shm_ip3");
}

#[test]
fn limits_default_clamp_and_stay_even() {
    assert_eq!(IpLimits::from_text("", ""), IpLimits::default());
    assert_eq!(IpLimits::default(), IpLimits { max_width: 1920, max_height: 1080 });
    assert_eq!(
        IpLimits::from_text(" 1281 ", "x"),
        IpLimits { max_width: 1280, max_height: 1080 }
    );
    assert_eq!(
        IpLimits::from_text("1", "99999"),
        IpLimits { max_width: 64, max_height: 4320 }
    );
}

#[test]
fn the_output_size_fits_the_limits_and_keeps_the_aspect() {
    let limits = IpLimits::default();
    assert_eq!(output_size(1920, 1080, limits), Some((1920, 1080)));
    assert_eq!(output_size(1280, 720, limits), Some((1280, 720)), "never upscaled");
    assert_eq!(output_size(2560, 1440, limits), Some((1920, 1080)));
    assert_eq!(output_size(3840, 2160, limits), Some((1920, 1080)));
    assert_eq!(output_size(2592, 1944, limits), Some((1440, 1080)), "4:3 bound by height");
    assert_eq!(output_size(1080, 1920, limits), Some((606, 1080)), "portrait, even width");
    assert_eq!(output_size(705, 481, limits), Some((704, 480)), "odd sides made even");
    assert_eq!(output_size(1, 1, limits), Some((2, 2)));
    assert_eq!(output_size(0, 1080, limits), None);
    assert_eq!(output_size(1920, 0, limits), None);
}

// ── URLs and arguments ───────────────────────────────────────────────────────

#[test]
fn credentials_are_percent_encoded_into_the_userinfo() {
    assert_eq!(percent_encode(PASSWORD), PASSWORD_ENCODED);
    assert_eq!(percent_encode("a-Z_0.9~"), "a-Z_0.9~");
    assert_eq!(
        authenticated_url(&camera()),
        format!("rtsp://fake%20user:{PASSWORD_ENCODED}@192.0.2.10:554/cam/realmonitor?channel=1")
    );
    let anonymous = IpCameraConfig { url: "RTSP://192.0.2.10/s".to_string(), ..Default::default() };
    assert_eq!(authenticated_url(&anonymous), "rtsp://192.0.2.10/s");
    let password_only = IpCameraConfig { password: PASSWORD.to_string(), ..anonymous };
    assert_eq!(authenticated_url(&password_only), "rtsp://192.0.2.10/s", "empty user = no auth");
}

#[test]
fn a_camera_is_labelled_by_host_and_port_only() {
    assert_eq!(url_label(&camera().url), "192.0.2.10:554");
    assert_eq!(url_label("rtsp://camera.example/live?token=x"), "camera.example");
    assert_eq!(
        url_label("rtsp://198.51.100.207:554/cam/realmonitor?channel=1&subtype=0"),
        "198.51.100.207:554"
    );
    assert_eq!(url_label("rtsp://[2001:db8::1]:8554/a"), "[2001:db8::1]:8554");
    assert_eq!(url_label(&authenticated_url(&camera())), "192.0.2.10:554");
}

#[test]
fn ffprobe_and_ffmpeg_arguments_carry_the_url_once_and_the_socket_timeout() {
    let url = authenticated_url(&camera());
    let timing = Timing::default();

    let probe = ffprobe_args(&url, &timing);
    assert_eq!(probe.last(), Some(&url));
    let pairs: Vec<_> = probe.windows(2).map(|w| (w[0].as_str(), w[1].as_str())).collect();
    assert!(pairs.contains(&("-rtsp_transport", "tcp")));
    assert!(pairs.contains(&("-timeout", "5000000")));
    assert!(pairs.contains(&("-of", "csv=p=0")));

    let ffmpeg = ffmpeg_args(&url, 1280, 720, &timing);
    let pairs: Vec<_> = ffmpeg.windows(2).map(|w| (w[0].as_str(), w[1].as_str())).collect();
    assert!(pairs.contains(&("-i", url.as_str())));
    assert!(pairs.contains(&("-timeout", "5000000")));
    assert!(pairs.contains(&("-vf", "scale=1280:720")));
    assert!(pairs.contains(&("-pix_fmt", "bgr24")));
    assert!(pairs.contains(&("-f", "rawvideo")));
    assert_eq!(ffmpeg.last().map(String::as_str), Some("-"));
    // Input options precede `-i`; output options follow it.
    let input_at = ffmpeg.iter().position(|a| a == "-i").unwrap();
    for input_option in ["-rtsp_transport", "-timeout", "-fflags", "-flags"] {
        assert!(ffmpeg.iter().position(|a| a == input_option).unwrap() < input_at);
    }
    for output_option in ["-an", "-vf", "-pix_fmt", "-f"] {
        assert!(ffmpeg.iter().position(|a| a == output_option).unwrap() > input_at);
    }

    for args in [&probe, &ffmpeg] {
        assert_eq!(args.iter().filter(|a| a.contains(PASSWORD_ENCODED)).count(), 1);
        assert!(!args.iter().any(|a| a.contains(PASSWORD)), "never un-encoded");
    }
}

#[test]
fn probe_output_parses_width_and_height() {
    assert_eq!(parse_probe_output("1920,1080\n"), Some((1920, 1080)));
    assert_eq!(parse_probe_output("1280,720,\n"), Some((1280, 720)));
    assert_eq!(parse_probe_output("\nN/A,N/A\n640,480\n"), Some((640, 480)));
    assert_eq!(parse_probe_output(""), None);
    assert_eq!(parse_probe_output("1920\n"), None);
}

// ── redaction ────────────────────────────────────────────────────────────────

#[test]
fn userinfo_is_redacted_from_any_url_in_a_line() {
    assert_eq!(
        redact_userinfo("rtsp://fake:secret@192.0.2.10:554/s: Connection refused"),
        "rtsp://***@192.0.2.10:554/s: Connection refused"
    );
    assert_eq!(
        redact_userinfo("a rtsp://u:p@192.0.2.1/x and tcp://u2:p2@192.0.2.2 end"),
        "a rtsp://***@192.0.2.1/x and tcp://***@192.0.2.2 end"
    );
    assert_eq!(redact_userinfo("rtsp://u:p@q@192.0.2.1/"), "rtsp://***@192.0.2.1/");
    assert_eq!(
        redact_userinfo("rtsp://192.0.2.1/path@weird"),
        "rtsp://192.0.2.1/path@weird",
        "an @ after the host is not userinfo"
    );
    assert_eq!(redact_userinfo("no url here"), "no url here");
    assert_eq!(redact_userinfo("ends with ://"), "ends with ://");
}

#[test]
fn the_redactor_masks_the_password_raw_and_encoded_everywhere() {
    let redactor = Redactor::new(&camera());
    let lines = [
        format!("[rtsp @ 0x1] {}: 401 Unauthorized", authenticated_url(&camera())),
        format!("password was {PASSWORD} (encoded {PASSWORD_ENCODED})"),
        format!("Option user:{PASSWORD} not found"),
    ];
    for line in lines {
        let redacted = redactor.redact(&line);
        assert!(!redacted.contains(PASSWORD), "{redacted}");
        assert!(!redacted.contains(PASSWORD_ENCODED), "{redacted}");
        assert!(!redacted.contains("FAKE"), "{redacted}");
    }
    let anonymous = Redactor::new(&IpCameraConfig::default());
    assert_eq!(anonymous.redact("plain line"), "plain line");
}

// ── classification ───────────────────────────────────────────────────────────

fn lines(text: &[&str]) -> Vec<String> {
    text.iter().map(|s| s.to_string()).collect()
}

#[test]
fn stderr_is_classified_into_health_details() {
    let cases: [(&[&str], CameraHealthDetail); 11] = [
        (&["[rtsp @ 0x1] method DESCRIBE failed: 401 Unauthorized"], CameraHealthDetail::Unauthorized),
        (&["Server returned 401 Unauthorized (authorization failed)"], CameraHealthDetail::Unauthorized),
        (&["[tcp @ 0x1] Connection to tcp://192.0.2.1:554?timeout=5000000 failed: Connection refused"], CameraHealthDetail::Unreachable),
        (&["[tcp @ 0x1] Connection to tcp://192.0.2.1:554 failed: Connection timed out"], CameraHealthDetail::Unreachable),
        (&["rtsp://***@192.0.2.1/s: No route to host"], CameraHealthDetail::Unreachable),
        (&["rtsp://192.0.2.1/s: Network is unreachable"], CameraHealthDetail::Unreachable),
        (&["Error in the pull function: Connection reset by peer"], CameraHealthDetail::Unreachable),
        (&["[tcp @ 0x1] Failed to resolve hostname camera.example: Name or service not known"], CameraHealthDetail::Unreachable),
        (&["rtsp://192.0.2.1:4010/s: Invalid data found when processing input"], CameraHealthDetail::BadStream),
        (&["method DESCRIBE failed: 404 Not Found"], CameraHealthDetail::BadStream),
        (&[], CameraHealthDetail::BadStream),
    ];
    for (text, detail) in cases {
        assert_eq!(classify(&lines(text), "ffmpeg exited").detail(), detail, "{text:?}");
    }
}

#[test]
fn classification_keeps_a_short_excerpt_or_the_fallback() {
    let long = format!("Connection refused {}", "x".repeat(500));
    match classify(&lines(&["first", &long]), "fallback") {
        StreamError::Unreachable(why) => assert_eq!(why.len(), EXCERPT_MAX),
        other => panic!("{other:?}"),
    }
    assert_eq!(
        classify(&lines(&["", "  "]), "ffmpeg exited (exit status: 1)"),
        StreamError::BadStream("ffmpeg exited (exit status: 1)".to_string())
    );
    assert_eq!(
        classify(&lines(&["decode error", ""]), "fallback"),
        StreamError::BadStream("decode error".to_string())
    );
}

// ── rawvideo reader ──────────────────────────────────────────────────────────

/// A `Read` that hands out `data` in the given chunk sizes, then EOF.
struct Chunked {
    data: Vec<u8>,
    at: usize,
    chunks: Vec<usize>,
    next: usize,
}

impl Chunked {
    fn new(data: Vec<u8>, chunks: &[usize]) -> Self {
        Self { data, at: 0, chunks: chunks.to_vec(), next: 0 }
    }
}

impl Read for Chunked {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let want = self.chunks.get(self.next).copied().unwrap_or(usize::MAX);
        self.next += 1;
        let n = want.min(buf.len()).min(self.data.len() - self.at);
        buf[..n].copy_from_slice(&self.data[self.at..self.at + n]);
        self.at += n;
        Ok(n)
    }
}

/// Drive the reader to EOF; the frames it completed and the final progress.
fn read_all(reader: &mut RawFrameReader, src: &mut dyn Read) -> (Vec<Vec<u8>>, ReadProgress) {
    let mut frames = Vec::new();
    loop {
        match reader.read_some(src).unwrap() {
            ReadProgress::Frame => frames.push(reader.frame().to_vec()),
            ReadProgress::Partial => {}
            eof @ ReadProgress::Eof { .. } => return (frames, eof),
        }
    }
}

#[test]
fn whole_frames_are_assembled_from_partial_reads() {
    let data: Vec<u8> = (0..24u8).collect();
    let mut src = Chunked::new(data.clone(), &[1, 5, 2, 7, 3, 6]);
    let mut reader = RawFrameReader::new(6);
    let (frames, end) = read_all(&mut reader, &mut src);
    assert_eq!(frames, data.chunks(6).map(<[u8]>::to_vec).collect::<Vec<_>>());
    assert_eq!(end, ReadProgress::Eof { partial: 0 });
}

#[test]
fn an_exact_size_read_is_one_frame() {
    let mut src = Chunked::new(vec![7; 12], &[12]);
    let mut reader = RawFrameReader::new(12);
    assert_eq!(reader.read_some(&mut src).unwrap(), ReadProgress::Frame);
    assert_eq!(reader.frame(), &[7; 12]);
    assert_eq!(reader.read_some(&mut src).unwrap(), ReadProgress::Eof { partial: 0 });
}

#[test]
fn a_short_final_frame_is_discarded() {
    let mut src = Chunked::new((0..10u8).collect(), &[4, 4, 4]);
    let mut reader = RawFrameReader::new(6);
    let (frames, end) = read_all(&mut reader, &mut src);
    assert_eq!(frames, vec![(0..6u8).collect::<Vec<_>>()]);
    assert_eq!(end, ReadProgress::Eof { partial: 4 });
}

#[test]
fn interrupted_reads_are_not_errors() {
    struct Interrupted(Cell<bool>);
    impl Read for Interrupted {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            if self.0.replace(false) {
                return Err(std::io::ErrorKind::Interrupted.into());
            }
            buf[0] = 1;
            Ok(1)
        }
    }
    let mut src = Interrupted(Cell::new(true));
    let mut reader = RawFrameReader::new(1);
    assert_eq!(reader.read_some(&mut src).unwrap(), ReadProgress::Partial);
    assert_eq!(reader.read_some(&mut src).unwrap(), ReadProgress::Frame);
}

// ── health aggregation and reconciliation ────────────────────────────────────

fn status(status: &'static str, detail: CameraHealthDetail) -> CameraStatus {
    CameraStatus { status, detail }
}

#[test]
fn any_capturing_camera_makes_the_aggregate_capturing() {
    let unauthorized = status("no_camera", CameraHealthDetail::Unauthorized);
    assert_eq!(aggregate(&[]), CameraStatus::CONNECTING);
    assert_eq!(aggregate(&[unauthorized, CameraStatus::CAPTURING]), CameraStatus::CAPTURING);
    assert_eq!(
        aggregate(&[unauthorized, status("no_camera", CameraHealthDetail::Unreachable)]),
        unauthorized,
        "otherwise the first camera's status and detail"
    );
    assert_eq!(aggregate(&[CameraStatus::CONNECTING, unauthorized]), CameraStatus::CONNECTING);
}

#[test]
fn the_aggregate_health_carries_the_camera_count() {
    let health = CameraStatus::CAPTURING.health(3);
    assert_eq!(health.status, "capturing");
    assert_eq!(health.detail, CameraHealthDetail::Unspecified as i32);
    assert_eq!(health.ip_camera_count, 3);
}

struct FakeWorker {
    camera: IpCameraConfig,
    stopped: Arc<AtomicBool>,
}

impl CameraWorker for FakeWorker {
    fn camera(&self) -> &IpCameraConfig {
        &self.camera
    }
    fn signal_stop(&self) {
        self.stopped.store(true, Ordering::Relaxed);
    }
}

fn cam(n: u8) -> IpCameraConfig {
    IpCameraConfig { url: format!("rtsp://192.0.2.{n}/s"), ..camera() }
}

#[test]
fn reconcile_restarts_only_changed_new_or_removed_positions() {
    let mut workers: Vec<FakeWorker> = Vec::new();
    let flags: std::cell::RefCell<Vec<Arc<AtomicBool>>> = Default::default();
    let mut spawn = |_: usize, camera: &IpCameraConfig| {
        let stopped = Arc::new(AtomicBool::new(false));
        flags.borrow_mut().push(stopped.clone());
        FakeWorker { camera: camera.clone(), stopped }
    };

    assert_eq!(reconcile(&mut workers, &[cam(1), cam(2), cam(3)], &mut spawn), vec![0, 1, 2]);
    assert!(reconcile(&mut workers, &[cam(1), cam(2), cam(3)], &mut spawn).is_empty());

    // Only the password of position 1 changed.
    let edited = IpCameraConfig { password: "0THER-FAKE".to_string(), ..cam(2) };
    assert_eq!(reconcile(&mut workers, &[cam(1), edited.clone(), cam(3)], &mut spawn), vec![1]);
    {
        let flags = flags.borrow();
        assert!(flags[1].load(Ordering::Relaxed), "the old worker was stopped");
        assert!(!flags[0].load(Ordering::Relaxed) && !flags[2].load(Ordering::Relaxed));
    }

    // A fourth camera appears, then the list shrinks to one.
    assert_eq!(
        reconcile(&mut workers, &[cam(1), edited.clone(), cam(3), cam(4)], &mut spawn),
        vec![3]
    );
    assert!(reconcile(&mut workers, &[cam(1)], &mut spawn).is_empty());
    assert_eq!(workers.len(), 1);
    let flags = flags.borrow();
    assert!(!flags[0].load(Ordering::Relaxed), "position 0 kept running throughout");
    assert!(flags[2].load(Ordering::Relaxed) && flags[3].load(Ordering::Relaxed));
    assert!(flags[4].load(Ordering::Relaxed), "the fourth camera was stopped too");
}

// ── worker over a fake decoder ───────────────────────────────────────────────

/// Serves `frames` frames of `size`, then `then` (or silence when `None`).
struct FakeDecoder {
    start: Result<(u32, u32), StreamError>,
    frames: usize,
    then: Option<StreamError>,
    buf: Vec<u8>,
    stopped: Arc<AtomicBool>,
}

impl FakeDecoder {
    fn boxed(
        start: Result<(u32, u32), StreamError>,
        frames: usize,
        then: Option<StreamError>,
    ) -> (Box<dyn FrameDecoder>, Arc<AtomicBool>) {
        let stopped = Arc::new(AtomicBool::new(false));
        let decoder =
            Self { start, frames, then, buf: Vec::new(), stopped: stopped.clone() };
        (Box::new(decoder), stopped)
    }
}

impl FrameDecoder for FakeDecoder {
    fn start(
        &mut self,
        _should_stop: &mut dyn FnMut() -> bool,
    ) -> Result<Option<(u32, u32)>, StreamError> {
        let (w, h) = self.start.clone()?;
        self.buf = vec![0; w as usize * h as usize * 3];
        Ok(Some((w, h)))
    }

    fn next_frame(&mut self, wait: Duration) -> Result<Option<&[u8]>, StreamError> {
        std::thread::sleep(wait.min(Duration::from_millis(2)));
        if self.frames > 0 {
            self.frames -= 1;
            return Ok(Some(&self.buf));
        }
        match self.then.clone() {
            Some(err) => Err(err),
            None => Ok(None),
        }
    }

    fn stop(&mut self) {
        self.stopped.store(true, Ordering::Relaxed);
    }
}

fn fast_timing() -> Timing {
    Timing {
        first_frame: Duration::from_millis(100),
        stall: Duration::from_millis(100),
        poll: Duration::from_millis(5),
        ..Timing::default()
    }
}

fn small_limits() -> IpLimits {
    IpLimits { max_width: 64, max_height: 64 }
}

/// Poll until the worker reports `want`, or fail after two seconds.
fn wait_for(worker: &IpWorker, want: CameraStatus) {
    let until = Instant::now() + Duration::from_secs(2);
    while worker.status() != want {
        assert!(Instant::now() < until, "still {:?}, wanted {want:?}", worker.status());
        std::thread::sleep(Duration::from_millis(5));
    }
}

fn base_name(tag: &str) -> String {
    format!("conecsa-rtsp-test-{}-{tag}", std::process::id())
}

fn ring_exists(name: &str) -> bool {
    std::path::Path::new(&format!("/dev/shm/{name}")).exists()
}

#[test]
fn a_streaming_camera_reports_capturing_and_its_ring_goes_away_on_drop() {
    let base = base_name("ok");
    let (decoder, stopped) = FakeDecoder::boxed(Ok((32, 16)), usize::MAX, None);
    let worker = IpWorker::spawn(1, &camera(), &base, small_limits(), fast_timing(), decoder);
    wait_for(&worker, CameraStatus::CAPTURING);
    assert!(ring_exists(&ring_name(&base, 1)));
    drop(worker);
    assert!(stopped.load(Ordering::Relaxed), "the decoder was stopped");
    assert!(!ring_exists(&ring_name(&base, 1)), "the ring was unlinked");
}

#[test]
fn a_rejected_login_reports_unauthorized_on_the_camera_ring() {
    let base = base_name("401");
    let (decoder, _) = FakeDecoder::boxed(Err(StreamError::Unauthorized), 0, None);
    let worker = IpWorker::spawn(0, &camera(), &base, small_limits(), fast_timing(), decoder);
    wait_for(&worker, status("no_camera", CameraHealthDetail::Unauthorized));
    assert!(
        ring_exists(&ring_name(&base, 0)),
        "the provisional ring carries the health before any stream size is known"
    );
}

#[test]
fn a_camera_without_frames_is_stalled_and_one_that_stops_is_too() {
    let base = base_name("stall");
    let (decoder, _) = FakeDecoder::boxed(Ok((32, 16)), 0, None);
    let worker = IpWorker::spawn(2, &camera(), &base, small_limits(), fast_timing(), decoder);
    wait_for(&worker, status("no_camera", CameraHealthDetail::Stalled));

    let (decoder, _) = FakeDecoder::boxed(Ok((32, 16)), 3, None);
    let worker = IpWorker::spawn(3, &camera(), &base, small_limits(), fast_timing(), decoder);
    wait_for(&worker, status("no_camera", CameraHealthDetail::Stalled));
}

#[test]
fn a_stream_error_after_frames_is_reported_with_its_detail() {
    let base = base_name("bad");
    let (decoder, _) = FakeDecoder::boxed(
        Ok((32, 16)),
        2,
        Some(StreamError::BadStream("decode error".to_string())),
    );
    let worker = IpWorker::spawn(0, &camera(), &base, small_limits(), fast_timing(), decoder);
    wait_for(&worker, status("no_camera", CameraHealthDetail::BadStream));
}
