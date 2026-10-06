// SPDX-FileCopyrightText: 2026 Conecsa
//
// SPDX-License-Identifier: AGPL-3.0-only

//! IP camera source: up to four RTSP cameras, each decoded by its own `ffmpeg`
//! subprocess into raw BGR frames and published into its own SHM ring,
//! `<SHM_NAME>_ip<N>` (N = the camera's 0-based position in the list).
//!
//! The main ring (`SHM_NAME`) carries no frames in this mode: the supervisor
//! loop keeps polling its config region, starts and stops one worker thread
//! per camera, and publishes the aggregate health there together with the
//! number of configured cameras.
//!
//! Camera credentials are secrets. They only travel inside the URL handed to
//! ffmpeg/ffprobe; every line those tools print is redacted before it is kept,
//! and log lines name a camera by its position and `host[:port]` only.

use std::collections::VecDeque;
use std::io::{ErrorKind, Read};
use std::os::fd::AsRawFd;
use std::os::unix::process::CommandExt;
use std::process::{Child, ChildStderr, ChildStdout, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use super::super::config::{IP_CAMERAS_MAX, IpCameraConfig, strip_rtsp_scheme};
use super::super::shm::proto::{CameraHealthDetail, HealthStatus};
use super::super::shm::ShmProducer;
use super::super::{CameraConfig, CameraSource, WebcamServer};
use super::network::{RetryState, StreamError};

/// Default cap on the decoded frame size (`IPCAM_MAX_WIDTH`/`IPCAM_MAX_HEIGHT`).
const MAX_WIDTH_DEFAULT: u32 = 1920;
const MAX_HEIGHT_DEFAULT: u32 = 1080;
/// Bounds accepted for those variables.
const MAX_SIDE_MIN: u32 = 64;
const MAX_WIDTH_LIMIT: u32 = 7680;
const MAX_HEIGHT_LIMIT: u32 = 4320;
/// Stderr lines kept per subprocess (the tail is what classifies a failure).
const STDERR_TAIL_LINES: usize = 16;
/// Longest stderr line kept; longer lines are cut.
const STDERR_LINE_MAX: usize = 300;
/// Longest stderr excerpt carried into a `StreamError` (and so a log line).
const EXCERPT_MAX: usize = 160;
/// Most ffprobe stdout read (it prints one short CSV line).
const PROBE_STDOUT_MAX: u64 = 4096;
/// Pipe size requested for ffmpeg's stdout, so one frame takes fewer reads.
const PIPE_SIZE: libc::c_int = 1 << 20;
/// How often the supervisor polls the config and refreshes the health.
const SUPERVISOR_POLL: Duration = Duration::from_millis(100);
/// How often a retry wait looks at the stop flag.
const RETRY_POLL: Duration = Duration::from_millis(100);
/// Status strings of `HealthStatus.status`.
const STARTING: &str = "starting";
const CAPTURING: &str = "capturing";
const NO_CAMERA: &str = "no_camera";

/// The SHM segment name of the camera at `index` (0-based list position).
pub(crate) fn ring_name(base: &str, index: usize) -> String {
    format!("{base}_ip{index}")
}

/// Unlink every camera ring name. Harmless for names that do not exist; used
/// when the IP source stops and at start-up for segments a crash left behind.
pub(crate) fn unlink_camera_rings(base: &str) {
    for index in 0..IP_CAMERAS_MAX {
        if let Ok(name) = std::ffi::CString::new(format!("/{}", ring_name(base, index))) {
            // SAFETY: a valid NUL-terminated name; failure (ENOENT) is ignored.
            unsafe {
                libc::shm_unlink(name.as_ptr());
            }
        }
    }
}

// ── limits, timing ───────────────────────────────────────────────────────────

/// Largest frame the decoder outputs; bigger streams are scaled down.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct IpLimits {
    pub max_width: u32,
    pub max_height: u32,
}

impl Default for IpLimits {
    fn default() -> Self {
        Self { max_width: MAX_WIDTH_DEFAULT, max_height: MAX_HEIGHT_DEFAULT }
    }
}

impl IpLimits {
    /// `IPCAM_MAX_WIDTH`/`IPCAM_MAX_HEIGHT`, clamped and made even; an unset
    /// or unparsable value keeps the default.
    pub(crate) fn from_env() -> Self {
        Self::from_text(
            &std::env::var("IPCAM_MAX_WIDTH").unwrap_or_default(),
            &std::env::var("IPCAM_MAX_HEIGHT").unwrap_or_default(),
        )
    }

    pub(crate) fn from_text(width: &str, height: &str) -> Self {
        let side = |text: &str, default: u32, limit: u32| {
            text.trim()
                .parse::<u32>()
                .map_or(default, |v| v.clamp(MAX_SIDE_MIN, limit) & !1)
        };
        Self {
            max_width: side(width, MAX_WIDTH_DEFAULT, MAX_WIDTH_LIMIT),
            max_height: side(height, MAX_HEIGHT_DEFAULT, MAX_HEIGHT_LIMIT),
        }
    }
}

/// The decoded frame size for a `src_w`×`src_h` stream: scaled down (aspect
/// kept) to fit the limits, never up, both sides even and at least 2.
/// `None` for a stream without a size.
pub(crate) fn output_size(src_w: u32, src_h: u32, limits: IpLimits) -> Option<(u32, u32)> {
    if src_w == 0 || src_h == 0 {
        return None;
    }
    let scale = (f64::from(limits.max_width) / f64::from(src_w))
        .min(f64::from(limits.max_height) / f64::from(src_h))
        .min(1.0);
    let side = |v: u32| ((f64::from(v) * scale).floor() as u32 & !1).max(2);
    Some((side(src_w), side(src_h)))
}

/// Every deadline of one camera session; the tests shrink them.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Timing {
    /// RTSP socket I/O timeout handed to ffprobe/ffmpeg (`-timeout`).
    pub socket: Duration,
    /// Longest ffprobe may take before it is killed.
    pub probe: Duration,
    /// Longest wait for the first frame after ffmpeg started.
    pub first_frame: Duration,
    /// Longest gap between two frames once streaming.
    pub stall: Duration,
    /// Cadence at which the stop flag and the deadlines are checked.
    pub poll: Duration,
}

impl Default for Timing {
    fn default() -> Self {
        Self {
            socket: Duration::from_secs(5),
            probe: Duration::from_secs(15),
            first_frame: Duration::from_secs(10),
            stall: Duration::from_secs(5),
            poll: Duration::from_millis(250),
        }
    }
}

// ── URLs, arguments, redaction ───────────────────────────────────────────────

/// Percent-encode everything but RFC 3986 unreserved characters.
pub(crate) fn percent_encode(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for b in text.bytes() {
        if b.is_ascii_alphanumeric() || b"-._~".contains(&b) {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

/// The URL handed to ffmpeg: the camera URL (scheme lower-cased) with the
/// percent-encoded credentials as userinfo; an empty username means no
/// authentication. A secret — never log it.
pub(crate) fn authenticated_url(camera: &IpCameraConfig) -> String {
    let rest = strip_rtsp_scheme(&camera.url).unwrap_or(&camera.url);
    if camera.username.is_empty() {
        return format!("rtsp://{rest}");
    }
    format!(
        "rtsp://{}:{}@{rest}",
        percent_encode(&camera.username),
        percent_encode(&camera.password)
    )
}

/// `host[:port]` of a (validated, credential-free) camera URL — what log
/// lines name a camera by, so a token in the path never reaches them.
pub(crate) fn url_label(url: &str) -> &str {
    let rest = strip_rtsp_scheme(url).unwrap_or(url);
    let rest = &rest[..rest.find(['/', '?', '#']).unwrap_or(rest.len())];
    rest.rsplit_once('@').map_or(rest, |(_, host)| host)
}

fn timeout_us(socket: Duration) -> String {
    socket.as_micros().to_string()
}

/// ffprobe arguments that print `width,height` of the first video stream.
pub(crate) fn ffprobe_args(url: &str, timing: &Timing) -> Vec<String> {
    [
        "-v", "error",
        "-rtsp_transport", "tcp",
        "-timeout", &timeout_us(timing.socket),
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0",
        url,
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

/// ffmpeg arguments that decode the first video stream to `w`×`h` BGR24
/// rawvideo on stdout, one frame per decoded picture.
pub(crate) fn ffmpeg_args(url: &str, w: u32, h: u32, timing: &Timing) -> Vec<String> {
    [
        "-nostdin",
        "-hide_banner",
        "-loglevel", "error",
        "-rtsp_transport", "tcp",
        "-timeout", &timeout_us(timing.socket),
        "-fflags", "nobuffer",
        "-flags", "low_delay",
        "-i", url,
        "-map", "0:v:0",
        "-an",
        "-vf", &format!("scale={w}:{h}"),
        "-fps_mode", "passthrough",
        "-pix_fmt", "bgr24",
        "-f", "rawvideo",
        "-",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

/// Parse ffprobe's `width,height` output (first line that holds both).
pub(crate) fn parse_probe_output(text: &str) -> Option<(u32, u32)> {
    text.lines().find_map(|line| {
        let mut fields = line.split(',').map(str::trim).filter(|f| !f.is_empty());
        let w = fields.next()?.parse().ok()?;
        let h = fields.next()?.parse().ok()?;
        Some((w, h))
    })
}

/// Replace the userinfo of every `scheme://user:pass@host` in `line` with
/// `***`, whatever the scheme.
pub(crate) fn redact_userinfo(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let mut rest = line;
    while let Some(at) = rest.find("://") {
        let start = at + 3;
        out.push_str(&rest[..start]);
        let tail = &rest[start..];
        let end = tail
            .find(|c: char| c.is_whitespace() || "/?#\"'<>".contains(c))
            .unwrap_or(tail.len());
        match tail[..end].rfind('@') {
            Some(userinfo_end) => {
                out.push_str("***");
                rest = &tail[userinfo_end..];
            }
            None => rest = tail,
        }
    }
    out.push_str(rest);
    out
}

/// Scrubs a camera's credentials out of subprocess output.
#[derive(Debug, Clone, Default)]
pub(crate) struct Redactor {
    /// The password, raw and percent-encoded, longest first.
    secrets: Vec<String>,
}

impl Redactor {
    pub(crate) fn new(camera: &IpCameraConfig) -> Self {
        let mut secrets = Vec::new();
        if !camera.password.is_empty() {
            secrets.push(camera.password.clone());
            let encoded = percent_encode(&camera.password);
            if encoded != camera.password {
                secrets.push(encoded);
            }
        }
        secrets.sort_by_key(|s| std::cmp::Reverse(s.len()));
        Self { secrets }
    }

    /// The line with every URL userinfo and every copy of the password masked.
    pub(crate) fn redact(&self, line: &str) -> String {
        let mut out = redact_userinfo(line);
        for secret in &self.secrets {
            out = out.replace(secret.as_str(), "***");
        }
        out
    }
}

// ── stderr classification ────────────────────────────────────────────────────

/// Cut `text` to at most `max` bytes on a character boundary.
fn truncate(text: &str, max: usize) -> &str {
    if text.len() <= max {
        return text;
    }
    let mut end = max;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    &text[..end]
}

/// True when `line` holds HTTP/RTSP status 401 as a word (not a port `:4010`).
fn has_status_401(line: &str) -> bool {
    line.match_indices("401").any(|(at, _)| {
        let before = line[..at].chars().next_back();
        let after = line[at + 3..].chars().next();
        before.is_some_and(char::is_whitespace) && !after.is_some_and(|c| c.is_ascii_digit())
    })
}

const UNREACHABLE_MARKERS: [&str; 8] = [
    "connection refused",
    "timed out",
    "no route to host",
    "network is unreachable",
    "connection reset",
    "failed to resolve hostname",
    "name or service not known",
    "temporary failure in name resolution",
];

/// Classify the (already redacted) stderr tail of a failed ffprobe/ffmpeg run.
/// `fallback` describes the failure when stderr says nothing useful.
pub(crate) fn classify(lines: &[String], fallback: &str) -> StreamError {
    let excerpt = |pred: &dyn Fn(&str) -> bool| {
        lines
            .iter()
            .rev()
            .find(|l| pred(l))
            .map(|l| truncate(l.trim(), EXCERPT_MAX).to_string())
    };
    if lines
        .iter()
        .any(|l| l.to_ascii_lowercase().contains("unauthorized") || has_status_401(l))
    {
        return StreamError::Unauthorized;
    }
    if let Some(line) = excerpt(&|l| {
        let lower = l.to_ascii_lowercase();
        UNREACHABLE_MARKERS.iter().any(|m| lower.contains(m))
    }) {
        return StreamError::Unreachable(line);
    }
    let why = excerpt(&|l| !l.trim().is_empty()).unwrap_or_else(|| fallback.to_string());
    StreamError::BadStream(why)
}

/// Collects the redacted tail of a subprocess's stderr on its own thread, so
/// the pipe never fills and blocks the child.
struct StderrTail {
    lines: Arc<Mutex<VecDeque<String>>>,
    thread: Option<JoinHandle<()>>,
}

impl StderrTail {
    fn spawn(mut stderr: ChildStderr, redactor: Redactor) -> Self {
        let lines = Arc::new(Mutex::new(VecDeque::with_capacity(STDERR_TAIL_LINES)));
        let sink = lines.clone();
        let thread = std::thread::Builder::new()
            .name("ipcam-stderr".to_string())
            .spawn(move || {
                let push = |raw: &[u8]| {
                    if raw.is_empty() {
                        return;
                    }
                    let text = redactor.redact(&String::from_utf8_lossy(raw));
                    let mut tail = sink.lock().unwrap();
                    if tail.len() == STDERR_TAIL_LINES {
                        tail.pop_front();
                    }
                    tail.push_back(truncate(&text, STDERR_LINE_MAX).to_string());
                };
                let mut line = Vec::new();
                let mut chunk = [0u8; 4096];
                loop {
                    let n = match stderr.read(&mut chunk) {
                        Ok(0) => break,
                        Ok(n) => n,
                        Err(e) if e.kind() == ErrorKind::Interrupted => continue,
                        Err(_) => break,
                    };
                    for &b in &chunk[..n] {
                        if b == b'\n' || b == b'\r' {
                            push(&line);
                            line.clear();
                        } else if line.len() < 4 * STDERR_LINE_MAX {
                            line.push(b);
                        }
                    }
                }
                push(&line);
            })
            .ok();
        Self { lines, thread }
    }

    /// Wait for the stream to end (the child must be gone) and return the tail.
    fn finish(mut self) -> Vec<String> {
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
        self.lines.lock().unwrap().drain(..).collect()
    }
}

/// A ffprobe/ffmpeg command that dies with the thread that spawned it, so a
/// crashed or killed webcam-server never leaves a decoder running.
fn decoder_command(program: &str, args: Vec<String>) -> Command {
    let mut command = Command::new(program);
    command.args(args).stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped());
    // SAFETY: prctl is async-signal-safe and touches no parent state.
    unsafe {
        command.pre_exec(|| {
            if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL) != 0 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }
    command
}

/// Kill and reap a child; never leaves a zombie.
fn kill_and_reap(child: &mut Child) {
    let _ = child.kill();
    let _ = child.wait();
}

// ── rawvideo frames ──────────────────────────────────────────────────────────

/// What one read did to the frame being assembled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ReadProgress {
    /// A whole frame is ready in [`RawFrameReader::frame`].
    Frame,
    /// Some or no bytes arrived; the frame is still incomplete.
    Partial,
    /// The source ended; `partial` bytes of an unfinished frame are discarded.
    Eof { partial: usize },
}

/// Assembles fixed-size rawvideo frames from whatever pieces a pipe delivers.
pub(crate) struct RawFrameReader {
    frame: Vec<u8>,
    filled: usize,
}

impl RawFrameReader {
    pub(crate) fn new(frame_len: usize) -> Self {
        Self { frame: vec![0; frame_len], filled: 0 }
    }

    /// One `read` from `src` into the frame. After `Frame`, the next call
    /// starts the next frame.
    pub(crate) fn read_some(&mut self, src: &mut dyn Read) -> std::io::Result<ReadProgress> {
        if self.filled == self.frame.len() {
            self.filled = 0;
        }
        match src.read(&mut self.frame[self.filled..]) {
            Ok(0) => {
                let partial = self.filled;
                self.filled = 0;
                Ok(ReadProgress::Eof { partial })
            }
            Ok(n) => {
                self.filled += n;
                Ok(if self.filled == self.frame.len() {
                    ReadProgress::Frame
                } else {
                    ReadProgress::Partial
                })
            }
            Err(e) if matches!(e.kind(), ErrorKind::Interrupted | ErrorKind::WouldBlock) => {
                Ok(ReadProgress::Partial)
            }
            Err(e) => Err(e),
        }
    }

    /// The last complete frame (valid right after `Frame`).
    pub(crate) fn frame(&self) -> &[u8] {
        &self.frame
    }
}

// ── decoder ──────────────────────────────────────────────────────────────────

/// A source of fixed-size BGR24 frames for one camera.
pub(crate) trait FrameDecoder: Send {
    /// Learn the stream size and start decoding. `Ok(None)` when
    /// `should_stop` asked to stop first. Returns the output frame size.
    fn start(
        &mut self,
        should_stop: &mut dyn FnMut() -> bool,
    ) -> Result<Option<(u32, u32)>, StreamError>;

    /// Wait up to `wait` for the next complete frame (`Ok(None)` = not yet).
    /// An error ends the session; the decoder is stopped by then.
    fn next_frame(&mut self, wait: Duration) -> Result<Option<&[u8]>, StreamError>;

    /// Stop decoding: kill and reap any subprocess. Idempotent.
    fn stop(&mut self);
}

/// `FrameDecoder` over `ffprobe` + an `ffmpeg` subprocess.
pub(crate) struct FfmpegDecoder {
    /// Camera URL with credentials — a secret.
    url: String,
    redactor: Redactor,
    limits: IpLimits,
    timing: Timing,
    child: Option<Child>,
    stdout: Option<ChildStdout>,
    stderr: Option<StderrTail>,
    reader: RawFrameReader,
}

impl FfmpegDecoder {
    pub(crate) fn new(camera: &IpCameraConfig, limits: IpLimits, timing: Timing) -> Self {
        Self {
            url: authenticated_url(camera),
            redactor: Redactor::new(camera),
            limits,
            timing,
            child: None,
            stdout: None,
            stderr: None,
            reader: RawFrameReader::new(0),
        }
    }

    fn spawn_error(tool: &str, e: &std::io::Error) -> StreamError {
        StreamError::BadStream(format!("cannot start {tool} ({})", e.kind()))
    }

    /// Run ffprobe under the probe deadline; the stream's coded size.
    fn probe(
        &mut self,
        should_stop: &mut dyn FnMut() -> bool,
    ) -> Result<Option<(u32, u32)>, StreamError> {
        let mut child = decoder_command("ffprobe", ffprobe_args(&self.url, &self.timing))
            .spawn()
            .map_err(|e| Self::spawn_error("ffprobe", &e))?;
        let stderr = child.stderr.take().map(|s| StderrTail::spawn(s, self.redactor.clone()));
        let stdout = child.stdout.take().map(|mut out| {
            std::thread::spawn(move || {
                let mut text = String::new();
                let _ = (&mut out).take(PROBE_STDOUT_MAX).read_to_string(&mut text);
                text
            })
        });

        /// How the ffprobe wait ended.
        enum Exit {
            Status(std::process::ExitStatus),
            Stopped,
            Deadline,
            WaitFailed,
        }
        let deadline = Instant::now() + self.timing.probe;
        let exit = loop {
            match child.try_wait() {
                Ok(Some(status)) => break Exit::Status(status),
                Ok(None) => {}
                Err(_) => break Exit::WaitFailed,
            }
            if should_stop() {
                break Exit::Stopped;
            }
            if Instant::now() >= deadline {
                break Exit::Deadline;
            }
            std::thread::sleep(Duration::from_millis(50));
        };
        if !matches!(exit, Exit::Status(_)) {
            kill_and_reap(&mut child);
        }
        let output = stdout.and_then(|t| t.join().ok()).unwrap_or_default();
        let lines = stderr.map(StderrTail::finish).unwrap_or_default();
        match exit {
            Exit::Stopped => return Ok(None),
            Exit::WaitFailed => {
                return Err(StreamError::BadStream("ffprobe could not be waited on".to_string()));
            }
            // Killed at the deadline: whatever it printed may still say why.
            Exit::Deadline => {
                return Err(match classify(&lines, "") {
                    StreamError::BadStream(_) => StreamError::Stalled,
                    known => known,
                });
            }
            Exit::Status(status) if !status.success() => {
                return Err(classify(&lines, &format!("ffprobe failed ({status})")));
            }
            Exit::Status(_) => {}
        }
        parse_probe_output(&output)
            .map(Some)
            .ok_or_else(|| StreamError::BadStream("the stream has no video size".to_string()))
    }

    /// ffmpeg ended (stdout closed or a read failed): reap it and say why.
    fn finish(&mut self, fallback: &str) -> StreamError {
        let mut status = None;
        if let Some(mut child) = self.child.take() {
            let deadline = Instant::now() + Duration::from_secs(1);
            while status.is_none() && Instant::now() < deadline {
                match child.try_wait() {
                    Ok(Some(s)) => status = Some(s),
                    Ok(None) => std::thread::sleep(Duration::from_millis(20)),
                    Err(_) => break,
                }
            }
            if status.is_none() {
                kill_and_reap(&mut child);
            }
        }
        self.stdout = None;
        let lines = self.stderr.take().map(StderrTail::finish).unwrap_or_default();
        let fallback = match status {
            Some(s) => format!("{fallback}; ffmpeg exited ({s})"),
            None => fallback.to_string(),
        };
        classify(&lines, &fallback)
    }
}

impl FrameDecoder for FfmpegDecoder {
    fn start(
        &mut self,
        should_stop: &mut dyn FnMut() -> bool,
    ) -> Result<Option<(u32, u32)>, StreamError> {
        self.stop();
        let Some((src_w, src_h)) = self.probe(should_stop)? else {
            return Ok(None);
        };
        let (w, h) = output_size(src_w, src_h, self.limits)
            .ok_or_else(|| StreamError::BadStream("the stream has no video size".to_string()))?;

        let mut child = decoder_command("ffmpeg", ffmpeg_args(&self.url, w, h, &self.timing))
            .spawn()
            .map_err(|e| Self::spawn_error("ffmpeg", &e))?;
        self.stderr = child.stderr.take().map(|s| StderrTail::spawn(s, self.redactor.clone()));
        if let Some(out) = &child.stdout {
            // SAFETY: a valid pipe fd owned by `child`; failure is harmless.
            unsafe {
                libc::fcntl(out.as_raw_fd(), libc::F_SETPIPE_SZ, PIPE_SIZE);
            }
        }
        self.stdout = child.stdout.take();
        self.child = Some(child);
        self.reader = RawFrameReader::new(w as usize * h as usize * 3);
        Ok(Some((w, h)))
    }

    fn next_frame(&mut self, wait: Duration) -> Result<Option<&[u8]>, StreamError> {
        let Some(stdout) = self.stdout.as_mut() else {
            return Err(StreamError::BadStream("decoder not started".to_string()));
        };
        let mut pfd = libc::pollfd { fd: stdout.as_raw_fd(), events: libc::POLLIN, revents: 0 };
        let wait_ms = libc::c_int::try_from(wait.as_millis()).unwrap_or(libc::c_int::MAX);
        // SAFETY: one valid pollfd for the duration of the call.
        let ready = unsafe { libc::poll(&mut pfd, 1, wait_ms) };
        if ready == 0 {
            return Ok(None);
        }
        if ready < 0 {
            let err = std::io::Error::last_os_error();
            if err.kind() == ErrorKind::Interrupted {
                return Ok(None);
            }
            return Err(self.finish(&format!("poll failed ({})", err.kind())));
        }
        match self.reader.read_some(stdout) {
            Ok(ReadProgress::Frame) => Ok(Some(self.reader.frame())),
            Ok(ReadProgress::Partial) => Ok(None),
            Ok(ReadProgress::Eof { .. }) => Err(self.finish("the stream ended")),
            Err(e) => Err(self.finish(&format!("read failed ({})", e.kind()))),
        }
    }

    fn stop(&mut self) {
        if let Some(mut child) = self.child.take() {
            kill_and_reap(&mut child);
        }
        self.stdout = None;
        if let Some(stderr) = self.stderr.take() {
            stderr.finish();
        }
    }
}

impl Drop for FfmpegDecoder {
    fn drop(&mut self) {
        self.stop();
    }
}

// ── per-camera worker ────────────────────────────────────────────────────────

/// One camera's health, as published on its ring.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CameraStatus {
    pub status: &'static str,
    pub detail: CameraHealthDetail,
}

impl CameraStatus {
    pub(crate) const CONNECTING: Self =
        Self { status: STARTING, detail: CameraHealthDetail::Connecting };
    pub(crate) const CAPTURING: Self =
        Self { status: CAPTURING, detail: CameraHealthDetail::Unspecified };

    fn failed(err: &StreamError) -> Self {
        Self { status: NO_CAMERA, detail: err.detail() }
    }

    fn health(self, ip_camera_count: u32) -> HealthStatus {
        HealthStatus {
            status: self.status.to_string(),
            detail: self.detail as i32,
            ip_camera_count,
        }
    }
}

/// The main ring's health: `capturing` while any camera captures, otherwise
/// the first camera's status and detail (`starting`/`connecting` before any
/// camera reported).
pub(crate) fn aggregate(cameras: &[CameraStatus]) -> CameraStatus {
    if cameras.iter().any(|c| c.status == CAPTURING) {
        return CameraStatus::CAPTURING;
    }
    cameras.first().copied().unwrap_or(CameraStatus::CONNECTING)
}

/// Everything one worker thread needs.
struct WorkerContext {
    index: usize,
    ring_name: String,
    label: String,
    limits: IpLimits,
    timing: Timing,
    stop: Arc<AtomicBool>,
    status: Arc<Mutex<CameraStatus>>,
}

/// The camera's ring and the frame size it was created for.
struct Ring {
    shm: ShmProducer,
    size: (u32, u32),
}

impl WorkerContext {
    fn stopping(&self) -> bool {
        self.stop.load(Ordering::Relaxed)
    }

    /// Make `ring` exist at `size`, re-creating it when the size changed.
    fn ensure_ring(&self, ring: &mut Option<Ring>, size: (u32, u32)) -> bool {
        if ring.as_ref().is_some_and(|r| r.size == size) {
            return true;
        }
        // Drop (unlink) the old segment before the name is created again.
        *ring = None;
        match ShmProducer::new_exact(&self.ring_name, size.0, size.1) {
            Ok(shm) => {
                *ring = Some(Ring { shm, size });
                true
            }
            Err(e) => {
                eprintln!("[webcam] IP camera {} cannot create SHM /{}: {e}", self.index + 1, self.ring_name);
                false
            }
        }
    }

    fn report(&self, ring: &Option<Ring>, status: CameraStatus) {
        if let Some(ring) = ring {
            ring.shm.publish_health(&status.health(0));
        }
        *self.status.lock().unwrap() = status;
    }

    /// Wait out a retry delay unless asked to stop.
    fn wait(&self, delay: Duration) {
        let until = Instant::now() + delay;
        while !self.stopping() && Instant::now() < until {
            std::thread::sleep(RETRY_POLL.min(until.saturating_duration_since(Instant::now())));
        }
    }

    /// The worker thread: connect, stream, retry with back-off, until stopped.
    fn run(&self, decoder: &mut dyn FrameDecoder) {
        // Provisional ring at the largest output size, so this camera's health
        // is readable before its stream size is known.
        let mut ring = None;
        self.ensure_ring(&mut ring, (self.limits.max_width, self.limits.max_height));
        let mut retry = RetryState::default();
        while !self.stopping() {
            if retry.is_fresh() {
                self.report(&ring, CameraStatus::CONNECTING);
            }
            match self.session(decoder, &mut ring, &mut retry) {
                Ok(()) => break,
                Err(e) => {
                    let (delay, log) = retry.on_failure(&e);
                    if log {
                        eprintln!(
                            "[webcam] IP camera {} ({}) {e}; retrying",
                            self.index + 1,
                            self.label
                        );
                    }
                    self.report(&ring, CameraStatus::failed(&e));
                    self.wait(delay);
                }
            }
        }
        decoder.stop();
        // `ring` drops here: the segment is unlinked.
    }

    /// One decoder session. `Ok` only when asked to stop.
    fn session(
        &self,
        decoder: &mut dyn FrameDecoder,
        ring: &mut Option<Ring>,
        retry: &mut RetryState,
    ) -> Result<(), StreamError> {
        let Some(size) = decoder.start(&mut || self.stopping())? else {
            return Ok(());
        };
        if !self.ensure_ring(ring, size) {
            decoder.stop();
            return Err(StreamError::BadStream("cannot create the camera ring".to_string()));
        }
        let started = Instant::now();
        let mut last_frame: Option<Instant> = None;
        loop {
            if self.stopping() {
                decoder.stop();
                return Ok(());
            }
            if let Some(frame) = decoder.next_frame(self.timing.poll)? {
                if let Some(ring) = ring.as_ref() {
                    ring.shm.publish_frame_bgr(frame, size.0, size.1);
                }
                last_frame = Some(Instant::now());
                if retry.on_frame() {
                    eprintln!(
                        "[webcam] IP camera {} ({}) streaming {}x{}",
                        self.index + 1,
                        self.label,
                        size.0,
                        size.1
                    );
                    self.report(ring, CameraStatus::CAPTURING);
                }
            }
            let stalled = match last_frame {
                None => started.elapsed() >= self.timing.first_frame,
                Some(at) => at.elapsed() >= self.timing.stall,
            };
            if stalled {
                decoder.stop();
                return Err(StreamError::Stalled);
            }
        }
    }
}

/// What the supervisor needs from a running camera worker.
pub(crate) trait CameraWorker {
    fn camera(&self) -> &IpCameraConfig;
    /// Ask the worker to stop; dropping it then waits for it.
    fn signal_stop(&self);
}

/// A camera's worker thread. Dropping it stops and joins the thread, which
/// kills ffmpeg and unlinks the camera's ring.
pub(crate) struct IpWorker {
    camera: IpCameraConfig,
    stop: Arc<AtomicBool>,
    status: Arc<Mutex<CameraStatus>>,
    thread: Option<JoinHandle<()>>,
}

impl IpWorker {
    pub(crate) fn spawn(
        index: usize,
        camera: &IpCameraConfig,
        base_name: &str,
        limits: IpLimits,
        timing: Timing,
        mut decoder: Box<dyn FrameDecoder>,
    ) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let status = Arc::new(Mutex::new(CameraStatus::CONNECTING));
        let ctx = WorkerContext {
            index,
            ring_name: ring_name(base_name, index),
            label: url_label(&camera.url).to_string(),
            limits,
            timing,
            stop: stop.clone(),
            status: status.clone(),
        };
        let thread = std::thread::Builder::new()
            .name(format!("ipcam{index}"))
            .spawn(move || ctx.run(decoder.as_mut()));
        let thread = match thread {
            Ok(t) => Some(t),
            Err(e) => {
                eprintln!("[webcam] IP camera {} worker did not start: {e}", index + 1);
                *status.lock().unwrap() = CameraStatus {
                    status: NO_CAMERA,
                    detail: CameraHealthDetail::BadStream,
                };
                None
            }
        };
        Self { camera: camera.clone(), stop, status, thread }
    }

    pub(crate) fn status(&self) -> CameraStatus {
        *self.status.lock().unwrap()
    }
}

impl CameraWorker for IpWorker {
    fn camera(&self) -> &IpCameraConfig {
        &self.camera
    }

    fn signal_stop(&self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}

impl Drop for IpWorker {
    fn drop(&mut self) {
        self.signal_stop();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// Bring `workers` in line with `desired`: a position whose camera entry
/// (URL, user or password) changed or disappeared is stopped, a new or changed
/// position is spawned, every other worker keeps running. Returns the
/// positions that were (re)spawned.
pub(crate) fn reconcile<W: CameraWorker>(
    workers: &mut Vec<W>,
    desired: &[IpCameraConfig],
    spawn: &mut dyn FnMut(usize, &IpCameraConfig) -> W,
) -> Vec<usize> {
    let stale: Vec<usize> = (0..workers.len())
        .filter(|&i| desired.get(i) != Some(workers[i].camera()))
        .collect();
    if stale.is_empty() && workers.len() == desired.len() {
        return Vec::new();
    }
    // Signal every stale worker first so they wind down in parallel.
    for &i in &stale {
        workers[i].signal_stop();
    }
    let mut old: Vec<Option<W>> = std::mem::take(workers).into_iter().map(Some).collect();
    for &i in &stale {
        drop(old[i].take());
    }
    let mut spawned = Vec::new();
    for (i, camera) in desired.iter().enumerate() {
        match old.get_mut(i).and_then(Option::take) {
            Some(worker) => workers.push(worker),
            None => {
                workers.push(spawn(i, camera));
                spawned.push(i);
            }
        }
    }
    spawned
}

/// Stop every worker, in parallel.
fn stop_all<W: CameraWorker>(workers: &mut Vec<W>) {
    for worker in workers.iter() {
        worker.signal_stop();
    }
    workers.clear();
}

impl WebcamServer {
    /// Supervise the IP cameras until a restart is requested (a source change).
    ///
    /// Keeps polling the main ring's config, reconciles one worker per camera
    /// with the configured list, and publishes the aggregate health plus the
    /// camera count on the main ring whenever either changes. On return every
    /// worker has stopped and every camera ring is unlinked.
    pub(crate) fn run_ip_cameras_loop(
        shm: &Arc<ShmProducer>,
        config_arc: &Arc<Mutex<CameraConfig>>,
        restart_flag: &AtomicBool,
        rgb_hardware_supported: &AtomicBool,
    ) {
        let limits = IpLimits::from_env();
        let timing = Timing::default();
        let base_name = config_arc.lock().unwrap().shm_name.clone();
        eprintln!(
            "[webcam] IP camera source active (frames up to {}x{}, rings /{}_ip<N>)",
            limits.max_width, limits.max_height, base_name
        );
        let mut workers: Vec<IpWorker> = Vec::new();
        let mut published: Option<(CameraStatus, u32)> = None;

        loop {
            if let Some(proto_cfg) = shm.poll_config() {
                Self::apply_shm_config(
                    &proto_cfg,
                    config_arc,
                    shm,
                    restart_flag,
                    rgb_hardware_supported,
                );
            }
            if restart_flag.load(Ordering::Relaxed) {
                break;
            }
            let desired = {
                let cfg = config_arc.lock().unwrap();
                if cfg.source != CameraSource::Ip {
                    break;
                }
                cfg.ip_cameras.clone()
            };
            let restarted = reconcile(&mut workers, &desired, &mut |index, camera| {
                IpWorker::spawn(
                    index,
                    camera,
                    &base_name,
                    limits,
                    timing,
                    Box::new(FfmpegDecoder::new(camera, limits, timing)),
                )
            });
            for index in restarted {
                eprintln!(
                    "[webcam] IP camera {} ({}) started",
                    index + 1,
                    url_label(&desired[index].url)
                );
            }

            let statuses: Vec<CameraStatus> = workers.iter().map(IpWorker::status).collect();
            let current = (aggregate(&statuses), desired.len() as u32);
            if published != Some(current) {
                shm.publish_health(&current.0.health(current.1));
                published = Some(current);
            }
            std::thread::sleep(SUPERVISOR_POLL);
        }

        stop_all(&mut workers);
        unlink_camera_rings(&base_name);
        eprintln!("[webcam] IP camera source stopped");
    }
}

#[cfg(test)]
mod tests;
