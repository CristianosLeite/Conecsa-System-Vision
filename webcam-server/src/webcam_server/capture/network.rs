// SPDX-FileCopyrightText: 2026 Conecsa
//
// SPDX-License-Identifier: AGPL-3.0-only

//! Network capture source: pulls a remote camera's MJPEG stream (remote camera protocol
//! version 1, `docs/services/remote-camera-protocol.md`) over plain TCP and
//! publishes every complete JPEG into the SHM ring, exactly like a local MJPEG
//! camera.
//!
//! Everything here is bounded: header sizes, the frame size (one SHM slot), the
//! connect time, the silence the peer may keep and the time one frame may take.
//! The stream token is a secret — nothing in this module logs the request, and
//! no error carries it.

use std::io::{ErrorKind, Read, Write};
use std::net::{Ipv4Addr, SocketAddr, SocketAddrV4, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use super::super::shm::proto::CameraHealthDetail;
use super::super::shm::ShmProducer;
use super::super::{CameraConfig, WebcamServer};

/// The only multipart boundary protocol version 1 allows.
pub(crate) const BOUNDARY: &str = "conecsaframe";
/// Status line plus response headers.
pub(crate) const RESPONSE_HEADER_MAX: usize = 8 * 1024;
/// Delimiter line plus one part's headers.
pub(crate) const PART_HEADER_MAX: usize = 1024;
/// Socket read size; also the most the parser buffer can hold beyond the unit
/// it is assembling.
const READ_CHUNK: usize = 64 * 1024;
/// Pause between connection attempts: 1 s, 2 s, then 5 s until a frame arrives.
const RETRY_DELAYS: [Duration; 3] = [
    Duration::from_secs(1),
    Duration::from_secs(2),
    Duration::from_secs(5),
];
/// How often a retry wait looks at the restart flag and the SHM config.
const RETRY_POLL: Duration = Duration::from_millis(100);

/// Why a network session ended. Never carries the token.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum StreamError {
    /// HTTP 401: the remote camera rejected the token.
    Unauthorized,
    /// HTTP 429: the remote camera locked authentication out.
    RateLimited { retry_after_secs: Option<u64> },
    /// Connect failure, reset, or the peer closed the stream.
    Unreachable(String),
    /// Connected, but the peer went silent or dripped a frame too slowly.
    Stalled,
    /// Invalid HTTP, multipart framing or JPEG payload.
    BadStream(String),
}

impl StreamError {
    pub(crate) fn detail(&self) -> CameraHealthDetail {
        match self {
            Self::Unauthorized => CameraHealthDetail::Unauthorized,
            Self::RateLimited { .. } => CameraHealthDetail::RateLimited,
            Self::Unreachable(_) => CameraHealthDetail::Unreachable,
            Self::Stalled => CameraHealthDetail::Stalled,
            Self::BadStream(_) => CameraHealthDetail::BadStream,
        }
    }
}

impl std::fmt::Display for StreamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unauthorized => write!(f, "unauthorized (HTTP 401)"),
            Self::RateLimited { retry_after_secs: Some(s) } => {
                write!(f, "rate limited (HTTP 429, retry after {s} s)")
            }
            Self::RateLimited { retry_after_secs: None } => write!(f, "rate limited (HTTP 429)"),
            Self::Unreachable(why) => write!(f, "unreachable ({why})"),
            Self::Stalled => write!(f, "stalled"),
            Self::BadStream(why) => write!(f, "bad stream ({why})"),
        }
    }
}

impl std::error::Error for StreamError {}

fn bad(why: impl Into<String>) -> StreamError {
    StreamError::BadStream(why.into())
}

/// The exact protocol-v1 request. The token goes out normalized (no hyphens,
/// upper case). The result is a secret: never log it.
pub(crate) fn build_request(host: &str, port: u32, token: &str) -> Vec<u8> {
    format!(
        "GET /stream.mjpeg HTTP/1.1\r\n\
         Host: {host}:{port}\r\n\
         Authorization: Bearer {token}\r\n\
         Accept: multipart/x-mixed-replace\r\n\
         Connection: close\r\n\
         \r\n"
    )
    .into_bytes()
}

// ── incremental parser ───────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    /// Waiting for the status line and response headers.
    ResponseHeaders,
    /// Waiting for a delimiter line and the part headers.
    PartHeaders,
    /// Waiting for `len` payload bytes and the CRLF that follows them.
    Payload { len: usize },
}

/// Incremental MJPEG-over-HTTP parser. Bytes go in with [`feed`](Self::feed)
/// in whatever pieces the socket delivers; complete frames come out of
/// [`next_frame`](Self::next_frame). Partial input survives between calls, so a
/// read timeout never loses data (which is why this is not built on
/// `read_exact`).
pub(crate) struct MjpegReader {
    buf: Vec<u8>,
    /// Prefix of `buf` already handed out as a frame; dropped on the next call.
    consumed: usize,
    /// Where the header-terminator search resumes, so a slow header is not
    /// rescanned from the start on every read.
    scan_from: usize,
    state: State,
    max_frame: usize,
}

impl MjpegReader {
    /// `max_frame` is the largest payload accepted (one SHM slot).
    pub(crate) fn new(max_frame: usize) -> Self {
        Self {
            buf: Vec::with_capacity(READ_CHUNK),
            consumed: 0,
            scan_from: 0,
            state: State::ResponseHeaders,
            max_frame,
        }
    }

    pub(crate) fn feed(&mut self, bytes: &[u8]) {
        self.buf.extend_from_slice(bytes);
    }

    /// True between units: nothing buffered, so nothing is "in flight".
    pub(crate) fn is_idle(&self) -> bool {
        self.buf.len() == self.consumed
    }

    /// Advance as far as the buffered bytes allow. `Ok(None)` means more input
    /// is needed; call again after every `Ok(Some(_))`, one read can hold
    /// several frames.
    pub(crate) fn next_frame(&mut self) -> Result<Option<&[u8]>, StreamError> {
        if self.consumed > 0 {
            self.buf.drain(..self.consumed);
            self.consumed = 0;
        }
        let len = loop {
            match self.state {
                State::ResponseHeaders => {
                    let Some(end) = self.header_block_end(RESPONSE_HEADER_MAX, "response")? else {
                        return Ok(None);
                    };
                    parse_response_headers(&self.buf[..end])?;
                    self.advance(end + 4, State::PartHeaders);
                }
                State::PartHeaders => {
                    let Some(end) = self.header_block_end(PART_HEADER_MAX, "part")? else {
                        return Ok(None);
                    };
                    let len = parse_part_headers(&self.buf[..end], self.max_frame)?;
                    self.advance(end + 4, State::Payload { len });
                }
                State::Payload { len } => {
                    if self.buf.len() < len + 2 {
                        return Ok(None);
                    }
                    if &self.buf[len..len + 2] != b"\r\n" {
                        return Err(bad("bytes after the declared Content-Length"));
                    }
                    let jpeg = &self.buf[..len];
                    if len < 4 || !jpeg.starts_with(&[0xFF, 0xD8]) || !jpeg.ends_with(&[0xFF, 0xD9])
                    {
                        return Err(bad("payload is not a complete JPEG"));
                    }
                    break len;
                }
            }
        };
        self.consumed = len + 2;
        self.scan_from = 0;
        self.state = State::PartHeaders;
        Ok(Some(&self.buf[..len]))
    }

    fn advance(&mut self, by: usize, next: State) {
        self.buf.drain(..by);
        self.scan_from = 0;
        self.state = next;
    }

    /// Index of the blank line ending a header block, `None` while it has not
    /// arrived, or an error once `limit` bytes hold no terminator.
    fn header_block_end(&mut self, limit: usize, what: &str) -> Result<Option<usize>, StreamError> {
        let found = self.buf[self.scan_from..]
            .windows(4)
            .position(|w| w == b"\r\n\r\n")
            .map(|at| at + self.scan_from);
        match found {
            Some(end) if end + 4 <= limit => Ok(Some(end)),
            Some(_) => Err(bad(format!("{what} headers exceed {limit} bytes"))),
            None if self.buf.len() >= limit => {
                Err(bad(format!("{what} headers exceed {limit} bytes")))
            }
            None => {
                self.scan_from = self.buf.len().saturating_sub(3);
                Ok(None)
            }
        }
    }
}

/// A header block: its first line, then `(lower-case name, value)` pairs.
type HeaderBlock<'a> = (&'a str, Vec<(String, &'a str)>);

/// Split a header block into its first line and its headers.
fn header_lines(block: &[u8]) -> Result<HeaderBlock<'_>, StreamError> {
    let text = std::str::from_utf8(block).map_err(|_| bad("headers are not valid UTF-8"))?;
    let mut lines = text.split("\r\n");
    let first = lines.next().unwrap_or("");
    let mut headers = Vec::new();
    for line in lines {
        let (name, value) = line.split_once(':').ok_or_else(|| bad("header line without ':'"))?;
        headers.push((name.trim().to_ascii_lowercase(), value.trim()));
    }
    Ok((first, headers))
}

fn parse_response_headers(block: &[u8]) -> Result<(), StreamError> {
    let (status_line, headers) = header_lines(block)?;
    let mut parts = status_line.split_whitespace();
    let version = parts.next().unwrap_or("");
    if !version.starts_with("HTTP/1.") {
        return Err(bad("not an HTTP/1.x response"));
    }
    let status: u16 = parts
        .next()
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| bad("invalid HTTP status line"))?;
    let header = |name: &str| headers.iter().find(|(n, _)| n == name).map(|(_, v)| *v);

    match status {
        200 => {}
        401 => return Err(StreamError::Unauthorized),
        429 => {
            return Err(StreamError::RateLimited {
                retry_after_secs: header("retry-after").and_then(|v| v.parse().ok()),
            });
        }
        other => return Err(bad(format!("HTTP status {other}"))),
    }
    if header("transfer-encoding").is_some() {
        return Err(bad("Transfer-Encoding is not supported"));
    }

    let content_type = header("content-type").ok_or_else(|| bad("missing Content-Type"))?;
    let mut params = content_type.split(';');
    let media_type = params.next().unwrap_or("").trim();
    if !media_type.eq_ignore_ascii_case("multipart/x-mixed-replace") {
        return Err(bad("Content-Type is not multipart/x-mixed-replace"));
    }
    let boundary = params
        .filter_map(|p| p.split_once('='))
        .find(|(name, _)| name.trim().eq_ignore_ascii_case("boundary"))
        .map(|(_, value)| value.trim())
        .ok_or_else(|| bad("Content-Type has no boundary"))?;
    // The boundary may be a token or a quoted string; the value is fixed.
    let boundary = boundary
        .strip_prefix('"')
        .and_then(|b| b.strip_suffix('"'))
        .unwrap_or(boundary);
    if boundary != BOUNDARY {
        return Err(bad("unexpected multipart boundary"));
    }
    Ok(())
}

/// Validate a delimiter line plus part headers; returns the payload length.
fn parse_part_headers(block: &[u8], max_frame: usize) -> Result<usize, StreamError> {
    let (delimiter, headers) = header_lines(block)?;
    if delimiter != format!("--{BOUNDARY}") {
        return Err(bad("expected the multipart delimiter"));
    }
    let mut lengths = headers.iter().filter(|(n, _)| n == "content-length");
    let value = lengths.next().ok_or_else(|| bad("part without Content-Length"))?.1;
    if lengths.next().is_some() {
        return Err(bad("duplicate Content-Length"));
    }
    // Digits only: `parse` alone would accept a leading '+'.
    if value.is_empty() || !value.bytes().all(|b| b.is_ascii_digit()) {
        return Err(bad("invalid Content-Length"));
    }
    let len: usize = value.parse().map_err(|_| bad("invalid Content-Length"))?;
    if len == 0 || len > max_frame {
        return Err(bad(format!("Content-Length {len} outside 1..={max_frame}")));
    }
    Ok(len)
}

// ── session ──────────────────────────────────────────────────────────────────

/// Every deadline of one session; the tests shrink them.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Timing {
    pub connect: Duration,
    /// Socket read timeout — the cadence at which `should_stop` is consulted.
    pub read: Duration,
    /// Longest silence tolerated from a connected peer.
    pub stall: Duration,
    /// Longest time one unit (headers or a frame) may take from first byte to
    /// last, so a peer dripping a byte at a time cannot hold the loop forever.
    pub frame: Duration,
}

impl Default for Timing {
    fn default() -> Self {
        Self {
            connect: Duration::from_secs(3),
            read: Duration::from_millis(250),
            stall: Duration::from_secs(3),
            frame: Duration::from_secs(5),
        }
    }
}

/// Connect, authenticate and stream until `should_stop` says so (`Ok`) or the
/// session fails (`Err`). `on_frame` gets every complete JPEG.
pub(crate) fn run_session(
    addr: SocketAddr,
    request: &[u8],
    max_frame: usize,
    timing: Timing,
    on_frame: &mut dyn FnMut(&[u8]),
    should_stop: &mut dyn FnMut() -> bool,
) -> Result<(), StreamError> {
    let unreachable = |e: std::io::Error| StreamError::Unreachable(e.kind().to_string());

    let mut stream = TcpStream::connect_timeout(&addr, timing.connect).map_err(unreachable)?;
    stream.set_read_timeout(Some(timing.read)).map_err(unreachable)?;
    stream.set_write_timeout(Some(timing.connect)).map_err(unreachable)?;
    let _ = stream.set_nodelay(true);
    stream.write_all(request).map_err(unreachable)?;

    let mut reader = MjpegReader::new(max_frame);
    let mut chunk = vec![0u8; READ_CHUNK];
    let mut last_byte = Instant::now();
    let mut unit_started = last_byte;

    loop {
        if should_stop() {
            return Ok(());
        }
        match stream.read(&mut chunk) {
            Ok(0) => return Err(StreamError::Unreachable("closed by peer".to_string())),
            Ok(n) => {
                let now = Instant::now();
                last_byte = now;
                if reader.is_idle() {
                    unit_started = now;
                }
                reader.feed(&chunk[..n]);
                while let Some(frame) = reader.next_frame()? {
                    on_frame(frame);
                    // Whatever is left over belongs to a new unit.
                    unit_started = now;
                }
            }
            Err(e) if matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut) => {}
            Err(e) if e.kind() == ErrorKind::Interrupted => {}
            Err(e) => return Err(unreachable(e)),
        }
        let now = Instant::now();
        if now.duration_since(last_byte) >= timing.stall
            || (!reader.is_idle() && now.duration_since(unit_started) >= timing.frame)
        {
            return Err(StreamError::Stalled);
        }
    }
}

// ── retry policy ─────────────────────────────────────────────────────────────

/// Back-off and log de-duplication across connection attempts.
#[derive(Debug, Default)]
pub(crate) struct RetryState {
    failures: usize,
    last_logged: Option<CameraHealthDetail>,
    streaming: bool,
}

impl RetryState {
    /// True until the first failure after start-up or after a healthy session.
    pub(crate) fn is_fresh(&self) -> bool {
        self.failures == 0
    }

    /// A valid frame arrived: the back-off starts over. Returns true for the
    /// first frame of a session — the transition worth a log line and a
    /// `capturing` health report.
    pub(crate) fn on_frame(&mut self) -> bool {
        let started = !self.streaming;
        self.streaming = true;
        self.failures = 0;
        self.last_logged = None;
        started
    }

    /// A session failed: how long to wait, and whether to log it (only when
    /// the failure kind changed — not once per retry).
    pub(crate) fn on_failure(&mut self, err: &StreamError) -> (Duration, bool) {
        let delay = RETRY_DELAYS[self.failures.min(RETRY_DELAYS.len() - 1)];
        self.failures += 1;
        self.streaming = false;
        let log = self.last_logged != Some(err.detail());
        self.last_logged = Some(err.detail());
        (delay, log)
    }
}

/// The one log line a failure transition gets. Built from the endpoint and the
/// error only, so the token cannot appear in it.
pub(crate) fn transition_line(host: &str, port: u32, err: &StreamError) -> String {
    format!("[webcam] Network camera {host}:{port} {err}; retrying")
}

impl WebcamServer {
    /// Stream from the configured remote camera until a restart is requested.
    ///
    /// Same contract as the V4L2 loops: `Ok(())` only when a restart was
    /// requested, `Err` when the source failed. The caller owns health
    /// reporting for failures and the retry wait.
    pub(crate) fn try_network_mjpeg_loop(
        cfg: &CameraConfig,
        shm: &Arc<ShmProducer>,
        config_arc: &Arc<Mutex<CameraConfig>>,
        restart_flag: &AtomicBool,
        rgb_hardware_supported: &AtomicBool,
        retry: &mut RetryState,
    ) -> Result<(), StreamError> {
        let ip: Ipv4Addr = cfg
            .network_host
            .parse()
            .map_err(|_| StreamError::Unreachable("no valid network_host".to_string()))?;
        let port = u16::try_from(cfg.network_port)
            .ok()
            .filter(|p| *p != 0)
            .ok_or_else(|| StreamError::Unreachable("no valid network_port".to_string()))?;
        let addr = SocketAddr::V4(SocketAddrV4::new(ip, port));
        let request = build_request(&cfg.network_host, cfg.network_port, &cfg.network_token);

        let mut on_frame = |jpeg: &[u8]| {
            // Dimensions stay unset in the SHM header: consumers decode them
            // from the JPEG, so the remote camera may change resolution freely.
            shm.publish_frame_jpeg(jpeg);
            if retry.on_frame() {
                eprintln!(
                    "[webcam] Network camera {}:{} streaming",
                    cfg.network_host, cfg.network_port
                );
                Self::publish_health_detail(shm, "capturing", CameraHealthDetail::Unspecified);
            }
        };
        let mut should_stop = || {
            if let Some(proto_cfg) = shm.poll_config() {
                Self::apply_shm_config(
                    &proto_cfg,
                    config_arc,
                    shm,
                    restart_flag,
                    rgb_hardware_supported,
                );
            }
            restart_flag.load(Ordering::Relaxed)
        };
        run_session(
            addr,
            &request,
            shm.slot_size(),
            Timing::default(),
            &mut on_frame,
            &mut should_stop,
        )
    }

    /// Wait out a retry delay, still honouring restarts and config changes.
    pub(crate) fn network_retry_wait(
        delay: Duration,
        shm: &Arc<ShmProducer>,
        config_arc: &Arc<Mutex<CameraConfig>>,
        restart_flag: &AtomicBool,
        rgb_hardware_supported: &AtomicBool,
    ) {
        let retry_at = Instant::now() + delay;
        while !restart_flag.load(Ordering::Relaxed) && Instant::now() < retry_at {
            if let Some(proto_cfg) = shm.poll_config() {
                Self::apply_shm_config(
                    &proto_cfg,
                    config_arc,
                    shm,
                    restart_flag,
                    rgb_hardware_supported,
                );
            }
            std::thread::sleep(RETRY_POLL);
        }
    }
}

#[cfg(test)]
mod tests;
