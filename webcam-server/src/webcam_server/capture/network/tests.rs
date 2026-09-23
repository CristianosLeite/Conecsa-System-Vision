// SPDX-FileCopyrightText: 2026 Conecsa
//
// SPDX-License-Identifier: AGPL-3.0-only

//! Unit tests for the network MJPEG source: the protocol-v1 golden bytes, the
//! incremental parser, the session deadlines and the retry policy.
use super::*;

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::thread;

/// Protocol version 1, byte for byte. The Android repository carries the same
/// two files; a change here is a protocol change, not a fixture refresh.
const GOLDEN_REQUEST: &[u8] = include_bytes!("fixtures/protocol_v1_request.bin");
const GOLDEN_RESPONSE: &[u8] = include_bytes!("fixtures/protocol_v1_response.bin");

const FRAME_ONE: &[u8] = b"\xff\xd8conecsa-frame-one\xff\xd9";
const FRAME_TWO: &[u8] = b"\xff\xd8conecsa-frame-two!\xff\xd9";
/// Visibly fake; the same value the golden request carries.
const TOKEN: &str = "TESTT0KEN123";
const MAX_FRAME: usize = 4096;

const OK_HEADERS: &str = "HTTP/1.1 200 OK\r\n\
    Content-Type: multipart/x-mixed-replace; boundary=conecsaframe\r\n\
    Connection: close\r\n\r\n";

fn part(headers: &str, payload: &[u8]) -> Vec<u8> {
    let mut out = format!("--conecsaframe\r\n{headers}\r\n").into_bytes();
    out.extend_from_slice(payload);
    out.extend_from_slice(b"\r\n");
    out
}

fn stream_of(parts: &[Vec<u8>]) -> Vec<u8> {
    let mut out = OK_HEADERS.as_bytes().to_vec();
    parts.iter().for_each(|p| out.extend_from_slice(p));
    out
}

/// Feed `pieces` in order and collect every frame, or the first error.
fn parse(pieces: &[&[u8]]) -> Result<Vec<Vec<u8>>, StreamError> {
    let mut reader = MjpegReader::new(MAX_FRAME);
    let mut frames = Vec::new();
    for piece in pieces {
        reader.feed(piece);
        while let Some(frame) = reader.next_frame()? {
            frames.push(frame.to_vec());
        }
    }
    Ok(frames)
}

fn bad_stream(result: Result<Vec<Vec<u8>>, StreamError>) -> String {
    match result {
        Err(StreamError::BadStream(why)) => why,
        other => panic!("expected BadStream, got {other:?}"),
    }
}

// ── request ──────────────────────────────────────────────────────────────────

#[test]
fn the_request_matches_the_golden_bytes() {
    assert_eq!(build_request("192.0.2.1", 8080, TOKEN), GOLDEN_REQUEST);
}

#[test]
fn the_request_carries_host_and_bearer_authorization() {
    let request = String::from_utf8(build_request("192.0.2.7", 9000, TOKEN)).unwrap();
    assert!(request.starts_with("GET /stream.mjpeg HTTP/1.1\r\n"));
    assert!(request.contains("\r\nHost: 192.0.2.7:9000\r\n"), "HTTP/1.1 requires Host");
    assert!(request.contains(&format!("\r\nAuthorization: Bearer {TOKEN}\r\n")));
    assert!(request.ends_with("\r\n\r\n"));
}

// ── parser: happy paths ──────────────────────────────────────────────────────

#[test]
fn the_golden_response_yields_both_frames_from_one_read() {
    let frames = parse(&[GOLDEN_RESPONSE]).unwrap();
    assert_eq!(frames, vec![FRAME_ONE.to_vec(), FRAME_TWO.to_vec()]);
}

#[test]
fn every_split_point_yields_the_same_frames() {
    // Covers a split inside the status line, a header, each CRLF, the
    // delimiter, the payload and the trailing CRLF.
    for at in 0..=GOLDEN_RESPONSE.len() {
        let (head, tail) = GOLDEN_RESPONSE.split_at(at);
        let frames = parse(&[head, tail]).unwrap_or_else(|e| panic!("split at {at}: {e}"));
        assert_eq!(frames.len(), 2, "split at {at}");
        assert_eq!(frames[1], FRAME_TWO, "split at {at}");
    }
}

#[test]
fn a_byte_at_a_time_yields_the_same_frames() {
    let pieces: Vec<&[u8]> = GOLDEN_RESPONSE.chunks(1).collect();
    assert_eq!(parse(&pieces).unwrap().len(), 2);
}

#[test]
fn the_reader_is_idle_only_between_units() {
    let mut reader = MjpegReader::new(MAX_FRAME);
    assert!(reader.is_idle());
    reader.feed(&GOLDEN_RESPONSE[..10]);
    assert!(reader.next_frame().unwrap().is_none());
    assert!(!reader.is_idle(), "a partial header is in flight");
    reader.feed(&GOLDEN_RESPONSE[10..]);
    while reader.next_frame().unwrap().is_some() {}
    assert!(reader.is_idle());
}

#[test]
fn header_names_are_case_insensitive_and_the_boundary_may_be_quoted() {
    let mut stream = b"HTTP/1.1 200 OK\r\n\
        content-TYPE: Multipart/X-Mixed-Replace; BOUNDARY=\"conecsaframe\"\r\n\r\n"
        .to_vec();
    stream.extend(part("CONTENT-length: 21\r\nx-unknown: ignored\r\n", FRAME_ONE));
    assert_eq!(parse(&[&stream]).unwrap(), vec![FRAME_ONE.to_vec()]);
}

// ── parser: response rejection ───────────────────────────────────────────────

#[test]
fn http_401_is_unauthorized() {
    let result = parse(&[b"HTTP/1.1 401 Unauthorized\r\nConnection: close\r\n\r\n"]);
    assert_eq!(result, Err(StreamError::Unauthorized));
}

#[test]
fn http_429_is_rate_limited_with_its_retry_after() {
    let result = parse(&[b"HTTP/1.1 429 Too Many Requests\r\nRetry-After: 30\r\n\r\n"]);
    assert_eq!(result, Err(StreamError::RateLimited { retry_after_secs: Some(30) }));
    let result = parse(&[b"HTTP/1.1 429 Too Many Requests\r\n\r\n"]);
    assert_eq!(result, Err(StreamError::RateLimited { retry_after_secs: None }));
}

#[test]
fn any_other_status_is_a_bad_stream() {
    for status in ["404 Not Found", "405 Method Not Allowed", "500 Oops", "204 No Content"] {
        let response = format!("HTTP/1.1 {status}\r\n\r\n");
        assert!(bad_stream(parse(&[response.as_bytes()])).contains("HTTP status"));
    }
    assert!(bad_stream(parse(&[b"ICY 200 OK\r\n\r\n"])).contains("HTTP/1.x"));
}

#[test]
fn a_chunked_response_is_rejected() {
    let response = b"HTTP/1.1 200 OK\r\n\
        Content-Type: multipart/x-mixed-replace; boundary=conecsaframe\r\n\
        Transfer-Encoding: chunked\r\n\r\n";
    assert!(bad_stream(parse(&[response])).contains("Transfer-Encoding"));
}

#[test]
fn a_wrong_content_type_or_boundary_is_rejected() {
    let cases: [&[u8]; 4] = [
        b"HTTP/1.1 200 OK\r\nContent-Type: image/jpeg\r\n\r\n",
        b"HTTP/1.1 200 OK\r\nContent-Type: multipart/x-mixed-replace\r\n\r\n",
        b"HTTP/1.1 200 OK\r\nContent-Type: multipart/x-mixed-replace; boundary=frame\r\n\r\n",
        b"HTTP/1.1 200 OK\r\nConnection: close\r\n\r\n",
    ];
    for response in cases {
        bad_stream(parse(&[response]));
    }
}

// ── parser: part rejection ───────────────────────────────────────────────────

#[test]
fn content_length_must_be_present_once_valid_and_in_range() {
    let too_big = format!("Content-Length: {}\r\n", MAX_FRAME + 1);
    let cases = [
        ("Content-Type: image/jpeg\r\n", "without Content-Length"),
        ("Content-Length: 22\r\nContent-Length: 22\r\n", "duplicate"),
        ("Content-Length: 22\r\ncontent-length: 23\r\n", "duplicate"),
        ("Content-Length: +22\r\n", "invalid"),
        ("Content-Length: 0x16\r\n", "invalid"),
        ("Content-Length: \r\n", "invalid"),
        ("Content-Length: 99999999999999999999999\r\n", "invalid"),
        ("Content-Length: 0\r\n", "outside"),
        (too_big.as_str(), "outside"),
    ];
    for (headers, expected) in cases {
        let why = bad_stream(parse(&[&stream_of(&[part(headers, FRAME_ONE)])]));
        assert!(why.contains(expected), "{headers:?} gave {why:?}");
    }
}

#[test]
fn a_frame_exactly_filling_the_limit_is_accepted() {
    let mut jpeg = vec![0u8; MAX_FRAME];
    jpeg[..2].copy_from_slice(&[0xFF, 0xD8]);
    jpeg[MAX_FRAME - 2..].copy_from_slice(&[0xFF, 0xD9]);
    let headers = format!("Content-Length: {MAX_FRAME}\r\n");
    assert_eq!(parse(&[&stream_of(&[part(&headers, &jpeg)])]).unwrap().len(), 1);
}

#[test]
fn a_truncated_jpeg_is_not_published() {
    // The declared length is honoured, but the payload lacks the EOI marker.
    let why = bad_stream(parse(&[&stream_of(&[part("Content-Length: 20\r\n", &FRAME_ONE[..20])])]));
    assert!(why.contains("complete JPEG"));
    // ... or the SOI marker.
    let why = bad_stream(parse(&[&stream_of(&[part("Content-Length: 19\r\n", &FRAME_ONE[2..])])]));
    assert!(why.contains("complete JPEG"));
}

#[test]
fn bytes_after_the_declared_length_are_rejected() {
    let mut payload = FRAME_ONE.to_vec();
    payload.extend_from_slice(b"xx");
    let why = bad_stream(parse(&[&stream_of(&[part("Content-Length: 21\r\n", &payload)])]));
    assert!(why.contains("after the declared"));
}

#[test]
fn a_part_must_start_with_the_delimiter() {
    let mut stream = OK_HEADERS.as_bytes().to_vec();
    stream.extend_from_slice(b"--otherboundary\r\nContent-Length: 21\r\n\r\n");
    assert!(bad_stream(parse(&[&stream])).contains("delimiter"));
}

#[test]
fn a_good_frame_before_a_bad_one_is_still_delivered() {
    let stream = stream_of(&[
        part("Content-Length: 21\r\n", FRAME_ONE),
        part("Content-Type: image/jpeg\r\n", FRAME_TWO),
    ]);
    let mut reader = MjpegReader::new(MAX_FRAME);
    reader.feed(&stream);
    assert_eq!(reader.next_frame().unwrap(), Some(FRAME_ONE));
    assert!(reader.next_frame().is_err());
}

// ── parser: size limits ──────────────────────────────────────────────────────

#[test]
fn response_headers_are_limited_to_8_kib() {
    let filler = format!("X-Filler: {}\r\n", "a".repeat(RESPONSE_HEADER_MAX));
    let response = format!("HTTP/1.1 200 OK\r\n{filler}\r\n");
    assert!(bad_stream(parse(&[response.as_bytes()])).contains("response headers exceed"));
    // A line that never ends fails at the limit instead of buffering forever.
    let endless = vec![b'a'; RESPONSE_HEADER_MAX];
    assert!(bad_stream(parse(&[&endless])).contains("response headers exceed"));
    assert!(parse(&[&endless[..RESPONSE_HEADER_MAX - 1]]).unwrap().is_empty());
}

#[test]
fn part_headers_are_limited_to_1_kib() {
    let filler = format!("Content-Length: 22\r\nX-Filler: {}\r\n", "a".repeat(PART_HEADER_MAX));
    let stream = stream_of(&[part(&filler, FRAME_ONE)]);
    assert!(bad_stream(parse(&[&stream])).contains("part headers exceed"));
}

// ── session I/O ──────────────────────────────────────────────────────────────

/// Short deadlines so the I/O tests stay fast.
fn fast() -> Timing {
    Timing {
        connect: Duration::from_millis(500),
        read: Duration::from_millis(20),
        stall: Duration::from_millis(200),
        frame: Duration::from_millis(400),
    }
}

/// A one-connection fake remote camera. `serve` gets the request bytes and the socket.
fn fake_camera(
    serve: impl FnOnce(Vec<u8>, TcpStream) + Send + 'static,
) -> (SocketAddr, thread::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let handle = thread::spawn(move || {
        let (mut socket, _) = listener.accept().unwrap();
        let mut request = Vec::new();
        let mut byte = [0u8; 1];
        while !request.ends_with(b"\r\n\r\n") && socket.read(&mut byte).unwrap_or(0) == 1 {
            request.push(byte[0]);
        }
        serve(request, socket);
    });
    (addr, handle)
}

fn run(addr: SocketAddr, timing: Timing) -> (Result<(), StreamError>, Vec<Vec<u8>>) {
    let mut frames = Vec::new();
    let result = run_session(
        addr,
        GOLDEN_REQUEST,
        MAX_FRAME,
        timing,
        &mut |jpeg| frames.push(jpeg.to_vec()),
        &mut || false,
    );
    (result, frames)
}

#[test]
fn a_session_sends_the_request_and_delivers_frames_until_the_peer_closes() {
    let (addr, camera) = fake_camera(|request, mut socket| {
        assert_eq!(request, GOLDEN_REQUEST);
        socket.write_all(GOLDEN_RESPONSE).unwrap();
    });
    let (result, frames) = run(addr, fast());
    camera.join().unwrap();
    assert_eq!(frames, vec![FRAME_ONE.to_vec(), FRAME_TWO.to_vec()]);
    assert_eq!(result, Err(StreamError::Unreachable("closed by peer".to_string())));
}

#[test]
fn a_refused_connection_is_unreachable() {
    // Bind then drop: the port is free and nothing listens on it.
    let addr = TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap();
    let (result, frames) = run(addr, fast());
    assert!(matches!(result, Err(StreamError::Unreachable(_))), "{result:?}");
    assert!(frames.is_empty());
}

#[test]
fn a_wrong_token_surfaces_as_unauthorized() {
    let (addr, camera) = fake_camera(|_, mut socket| {
        socket.write_all(b"HTTP/1.1 401 Unauthorized\r\nConnection: close\r\n\r\n").unwrap();
    });
    let (result, _) = run(addr, fast());
    camera.join().unwrap();
    assert_eq!(result, Err(StreamError::Unauthorized));
}

#[test]
fn a_silent_peer_stalls_at_the_deadline() {
    let (addr, camera) = fake_camera(|_, mut socket| {
        socket.write_all(GOLDEN_RESPONSE).unwrap();
        thread::sleep(Duration::from_millis(600)); // keep the socket open, say nothing
    });
    let started = Instant::now();
    let (result, frames) = run(addr, fast());
    assert_eq!(result, Err(StreamError::Stalled));
    assert_eq!(frames.len(), 2, "frames before the silence were delivered");
    assert!(started.elapsed() < Duration::from_millis(550), "bounded by the stall deadline");
    camera.join().unwrap();
}

#[test]
fn a_slow_drip_is_bounded_by_the_frame_deadline() {
    // One byte every 50 ms never trips the 200 ms silence deadline, so only
    // the per-unit deadline can end this.
    let (addr, camera) = fake_camera(|_, mut socket| {
        for byte in GOLDEN_RESPONSE.iter().take(40) {
            if socket.write_all(&[*byte]).is_err() {
                break;
            }
            thread::sleep(Duration::from_millis(50));
        }
    });
    let started = Instant::now();
    let (result, frames) = run(addr, fast());
    assert_eq!(result, Err(StreamError::Stalled));
    assert!(frames.is_empty());
    assert!(started.elapsed() < Duration::from_millis(900), "took {:?}", started.elapsed());
    camera.join().unwrap();
}

#[test]
fn a_restart_request_ends_a_stalled_session_cleanly() {
    let (addr, camera) = fake_camera(|_, _socket| thread::sleep(Duration::from_millis(300)));
    let slow_to_stall = Timing { stall: Duration::from_secs(30), ..fast() };
    let mut polls = 0;
    let result = run_session(addr, GOLDEN_REQUEST, MAX_FRAME, slow_to_stall, &mut |_| {}, &mut || {
        polls += 1;
        polls > 3
    });
    assert_eq!(result, Ok(()), "Ok means a restart was requested, as in the V4L2 loops");
    camera.join().unwrap();
}

#[test]
fn an_oversized_part_ends_the_session_before_it_is_buffered() {
    let (addr, camera) = fake_camera(|_, mut socket| {
        let headers = format!("Content-Length: {}\r\n", MAX_FRAME + 1);
        let _ = socket.write_all(&stream_of(&[part(&headers, &vec![0u8; MAX_FRAME + 1])]));
    });
    let (result, frames) = run(addr, fast());
    camera.join().unwrap();
    assert!(matches!(result, Err(StreamError::BadStream(_))), "{result:?}");
    assert!(frames.is_empty());
}

// ── retry policy and secrecy ─────────────────────────────────────────────────

#[test]
fn retries_back_off_1_2_5_and_reset_after_a_frame() {
    let mut retry = RetryState::default();
    assert!(retry.is_fresh());
    let delays: Vec<u64> =
        (0..5).map(|_| retry.on_failure(&StreamError::Stalled).0.as_secs()).collect();
    assert_eq!(delays, vec![1, 2, 5, 5, 5]);
    assert!(!retry.is_fresh());

    assert!(retry.on_frame(), "the first frame of a session is a transition");
    assert!(!retry.on_frame(), "later frames are not");
    assert!(retry.is_fresh());
    assert_eq!(retry.on_failure(&StreamError::Stalled).0.as_secs(), 1);
}

#[test]
fn a_repeated_failure_is_logged_once_per_transition() {
    let mut retry = RetryState::default();
    let unreachable = StreamError::Unreachable("connection refused".to_string());
    assert!(retry.on_failure(&unreachable).1);
    assert!(!retry.on_failure(&unreachable).1, "same state: no second line");
    assert!(retry.on_failure(&StreamError::Unauthorized).1, "a new state logs");
    retry.on_frame();
    assert!(retry.on_failure(&StreamError::Unauthorized).1, "logs again after a recovery");
}

#[test]
fn failures_map_to_their_health_detail() {
    let cases = [
        (StreamError::Unauthorized, CameraHealthDetail::Unauthorized),
        (StreamError::RateLimited { retry_after_secs: None }, CameraHealthDetail::RateLimited),
        (StreamError::Unreachable(String::new()), CameraHealthDetail::Unreachable),
        (StreamError::Stalled, CameraHealthDetail::Stalled),
        (StreamError::BadStream(String::new()), CameraHealthDetail::BadStream),
    ];
    for (err, detail) in cases {
        assert_eq!(err.detail(), detail);
    }
}

#[test]
fn no_error_or_log_line_of_a_session_contains_the_token() {
    // A remote camera that echoes the whole request — token included — back as junk.
    let (addr, camera) = fake_camera(|request, mut socket| {
        let _ = socket.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: ");
        let _ = socket.write_all(&request);
        let _ = socket.write_all(b"\r\n\r\n");
    });
    let (result, _) = run(addr, fast());
    camera.join().unwrap();
    let err = result.unwrap_err();
    for text in [format!("{err}"), format!("{err:?}"), transition_line("192.0.2.1", 8080, &err)] {
        assert!(!text.contains(TOKEN), "token leaked into {text:?}");
    }
}

// ── config polling during a session and during a retry delay ────────────────

use super::super::super::shm::{proto, ShmProducer};
use super::super::super::{CameraConfig, CameraSource, WebcamServer};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

fn producer(tag: &str) -> Arc<ShmProducer> {
    let name = format!("conecsa-shm-network-{tag}-{}", std::process::id());
    Arc::new(ShmProducer::new(&name, 64, 64).expect("create segment"))
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

/// An administrator moving the remote camera to another address.
fn moved_to(host: &str) -> proto::CameraConfig {
    proto::CameraConfig {
        source: Some(proto::CameraSource::Network as i32),
        network_host: host.to_string(),
        network_port: 8080,
        network_token: TOKEN.to_string(),
        ..Default::default()
    }
}

fn write_after(shm: &Arc<ShmProducer>, delay: Duration, cfg: proto::CameraConfig) -> thread::JoinHandle<()> {
    let writer = Arc::clone(shm);
    thread::spawn(move || {
        thread::sleep(delay);
        writer.write_config_for_test(&cfg);
    })
}

#[test]
fn a_config_change_during_a_retry_delay_is_applied_and_ends_the_wait() {
    let shm = producer("retry");
    let config = Arc::new(Mutex::new(network_config()));
    let restart = AtomicBool::new(false);
    let rgb = AtomicBool::new(false);
    let writer = write_after(&shm, Duration::from_millis(50), moved_to("192.0.2.2"));

    let started = Instant::now();
    WebcamServer::network_retry_wait(Duration::from_secs(5), &shm, &config, &restart, &rgb);
    writer.join().unwrap();

    assert!(started.elapsed() < Duration::from_secs(2), "the wait ended with the change, not the delay");
    assert!(restart.load(Ordering::Relaxed), "a new address restarts the source");
    let cfg = config.lock().unwrap();
    assert_eq!((cfg.source, cfg.network_host.as_str()), (CameraSource::Network, "192.0.2.2"));
}

#[test]
fn a_restart_flag_ends_the_retry_wait() {
    let shm = producer("restart");
    let config = Arc::new(Mutex::new(network_config()));
    let restart = AtomicBool::new(true);
    let rgb = AtomicBool::new(false);
    let started = Instant::now();
    WebcamServer::network_retry_wait(Duration::from_secs(5), &shm, &config, &restart, &rgb);
    assert!(started.elapsed() < Duration::from_secs(1));
    assert_eq!(config.lock().unwrap().network_host, "192.0.2.1", "nothing was applied");
}

#[test]
fn a_config_change_during_a_session_is_seen_by_should_stop() {
    // A peer that sent its headers and then goes quiet: the loop polls the
    // config on every read timeout and the new address ends the session.
    let (addr, camera) = fake_camera(|_, mut socket| {
        let _ = socket.write_all(OK_HEADERS.as_bytes());
        thread::sleep(Duration::from_millis(600));
    });
    let shm = producer("session");
    let config = Arc::new(Mutex::new(network_config()));
    let restart = AtomicBool::new(false);
    let rgb = AtomicBool::new(false);
    let writer = write_after(&shm, Duration::from_millis(50), moved_to("192.0.2.3"));

    let patient = Timing { stall: Duration::from_secs(30), ..fast() };
    let mut should_stop = || {
        if let Some(proto_cfg) = shm.poll_config() {
            WebcamServer::apply_shm_config(&proto_cfg, &config, &shm, &restart, &rgb);
        }
        restart.load(Ordering::Relaxed)
    };
    let result = run_session(addr, GOLDEN_REQUEST, MAX_FRAME, patient, &mut |_| {}, &mut should_stop);
    writer.join().unwrap();
    camera.join().unwrap();

    assert_eq!(result, Ok(()), "the session ended for a restart, not a stall");
    assert_eq!(config.lock().unwrap().network_host, "192.0.2.3");
}
