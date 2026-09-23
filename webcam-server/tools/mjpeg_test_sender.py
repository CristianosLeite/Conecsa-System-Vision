#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Remote camera protocol v1 test sender (standard library only).

Stands in for the Android app so the whole capture path — webcam-server, SHM,
inference, gateway, device screen — can be exercised from a PC. It serves
``GET /stream.mjpeg`` exactly as ``docs/services/remote-camera-protocol.md``
describes, and can misbehave on purpose (``--fault``) to drive every health
detail the webcam-server reports.

Bind it to the machine's LAN address, not ``127.0.0.1``: inside the dev compose
loopback is the webcam-server container itself, and the device API rejects a
loopback camera address.

    python3 webcam-server/tools/mjpeg_test_sender.py --bind 192.0.2.10 --token ABCD-1234

The token is never printed; only whether a request carried the right one.
"""
import argparse
import base64
import hmac
import itertools
import os
import socket
import sys
import time

BOUNDARY = "conecsaframe"
REQUEST_HEADER_MAX = 8 * 1024
LOCKOUT_FAILURES = 5
LOCKOUT_SECS = 30

FAULTS = (
    "none", "stall", "truncated-jpeg", "oversized", "missing-length", "duplicate-length",
    "wrong-boundary", "chunked", "slow-headers", "trailing-bytes", "slow-drip",
)

# A 320x240 test pattern, used when --images is not given.
_BUILTIN_JPEG = base64.b64decode(
    "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"
)


def normalize_token(token: str) -> str:
    """Hyphens are presentation only and case does not matter (Crockford base32)."""
    return token.replace("-", "").upper()


def load_frames(directory):
    if not directory:
        return [_BUILTIN_JPEG]
    frames = []
    for name in sorted(os.listdir(directory)):
        if name.lower().endswith((".jpg", ".jpeg")):
            with open(os.path.join(directory, name), "rb") as fh:
                frames.append(fh.read())
    if not frames:
        sys.exit(f"no .jpg/.jpeg files in {directory}")
    return frames


def read_request(conn):
    """Return (method, path, headers) or None for a malformed/oversized request."""
    conn.settimeout(5)
    data = b""
    while b"\r\n\r\n" not in data:
        if len(data) > REQUEST_HEADER_MAX:
            return None
        chunk = conn.recv(1024)
        if not chunk:
            return None
        data += chunk
    lines = data.split(b"\r\n\r\n", 1)[0].decode("latin-1").split("\r\n")
    parts = lines[0].split()
    if len(parts) != 3:
        return None
    headers = {}
    for line in lines[1:]:
        name, _, value = line.partition(":")
        headers[name.strip().lower()] = value.strip()
    return parts[0], parts[1], headers


def respond(conn, status, reason, extra=""):
    conn.sendall(f"HTTP/1.1 {status} {reason}\r\n{extra}Connection: close\r\n\r\n".encode())


class Lockout:
    """Five bad tokens lock authentication for 30 s, on a monotonic clock."""

    def __init__(self):
        self.failures = 0
        self.until = 0.0

    def remaining(self):
        return max(0, int(self.until - time.monotonic() + 0.999))

    def failed(self):
        self.failures += 1
        if self.failures >= LOCKOUT_FAILURES:
            self.until = time.monotonic() + LOCKOUT_SECS
            self.failures = 0

    def succeeded(self):
        self.failures = 0


def part(jpeg, seq, fault):
    headers = ["Content-Type: image/jpeg"]
    if fault == "oversized":
        headers.append(f"Content-Length: {1 << 30}")
    elif fault != "missing-length":
        headers.append(f"Content-Length: {len(jpeg)}")
    if fault == "duplicate-length":
        headers.append(f"Content-Length: {len(jpeg)}")
    headers += [f"X-Frame-Seq: {seq}", "X-Encode-Ms: 0"]
    if fault == "truncated-jpeg":
        jpeg = jpeg[:-2] + b"\x00\x00"  # right length, no EOI marker
    tail = b"xx\r\n" if fault == "trailing-bytes" else b"\r\n"
    head = f"--{BOUNDARY}\r\n" + "\r\n".join(headers) + "\r\n\r\n"
    return head.encode() + jpeg + tail


def stream(conn, args, frames):
    fault = args.fault
    boundary = "someotherboundary" if fault == "wrong-boundary" else BOUNDARY
    head = f"HTTP/1.1 200 OK\r\nContent-Type: multipart/x-mixed-replace; boundary={boundary}\r\n"
    if fault == "chunked":
        head += "Transfer-Encoding: chunked\r\n"
    head = (head + "Connection: close\r\n\r\n").encode()
    conn.settimeout(10)
    if fault == "slow-headers":
        for byte in head:
            conn.sendall(bytes([byte]))
            time.sleep(0.5)
    else:
        conn.sendall(head)

    interval = 1.0 / args.fps
    next_at = time.monotonic()
    for seq, jpeg in enumerate(itertools.cycle(frames), start=1):
        if fault == "stall" and seq > args.good_frames:
            print("fault: stalling (socket open, no bytes)")
            time.sleep(3600)
        active = fault if seq > args.good_frames else "none"
        data = part(jpeg, seq, active)
        if active == "slow-drip":
            for byte in data:
                conn.sendall(bytes([byte]))
                time.sleep(0.2)
        else:
            conn.sendall(data)
        next_at += interval
        time.sleep(max(0.0, next_at - time.monotonic()))


def main():
    parser = argparse.ArgumentParser(description="Remote camera protocol v1 test sender")
    parser.add_argument("--bind", required=True, help="LAN IPv4 address to listen on")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--token", required=True, help="stream token, e.g. ABCD-1234")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--images", help="directory of .jpg files to cycle (default: test pattern)")
    parser.add_argument("--fault", choices=FAULTS, default="none")
    parser.add_argument("--good-frames", type=int, default=30,
                        help="valid frames sent before a per-frame fault starts")
    parser.add_argument("--status", type=int, default=200,
                        help="answer every authenticated request with this status instead")
    args = parser.parse_args()

    token = normalize_token(args.token).encode()
    frames = load_frames(args.images)
    lockout = Lockout()

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.bind, args.port))
    server.listen(1)
    print(f"serving {len(frames)} frame(s) at {args.fps:g} fps on {args.bind}:{args.port} "
          f"(fault: {args.fault}, token: <redacted>)")

    # One client at a time: a new connection is served after the previous ends.
    while True:
        conn, peer = server.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            request = read_request(conn)
            if request is None:
                respond(conn, 400, "Bad Request")
                continue
            method, path, headers = request
            if path != "/stream.mjpeg":
                respond(conn, 404, "Not Found")
            elif method != "GET":
                respond(conn, 405, "Method Not Allowed", "Allow: GET\r\n")
            elif lockout.remaining():
                print(f"{peer[0]}: locked out")
                respond(conn, 429, "Too Many Requests", f"Retry-After: {lockout.remaining()}\r\n")
            else:
                scheme, _, offered = headers.get("authorization", "").partition(" ")
                offered = normalize_token(offered).encode()
                if scheme.lower() != "bearer" or not hmac.compare_digest(offered, token):
                    lockout.failed()
                    print(f"{peer[0]}: bad or missing token")
                    respond(conn, 401, "Unauthorized", "WWW-Authenticate: Bearer\r\n")
                elif args.status != 200:
                    respond(conn, args.status, "Test Status")
                else:
                    lockout.succeeded()
                    print(f"{peer[0]}: authenticated, streaming")
                    stream(conn, args, frames)
        except (OSError, socket.timeout) as exc:
            print(f"{peer[0]}: disconnected ({exc.__class__.__name__})")
        finally:
            conn.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
