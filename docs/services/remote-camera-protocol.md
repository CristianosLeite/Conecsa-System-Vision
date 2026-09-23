# Remote camera protocol (version 1)

How a remote camera hands its video to a device. The remote camera — a phone
running the Conecsa camera app — runs a small HTTP server and serves a
Motion-JPEG stream; the device's [webcam-server](webcam-server.md) connects to
it, **pulls** the stream and publishes every complete JPEG into the camera
shared-memory ring, exactly as it does for a local MJPEG camera. Nothing
downstream — inference, the API gateway, training — knows the frames came over
the network.

The device opens no listener for this: the api-gateway stays its only HTTP
surface, and the remote camera never connects to the device.

This is **protocol version 1**. It is identified by its fixed URL rather than a
negotiation header; an incompatible revision gets a new URL.

## Transport and trust

Plain HTTP/1.1 over TCP, IPv4 only, on a **direct WPA2 link** between the remote
camera and the device (the remote camera's hotspot, or the device's own access point). On that
link the 802.11 layer already encrypts and retransmits, which is the only reason
unencrypted HTTP is acceptable here. The bearer token authenticates the device to
the remote camera; it does not provide confidentiality. If this stream ever has to cross
a shared network, it needs transport encryption first.

The token is still a secret. Neither peer logs the `Authorization` header or the
complete request, and the device never returns the token from an API, an event or
the audit trail.

## Request

The device sends exactly:

```http
GET /stream.mjpeg HTTP/1.1
Host: 192.0.2.1:8080
Authorization: Bearer <token>
Accept: multipart/x-mixed-replace
Connection: close

```

Every line ends in CRLF, and an empty line ends the request. `192.0.2.1:8080` is
a documentation address; on the wire `Host` carries the configured IPv4 literal
and port.

### Token

The remote camera generates the token; an administrator types it into the device screen.
It is Crockford base32 — the digits and the upper-case letters except `I`, `L`,
`O` and `U` — 8 to 32 characters. Hyphens are presentation only and case does not
matter, so both peers normalize before comparing: strip hyphens, upper-case. The
device sends the normalized form (`ABCD-1234` goes out as `ABCD1234`). The remote
camera compares in constant time.

## Response

```http
HTTP/1.1 200 OK
Content-Type: multipart/x-mixed-replace; boundary=conecsaframe
Connection: close

--conecsaframe
Content-Type: image/jpeg
Content-Length: <N>
X-Frame-Seq: <optional unsigned integer>
X-Encode-Ms: <optional camera-local duration>

<exactly N JPEG bytes>
```

Again every line ends in CRLF. The `N` payload bytes are followed by one CRLF,
then the next `--conecsaframe` delimiter line. The stream has no end: the remote
camera keeps sending parts until either side closes the connection.

## Rules

- Header names — in the response and in each part — are compared
  case-insensitively. Unknown headers are ignored, within the size limits below.
- The response `Content-Type` must be `multipart/x-mixed-replace` with
  `boundary=conecsaframe`. The boundary may be written as a token or as a quoted
  string; any other value is rejected.
- `Transfer-Encoding` is not supported. A response that carries it, `chunked`
  included, is rejected.
- Each part carries exactly one `Content-Length`: decimal digits only, and
  `0 < N <= slot size`, the capacity of one shared-memory frame slot
  (`SHM_SLOT_MIN_BYTES`, 16 MB in the shipped compose file). A missing, repeated,
  malformed or out-of-range length is an error — an oversized part is refused
  before any of it is buffered.
- A frame is published only when its payload starts with the JPEG SOI marker
  (`FF D8`), ends with EOI (`FF D9`) and is followed by CRLF. A truncated image is
  never published: the consumers would decode garbage.
- The status line plus response headers are limited to **8 KiB**, a delimiter
  line plus part headers to **1 KiB**. A header block that does not end within
  its limit is malformed.
- JPEG dimensions are not carried anywhere in the protocol, and the device leaves
  them unset in the shared-memory header: consumers read them from the JPEG. The
  remote camera may therefore change resolution between frames.
- `X-Frame-Seq` is diagnostic only. `X-Encode-Ms` is a duration measured entirely
  on the remote camera's monotonic clock. The device stamps each frame's arrival with its
  **own** monotonic clock; a value from one machine's clock is never subtracted
  from the other's.

## Server behaviour

| Request | Answer |
|---|---|
| Correct token on `GET /stream.mjpeg` | `200` and the stream |
| Missing or wrong token | `401` |
| Any other path | `404` |
| Any other method | `405` |
| Authentication locked out | `429` with `Retry-After` |

Five failed authentications lock **all** authentication for 30 seconds. The
failure counter and the deadline use a monotonic clock, and the counter resets
after a successful authentication.

One consumer at a time: a newly authenticated client closes and replaces the
previous stream, and only one writer ever owns a stream socket. The remote
camera keeps only the newest frame it has not yet written, so a slow link drops
frames instead of building latency.

## Client behaviour

The device's reader is synchronous and bounded on every axis:

| Limit | Value |
|---|---|
| Connect timeout | 3 s |
| Socket read timeout (how often it checks for a config change or restart) | 250 ms |
| Silence tolerated from a connected remote camera | 3 s |
| Time one unit — the headers, or one frame — may take from first byte to last | 5 s |
| Pause before reconnecting | 1 s, then 2 s, then 5 s; back to 1 s after a valid frame |

It reports what it sees through the camera health in shared memory
(`HealthStatus` in [`proto/shm.proto`](../reference/proto.md)):

| Condition | `status` | `detail` |
|---|---|---|
| The first attempt after a start, a restart or a source change | `starting` | `connecting` |
| A valid frame arrived, and while frames keep flowing | `capturing` | unspecified |
| HTTP 401 | `no_camera` | `unauthorized` |
| HTTP 429 | `no_camera` | `rate_limited` |
| Connection refused, timed out, reset, or closed by the remote camera | `no_camera` | `unreachable` |
| Connected but silent past the deadline, or a frame dripped too slowly | `no_camera` | `stalled` |
| Invalid HTTP, multipart framing or JPEG payload, or any HTTP status other than 200, 401 and 429 | `no_camera` | `bad_stream` |

While it is retrying, the last failure stays visible rather than flickering back
to `connecting` on every attempt, including after a healthy session drops. The
back-off is the fixed 1 s, 2 s, 5 s sequence; a `Retry-After` header does not
change it. Any status other than `capturing` gates
detection, as it does for a local camera. A failed network source is **never**
replaced by a local camera on its own: a silent fallback would leave the operator
unsure which camera is being inspected.

The device logs one line per state change, not one per retry, and never the token.

## Testing without a remote camera

`webcam-server/tools/mjpeg_test_sender.py` is a standard-library stand-in for the
app. It serves the protocol above and can misbehave on purpose:

```bash
python3 webcam-server/tools/mjpeg_test_sender.py --bind <LAN address> --token ABCD-1234
python3 webcam-server/tools/mjpeg_test_sender.py --bind <LAN address> --token ABCD-1234 --fault stall
```

`--fault` takes `stall`, `truncated-jpeg`, `oversized`, `missing-length`,
`duplicate-length`, `wrong-boundary`, `chunked`, `slow-headers`, `slow-drip` and
`trailing-bytes`; `--status` answers with a chosen status code; `--images` cycles
a directory of JPEG files instead of the built-in test pattern.

Bind it to the machine's **LAN address**, not `127.0.0.1`: inside the development
compose stack, loopback is the webcam-server container itself.

The exact request and response bytes are pinned as golden fixtures in
`webcam-server/src/webcam_server/capture/network/fixtures/`. A change to those
files is a protocol change and needs a review, not a fixture refresh.
