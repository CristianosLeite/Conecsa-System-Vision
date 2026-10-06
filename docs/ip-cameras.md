# IP cameras

A device can inspect up to **four RTSP IP cameras** instead of its USB camera.
The cameras are shown together as a **mosaic** on the device screen's live
stream, and inference runs on every camera of the set before the mosaic is
published as one frame.

| Cameras | Layout |
|---|---|
| 1 | the camera fills the stream |
| 2 | side by side, half each |
| 3–4 | 2 × 2 (with three cameras the fourth cell stays dark) |

The layout follows the number of **configured** cameras, not the number online:
a camera that drops keeps its cell, which turns dark until it is back. Detection
areas are drawn on the whole mosaic, so they keep covering the same camera
while another one is offline. Changing the number of cameras changes the
layout, and areas drawn before then may cover a different camera: redraw them.

## Connecting

1. Make sure the device reaches the cameras on the network. Check a stream from
   a computer first, for example
   `ffplay -rtsp_transport tcp "rtsp://192.0.2.10:554/stream"`.
2. On the device screen open **Camera Settings** and choose **IP cameras**.
3. For each camera enter a name, its RTSP address, a user name and a password.
   The address must not contain credentials (`rtsp://user:pass@…` is refused):
   the user name and password have their own fields. The order of the list is
   the order of the mosaic.
4. Apply. Each camera's line shows its state (connecting, capturing, wrong
   password, unreachable, no frames, invalid stream) and refreshes while the
   panel is open.

The password is **write-only**: the device reports only whether one is stored.
Leave the field untouched to keep it; clear it to remove it. Switching to
**Local camera** or **Remote camera** keeps the IP camera list stored on the
device, so returning to it needs no re-entry. Only an administrator can change
the source. The list belongs to the device and is not part of any model's
settings.

!!! tip "Choosing a stream"
    Most cameras offer a main stream and a lower-resolution sub-stream (for
    example `subtype=0` and `subtype=1` on many models). The device decodes
    every camera in software and scales it to at most
    `IPCAM_MAX_WIDTH` × `IPCAM_MAX_HEIGHT` (1920 × 1080); from three cameras
    on, prefer the sub-stream to keep the processor load down.

## How it works

- The `webcam-server` runs one `ffmpeg` process per camera (RTSP over TCP),
  decodes it and publishes the frames into a shared-memory ring of its own,
  `<SHM_NAME>_ip<N>`. Its main ring then carries no frames: its health reports
  how many cameras are configured, and the first camera's state when none
  captures. A camera that fails is retried after 1, 2 and 5 seconds, by
  itself; it never falls back to the USB camera.
- The `inference-service` composes the latest frame of every camera into the
  mosaic. Each camera is letterboxed into a cell of
  `MOSAIC_CELL_WIDTH` × `MOSAIC_CELL_HEIGHT` (1280 × 720), and the model runs on
  each camera's picture on its own — tiled exactly as a single camera of that
  size would be — so boxes never merge across cameras. The boxes are then
  drawn on the mosaic, which is the processed stream, the snapshot frame and
  the clean `raw_frame`.
- The raw video feed, the training preview and dataset captures show the same
  mosaic.

The work grows with the number of cameras: four cameras cost about four times
the inference of one, so expect the processed stream's frame rate to drop
accordingly.

## Results per camera

With IP cameras, every item of `GET /api/v1/detections/snapshot` carries
`camera` (the camera's stable id) and `camera_name`, next to its `bbox`, which
stays normalized to the whole mosaic. The snapshot adds `cameras`, the layout:

```json
"cameras": [
  {"id": "cam-3f9a1c2e", "name": "Dock 1", "rect": [0.0, 0.0, 0.5, 1.0]},
  {"id": "cam-0b71d5aa", "name": "Dock 2", "rect": [0.5, 0.0, 1.0, 1.0]}
]
```

A classification model classifies each camera on its own: the snapshot then
carries `per_camera` (`[{camera, camera_name, candidates}]`) instead of a single
`candidates` list, and the counter counts each camera's class changes.

## Configuration through the API

`POST /api/v1/camera/config` (administrator):

```json
{"source": "ip",
 "ip_cameras": [
   {"id": "cam-3f9a1c2e", "name": "Dock 1", "url": "rtsp://192.0.2.10:554/stream1",
    "username": "operator"},
   {"name": "Dock 2", "url": "rtsp://192.0.2.11/stream1",
    "username": "operator", "password": "example-only"}
 ]}
```

`ip_cameras` is the complete, ordered list (1 to 4 cameras). An entry without
`id` is a new camera; an entry with one must name a stored camera and keeps its
password when `password` is omitted. `password: ""` removes it. Limits: name 40
bytes, address 256 bytes (`rtsp://`, no credentials, no loopback), user name
and password 64 bytes each. See the [HTTP API reference](api-reference.md#camera).

## Security

RTSP is plain text: the camera password is sent with Digest or Basic
authentication and the video is not encrypted. Keep the cameras on a network
segment you control. On the device the passwords are stored owner-only in
`camera_source.json` and never appear in a log or an API response; they do
appear on the command line of the `ffmpeg` processes inside the
`webcam-server` container.

## Not supported

More than four cameras, ONVIF discovery, PTZ control, audio, recording,
per-camera detection areas and per-camera dataset capture (a capture stores
the whole mosaic).
