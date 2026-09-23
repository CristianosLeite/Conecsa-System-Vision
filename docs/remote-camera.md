# Remote camera

A device can inspect with a remote camera instead of a camera plugged into it,
with no router involved. The remote camera is a phone running the Conecsa camera
app, which serves the camera as a video stream; the device **pulls** that stream
over a direct Wi-Fi link and feeds it into the same pipeline as a local camera.
Detection, classification, segmentation, training previews and the live video
all work unchanged.

The device opens no new port for this, and the remote camera never connects to
the device. The wire format is specified in the
[remote camera protocol](services/remote-camera-protocol.md).

## Connecting

1. **Create the link.** Turn on the remote camera's hotspot and join it from the
   device screen's network settings, like any other Wi-Fi network. The single
   radio is then taken by the hotspot, so the device should reach the hub over
   Ethernet.
2. **Start the app** on the remote camera. It shows its address, port and an
   `XXXX-XXXX` token.
3. **Select the source.** On the device screen open **Camera Settings**, choose
   **Remote camera**, and enter the address, port and token. The token field
   formats what is typed as upper-case groups of four, so hyphens and case do
   not matter; a letter the alphabet excludes (I, L, O, U) stays visible and
   is refused on Apply rather than silently corrected. On a hotspot the
   remote camera is the Wi-Fi gateway, so the address is suggested — check it
   against the app before applying. Apply: the capture reconnects by itself.
   While the device's own access point is off, Apply first asks whether to
   turn it on (see below); **Apply without access point** is the hotspot case.
   When the access point's state is unknown (the hardware agent unreachable)
   Apply does not ask and applies straight away.

Switching back to **Local camera** is the same control. The remote camera's
address and token stay stored on the device, so returning to it needs no
re-entry: the stored token sits behind a **Change token** link, and a new one
is typed only after pressing it.

Only an administrator can change the source.

## What changes on the device screen

The capture device, resolution, frame rate, 3D stereo overlay and the image
adjustments (exposure, RGB, gamma, gain) are V4L2 camera controls. They mean
nothing to a remote camera, so they are hidden while it is the source; lock focus
and exposure in the app instead. Resolution is whatever the remote camera sends —
the pipeline reads it from each frame.

**Camera Status** shows whether video is arriving and, when it is not, why:

| Shown | Meaning | What to do |
|---|---|---|
| Connecting to the remote camera | First connection attempt in progress | Wait a few seconds |
| The remote camera rejected the token | Wrong or regenerated token | Re-enter the token the app shows |
| Too many failed attempts | The app locks authentication for 30 s after five bad tokens | Fix the token, wait, it retries by itself |
| The remote camera cannot be reached | Wrong address/port, app closed, or the link is down | Check the hotspot and the address in the app |
| The remote camera stopped sending video | App in the background, screen-off restrictions, or a weak link | Bring the app forward; move the remote camera closer |
| The remote camera sent an invalid video stream | Not the Conecsa app, or an incompatible version | Check the port and the app version |

The device retries on its own — 1 s, 2 s, then every 5 s — and recovers without
any restart when the remote camera comes back. It **never** switches to the local
camera by itself: a silent fallback would leave the operator unsure which camera
is being inspected. While the remote camera is not delivering video, detection
stays gated exactly as it does for an unplugged camera.

## Using the device as the access point

When the remote camera's hotspot is not an option, the device can serve the
Wi-Fi network itself. The network is named after the device. The shortest path
is from **Camera Settings**: choose **Remote camera** and press Apply while the
access point is off — a confirmation shows the warning below, asks for a
passphrase and a channel (the list holds Automatic plus the channels the radio
may start on right now; leave it on Automatic), and **Turn on and apply**
starts the access point. Join it from the remote camera. While **Remote
camera** is selected the form follows the access point every 3 s and, once
exactly one remote camera has joined and holds a lease, suggests its address
into an empty address field (or one still showing a suggestion; an address
the operator typed is never replaced), so the sequence is turn on, join,
check the suggested address against the app, apply. If the address, port and
token were already in the form, the source is applied right after the access
point starts. The same access point can also be managed from
**Settings → Network → Access point**, which lists the joined remote cameras
with their addresses and turns it off. A start that ends with "cannot start an
access point right now" names the channels that can; the radio's regulatory
flags move with the Wi-Fi networks it has heard, which is why Automatic is the
default.

In the remote camera app, choose **Device access point** as the connection,
enter the network name (the device id) and the passphrase, and press **Join and
start**. Android asks once whether the app may connect to that network; while
the phone is joined it is off its usual Wi-Fi network (mobile data is not
affected). The app then shows the address it got from the device, the same one
the Access point panel lists. If the device turns the access point off, or the
phone loses it, the app stops streaming and says so rather than serving on
another network; **Stop and leave** returns the phone to its own Wi-Fi.

The device has one radio, so while the access point is on it is **off its
Wi-Fi network** and the hub reaches it only over the wired port; the button is
disabled until the wired port has a link and an IPv4 address (a link-local one
counts). The access point is never persisted: it
turns itself off if no remote camera joins within five minutes, on an agent
restart and on a reboot, and a failed start rolls the radio back to station mode.
The address range is `AP_ADDRESS_CIDR` (see [configuration](configuration.md));
the mechanics are in the [os-base hardware agent](services/os-hardware-agent.md#wi-fi-access-point)
page.

## Where the source is stored

The capture source belongs to the **device**, not to a model. It is kept in
`camera_source.json` in the models volume, applied at start-up even with no model
selected, and left alone by a model switch. It never appears in a model's
`*.settings.json`.

The token is write-only. It is stored in that file (owner-readable only) and sent
to the remote camera, and nowhere else: no API response, event, audit entry or
log line carries it. `GET /api/v1/camera/devices` reports only
`network_token_set`. A request that omits `network_token` keeps the stored token;
an empty one is rejected.

If the webcam-server restarts on its own, the inference-service publishes the
source to it again, so the device does not quietly return to the local camera.

## Security

The stream is plain HTTP, which is acceptable **only** on the direct WPA2 link
described here: the Wi-Fi layer encrypts it and nobody else is on the network.
The token stops another client from pulling the stream; it is not encryption. Do
not run the remote camera across a shared LAN.

## Trying it without a remote camera

`webcam-server/tools/mjpeg_test_sender.py` stands in for the app (see the
[protocol page](services/remote-camera-protocol.md#testing-without-a-remote-camera)).
Run it on a computer on the device's network, bound to that computer's **LAN
address**, and enter that address on the device screen. `127.0.0.1` does not work, by design: the API rejects a loopback camera
address, and inside the compose stack loopback is the webcam-server container
itself.

## Not supported

Audio, recording, remote control of the remote camera from the device, more than
one remote camera at a time, discovery, iOS, streaming over the Internet, and
H.264/RTSP/WebRTC.
