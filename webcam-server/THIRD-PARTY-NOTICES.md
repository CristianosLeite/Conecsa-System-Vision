# Third-party notices: webcam-server image

Hand-maintained; update it when `Dockerfile.webcam-server` installs or drops a
runtime package. The Rust crates linked into `webcam_server` are all under
permissive licenses (MIT, Apache-2.0, BSD, IJG); `cargo metadata` on
`webcam-server/Cargo.lock` lists them with their licenses.

## Programs run as subprocesses (not linked)

| Package | License | Source |
|---|---|---|
| ffmpeg (Debian bookworm, `7:5.1.x`, built with `--enable-gpl`) | GPL-2.0-or-later | https://ffmpeg.org/ — Debian source package `ffmpeg`, full copyright file at `/usr/share/doc/ffmpeg/copyright` inside the image |

`webcam_server` starts `ffmpeg` (and `ffprobe`) as separate processes to decode
the RTSP IP camera source and reads their raw frames over a pipe. Nothing from
FFmpeg is linked into the AGPL-3.0-only `webcam_server` binary, so the two
programs keep their own licenses. The Debian build enables the GPL components,
so the `ffmpeg` binary as shipped is GPL-2.0-or-later; its corresponding source
is Debian's `ffmpeg` source package for the installed version.

## Debian runtime libraries

`libssl3` (Apache-2.0), `libv4l-0` (LGPL-2.1-or-later), `v4l-utils`
(GPL-2.0-or-later, command-line tools only) and `ca-certificates` (MPL-2.0 /
GPL-2.0-or-later) come from Debian bookworm, each with its copyright file under
`/usr/share/doc/<package>/` in the image.
