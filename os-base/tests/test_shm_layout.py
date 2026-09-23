# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Pins the SHM header layouts so the lockstep rule is not prose only.

The camera ring is written by Rust (``webcam-server/src/webcam_server/shm.rs``)
and read by Python (``conecsa_shm.camera_ring``); the constants are duplicated
by hand, so this test parses the Rust source and compares both tables. The
processed ring is Python on both ends, so only its internal consistency is
checked.
"""
import re
from pathlib import Path

import pytest
from conecsa_shm import camera_ring, processed_ring

SHM_RS = (
    Path(__file__).resolve().parents[2] / "webcam-server" / "src" / "webcam_server" / "shm.rs"
)

# Field widths in bytes; every other header field is a u32.
CAMERA_WIDTHS = {
    "FRAME_WRITE_SEQ": 8,
    "CONFIG_PAYLOAD": camera_ring.CONFIG_PAYLOAD_MAX,
    "HEALTH_PAYLOAD": camera_ring.HEALTH_PAYLOAD_MAX,
}
PROCESSED_WIDTHS = {"WRITE_SEQ": 8}
U64_FIELDS = ("FRAME_WRITE_SEQ", "WRITE_SEQ")


def _int(literal: str) -> int:
    return int(literal.replace("_", ""), 0)


@pytest.fixture(scope="module")
def rust_src() -> str:
    if not SHM_RS.is_file():
        pytest.skip("webcam-server source not present")
    return SHM_RS.read_text()


def _rust_offsets(src: str) -> dict:
    body = re.search(r"^mod off \{(.*?)^\}", src, re.S | re.M)
    assert body, "mod off not found in shm.rs"
    return {n: int(v) for n, v in re.findall(r"pub const (\w+): usize = (\d+);", body.group(1))}


def _rust_const(src: str, name: str) -> int:
    m = re.search(rf"^(?:pub )?const {name}: \w+ = ([0-9A-Fa-fx_]+);", src, re.M)
    assert m, f"{name} not found in shm.rs"
    return _int(m.group(1))


def _python_offsets(module, prefix: str) -> dict:
    return {n[len(prefix):]: getattr(module, n) for n in dir(module) if n.startswith(prefix)}


def _assert_packed(offsets: dict, widths: dict, header_size: int) -> None:
    regions = sorted((off, off + widths.get(name, 4), name) for name, off in offsets.items())
    for (_, end, name), (start, _, nxt) in zip(regions, regions[1:], strict=False):
        assert end <= start, f"{name} overlaps {nxt}"
    assert regions[-1][1] <= header_size, f"{regions[-1][2]} runs past the header"
    for name in U64_FIELDS:
        if name in offsets:
            assert offsets[name] % 8 == 0, f"u64 {name} is not 8-byte aligned"


class TestCameraRingMatchesRust:
    def test_offsets_match_by_name(self, rust_src):
        assert _python_offsets(camera_ring, "OFF_") == _rust_offsets(rust_src)

    @pytest.mark.parametrize(
        "name",
        [
            "SHM_MAGIC",
            "SHM_VERSION",
            "HEADER_SIZE",
            "CONFIG_PAYLOAD_MAX",
            "HEALTH_PAYLOAD_MAX",
            "FORMAT_RAW_RGB",
            "FORMAT_JPEG",
        ],
    )
    def test_constant_matches(self, rust_src, name):
        assert getattr(camera_ring, name) == _rust_const(rust_src, name)

    def test_header_fields_do_not_overlap(self):
        _assert_packed(_python_offsets(camera_ring, "OFF_"), CAMERA_WIDTHS, camera_ring.HEADER_SIZE)


class TestProcessedRingLayout:
    def test_header_fields_do_not_overlap(self):
        _assert_packed(
            _python_offsets(processed_ring, "_OFF_"), PROCESSED_WIDTHS, processed_ring.HEADER_SIZE
        )

    def test_magic_differs_from_the_camera_ring(self):
        assert processed_ring.MAGIC != camera_ring.SHM_MAGIC

    def test_protocol_version_moves_with_the_camera_ring(self):
        assert processed_ring.VERSION == camera_ring.SHM_VERSION
