# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Serialized ``CameraConfig`` size against the SHM config region.

The config region of the camera segment header holds 4 KiB (protocol version
3) and ``write_config_bytes`` drops anything larger, so the message size is part
of the contract. The same figures are pinned on the producer side in
``webcam-server/src/webcam_server/shm.rs``.
"""
import shm_pb2
from conecsa_shm.camera_ring import CONFIG_PAYLOAD_MAX, HEALTH_PAYLOAD_MAX

UINT32_MAX = 2**32 - 1
INT32_MAX = 2**31 - 1

_NUMERIC_FIELDS = (
    "camera_index", "width", "height", "framerate", "exposure_time",
    "rgb_red", "rgb_green", "rgb_blue", "gamma", "gain",
)


def _config(**network) -> shm_pb2.CameraConfig:
    """Fields 1-11 at their wire maximum, plus the given source/network fields."""
    numbers = {name: UINT32_MAX for name in _NUMERIC_FIELDS}
    return shm_pb2.CameraConfig(auto_exposure=True, **numbers, **network)


def test_validated_maxima_take_exactly_119_bytes():
    cfg = _config(
        source=shm_pb2.CAMERA_SOURCE_NETWORK,
        network_host="255.255.255.255",
        network_port=65_535,
        network_token="Z" * 32,
    )
    assert len(cfg.SerializeToString()) == 119


def _four_ip_cameras_at_their_maxima() -> list:
    return [
        shm_pb2.IpCamera(url="rtsp://" + "h" * 249, username="u" * 64, password="p" * 64)
        for _ in range(4)
    ]


def test_the_worst_case_config_fits_the_region_with_room_to_spare():
    # Every number at its wire maximum, the network fields and four IP cameras
    # at their validated maxima: under half of the region.
    cfg = _config(
        source=INT32_MAX,
        network_host="255.255.255.255",
        network_port=UINT32_MAX,
        network_token="Z" * 32,
        ip_cameras=_four_ip_cameras_at_their_maxima(),
    )
    assert len(cfg.SerializeToString()) == 1705
    assert len(cfg.SerializeToString()) <= CONFIG_PAYLOAD_MAX // 2


def test_a_negative_source_is_a_ten_byte_varint_so_writers_validate_the_enum():
    # The size check in write_config_bytes is only a backstop; rejecting
    # unknown values is the rule.
    positive = _config(source=INT32_MAX)
    negative = _config(source=-1)
    assert len(negative.SerializeToString()) == len(positive.SerializeToString()) + 5


def test_a_message_without_source_carries_no_source_on_the_wire():
    # Explicit presence: a writer that only tunes the local camera must be
    # distinguishable from one that selects CAMERA_SOURCE_LOCAL (value 0).
    tuning_only = _config()
    assert not tuning_only.HasField("source")

    local = _config(source=shm_pb2.CAMERA_SOURCE_LOCAL)
    assert local.HasField("source")
    assert len(local.SerializeToString()) == len(tuning_only.SerializeToString()) + 2

    decoded = shm_pb2.CameraConfig.FromString(tuning_only.SerializeToString())
    assert not decoded.HasField("source")


def test_health_with_a_detail_fits_its_region():
    health = shm_pb2.HealthStatus(
        status="no_camera", detail=shm_pb2.CAMERA_HEALTH_DETAIL_BAD_STREAM
    )
    assert len(health.SerializeToString()) <= HEALTH_PAYLOAD_MAX


def test_the_protobuf_free_count_reader_matches_the_stubs():
    # conecsa_shm.mosaic scans the health payload by hand (no protobuf there).
    from conecsa_shm.mosaic import ip_camera_count_of

    assert ip_camera_count_of(None) == 0
    assert ip_camera_count_of(shm_pb2.HealthStatus(status="capturing").SerializeToString()) == 0
    for count in (1, 4, 300):
        health = shm_pb2.HealthStatus(
            status="no_camera", detail=shm_pb2.CAMERA_HEALTH_DETAIL_BAD_STREAM,
            ip_camera_count=count)
        assert ip_camera_count_of(health.SerializeToString()) == count
    assert ip_camera_count_of(b"\x0a\xff") == 0  # truncated
