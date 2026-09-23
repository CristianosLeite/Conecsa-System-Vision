# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Serialized ``CameraConfig`` size against the SHM config region.

The config region of the camera segment header holds 128 bytes and
``write_config_bytes`` drops anything larger, so the message size is part of the
contract. The same figures are pinned on the producer side in
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


def test_every_number_at_uint32_max_still_fits_the_region():
    # The worst case leaves 3 bytes and tag 16 onward costs a two-byte tag, so
    # the next CameraConfig field needs a larger region: a layout change and an
    # SHM_VERSION bump.
    cfg = _config(
        source=INT32_MAX,
        network_host="255.255.255.255",
        network_port=UINT32_MAX,
        network_token="Z" * 32,
    )
    assert len(cfg.SerializeToString()) == 125
    assert len(cfg.SerializeToString()) <= CONFIG_PAYLOAD_MAX


def test_a_negative_source_does_not_fit_so_writers_validate_the_enum():
    # A negative enum value is a 10-byte varint. The size check in
    # write_config_bytes is the backstop; rejecting unknown values is the rule.
    cfg = _config(
        source=-1,
        network_host="255.255.255.255",
        network_port=UINT32_MAX,
        network_token="Z" * 32,
    )
    assert len(cfg.SerializeToString()) == 130
    assert len(cfg.SerializeToString()) > CONFIG_PAYLOAD_MAX


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
