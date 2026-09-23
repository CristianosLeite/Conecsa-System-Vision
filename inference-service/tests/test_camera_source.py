# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Device-level capture source: validation, persistence and the SHM transaction.

The source (local camera or a remote camera's stream) belongs to the device, not to a
model, and carries a secret token. These tests pin the three properties that
follow: it survives model switches and restarts, a failed update changes
nothing, and the token never leaves through a read path or a log line.
"""
import json
import logging
import os
import stat
import time
from types import SimpleNamespace
from typing import Any

import pytest
import shm_pb2
from api.inference_grpc import ManagementControlServicer
from api.services.camera_source_store import CameraSourceStore
from api.services.config_validation import (
    LOCAL_CAMERA_KEYS,
    ConfigValidationError,
    normalize_network_token,
    validate_camera_patch,
    validate_network_host,
    validate_source_state,
)
from api.services.consumer_service import ConsumerService
from api.services.model_settings_service import ModelSettingsService
from api.services.video_service import VideoService

# Visibly fake stream tokens, as typed (hyphenated, mixed case) and normalized.
RAW_TOKEN = "test-t0ke-n123"
TOKEN = "TESTT0KEN123"
OTHER_TOKEN = "0THERT0KEN99"
HOST = "192.0.2.1"  # RFC 5737 documentation address

NETWORK = {"source": "network", "network_host": HOST, "network_port": 8080,
           "network_token": RAW_TOKEN}


class FakeConsumer:
    """Records what reaches SHM; ``attached=False`` models a missing segment."""

    def __init__(self, attached=True, detail=0, status="capturing"):
        self.attached = attached
        self.written = []
        self._health = SimpleNamespace(status=status, detail=detail)

    def write_config(self, cfg):
        if not self.attached:
            return False
        self.written.append(shm_pb2.CameraConfig.FromString(cfg.SerializeToString()))
        return True

    def read_health(self):
        return self._health

    @property
    def last(self):
        return self.written[-1]


def _codec():
    return SimpleNamespace(get_stereo_config=lambda: {}, set_stereo_config=lambda *a: None)


def _video(tmp_path, consumer=None):
    consumer = consumer or FakeConsumer()
    return VideoService(consumer, _codec(), model_directory=str(tmp_path)), consumer


def _stored(tmp_path):
    with open(tmp_path / CameraSourceStore.FILENAME) as fh:
        return json.load(fh)


# ── validation ───────────────────────────────────────────────────────────────

class TestValidation:
    @pytest.mark.parametrize("raw", ["test-t0ke-n123", "TESTT0KEN123", "TEST-T0KE-N123", "t-e-s-t-t0ken123"])
    def test_token_is_normalized(self, raw):
        assert normalize_network_token(raw) == TOKEN

    @pytest.mark.parametrize("raw", [
        "", "SHORT", "A" * 33, "HASLETTERI123", "HASLETTERL123", "HASLETTERO123",
        "HASLETTERU123", "with space1", "TESTT0KEN12!", None, 12345678,
    ])
    def test_bad_tokens_are_rejected_without_echoing_them(self, raw):
        with pytest.raises(ConfigValidationError) as err:
            normalize_network_token(raw)
        assert "network_token" in str(err.value)
        if isinstance(raw, str) and raw:
            assert raw not in str(err.value)

    @pytest.mark.parametrize("host", ["192.0.2.1", "10.98.76.20", "172.20.10.1", "169.254.7.9", " 192.168.43.1 "])
    def test_direct_link_addresses_are_accepted(self, host):
        assert validate_network_host(host) == host.strip()

    @pytest.mark.parametrize("host", [
        "0.0.0.0", "127.0.0.1", "127.8.8.8", "224.0.0.1", "239.1.2.3", "255.255.255.255",
        "camera.local", "192.0.2", "192.0.2.256", "192.0.2.01", "::1", "", None,
    ])
    def test_unusable_addresses_are_rejected(self, host):
        with pytest.raises(ConfigValidationError, match="network_host"):
            validate_network_host(host)

    @pytest.mark.parametrize("port", [0, 65_536, -1, "http", True])
    def test_port_bounds(self, port):
        with pytest.raises(ConfigValidationError, match="network_port"):
            validate_camera_patch({"network_port": port})

    @pytest.mark.parametrize("source", ["v4l2", "NETWORK", "", None, 1])
    def test_source_is_local_or_network_only(self, source):
        with pytest.raises(ConfigValidationError, match="source"):
            validate_camera_patch({"source": source})

    def test_a_source_patch_is_a_recognised_camera_update(self):
        patch = validate_camera_patch(dict(NETWORK))
        assert patch.source == {**NETWORK, "network_token": TOKEN}
        assert patch.webcam == {}

    def test_a_network_state_must_be_complete(self):
        for missing in ("network_host", "network_port", "network_token"):
            state = {**NETWORK, missing: None}
            with pytest.raises(ConfigValidationError, match="needs network_host"):
                validate_source_state(state)

    def test_a_local_state_may_keep_dormant_or_no_credentials(self):
        assert validate_source_state({"source": "local"}) == {
            "source": "local", "network_host": "", "network_port": 0, "network_token": ""}
        dormant = validate_source_state({**NETWORK, "source": "local"})
        assert dormant["network_token"] == TOKEN

    def test_source_keys_are_not_model_camera_keys(self):
        assert not set(LOCAL_CAMERA_KEYS) & {"source", "network_host", "network_port",
                                             "network_token"}


# ── store ────────────────────────────────────────────────────────────────────

class TestStore:
    def test_round_trip_is_owner_only_and_versioned(self, tmp_path):
        store = CameraSourceStore(str(tmp_path))
        assert store.load() is None
        state = validate_source_state(NETWORK)
        store.save(state)
        assert store.load() == state
        assert _stored(tmp_path)["version"] == 1
        mode = stat.S_IMODE(os.stat(tmp_path / CameraSourceStore.FILENAME).st_mode)
        assert mode == 0o600, "the file holds the token"

    def test_a_corrupt_file_is_quarantined(self, tmp_path):
        path = tmp_path / CameraSourceStore.FILENAME
        path.write_text("{not json")
        assert CameraSourceStore(str(tmp_path)).load() is None
        assert not path.exists()
        assert (tmp_path / (CameraSourceStore.FILENAME + ".corrupt")).exists()

    @pytest.mark.parametrize("payload", [
        {"version": 2, **NETWORK},
        {"version": 1, "source": "network", "network_host": HOST},
        {"version": 1, "source": "rtsp"},
        ["not", "an", "object"],
    ])
    def test_an_invalid_file_is_ignored_and_never_logged_with_its_token(
            self, tmp_path, payload, caplog):
        (tmp_path / CameraSourceStore.FILENAME).write_text(json.dumps(payload))
        with caplog.at_level(logging.DEBUG):
            assert CameraSourceStore(str(tmp_path)).load() is None
        assert RAW_TOKEN not in caplog.text and TOKEN not in caplog.text

    def test_restore_returns_to_the_previous_state_or_to_no_file(self, tmp_path):
        store = CameraSourceStore(str(tmp_path))
        first = validate_source_state(NETWORK)
        store.save(first)
        store.save(validate_source_state({"source": "local"}))
        store.restore(first)
        assert store.load() == first
        store.restore(None)
        assert store.load() is None


# ── the update transaction ───────────────────────────────────────────────────

class TestSourceUpdate:
    def test_switching_to_network_persists_publishes_and_reports(self, tmp_path):
        video, consumer = _video(tmp_path)
        assert video.apply_camera_update(dict(NETWORK)) == (True, "Camera configuration applied", 200)

        assert _stored(tmp_path) == {"version": 1, "source": "network", "network_host": HOST,
                                     "network_port": 8080, "network_token": TOKEN}
        sent = consumer.last
        assert sent.HasField("source") and sent.source == shm_pb2.CAMERA_SOURCE_NETWORK
        assert (sent.network_host, sent.network_port, sent.network_token) == (HOST, 8080, TOKEN)
        assert len(sent.SerializeToString()) <= 128

        info = video.list_camera_devices()
        assert info["current_source"] == "network"
        assert (info["current_network_host"], info["current_network_port"]) == (HOST, 8080)
        assert info["network_token_set"] is True

    def test_an_omitted_token_keeps_the_stored_one(self, tmp_path):
        video, consumer = _video(tmp_path)
        video.apply_camera_update(dict(NETWORK))
        ok, _, _ = video.apply_camera_update({"network_host": "192.0.2.7"})
        assert ok
        assert consumer.last.network_host == "192.0.2.7"
        assert consumer.last.network_token == TOKEN
        assert _stored(tmp_path)["network_token"] == TOKEN

    def test_an_empty_token_is_invalid_not_a_clear(self, tmp_path):
        video, consumer = _video(tmp_path)
        video.apply_camera_update(dict(NETWORK))
        ok, message, status = video.apply_camera_update({"network_token": ""})
        assert (ok, status) == (False, 400) and "network_token" in message
        assert _stored(tmp_path)["network_token"] == TOKEN
        assert len(consumer.written) == 1

    def test_switching_to_network_needs_a_complete_state(self, tmp_path):
        video, consumer = _video(tmp_path)
        ok, message, status = video.apply_camera_update(
            {"source": "network", "network_host": HOST, "network_port": 8080})
        assert (ok, status) == (False, 400) and "needs network_host" in message
        assert not (tmp_path / CameraSourceStore.FILENAME).exists()
        assert consumer.written == []

    def test_switching_to_local_keeps_dormant_credentials(self, tmp_path):
        video, consumer = _video(tmp_path)
        video.apply_camera_update(dict(NETWORK))
        assert video.apply_camera_update({"source": "local"})[0]
        assert consumer.last.source == shm_pb2.CAMERA_SOURCE_LOCAL
        assert consumer.last.network_token == TOKEN
        # ... so switching back needs no re-entry.
        assert video.apply_camera_update({"source": "network"})[0]
        assert consumer.last.source == shm_pb2.CAMERA_SOURCE_NETWORK

    def test_a_failed_publish_restores_the_file_and_leaves_memory_alone(self, tmp_path):
        video, consumer = _video(tmp_path)
        video.apply_camera_update(dict(NETWORK))
        consumer.attached = False

        ok, _, status = video.apply_camera_update(
            {"network_host": "192.0.2.9", "network_token": OTHER_TOKEN})
        assert (ok, status) == (False, 503)
        assert _stored(tmp_path)["network_host"] == HOST
        assert _stored(tmp_path)["network_token"] == TOKEN
        assert video.list_camera_devices()["current_network_host"] == HOST

    def test_a_failed_first_publish_leaves_no_file_behind(self, tmp_path):
        video, _ = _video(tmp_path, FakeConsumer(attached=False))
        assert video.apply_camera_update(dict(NETWORK))[2] == 503
        assert not (tmp_path / CameraSourceStore.FILENAME).exists()
        assert video.list_camera_devices()["current_source"] == "local"

    def test_a_failed_write_to_disk_publishes_nothing(self, tmp_path, monkeypatch):
        video, consumer = _video(tmp_path)

        def refuse(*_args, **_kwargs):
            raise OSError("disk full")
        monkeypatch.setattr("api.services.camera_source_store.atomic_write_json", refuse)

        ok, message, status = video.apply_camera_update(dict(NETWORK))
        assert (ok, status) == (False, 500)
        assert TOKEN not in message and RAW_TOKEN not in message
        assert consumer.written == []
        assert video.list_camera_devices()["current_source"] == "local"

    def test_source_and_tuning_in_one_body_go_out_as_one_message(self, tmp_path):
        video, consumer = _video(tmp_path)
        assert video.apply_camera_update({**NETWORK, "gain": 33})[0]
        assert len(consumer.written) == 1
        assert consumer.last.gain == 33 and consumer.last.network_host == HOST


# ── device scope: models, restarts, unconfigured devices ─────────────────────

class TestDeviceScope:
    def test_until_a_source_is_saved_messages_carry_none(self, tmp_path):
        # The webcam-server reads "no source" as "leave it alone", which keeps
        # its CAMERA_SOURCE bootstrap env in charge on an unconfigured device.
        video, consumer = _video(tmp_path)
        assert video.apply_camera_update({"gain": 5})[0]
        assert not consumer.last.HasField("source")
        video.publish_startup_source()
        assert len(consumer.written) == 1, "nothing to restore at boot"

    def test_tuning_updates_carry_the_saved_source(self, tmp_path):
        video, consumer = _video(tmp_path)
        video.apply_camera_update(dict(NETWORK))
        video.apply_camera_update({"gain": 9})
        assert consumer.last.gain == 9
        assert consumer.last.source == shm_pb2.CAMERA_SOURCE_NETWORK
        assert consumer.last.network_token == TOKEN

    def test_the_source_is_restored_at_boot_without_any_model(self, tmp_path):
        _video(tmp_path)[0].apply_camera_update(dict(NETWORK))

        video, consumer = _video(tmp_path)  # a restarted inference-service
        assert consumer.written == []
        video.publish_startup_source()
        assert consumer.last.source == shm_pb2.CAMERA_SOURCE_NETWORK
        assert consumer.last.network_token == TOKEN

    def test_model_settings_can_neither_hold_nor_select_a_source(self, tmp_path):
        video, consumer = _video(tmp_path)
        video.apply_camera_update(dict(NETWORK))

        assert set(video.get_model_camera_config()) == set(LOCAL_CAMERA_KEYS)

        config: Any = SimpleNamespace(CONFIDENCE_THRESHOLD=0.5, OVERLAY_THRESHOLD=0.5)
        settings = ModelSettingsService(config, video)
        path = str(tmp_path / "weights.settings.json")
        settings.switch_model(path)
        settings.save()
        text = open(path).read()
        assert TOKEN not in text and "network" not in text and "source" not in text

        # A hand-edited model file trying to switch the device to local:
        with open(path) as fh:
            data = json.load(fh)
        data["camera"].update({"source": "local", "network_host": "192.0.2.66", "gain": 21})
        with open(path, "w") as fh:
            json.dump(data, fh)
        settings.switch_model(path)
        assert consumer.last.gain == 21, "the tuning applies"
        assert consumer.last.source == shm_pb2.CAMERA_SOURCE_NETWORK
        assert consumer.last.network_host == HOST

    def test_a_recreated_segment_gets_the_config_again(self, tmp_path):
        video, consumer = _video(tmp_path, FakeConsumer(attached=False))
        video.republish()
        assert consumer.written == [], "nothing was ever meant to be published"

        _video(tmp_path)[0].apply_camera_update(dict(NETWORK))
        video, consumer = _video(tmp_path, FakeConsumer(attached=False))
        video.publish_startup_source()  # webcam-server not up yet: lost
        assert consumer.written == []
        consumer.attached = True
        video.republish()  # what ConsumerService runs when the segment appears
        assert consumer.last.source == shm_pb2.CAMERA_SOURCE_NETWORK


class TestAttachHook:
    def _consumer(self, generation) -> Any:
        consumer: Any = object.__new__(ConsumerService)  # no reader thread, no SHM
        consumer._shm = SimpleNamespace(attach_generation=generation)
        consumer._on_attach = None
        consumer._attach_seen = 0
        return consumer

    def test_the_hook_runs_once_per_mapping_including_an_earlier_one(self):
        consumer = self._consumer(generation=1)
        consumer._notify_attach()  # mapped before anyone registered: remembered
        calls = []
        consumer.set_on_attach(lambda: calls.append(1))
        consumer._notify_attach()
        consumer._notify_attach()
        assert calls == [1]
        consumer._shm.attach_generation = 2  # the producer restarted
        consumer._notify_attach()
        assert calls == [1, 1]

    def test_a_failing_hook_does_not_kill_the_reader(self):
        def explode() -> None:
            raise RuntimeError("hook failed")

        consumer = self._consumer(generation=1)
        consumer.set_on_attach(explode)
        consumer._notify_attach()


# ── health detail and secrecy ────────────────────────────────────────────────

class TestReporting:
    @pytest.mark.parametrize("number,text", [
        (0, "unspecified"), (1, "connecting"), (2, "unauthorized"), (3, "rate_limited"),
        (4, "unreachable"), (5, "stalled"), (6, "bad_stream"), (99, "unspecified"),
    ])
    def test_health_detail_maps_to_stable_text(self, tmp_path, number, text):
        video, _ = _video(tmp_path, FakeConsumer(detail=number, status="no_camera"))
        assert video.camera_detail() == text
        assert video.list_camera_devices()["camera_detail"] == text

    def test_every_proto_detail_has_text(self, tmp_path):
        for value in shm_pb2.CameraHealthDetail.DESCRIPTOR.values:
            video, _ = _video(tmp_path, FakeConsumer(detail=value.number))
            expected = value.name.replace("CAMERA_HEALTH_DETAIL_", "").lower()
            assert video.camera_detail() == expected

    def test_a_health_without_detail_reads_as_unspecified(self, tmp_path):
        consumer = FakeConsumer()
        consumer._health = SimpleNamespace(status="capturing")  # an older producer
        assert _video(tmp_path, consumer)[0].camera_detail() == "unspecified"

    def test_the_token_never_leaves_through_reads_errors_or_logs(self, tmp_path, caplog):
        with caplog.at_level(logging.DEBUG):
            video, consumer = _video(tmp_path)
            results = [
                video.apply_camera_update(dict(NETWORK)),
                video.apply_camera_update({"network_token": "bad token !!"}),
                video.apply_camera_update({"network_host": "127.0.0.1"}),
            ]
            consumer.attached = False
            results.append(video.apply_camera_update({"network_token": OTHER_TOKEN}))
            video.publish_startup_source()
            video.republish()
            visible = json.dumps([results, video.list_camera_devices(),
                                  video.get_current_camera_config(),
                                  video.get_model_camera_config(),
                                  video.get_webcam_server_config()])
        for secret in (RAW_TOKEN, TOKEN, OTHER_TOKEN, "bad token"):
            assert secret not in visible
            assert secret not in caplog.text
        assert "network_token" not in json.dumps(video.list_camera_devices()).replace(
            "network_token_set", "")


# ── gRPC servicer ────────────────────────────────────────────────────────────

class _Settings:
    def __init__(self):
        self.saves = 0

    def save(self):
        self.saves += 1


def _servicer(tmp_path):
    video, consumer = _video(tmp_path)
    settings = _Settings()
    app = SimpleNamespace(video_service=video, model_settings_service=settings)
    return ManagementControlServicer(app), settings, consumer


class TestHealthEvents:
    """The screen follows the camera through events, not by polling the device list."""

    class Events:
        def __init__(self):
            self.published: list = []

        def publish(self, event_type, keys, source, data):
            self.published.append((event_type, keys, source, data))

    def _watching(self, tmp_path, consumer):
        video, _ = _video(tmp_path, consumer)
        events = self.Events()
        video._health_events = events
        video._health_seen = None
        return video, events

    def test_only_a_change_is_published(self, tmp_path):
        consumer = FakeConsumer(status="capturing")
        video, events = self._watching(tmp_path, consumer)
        assert video._health_tick() is True
        assert video._health_tick() is False
        consumer._health = SimpleNamespace(status="no_camera", detail=shm_pb2.CAMERA_HEALTH_DETAIL_UNREACHABLE)
        assert video._health_tick() is True
        assert [e[3] for e in events.published] == [
            {"status": "capturing", "detail": "unspecified", "source": "local"},
            {"status": "no_camera", "detail": "unreachable", "source": "local"},
        ]
        assert events.published[0][:3] == ("camera_health_changed", ["camera_health"], "camera")

    def test_the_event_names_the_applied_source_and_never_the_token(self, tmp_path):
        video, events = self._watching(tmp_path, FakeConsumer(status="no_camera", detail=shm_pb2.CAMERA_HEALTH_DETAIL_UNAUTHORIZED))
        video.apply_camera_update({"source": "network", "network_host": "192.0.2.1", "network_port": 8080,
                                   "network_token": "TESTT0KEN123"})
        video._health_tick()
        (_, _, _, data), = events.published
        assert data == {"status": "no_camera", "detail": "unauthorized", "source": "network"}
        assert "TESTT0KEN123" not in json.dumps(events.published)

    def test_the_watch_thread_publishes_the_first_sample(self, tmp_path):
        video, _ = _video(tmp_path, FakeConsumer())
        events = self.Events()
        video.start_health_watch(events, interval=0.01)
        deadline = time.monotonic() + 2
        while not events.published and time.monotonic() < deadline:
            time.sleep(0.01)
        assert events.published and events.published[0][0] == "camera_health_changed"


class TestServicer:
    def test_a_source_update_does_not_touch_the_model_settings(self, tmp_path):
        servicer, settings, _ = _servicer(tmp_path)
        result = servicer.UpdateCamera(SimpleNamespace(json=json.dumps(NETWORK)), None)
        assert result.success
        assert settings.saves == 0
        assert TOKEN not in result.message and RAW_TOKEN not in result.message

    def test_a_tuning_update_still_persists_to_the_model(self, tmp_path):
        servicer, settings, _ = _servicer(tmp_path)
        servicer.UpdateCamera(SimpleNamespace(json=json.dumps({"gain": 4})), None)
        servicer.UpdateCamera(SimpleNamespace(json=json.dumps({**NETWORK, "gain": 5})), None)
        assert settings.saves == 2

    def test_get_camera_never_returns_the_token(self, tmp_path):
        servicer, _, _ = _servicer(tmp_path)
        servicer.UpdateCamera(SimpleNamespace(json=json.dumps(NETWORK)), None)
        body = servicer.GetCamera(None, None).json
        assert TOKEN not in body and RAW_TOKEN not in body
        assert json.loads(body)["network_token_set"] is True
