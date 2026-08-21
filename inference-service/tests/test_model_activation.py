"""Tests for the transactional model activation (REFACTORING.md H5).

A failed activation used to leave detection stopped, persist the broken model
as the boot default, and keep the new model's areas/settings stores switched
in. It must now restore the previous model, its stores, and its running state,
and persist nothing.
"""
import os
import threading

import pytest
from api.config import Config
from api.services.model_service import ModelService


class FakeDetectionService:
    """Scriptable detector: initialize() fails while `poison` is set."""

    def __init__(self, running=False):
        self.running = running
        self.poison = False
        self.initialized_with = []

    def stop(self):
        was = self.running
        self.running = False
        return was

    def start(self):
        self.running = True

    def initialize(self):
        if self.poison:
            raise RuntimeError("engine deserialization failed")
        self.initialized_with.append(True)


class FakeStore:
    def __init__(self):
        self.paths = []

    def switch(self, path):
        self.paths.append(path)


@pytest.fixture
def rig(tmp_path):
    config = Config()
    service = ModelService(config, str(tmp_path))
    detector = FakeDetectionService()
    areas, settings = FakeStore(), FakeStore()
    service.attach_detection_service(detector)
    service.attach_area_service(type("A", (), {"switch_storage": staticmethod(areas.switch)}))
    service.attach_settings_service(type("S", (), {"switch_model": staticmethod(settings.switch)}))
    for name in ("old.engine", "new.engine"):
        (tmp_path / name).write_bytes(b"x")
    ok, _, was = service.activate_model("old.engine")
    assert ok
    return service, detector, areas, settings, str(tmp_path)


class TestTransactionalActivation:
    def test_success_persists_exactly_the_activated_model(self, rig):
        service, detector, *_ , model_dir = rig
        ok, path, _ = service.activate_model("new.engine")
        assert ok
        assert service.load_persisted_current_model() == "new.engine"
        assert service.current_model == "new.engine"

    def test_failure_restores_model_stores_and_running_state(self, rig):
        service, detector, areas, settings, model_dir = rig
        detector.running = True
        detector.poison = True

        ok, error, was_running = service.activate_model("new.engine")

        assert not ok and "Failed to load model" in error
        assert was_running
        # State points back at the old model…
        assert service.current_model == "old.engine"
        assert service.config.MODEL_PATH == os.path.join(model_dir, "old.engine")
        assert service.config.CLASSES_FILE_PATH.endswith("old.txt")
        # …the scoped stores were re-pointed…
        assert areas.paths[-1].endswith("old.areas.json")
        assert settings.paths[-1].endswith("old.settings.json")
        # …and the boot default was never touched.
        assert service.load_persisted_current_model() == "old.engine"

    def test_failure_restarts_the_previous_runtime(self, rig):
        service, detector, *_ = rig
        detector.running = True

        real_initialize = detector.initialized_with
        calls = {"n": 0}

        def initialize_once_broken():
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("bad engine")
            real_initialize.append(True)

        detector.initialize = initialize_once_broken
        ok, _, _ = service.activate_model("new.engine")
        assert not ok
        # The rollback re-initialized the old model and restarted the loop.
        assert calls["n"] == 2
        assert detector.running is True

    def test_a_failed_selection_restarts_the_untouched_runtime(self, rig):
        service, detector, *_ = rig
        detector.running = True
        ok, error, was_running = service.activate_model("missing.engine")
        assert not ok and "not found" in error
        assert was_running
        assert detector.running is True
        assert service.current_model == "old.engine"

    def test_lifecycle_operations_are_serialized(self, rig):
        # A delete may never interleave with an activation of the same file:
        # with the op lock held by a slow activation, the delete must wait.
        service, detector, *_ = rig
        release = threading.Event()
        entered = threading.Event()

        original_initialize = detector.initialize

        def slow_initialize():
            entered.set()
            assert release.wait(5)
            original_initialize()

        detector.initialize = slow_initialize
        results = {}

        def activate():
            results["activate"] = service.activate_model("new.engine")

        def delete():
            entered.wait(5)
            results["delete"] = service.delete_model("new.engine")

        threads = [threading.Thread(target=activate), threading.Thread(target=delete)]
        for t in threads:
            t.start()

        # The activation is inside initialize(); the delete must be parked on
        # the op lock, not deleting the file out from under it.
        assert entered.wait(5)
        import time
        time.sleep(0.2)
        assert "delete" not in results, "delete must wait for the activation"
        release.set()
        for t in threads:
            t.join(10)

        ok, _, _ = results["activate"]
        assert ok, "activation must not lose its file to the delete"
        # By the time the delete ran, new.engine had become the active model.
        deleted_ok, message = results["delete"]
        assert not deleted_ok
        assert "active" in message
