# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""The application switch transaction and its concurrency contract.

``ModelService.set_task`` must serialize with model activation and with the
GPU handover (``release_runtime``) under the model lifecycle lock, apply its
steps in order, and leave the previous state intact when stopping detection
or writing ``application.json`` fails.
"""
import json
import threading
import time
from types import SimpleNamespace

import pytest
from api.config import Config
from api.services.application_service import ApplicationService
from api.services.errors import InvalidTask, PreconditionFailed
from api.services.model_service import ModelService


class FakeDetection:
    def __init__(self):
        self.is_running = False
        self.calls = []
        self.fail_stop = None
        self.stop_gate = None
        self.start_gate = None

    def stop(self):
        self.calls.append("stop")
        if self.stop_gate is not None:
            self.stop_gate()
        if self.fail_stop is not None:
            raise self.fail_stop
        was, self.is_running = self.is_running, False
        return was

    def start(self):
        self.calls.append("start")
        if self.start_gate is not None:
            self.start_gate()
        self.is_running = True

    def initialize(self):
        self.calls.append("initialize")

    def unload_runtime(self):
        self.calls.append("unload_runtime")

    def reset_results(self):
        self.calls.append("reset_results")


class FakeLabeling:
    def __init__(self, task=None):
        self.task = task
        self.unloads = 0

    def loaded_task(self):
        return self.task

    def unload(self):
        self.unloads += 1
        self.task = None


class FakeEvents:
    def __init__(self):
        self.events = []

    def publish(self, event_type, keys=None, source=None, data=None):
        self.events.append((event_type, keys, data))


def _model(directory, name, task):
    (directory / name).write_bytes(b"engine")
    stem = name.rsplit(".", 1)[0]
    (directory / f"{stem}.settings.json").write_text(json.dumps({"task": task}))


@pytest.fixture
def rig(tmp_path, monkeypatch):
    import api.runtime_management.worker_client as wc
    monkeypatch.setattr(wc, "release_all_workers", lambda: None)
    _model(tmp_path, "det.engine", "detect")
    app = ApplicationService(str(tmp_path), ["detect", "classify"])
    app.load_or_migrate()  # an existing installation → detect (migrated)
    service = ModelService(Config(), str(tmp_path))
    detection, labeling, events = FakeDetection(), FakeLabeling(), FakeEvents()
    conversions = SimpleNamespace(jobs=[])
    conversions.get_active_jobs = lambda: conversions.jobs
    areas, settings = [], []
    service.attach_application_service(app)
    service.attach_detection_service(detection)
    service.attach_labeling_service(labeling)
    service.attach_event_service(events)
    service.attach_conversion_service(conversions)
    service.attach_area_service(SimpleNamespace(switch_storage=areas.append))
    service.attach_settings_service(SimpleNamespace(switch_model=settings.append))
    ok, _, _ = service.activate_model("det.engine")
    assert ok
    detection.is_running = True
    detection.calls.clear()
    return SimpleNamespace(service=service, app=app, detection=detection, labeling=labeling,
                           events=events, conversions=conversions, areas=areas,
                           settings=settings, dir=tmp_path)


def _application_file(rig):
    return json.loads((rig.dir / "application.json").read_text())


class TestSwitch:
    def test_same_task_is_a_no_op(self, rig):
        assert rig.service.set_task("detect") is False
        assert rig.detection.calls == []
        assert rig.events.events == []
        assert _application_file(rig) == {"task": "detect", "migrated": True}

    def test_switch_deselects_a_model_of_the_previous_task(self, rig):
        assert rig.service.set_task("classify") is True
        # Ordered steps: stop, persist, deselect, drop runtime/results, notify.
        assert rig.detection.calls == ["stop", "unload_runtime", "reset_results"]
        assert _application_file(rig) == {"task": "classify"}
        assert rig.app.migrated is False
        assert rig.service.current_model == ""
        assert rig.service.config.MODEL_PATH == rig.service.config.DEFAULT_MODEL_PATH
        assert not (rig.dir / ".current_model").exists()
        assert rig.areas[-1].endswith("weights.areas.json")
        assert rig.settings[-1].endswith("weights.settings.json")
        # The model's own files belong to the model and are kept.
        assert (rig.dir / "det.engine").exists()
        assert json.loads((rig.dir / "det.settings.json").read_text()) == {"task": "detect"}
        assert rig.events.events == [
            ("application_changed", ["application", "models", "status"],
             {"task": "classify"})]
        # Detection is never restarted by a switch.
        assert rig.detection.is_running is False

    def test_list_models_reports_every_model_with_its_task(self, rig):
        _model(rig.dir, "cls.engine", "classify")
        tasks = {m.name: m.task for m in rig.service.list_models()}
        assert tasks == {"det.engine": "detect", "cls.engine": "classify"}

    def test_unknown_and_unsupported_tasks_change_nothing(self, rig):
        with pytest.raises(InvalidTask):
            rig.service.set_task("pose")
        with pytest.raises(PreconditionFailed, match="not available"):
            rig.service.set_task("segment")
        assert rig.detection.calls == []
        assert rig.app.task == "detect"

    def test_a_labeling_engine_of_another_task_is_unloaded(self, rig):
        rig.labeling.task = "detect"
        rig.service.set_task("classify")
        assert rig.labeling.unloads == 1

    def test_a_labeling_engine_of_the_new_task_stays(self, rig):
        rig.labeling.task = "classify"
        rig.service.set_task("classify")
        assert rig.labeling.unloads == 0


class TestFailures:
    def test_a_stop_failure_leaves_everything_as_it_was(self, rig):
        rig.detection.fail_stop = RuntimeError("pipeline wedged")
        with pytest.raises(RuntimeError, match="wedged"):
            rig.service.set_task("classify")
        assert _application_file(rig)["task"] == "detect"
        assert rig.service.current_model == "det.engine"
        assert rig.events.events == []

    def test_a_write_failure_restarts_detection_and_keeps_the_model(self, rig, monkeypatch):
        def refuse(task):
            raise OSError("disk full")

        monkeypatch.setattr(rig.app, "persist", refuse)
        with pytest.raises(OSError):
            rig.service.set_task("classify")
        assert rig.detection.calls == ["stop", "start"]
        assert rig.detection.is_running is True
        assert rig.app.task == "detect"
        assert rig.service.current_model == "det.engine"
        assert (rig.dir / ".current_model").read_text() == "det.engine"
        assert rig.events.events == []


class TestGpuHandover:
    def test_refused_while_the_runtime_is_released(self, rig):
        ok, _ = rig.service.release_runtime()
        assert ok and rig.service.runtime_released
        with pytest.raises(PreconditionFailed, match="training"):
            rig.service.set_task("classify")
        ok, _ = rig.service.resume_runtime()
        assert ok and not rig.service.runtime_released
        assert rig.service.set_task("classify") is True

    def test_an_exit_without_restart_ends_the_handover(self, rig):
        # The post-training handoff: the conversion keeps the
        # GPU, so nothing restarts, but the application type can change again.
        rig.service.release_runtime()
        rig.detection.calls.clear()
        ok, _ = rig.service.resume_runtime(restart_detection=False)
        assert ok and not rig.service.runtime_released
        assert rig.detection.calls == []
        assert rig.detection.is_running is False
        assert rig.service.set_task("classify") is True

    def test_an_explicit_start_ends_the_handover(self, rig):
        rig.service.release_runtime()
        rig.detection.calls.clear()
        assert rig.service.start_detection() is True
        # The released runtime is dropped, so the start initializes the model
        # configured now instead of reusing the pre-handover one.
        assert rig.detection.calls == ["unload_runtime", "start"]
        assert not rig.service.runtime_released
        assert rig.detection.is_running is True
        # Nothing to end on a plain start, and nothing to drop.
        rig.detection.calls.clear()
        assert rig.service.start_detection() is False
        assert rig.detection.calls == ["start"]
        assert rig.service.set_task("classify") is True

    def test_a_failed_start_keeps_the_handover(self, rig):
        rig.service.release_runtime()

        def refuse():
            raise RuntimeError("no engine")

        rig.detection.start_gate = refuse
        with pytest.raises(RuntimeError, match="no engine"):
            rig.service.start_detection()
        assert rig.service.runtime_released
        with pytest.raises(PreconditionFailed, match="training"):
            rig.service.set_task("classify")

    def test_a_release_waits_for_a_start_in_progress(self, rig):
        rig.service.release_runtime()
        entered, go = threading.Event(), threading.Event()

        def gate():
            entered.set()
            assert go.wait(5)

        rig.detection.start_gate = gate
        results = {}
        start = threading.Thread(target=lambda: results.update(
            start=rig.service.start_detection()))
        start.start()
        assert entered.wait(5)
        rig.detection.start_gate = None
        release = threading.Thread(target=lambda: results.update(
            release=rig.service.release_runtime()))
        release.start()
        time.sleep(0.2)
        assert "release" not in results, "the release must wait for the start"
        go.set()
        start.join(5)
        release.join(5)
        # The start ended the old handover; the later release began a new one.
        assert results["start"] is True
        assert results["release"][0] is True
        assert rig.service.runtime_released is True
        assert rig.detection.is_running is False

    def test_refused_while_a_conversion_runs(self, rig):
        rig.conversions.jobs = ["job"]
        with pytest.raises(PreconditionFailed, match="conversion"):
            rig.service.set_task("classify")
        assert rig.detection.calls == []

    def test_a_release_waits_for_a_switch_in_progress(self, rig):
        entered, go = threading.Event(), threading.Event()

        def gate():
            entered.set()
            assert go.wait(5)

        rig.detection.stop_gate = gate
        results = {}
        switch = threading.Thread(target=lambda: results.update(
            switch=rig.service.set_task("classify")))
        switch.start()
        assert entered.wait(5)
        rig.detection.stop_gate = None
        release = threading.Thread(target=lambda: results.update(
            release=rig.service.release_runtime()))
        release.start()
        time.sleep(0.2)
        assert "release" not in results, "the release must wait for the switch"
        go.set()
        switch.join(5)
        release.join(5)
        assert results["switch"] is True
        assert results["release"][0] is True
        assert rig.service.runtime_released is True

    def test_an_activation_during_a_switch_sees_the_new_task(self, rig):
        entered, go = threading.Event(), threading.Event()

        def gate():
            entered.set()
            assert go.wait(5)

        rig.detection.stop_gate = gate
        results = {}
        switch = threading.Thread(target=lambda: results.update(
            switch=rig.service.set_task("classify")))
        switch.start()
        assert entered.wait(5)
        rig.detection.stop_gate = None

        def activate():
            try:
                results["activate"] = rig.service.activate_model("det.engine")
            except PreconditionFailed as exc:
                results["activate"] = exc

        activation = threading.Thread(target=activate)
        activation.start()
        time.sleep(0.2)
        assert "activate" not in results
        go.set()
        switch.join(5)
        activation.join(5)
        assert isinstance(results["activate"], PreconditionFailed)
        assert "'detect' model" in str(results["activate"])


class TestActivationGate:
    def test_a_model_of_another_task_is_refused_before_anything_stops(self, rig):
        _model(rig.dir, "cls.engine", "classify")
        with pytest.raises(PreconditionFailed, match="'classify' model.*'detect' application"):
            rig.service.activate_model("cls.engine")
        assert rig.detection.calls == []
        assert rig.detection.is_running is True
        assert rig.service.current_model == "det.engine"

    def test_an_engine_upload_records_its_task_and_is_refused_on_mismatch(self, rig):
        class Upload:
            def save(self, path):
                with open(path, "wb") as fh:
                    fh.write(b"engine")

        body, status = rig.service.process_upload("new.engine", Upload(), task="classify")
        assert status == 409 and "'classify' model" in body["error"]
        assert json.loads((rig.dir / "new.settings.json").read_text()) == {"task": "classify"}
        assert rig.service.current_model == "det.engine"


class TestBootSelfHeal:
    def _service(self, tmp_path, task):
        (tmp_path / "application.json").write_text(json.dumps({"task": task}))
        app = ApplicationService(str(tmp_path), ["detect", "classify"])
        app.load_or_migrate()
        service = ModelService(Config(), str(tmp_path))
        service.attach_application_service(app)
        return service

    def test_a_marker_to_a_missing_engine_is_cleared(self, tmp_path):
        service = self._service(tmp_path, "detect")
        (tmp_path / ".current_model").write_text("gone.engine")
        assert service.heal_persisted_selection() is True
        assert not (tmp_path / ".current_model").exists()

    def test_a_marker_to_a_model_of_another_task_is_cleared(self, tmp_path):
        service = self._service(tmp_path, "detect")
        _model(tmp_path, "cls.engine", "classify")
        (tmp_path / ".current_model").write_text("cls.engine")
        assert service.heal_persisted_selection() is True
        assert service.load_persisted_current_model() == ""

    def test_any_marker_is_cleared_while_no_application_is_chosen(self, tmp_path):
        service = self._service(tmp_path, None)
        _model(tmp_path, "det.engine", "detect")
        (tmp_path / ".current_model").write_text("det.engine")
        assert service.heal_persisted_selection() is True

    def test_a_valid_marker_is_kept(self, tmp_path):
        service = self._service(tmp_path, "detect")
        _model(tmp_path, "det.engine", "detect")
        (tmp_path / ".current_model").write_text("det.engine")
        assert service.heal_persisted_selection() is False
        assert service.load_persisted_current_model() == "det.engine"

    def test_a_legacy_model_without_a_task_counts_as_detect(self, tmp_path):
        service = self._service(tmp_path, "detect")
        (tmp_path / "old.engine").write_bytes(b"x")
        (tmp_path / ".current_model").write_text("old.engine")
        assert service.heal_persisted_selection() is False
