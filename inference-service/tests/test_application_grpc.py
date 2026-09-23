# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""The application-type RPCs and the task gates on the model RPCs.

Refusals must reach the gateway as gRPC status codes (FAILED_PRECONDITION →
409, INVALID_ARGUMENT → 400), not as ``success=False`` replies.
"""
import json
from types import SimpleNamespace

import api.inference_grpc as ig
import grpc
import inference_pb2 as pb
import pytest
from api.config import Config
from api.services.application_service import ApplicationService
from api.services.model_service import ModelService


class FakeContext:
    def __init__(self):
        self.code = None
        self.details = ""

    def set_code(self, code):
        self.code = code

    def set_details(self, details):
        self.details = details


class FakeDetection:
    is_running = False

    def stop(self):
        return False

    def start(self):
        pass

    def initialize(self):
        pass

    def unload_runtime(self):
        pass

    def reset_results(self):
        pass


@pytest.fixture
def app(tmp_path):
    application_service = ApplicationService(str(tmp_path), ["detect"])
    application_service.load_or_migrate()  # blank device
    models = ModelService(Config(), str(tmp_path))
    models.attach_application_service(application_service)
    models.attach_detection_service(FakeDetection())
    models.attach_conversion_service(SimpleNamespace(get_active_jobs=lambda: []))
    events = []
    models.attach_event_service(SimpleNamespace(
        publish=lambda event, keys=None, source=None, data=None: events.append(event)))
    return SimpleNamespace(application_service=application_service, model_service=models,
                           config=Config(), events=events, dir=tmp_path)


class TestApplicationRpcs:
    def test_a_blank_device_reports_no_task(self, app):
        info = ig.ManagementControlServicer(app).GetApplication(pb.Empty(), FakeContext())
        assert info.task == "" and list(info.supported_tasks) == ["detect"]
        assert info.migrated is False

    def test_set_records_the_choice(self, app):
        ctx = FakeContext()
        info = ig.ManagementControlServicer(app).SetApplication(
            pb.SetApplicationRequest(task="detect"), ctx)
        assert ctx.code is None
        assert info.task == "detect" and info.migrated is False
        assert json.loads((app.dir / "application.json").read_text()) == {"task": "detect"}
        assert app.events == ["application_changed"]

    def test_unknown_task_is_invalid_argument(self, app):
        ctx = FakeContext()
        ig.ManagementControlServicer(app).SetApplication(
            pb.SetApplicationRequest(task="pose"), ctx)
        assert ctx.code == grpc.StatusCode.INVALID_ARGUMENT

    @pytest.mark.parametrize("task", ["classify", "segment"])
    def test_unsupported_task_is_failed_precondition(self, app, task, caplog):
        ctx = FakeContext()
        ig.ManagementControlServicer(app).SetApplication(
            pb.SetApplicationRequest(task=task), ctx)
        assert ctx.code == grpc.StatusCode.FAILED_PRECONDITION
        assert f"refused (FAILED_PRECONDITION): {ctx.details}" in caplog.text
        assert "later release" not in ctx.details  # service wording, UI localizes
        assert app.application_service.task is None

    def test_a_write_failure_is_internal(self, app, monkeypatch):
        def refuse(task):
            raise OSError("disk full")

        monkeypatch.setattr(app.application_service, "persist", refuse)
        ctx = FakeContext()
        ig.ManagementControlServicer(app).SetApplication(
            pb.SetApplicationRequest(task="detect"), ctx)
        assert ctx.code == grpc.StatusCode.INTERNAL
        assert "disk full" in ctx.details


class TestGpuHandoverRpcs:
    """An explicit Start and a ResumeRuntime without restart end the handover,
    so the application type can change again."""

    @pytest.fixture
    def rig(self, app, monkeypatch):
        import api.runtime_management.worker_client as wc
        monkeypatch.setattr(wc, "release_all_workers", lambda: None)
        detection = SimpleNamespace(starts=0, is_running=False)

        def start():
            detection.starts += 1
            detection.is_running = True

        detection.start = start
        detection.stop = lambda: False
        detection.unload_runtime = lambda: None
        app.model_service.attach_detection_service(detection)
        published = []
        app.event_service = SimpleNamespace(
            publish=lambda event, keys=None, source=None, data=None:
            published.append((event, data)))
        app.detection_service = detection
        app.application_service.persist("detect")
        return SimpleNamespace(app=app, detection=detection, published=published)

    def test_start_ends_a_handover_and_says_so(self, rig):
        ok, _ = rig.app.model_service.release_runtime()
        assert ok and rig.app.model_service.runtime_released
        r = ig.DetectionControlServicer(rig.app).Start(pb.Empty(), FakeContext())
        assert r.success and rig.detection.starts == 1
        assert not rig.app.model_service.runtime_released
        assert rig.published == [("runtime_changed", {"runtime_released": False})]

    def test_a_plain_start_publishes_nothing_extra(self, rig):
        r = ig.DetectionControlServicer(rig.app).Start(pb.Empty(), FakeContext())
        assert r.success and rig.published == []

    def test_resume_without_restart_ends_the_handover_only(self, rig):
        rig.app.model_service.release_runtime()
        r = ig.ManagementControlServicer(rig.app).ResumeRuntime(
            pb.ResumeRuntimeRequest(keep_detection_stopped=True), FakeContext())
        assert r.success and "left stopped" in r.message
        assert not rig.app.model_service.runtime_released
        assert rig.detection.starts == 0
        assert rig.published == [("runtime_changed", {"runtime_released": False})]

    def test_an_empty_resume_request_still_restarts_detection(self, rig):
        rig.app.model_service.current_model = "m.engine"
        rig.app.model_service.release_runtime()
        rig.detection.initialize = lambda: None
        r = ig.ManagementControlServicer(rig.app).ResumeRuntime(
            pb.ResumeRuntimeRequest(), FakeContext())
        assert r.success and r.message == "Runtime resumed"
        assert rig.detection.starts == 1

    def test_a_resume_with_no_model_selected_ends_the_handover_only(self, rig):
        # Regression: it initialized the config's default model, which a device
        # of another task refuses, so leaving training mode failed forever.
        rig.app.model_service.release_runtime()

        def refuse():
            raise AssertionError("no model is selected; nothing to initialize")

        rig.detection.initialize = refuse
        r = ig.ManagementControlServicer(rig.app).ResumeRuntime(
            pb.ResumeRuntimeRequest(), FakeContext())
        assert r.success and "no model selected" in r.message
        assert not rig.app.model_service.runtime_released
        assert rig.detection.starts == 0


class TestModelRpcs:
    def test_select_of_another_task_is_failed_precondition(self, app):
        app.application_service.persist("detect")
        (app.dir / "cls.engine").write_bytes(b"x")
        (app.dir / "cls.settings.json").write_text(json.dumps({"task": "classify"}))
        ctx = FakeContext()
        r = ig.ModelControlServicer(app).SelectModel(pb.ModelName(name="cls.engine"), ctx)
        assert not r.success
        assert ctx.code == grpc.StatusCode.FAILED_PRECONDITION
        assert "'classify'" in ctx.details and "'detect'" in ctx.details

    def test_list_models_carries_the_task(self, app):
        (app.dir / "old.engine").write_bytes(b"x")
        ml = ig.ModelControlServicer(app).ListModels(pb.Empty(), FakeContext())
        assert [(m.name, m.task) for m in ml.models] == [("old.engine", "detect")]

    def test_a_model_being_converted_lists_with_its_declared_task(self, app):
        # Regression: before the sidecar exists it listed as "detect".
        (app.dir / "seg.pt").write_bytes(b"x")
        (app.dir / "seg.onnx").write_bytes(b"x")
        jobs = [SimpleNamespace(pt_path=str(app.dir / "seg.pt"),
                                onnx_path=str(app.dir / "seg.onnx"), task="segment")]
        app.model_service.attach_conversion_service(SimpleNamespace(get_active_jobs=lambda: jobs))
        servicer = ig.ModelControlServicer(app)

        listed = servicer.ListModels(pb.Empty(), FakeContext())
        assert sorted((m.name, m.task) for m in listed.models) == [
            ("seg.onnx", "segment"), ("seg.pt", "segment")]

        jobs.clear()  # finished: the sidecar is the source again
        listed = servicer.ListModels(pb.Empty(), FakeContext())
        assert {m.task for m in listed.models} == {"detect"}

    def _upload(self, app, task):
        chunks = [pb.ModelChunk(meta=pb.ModelUploadMeta(filename="m.engine", task=task)),
                  pb.ModelChunk(chunk=b"engine")]
        return ig.ModelControlServicer(app).UploadModel(iter(chunks), FakeContext())

    def test_an_undeclared_upload_on_a_blank_device_is_refused(self, app):
        r = self._upload(app, "")
        assert r.http_status == 400 and "Declare the model's task" in r.json
        assert not (app.dir / "m.engine").exists()

    def test_an_unsupported_declared_task_is_refused_at_upload(self, app):
        app.application_service.persist("detect")
        r = self._upload(app, "classify")
        assert r.http_status == 409
        assert not (app.dir / "m.engine").exists()


class TestModelSettingRpcs:
    """The per-model setters save through ModelService.update_setting."""

    @pytest.fixture
    def servicer(self, app):
        app.detection_service = SimpleNamespace(config=app.model_service.config)
        return ig.DetectionControlServicer(app)

    def _settings(self, app, saved):
        app.model_service.attach_settings_service(SimpleNamespace(save=lambda only=None: saved))

    def test_a_saved_limit_is_a_success(self, app, servicer):
        self._settings(app, True)
        ctx = FakeContext()
        r = servicer.SetSegmentMaxInstances(pb.SegmentMaxInstancesRequest(max_instances=12), ctx)
        assert r.success and ctx.code is None
        assert app.model_service.config.SEGMENT_MAX_INSTANCES == 12

    def test_a_refused_limit_is_an_unsuccessful_result(self, app, servicer):
        self._settings(app, True)
        ctx = FakeContext()
        r = servicer.SetSegmentMaxInstances(pb.SegmentMaxInstancesRequest(max_instances=0), ctx)
        assert not r.success and ctx.code is None

    def test_an_unsaved_limit_is_internal_and_rolled_back(self, app, servicer):
        self._settings(app, False)
        before = app.model_service.config.SEGMENT_MAX_INSTANCES
        ctx = FakeContext()
        servicer.SetSegmentMaxInstances(pb.SegmentMaxInstancesRequest(max_instances=12), ctx)
        assert ctx.code == grpc.StatusCode.INTERNAL and "could not be saved" in ctx.details
        assert app.model_service.config.SEGMENT_MAX_INSTANCES == before

    def test_an_unsaved_overlay_threshold_is_internal(self, app, servicer):
        self._settings(app, False)
        ctx = FakeContext()
        servicer.SetOverlayThreshold(pb.ThresholdRequest(threshold=0.3), ctx)
        assert ctx.code == grpc.StatusCode.INTERNAL

    def test_the_face_settings_given_are_saved_and_the_others_left_alone(self, app, servicer):
        self._settings(app, True)
        cfg = app.model_service.config
        before = cfg.FACE_MATCH_THRESHOLD
        ctx = FakeContext()
        r = servicer.SetFaceSettings(pb.FaceSettingsRequest(min_size_px=80, max_faces=3), ctx)
        assert r.success and ctx.code is None
        assert (cfg.FACE_MIN_SIZE_PX, cfg.FACE_MAX_FACES) == (80, 3)
        assert cfg.FACE_MATCH_THRESHOLD == before

    def test_a_request_without_a_setting_is_an_unsuccessful_result(self, app, servicer):
        self._settings(app, True)
        ctx = FakeContext()
        r = servicer.SetFaceSettings(pb.FaceSettingsRequest(), ctx)
        assert not r.success and "No face setting" in r.message and ctx.code is None

    def test_a_refused_face_setting_is_an_unsuccessful_result(self, app, servicer):
        self._settings(app, True)
        cfg = app.model_service.config
        ctx = FakeContext()
        r = servicer.SetFaceSettings(pb.FaceSettingsRequest(max_faces=0), ctx)
        assert not r.success and ctx.code is None
        assert cfg.FACE_MAX_FACES != 0

    def test_a_refused_field_leaves_the_valid_ones_unapplied_too(self, app, servicer):
        saves = []
        app.model_service.attach_settings_service(
            SimpleNamespace(save=lambda only=None: saves.append(only) or True))
        cfg = app.model_service.config
        before = (cfg.FACE_MATCH_THRESHOLD, cfg.FACE_MIN_SIZE_PX, cfg.FACE_MAX_FACES)
        ctx = FakeContext()
        r = servicer.SetFaceSettings(
            pb.FaceSettingsRequest(match_threshold=0.5, min_size_px=80, max_faces=0), ctx)
        assert not r.success and "faces per frame" in r.message and ctx.code is None
        assert (cfg.FACE_MATCH_THRESHOLD, cfg.FACE_MIN_SIZE_PX, cfg.FACE_MAX_FACES) == before
        assert saves == []

    def test_several_face_settings_are_saved_in_one_write(self, app, servicer):
        saves = []
        app.model_service.attach_settings_service(
            SimpleNamespace(save=lambda only=None: saves.append(only) or True))
        r = servicer.SetFaceSettings(
            pb.FaceSettingsRequest(match_threshold=0.5, max_faces=2), FakeContext())
        assert r.success
        assert saves == [["FACE_MATCH_THRESHOLD", "FACE_MAX_FACES"]]

    def test_an_unsaved_face_setting_is_internal_and_rolled_back(self, app, servicer):
        self._settings(app, False)
        cfg = app.model_service.config
        before = cfg.FACE_MAX_FACES
        ctx = FakeContext()
        servicer.SetFaceSettings(pb.FaceSettingsRequest(max_faces=7), ctx)
        assert ctx.code == grpc.StatusCode.INTERNAL and "could not be saved" in ctx.details
        assert cfg.FACE_MAX_FACES == before


class TestStatusPerTaskSettings:
    """GetStatus carries a task's settings only while the device runs that task."""

    def _status(self, app, task):
        app.application_service.persist(task)
        cfg = app.model_service.config
        app.detection_service = SimpleNamespace(
            config=cfg, is_running=False, acceleration_type=lambda: "trt",
            runtime_api=lambda: "tensorrt", get_trigger_status=lambda: True,
            get_detection_count=lambda: 0)
        app.stats_service = SimpleNamespace(get_stats=lambda: SimpleNamespace(
            fps=0.0, inference_time=0.0, detections=0, frames_with_detections=0))
        app.video_service = SimpleNamespace(camera_connected=lambda: True)
        return ig.DetectionControlServicer(app).GetStatus(pb.Empty(), FakeContext())

    _FIELDS = ("segment_max_instances", "face_match_threshold", "face_min_size_px",
               "face_max_faces")

    @pytest.mark.parametrize("task,present", [
        ("detect", [False, False, False, False]),
        ("classify", [False, False, False, False]),
        ("segment", [True, False, False, False]),
        ("face", [False, True, True, True]),
    ])
    def test_only_the_running_task_settings_are_present(self, app, task, present):
        status = self._status(app, task)
        assert status.task == task
        assert [status.HasField(f) for f in self._FIELDS] == present
