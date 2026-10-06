# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""The shared GPU release (gateway.gpu.acquire_gpu) and the routes it gates.

The rule: only detection running, a training job in progress or a model
conversion in progress keep the GPU; every other holder (SAM3, the labeling
engine, idle TensorRT workers, a leftover training handover) is evicted for
the caller. Each purpose's row in ``POLICIES`` says which of the three block
it and what goes.
"""
import time
from types import SimpleNamespace

import grpc
import inference_pb2 as inf_pb
import pytest
import training_pb2 as trn_pb
from flask import Flask
from gateway import gpu
from gateway.controllers import models
from gateway.training import session


class FakeRpcError(grpc.RpcError):
    def __init__(self, code=grpc.StatusCode.UNAVAILABLE):
        self._code = code

    def code(self):
        return self._code

    def details(self):
        return "down"


class Device:
    """Fake training + inference gRPC surfaces; ``calls`` logs every RPC.

    Unloads flip the assistant's state, so the post-unload confirmation
    probe sees it gone — unless ``sticky``.
    """

    def __init__(self, *, job="idle", model_name="Parts", conversions=(), sam_loaded=True,
                 label_loaded=True, busy=inf_pb.GPU_HOLDER_NONE, training_down=None,
                 inference_down=False, probes_down=False, unload_ok=True, sticky=False,
                 running=False):
        self.calls = []
        self.job, self.model_name, self.conversions = job, model_name, conversions
        self.sam_loaded, self.label_loaded = sam_loaded, label_loaded
        self.busy, self.unload_ok, self.sticky = busy, unload_ok, sticky
        self.training_down, self.inference_down = training_down, inference_down
        self.probes_down = probes_down
        self.running = running

    def _log(self, name, down=None):
        self.calls.append(name)
        if down:
            raise FakeRpcError(down if isinstance(down, grpc.StatusCode)
                               else grpc.StatusCode.UNAVAILABLE)

    def clients(self):
        def get_training(_):
            self._log("GetTraining", self.training_down or self.probes_down)
            return SimpleNamespace(job_id="job-1", status=self.job, model_name=self.model_name)

        def get_sam_status(_):
            self._log("GetSamStatus", self.training_down)
            return SimpleNamespace(loaded=self.sam_loaded)

        def unload_sam(_):
            self._log("UnloadSam")
            if self.unload_ok and not self.sticky:
                self.sam_loaded = False
            return SimpleNamespace(success=self.unload_ok, message="worker stuck")

        def list_conversions(_):
            self._log("ListConversions", self.inference_down or self.probes_down)
            return SimpleNamespace(jobs=[SimpleNamespace(status=s) for s in self.conversions])

        def label_status(_):
            self._log("GetLabelModelStatus", self.inference_down)
            return SimpleNamespace(loaded=self.label_loaded)

        def unload_label(_):
            self._log("UnloadLabelModel")
            if self.unload_ok and not self.sticky:
                self.label_loaded = False
            return SimpleNamespace(success=self.unload_ok, message="worker stuck")

        def free_gpu(req):
            self._log(f"FreeGpu(stop={req.stop_detection}, handover={req.begin_handover})",
                      self.inference_down)
            return inf_pb.FreeGpuResult(success=self.busy == inf_pb.GPU_HOLDER_NONE,
                                        busy=self.busy, message="refused")

        def get_status(_):
            self._log("GetStatus", self.inference_down or self.probes_down)
            return SimpleNamespace(is_running=self.running)

        return SimpleNamespace(
            detection=SimpleNamespace(GetStatus=get_status),
            training=SimpleNamespace(GetTraining=get_training, GetSamStatus=get_sam_status,
                                     UnloadSam=unload_sam),
            model=SimpleNamespace(ListConversions=list_conversions,
                                  GetLabelModelStatus=label_status,
                                  UnloadLabelModel=unload_label),
            management=SimpleNamespace(FreeGpu=free_gpu))


@pytest.fixture
def device(monkeypatch):
    def make(**kw):
        d = Device(**kw)
        monkeypatch.setattr(gpu, "clients", d.clients())
        return d
    return make


P = gpu.Purpose
SAM_GONE = ["GetSamStatus", "UnloadSam", "GetSamStatus"]
LABEL_GONE = ["GetLabelModelStatus", "UnloadLabelModel", "GetLabelModelStatus"]


class TestEvictions:
    def test_application_frees_everything_without_stopping_detection(self, device):
        d = device()
        gpu.acquire_gpu(P.APPLICATION)
        assert d.calls == ["GetTraining", "ListConversions", "GetStatus", *SAM_GONE,
                           "FreeGpu(stop=False, handover=False)"]

    def test_training_frees_stops_and_marks_the_handover_in_one_call(self, device):
        d = device(job="training")  # a re-enter during a run is not refused
        gpu.acquire_gpu(P.TRAINING)
        # One inference-side operation: nothing can start between the release
        # and the handover mark.
        assert d.calls == ["ListConversions", *SAM_GONE, "FreeGpu(stop=True, handover=True)"]

    def test_sam_keeps_itself_and_frees_the_rest(self, device):
        d = device()
        gpu.acquire_gpu(P.SAM)
        assert d.calls == ["GetTraining", "ListConversions", "GetStatus",
                           "FreeGpu(stop=False, handover=False)"]

    def test_label_model_drops_sam_and_the_runtime(self, device):
        d = device()
        gpu.acquire_gpu(P.LABEL_MODEL)
        assert d.calls == ["GetTraining", "ListConversions", "GetStatus", *SAM_GONE,
                           "FreeGpu(stop=False, handover=False)"]

    @pytest.mark.parametrize("purpose", [P.DETECTION, P.MODEL_SELECT, P.ENGINE_UPLOAD])
    def test_detection_keeps_its_runtime_and_drops_the_assistants(self, device, purpose):
        # A restart must not re-deserialize the engine: no FreeGpu here.
        d = device()
        gpu.acquire_gpu(purpose)
        assert d.calls == ["GetTraining", "ListConversions", *SAM_GONE, *LABEL_GONE]

    def test_a_conversion_beside_another_drops_the_assistants(self, device):
        d = device(conversions=("converting_to_engine",))
        gpu.acquire_gpu(P.CONVERSION, upload_name="new.pt")
        assert d.calls == ["GetTraining", *SAM_GONE, *LABEL_GONE]

    def test_idle_assistants_are_only_probed(self, device):
        d = device(sam_loaded=False, label_loaded=False)
        gpu.acquire_gpu(P.DETECTION)
        assert "UnloadSam" not in d.calls and "UnloadLabelModel" not in d.calls


class TestFailedUnloads:
    # SAM3 is never unloaded for itself; the labeling engine is then FreeGpu's.
    @pytest.mark.parametrize("purpose", [p for p in P if p is not P.SAM])
    @pytest.mark.parametrize("kw", [{"unload_ok": False}, {"sticky": True}])
    def test_an_unload_that_does_not_happen_refuses(self, device, purpose, kw):
        d = device(**kw)
        with pytest.raises(gpu.GpuBusy) as exc:
            gpu.acquire_gpu(purpose, upload_name="new.pt")
        assert exc.value.status == 500 and exc.value.holder is None
        assert "Could not unload" in exc.value.message
        # Nothing is released or started on top of a resident assistant.
        assert not any(c.startswith("FreeGpu") for c in d.calls)

    @pytest.mark.parametrize("purpose", [P.DETECTION, P.MODEL_SELECT, P.ENGINE_UPLOAD,
                                         P.CONVERSION, P.TRAINING])
    def test_a_training_service_nobody_answers_on_has_no_sam(self, device, purpose):
        d = device(training_down=grpc.StatusCode.UNAVAILABLE, label_loaded=False)
        gpu.acquire_gpu(purpose, upload_name="new.pt")
        assert "UnloadSam" not in d.calls

    @pytest.mark.parametrize("purpose", [P.DETECTION, P.MODEL_SELECT, P.CONVERSION,
                                         P.TRAINING])
    @pytest.mark.parametrize("code", [grpc.StatusCode.DEADLINE_EXCEEDED,
                                      grpc.StatusCode.INTERNAL])
    def test_an_unreadable_sam_status_refuses(self, device, purpose, code):
        # The training probe fails open, but a service that is up and cannot
        # say whether SAM3 is resident is not a free GPU.
        d = device(training_down=code)
        with pytest.raises(gpu.GpuBusy) as exc:
            gpu.acquire_gpu(purpose, upload_name="new.pt")
        assert exc.value.status == 503
        assert not any(c.startswith("FreeGpu") for c in d.calls)

    def test_an_unreadable_labeling_status_refuses(self, device):
        device(inference_down=True, sam_loaded=False)
        with pytest.raises(gpu.GpuBusy) as exc:
            gpu.acquire_gpu(P.CONVERSION, upload_name="new.pt")
        assert exc.value.status == 503


BLOCKED_BY_TRAINING = [P.APPLICATION, P.DETECTION, P.SAM, P.LABEL_MODEL, P.MODEL_SELECT,
                       P.ENGINE_UPLOAD, P.CONVERSION]


class TestRefusals:
    @pytest.mark.parametrize("purpose", BLOCKED_BY_TRAINING)
    @pytest.mark.parametrize("job", gpu.ACTIVE_JOB_STATUSES)
    def test_a_training_job_blocks(self, device, purpose, job):
        d = device(job=job)
        with pytest.raises(gpu.GpuBusy) as exc:
            gpu.acquire_gpu(purpose, upload_name="someone-else.pt")
        assert exc.value.holder == gpu.TRAINING and exc.value.status == 409
        # Nothing was evicted.
        assert d.calls == ["GetTraining"]

    @pytest.mark.parametrize("purpose", [P.APPLICATION, P.SAM, P.LABEL_MODEL])
    def test_a_running_detection_blocks_before_anything_is_evicted(self, device, purpose):
        # A refused request keeps its assistant: detection is probed first.
        d = device(running=True)
        with pytest.raises(gpu.GpuBusy) as exc:
            gpu.acquire_gpu(purpose)
        assert exc.value.holder == gpu.DETECTION
        assert "UnloadSam" not in d.calls
        assert not any(c.startswith("FreeGpu") for c in d.calls)

    @pytest.mark.parametrize("purpose", [P.DETECTION, P.MODEL_SELECT, P.CONVERSION,
                                         P.TRAINING])
    def test_a_running_detection_does_not_block_its_own_purposes(self, device, purpose):
        device(running=True)
        gpu.acquire_gpu(purpose, upload_name="new.pt")

    def test_a_training_job_does_not_block_entering_training(self, device):
        device(job="training")
        gpu.acquire_gpu(P.TRAINING)

    @pytest.mark.parametrize("name", ["Parts.pt", "Parts.faces"])
    def test_the_jobs_own_result_upload_goes_through(self, device, name):
        device(job="uploading", model_name="Parts")
        gpu.acquire_gpu(P.CONVERSION, upload_name=name, upload_job="job-1")

    @pytest.mark.parametrize("job,name,upload_job", [
        ("training", "Parts.pt", "job-1"),      # not uploading yet: not its result
        ("preparing", "Parts.faces", "job-1"),
        ("uploading", "Other.pt", "job-1"),     # someone else's upload meanwhile
        ("uploading", "Parts.onnx", "job-1"),   # a training never uploads an .onnx
        ("uploading", "Parts.pt", None),        # the right name alone proves nothing
        ("uploading", "Parts.pt", "job-0"),     # another job's id
    ])
    def test_other_uploads_during_a_training_are_refused(self, device, job, name, upload_job):
        device(job=job, model_name="Parts")
        with pytest.raises(gpu.GpuBusy) as exc:
            gpu.acquire_gpu(P.CONVERSION, upload_name=name, upload_job=upload_job)
        assert exc.value.holder == gpu.TRAINING

    @pytest.mark.parametrize("job", ["uploading", "training", "preparing"])
    def test_an_engine_upload_during_a_training_is_refused(self, device, job):
        # A prebuilt engine is activated on arrival, even with the job's name.
        device(job=job, model_name="Parts")
        with pytest.raises(gpu.GpuBusy) as exc:
            gpu.acquire_gpu(P.ENGINE_UPLOAD)
        assert exc.value.holder == gpu.TRAINING

    @pytest.mark.parametrize("purpose", [P.APPLICATION, P.TRAINING, P.SAM, P.LABEL_MODEL,
                                         P.DETECTION, P.MODEL_SELECT, P.ENGINE_UPLOAD])
    @pytest.mark.parametrize("status", gpu.ACTIVE_CONVERSION_STATUSES)
    def test_a_conversion_blocks_before_anything_is_evicted(self, device, purpose, status):
        # A running build is found by the probe, so the caller never loses
        # its assistant (or its detection) to a request that is refused anyway.
        d = device(conversions=("done", status))
        with pytest.raises(gpu.GpuBusy) as exc:
            gpu.acquire_gpu(purpose)
        assert exc.value.holder == gpu.CONVERSION
        assert "UnloadSam" not in d.calls
        assert not any(c.startswith("FreeGpu") for c in d.calls)

    @pytest.mark.parametrize("purpose", [P.APPLICATION, P.SAM, P.LABEL_MODEL, P.TRAINING])
    @pytest.mark.parametrize("holder,name", [
        (inf_pb.GPU_HOLDER_DETECTION, gpu.DETECTION),
        (inf_pb.GPU_HOLDER_CONVERSION, gpu.CONVERSION),
    ])
    def test_the_inference_service_names_its_holder(self, device, purpose, holder, name):
        device(busy=holder)
        with pytest.raises(gpu.GpuBusy) as exc:
            gpu.acquire_gpu(purpose)
        assert exc.value.holder == name
        assert exc.value.message == gpu.BUSY_MESSAGES[name]

    def test_a_free_that_fails_without_a_holder_is_a_500(self, device):
        # E.g. a TensorRT worker that did not exit.
        d = device()
        gpu.clients.management.FreeGpu = lambda req: inf_pb.FreeGpuResult(
            success=False, message="Could not terminate the TensorRT workers on ports 5501")
        with pytest.raises(gpu.GpuBusy) as exc:
            gpu.acquire_gpu(P.TRAINING)
        assert exc.value.holder is None and exc.value.status == 500
        assert "5501" in exc.value.message
        assert d.calls == ["ListConversions", *SAM_GONE]

    @pytest.mark.parametrize("purpose", [P.APPLICATION, P.SAM, P.LABEL_MODEL])
    def test_strict_purposes_fail_closed(self, device, purpose):
        device(training_down=grpc.StatusCode.UNAVAILABLE)
        with pytest.raises(gpu.GpuBusy) as exc:
            gpu.acquire_gpu(purpose)
        assert exc.value.holder is None and exc.value.status == 503

    @pytest.mark.parametrize("purpose", [P.DETECTION, P.MODEL_SELECT, P.ENGINE_UPLOAD,
                                         P.CONVERSION])
    def test_fail_open_purposes_go_through(self, device, purpose):
        # A job or conversion registry that cannot be reached is not busy for
        # a start, a select or an upload.
        d = device(probes_down=True)
        gpu.acquire_gpu(purpose, upload_name="new.pt")
        assert "UnloadSam" in d.calls

    def test_an_unreachable_inference_service_propagates(self, device):
        # Fail-open probes pass; the release itself then cannot be made.
        device(inference_down=True)
        with pytest.raises(grpc.RpcError):
            gpu.acquire_gpu(P.TRAINING)

    def test_an_unreachable_inference_service_fails_a_strict_purpose_closed(self, device):
        d = device(inference_down=True)
        with pytest.raises(gpu.GpuBusy) as exc:
            gpu.acquire_gpu(P.APPLICATION)
        assert exc.value.status == 503 and "UnloadSam" not in d.calls

    def test_busy_response_names_the_holder(self):
        with Flask(__name__).app_context():
            resp = gpu.busy_response(gpu.GpuBusy(gpu.DETECTION, "stop it"))
        assert resp.status_code == 409
        assert resp.get_json() == {"error": "stop it", "gpu_busy": "detection"}


class TestLock:
    """One GPU operation at a time, from its checks through its RPC."""

    def test_a_second_operation_is_refused_after_the_wait(self):
        import threading

        held, done = threading.Event(), threading.Event()

        def hold():
            with gpu.gpu_lock():
                held.set()
                assert done.wait(5)

        worker = threading.Thread(target=hold)
        worker.start()
        assert held.wait(5)
        try:
            with pytest.raises(gpu.GpuBusy) as exc:
                with gpu.gpu_lock(wait=0.05):
                    pass
            assert exc.value.status == 503 and exc.value.message == gpu.LOCKED_MESSAGE
        finally:
            done.set()
            worker.join(5)
        with gpu.gpu_lock(wait=0):  # free again
            pass

    def test_a_session_holds_the_lock_through_the_operation(self, device):
        device()
        with gpu.gpu_session(P.APPLICATION):
            with pytest.raises(gpu.GpuBusy):
                with gpu.gpu_lock(wait=0):
                    pass

    def test_a_refused_session_releases_the_lock(self, device):
        device(job="training")
        with pytest.raises(gpu.GpuBusy):
            with gpu.gpu_session(P.APPLICATION):
                pass
        with gpu.gpu_lock(wait=0):
            pass


@pytest.fixture
def app():
    return Flask(__name__)


def _acquired(monkeypatch, module, busy=None):
    purposes = []

    def acquire(purpose, **kw):
        purposes.append(purpose)
        if busy is not None:
            raise busy

    # gpu_session calls gateway.gpu's own binding; routes that call
    # acquire_gpu directly hold an imported one.
    monkeypatch.setattr(gpu, "acquire_gpu", acquire)
    monkeypatch.setattr(module, "acquire_gpu", acquire, raising=False)
    return purposes


class TestRoutes:
    def test_training_enter_hands_the_gpu_over(self, app, monkeypatch):
        purposes = _acquired(monkeypatch, session)
        published, armed = [], []
        monkeypatch.setattr(session, "event_service", SimpleNamespace(
            publish=lambda event, keys=None, data=None: published.append((event, data))))
        monkeypatch.setattr(session, "tracker", SimpleNamespace(arm=lambda: armed.append(1)))
        with app.test_request_context("/api/v1/training/enter", method="POST"):
            resp = session.training_enter()
        assert resp.status_code == 200
        assert purposes == [P.TRAINING]
        assert published == [("detection_state_changed", {"is_running": False})]
        assert armed == [1]

    def test_the_watchdog_is_armed_and_disarmed_under_the_lock(self, app, monkeypatch):
        # An enter, a start or an exit racing another one must never leave a
        # handover without a watchdog, or a watchdog without a handover.

        seen = []

        def record(what):
            return lambda: seen.append((what, gpu._GPU_LOCK.locked()))

        tracker = SimpleNamespace(armed=True, arm=record("arm"), disarm=record("disarm"))
        _acquired(monkeypatch, session)
        monkeypatch.setattr(session, "tracker", tracker)
        monkeypatch.setattr(session, "event_service",
                            SimpleNamespace(publish=lambda *a, **k: None))
        monkeypatch.setattr(session, "_do_exit", lambda resume: (True, "ok"))
        with app.test_request_context("/api/v1/training/enter", method="POST"):
            assert session.training_enter().status_code == 200
        with app.test_request_context("/api/v1/training/exit", method="POST", json={}):
            assert session.training_exit().status_code == 200

        jobs_, _, _, _ = self._wire_train(monkeypatch, entered=True, start_fails=False)
        monkeypatch.setattr(jobs_, "tracker", tracker)
        assert self._post_train(app, jobs_).status_code == 202
        assert seen == [("arm", True), ("disarm", True), ("arm", True)]

    def test_a_failed_exit_keeps_the_watchdog_armed(self, app, monkeypatch):
        disarmed = []
        monkeypatch.setattr(session, "tracker", SimpleNamespace(
            disarm=lambda: disarmed.append(1)))
        monkeypatch.setattr(session, "_do_exit", lambda resume: (False, "SAM3 stuck"))
        with app.test_request_context("/api/v1/training/exit", method="POST", json={}):
            assert session.training_exit().status_code == 500
        assert disarmed == []

    def test_training_enter_refused_during_a_conversion(self, app, monkeypatch):
        _acquired(monkeypatch, session,
                  busy=gpu.GpuBusy(gpu.CONVERSION, gpu.BUSY_MESSAGES[gpu.CONVERSION]))
        armed = []
        monkeypatch.setattr(session, "tracker", SimpleNamespace(arm=lambda: armed.append(1)))
        with app.test_request_context("/api/v1/training/enter", method="POST"):
            resp = session.training_enter()
        assert resp.status_code == 409 and resp.get_json()["gpu_busy"] == "conversion"
        assert armed == []

    def _wire_train(self, monkeypatch, *, entered, start_fails, running=True):
        from gateway.training import jobs

        calls, undone = [], []

        def start_training(req):
            calls.append(("start", req.epochs))
            if start_fails:
                raise FakeRpcError(grpc.StatusCode.FAILED_PRECONDITION)
            return trn_pb.TrainingJob(job_id="j", status="preparing", model_name="m")

        monkeypatch.setattr(jobs, "clients", SimpleNamespace(
            training=SimpleNamespace(StartTraining=start_training),
            detection=SimpleNamespace(GetStatus=lambda _: SimpleNamespace(is_running=running))))
        monkeypatch.setattr(jobs, "tracker", SimpleNamespace(armed=entered, arm=lambda: None))
        monkeypatch.setattr("gateway.training.session._do_exit",
                            lambda resume: undone.append(resume) or (True, "ok"))
        purposes = _acquired(monkeypatch, jobs)
        return jobs, calls, undone, purposes

    def _post_train(self, app, jobs, **body):
        with app.test_request_context("/api/v1/training/train", method="POST",
                                      json={"dataset_id": "d", "model_name": "m", **body}):
            return jobs.training_start()

    def test_no_training_starts_while_the_inference_service_does_not_serve(
            self, app, monkeypatch):
        # A booting inference-service serves gRPC only after its restore, so
        # its training probe cannot go stale: FreeGpu fails, nothing starts.
        jobs, calls, undone, purposes = self._wire_train(
            monkeypatch, entered=False, start_fails=False)

        def unavailable(purpose, **kw):
            raise FakeRpcError(grpc.StatusCode.UNAVAILABLE)

        monkeypatch.setattr(jobs, "acquire_gpu", unavailable)
        assert self._post_train(app, jobs).status_code == 503
        assert calls == []

    def test_a_malformed_train_request_takes_no_gpu(self, app, monkeypatch):
        jobs, calls, undone, purposes = self._wire_train(
            monkeypatch, entered=False, start_fails=False)
        assert self._post_train(app, jobs, epochs="many").status_code == 400
        assert purposes == [] and calls == [] and undone == []

    @pytest.mark.parametrize("running", [True, False])
    def test_a_failed_start_outside_training_mode_undoes_the_handover(self, app, monkeypatch,
                                                                      running):
        jobs, calls, undone, purposes = self._wire_train(
            monkeypatch, entered=False, start_fails=True, running=running)
        assert self._post_train(app, jobs).status_code == 409
        assert purposes == [P.TRAINING]
        # Detection comes back only if it ran before the request.
        assert undone == [running]

    def test_training_mode_is_read_under_the_lock(self, app, monkeypatch):
        # A /training/enter that completes while this start waits for the
        # lock makes it a start inside training mode: no rollback.
        import threading

        jobs, _, undone, _ = self._wire_train(monkeypatch, entered=False, start_fails=True)
        results = {}

        def post():
            results["resp"] = self._post_train(app, jobs)

        with gpu.gpu_lock():
            worker = threading.Thread(target=post)
            worker.start()
            time.sleep(0.1)
            jobs.tracker.armed = True  # pyright: ignore[reportAttributeAccessIssue]
        worker.join(5)
        assert results["resp"].status_code == 409
        assert undone == []

    def test_a_failed_start_inside_training_mode_keeps_it(self, app, monkeypatch):
        jobs, calls, undone, _ = self._wire_train(monkeypatch, entered=True, start_fails=True)
        assert self._post_train(app, jobs).status_code == 409
        assert undone == []

    def _wire_models(self, monkeypatch):
        calls = []
        monkeypatch.setattr(models, "clients", SimpleNamespace(
            detection=SimpleNamespace(GetStatus=lambda _: SimpleNamespace(is_running=True)),
            model=SimpleNamespace(
                SelectModel=lambda m: calls.append(("select", m.name))
                or SimpleNamespace(success=True, message="/data/models/a.engine"),
                UploadModel=lambda stream: calls.append(("upload", None))
                or SimpleNamespace(json='{"status": "converting"}', http_status=202))))
        monkeypatch.setattr(models, "_publish_if_success", lambda resp, *a, **kw: resp)
        monkeypatch.setattr(models, "_publish_event", lambda *a, **kw: None)
        return calls

    def test_model_select_acquires_the_gpu(self, app, monkeypatch):
        calls = self._wire_models(monkeypatch)
        purposes = _acquired(monkeypatch, models)
        with app.test_request_context("/api/v1/model/select", method="POST",
                                      json={"model_name": "a.engine"}):
            resp = models.select_model()
        assert resp.status_code == 200
        assert purposes == [P.MODEL_SELECT] and calls == [("select", "a.engine")]

    def test_model_select_refused_during_training(self, app, monkeypatch):
        calls = self._wire_models(monkeypatch)
        _acquired(monkeypatch, models,
                  busy=gpu.GpuBusy(gpu.TRAINING, gpu.BUSY_MESSAGES[gpu.TRAINING]))
        with app.test_request_context("/api/v1/model/select", method="POST",
                                      json={"model_name": "a.engine"}):
            resp = models.select_model()
        assert resp.status_code == 409 and resp.get_json()["gpu_busy"] == "training"
        assert calls == []

    def _post_upload(self, app, name, headers=None):
        import io

        with app.test_request_context(
                "/api/v1/model", method="POST",
                data={"file": (io.BytesIO(b"model"), name)},
                content_type="multipart/form-data", headers=headers or {}):
            return models.upload_model()

    @pytest.mark.parametrize("name", ["best.pt", "graph.onnx", "staff.faces"])
    def test_a_converted_upload_names_itself(self, app, monkeypatch, name):
        calls = self._wire_models(monkeypatch)
        purposes = []
        monkeypatch.setattr(gpu, "acquire_gpu",
                            lambda purpose, **kw: purposes.append((purpose, kw)))
        assert self._post_upload(app, name).status_code == 202
        assert purposes == [(P.CONVERSION, {"upload_name": name, "upload_job": None})]
        assert calls == [("upload", None)]

    @pytest.mark.parametrize("hub_verified,expected", [(False, "job-1"), (True, None)])
    def test_only_an_internal_upload_names_its_training_job(self, app, monkeypatch,
                                                            hub_verified, expected):
        # The training-service sends the header on the internal network; a
        # hub-relayed request cannot claim a job's result exemption.
        self._wire_models(monkeypatch)
        monkeypatch.setattr(models, "_hub_verified", lambda: hub_verified)
        purposes = []
        monkeypatch.setattr(gpu, "acquire_gpu",
                            lambda purpose, **kw: purposes.append(kw.get("upload_job")))
        resp = self._post_upload(app, "Parts.pt", {gpu.TRAINING_JOB_HEADER: "job-1"})
        assert resp.status_code == 202
        assert purposes == [expected]

    @pytest.mark.parametrize("name", ["prebuilt.engine", "PREBUILT.PLAN"])
    def test_a_prebuilt_engine_upload_is_an_activation(self, app, monkeypatch, name):
        self._wire_models(monkeypatch)
        purposes = _acquired(monkeypatch, models)
        assert self._post_upload(app, name).status_code == 202
        assert purposes == [P.ENGINE_UPLOAD]

    @pytest.mark.parametrize("job", ["preparing", "training"])
    def test_a_prebuilt_engine_upload_during_training_is_refused(self, app, monkeypatch,
                                                                 device, job):
        calls = self._wire_models(monkeypatch)
        device(job=job)
        resp = self._post_upload(app, "prebuilt.engine")
        assert resp.status_code == 409 and resp.get_json()["gpu_busy"] == "training"
        assert calls == []
