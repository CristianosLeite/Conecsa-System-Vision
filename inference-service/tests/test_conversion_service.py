# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Unit tests for ConversionService pure helpers and job serialization."""
import time

import pytest
from api.services.conversion_service import (
    ConversionJob,
    ConversionService,
    ConversionStatus,
    _remove_file_safe,
)


class TestConversionStatus:
    def test_string_enum_values(self):
        assert ConversionStatus.PENDING.value == "pending"
        assert ConversionStatus.CONVERTING_TO_ONNX.value == "converting_to_onnx"
        assert ConversionStatus.CONVERTING_TO_ENGINE.value == "converting_to_engine"
        assert ConversionStatus.DONE.value == "done"
        assert ConversionStatus.FAILED.value == "failed"

    def test_is_str_subclass(self):
        assert ConversionStatus.DONE == "done"


class TestConversionJobDefaults:
    def test_defaults(self):
        job = ConversionJob(
            job_id="j1",
            original_filename="m.pt",
            pt_path="/m.pt",
            onnx_path="/m.onnx",
            engine_path="/m.engine",
        )
        assert job.status is ConversionStatus.PENDING
        assert job.progress == 0
        assert job.error is None
        assert job.engine_filename is None
        assert job.imgsz is None
        assert isinstance(job.started_at, float)
        assert isinstance(job.started_monotonic, float)
        assert job.elapsed_secs >= 0.0


class TestToDict:
    def test_serializes_status_value(self):
        job = ConversionJob("j1", "m.pt", "/m.pt", "/m.onnx", "/m.engine")
        job.status = ConversionStatus.CONVERTING_TO_ENGINE
        job.progress = 45
        d = ConversionService.to_dict(job)
        assert d["job_id"] == "j1"
        assert d["status"] == "converting_to_engine"  # .value, not the enum
        assert d["progress"] == 45
        assert set(d.keys()) == {
            "job_id",
            "original_filename",
            "status",
            "progress",
            "message",
            "error",
            "engine_filename",
            "imgsz",
            "train_geometry",
            "task",
            "started_at",
            "elapsed_secs",
        }

    def test_imgsz_is_serialized(self):
        pt_job = ConversionJob("j1", "m.pt", "/m.pt", "/m.onnx", "/m.engine", imgsz=1280)
        assert ConversionService.to_dict(pt_job)["imgsz"] == 1280
        # .onnx -> .engine-only jobs have no export size of their own.
        onnx_job = ConversionJob("j2", "m.onnx", "", "/m.onnx", "/m.engine")
        assert ConversionService.to_dict(onnx_job)["imgsz"] is None


class TestElapsedSecs:
    def test_is_measured_from_the_monotonic_clock(self):
        # The device has no RTC battery, so the hub steps CLOCK_REALTIME
        # whenever it notices drift — mid-conversion included. A job's age must
        # come from the monotonic clock, so a `started_at` that jumps (here to
        # the epoch) cannot move it.
        job = ConversionJob("j1", "m.pt", "/m.pt", "/m.onnx", "/m.engine")
        job.started_monotonic = time.monotonic() - 30.0
        job.started_at = 0.0

        assert job.elapsed_secs == pytest.approx(30.0, abs=1.0)
        assert ConversionService.to_dict(job)["elapsed_secs"] == pytest.approx(30.0, abs=1.0)

    def test_never_negative(self):
        job = ConversionJob("j1", "m.pt", "/m.pt", "/m.onnx", "/m.engine")
        job.started_monotonic = time.monotonic() + 100.0
        assert job.elapsed_secs == 0.0


class TestRemoveFileSafe:
    def test_removes_existing(self, tmp_path):
        f = tmp_path / "x.onnx"
        f.write_text("data")
        _remove_file_safe(str(f))
        assert not f.exists()

    def test_missing_file_is_noop(self, tmp_path):
        # Should not raise.
        _remove_file_safe(str(tmp_path / "absent.onnx"))


class TestStartPtConversion:
    def test_records_imgsz_on_the_job(self, monkeypatch, tmp_path):
        svc = ConversionService()
        # Keep the worker thread from touching torch/TensorRT.
        monkeypatch.setattr(ConversionService, "_run_job", lambda self, job_id: None)
        job = svc.start_pt_conversion(str(tmp_path / "m.pt"), "m.pt", str(tmp_path), imgsz=1280)
        assert job.imgsz == 1280
        assert svc.get_job(job.job_id) is job

    def test_records_the_declared_training_geometry(self, monkeypatch, tmp_path):
        svc = ConversionService()
        monkeypatch.setattr(ConversionService, "_run_job", lambda self, job_id: None)
        job = svc.start_pt_conversion(str(tmp_path / "m.pt"), "m.pt", str(tmp_path),
                                      imgsz=640, train_geometry="tiles:auto")
        assert job.train_geometry == "tiles:auto"
        assert svc.to_dict(job)["train_geometry"] == "tiles:auto"
        plain = svc.start_pt_conversion(str(tmp_path / "n.pt"), "n.pt", str(tmp_path))
        assert plain.train_geometry is None

    def test_onnx_only_job_has_no_imgsz(self, monkeypatch, tmp_path):
        svc = ConversionService()
        monkeypatch.setattr(ConversionService, "_run_job", lambda self, job_id: None)
        job = svc.start_onnx_conversion(str(tmp_path / "m.onnx"), "m.onnx", str(tmp_path))
        assert job.imgsz is None


class TestJobRegistry:
    def test_get_unknown_job(self):
        svc = ConversionService()
        assert svc.get_job("nope") is None

    def test_active_jobs_excludes_terminal(self):
        svc = ConversionService()
        active = ConversionJob("a", "a.pt", "", "", "")
        done = ConversionJob("b", "b.pt", "", "", "")
        done.status = ConversionStatus.DONE
        failed = ConversionJob("c", "c.pt", "", "", "")
        failed.status = ConversionStatus.FAILED
        svc._jobs = {"a": active, "b": done, "c": failed}
        ids = {j.job_id for j in svc.get_active_jobs()}
        assert ids == {"a"}


def _run_inline(monkeypatch, tmp_path, *, engine_ok=True, exported=None, task=None,
                onnx_shapes=None, onnx=False, built=None, from_checkpoint=False,
                exports=None):
    """Run one conversion job body inline with the converter and builder faked."""
    import api.services.conversion_service as cs

    def export(pt, onnx_path, imgsz, from_checkpoint=False):
        if exports is not None:
            exports.append((imgsz, from_checkpoint))
        open(onnx_path, "w").close()
        return exported if exported is not None else cs.ConverterOutput(
            [], "detect", [[1, 300, 6]])

    def build(onnx_path, engine):
        if built is not None:
            built.append(engine)
        if not engine_ok:
            raise RuntimeError("no workspace")
        open(engine, "w").close()

    monkeypatch.setattr(cs, "_convert_pt_to_onnx", export)
    monkeypatch.setattr(cs, "_build_engine_from_onnx", build)
    monkeypatch.setattr(cs, "_inspect_onnx", lambda path: onnx_shapes)
    # Enqueue without the worker thread, then run the job body inline.
    run_job = ConversionService._run_job
    monkeypatch.setattr(ConversionService, "_run_job", lambda self, job_id: None)
    svc = ConversionService()
    if onnx:
        src = tmp_path / "Teste.onnx"
        src.write_bytes(b"onnx")
        job = svc.start_onnx_conversion(str(src), "Teste.onnx", str(tmp_path), task=task)
    else:
        src = tmp_path / "Teste.pt"
        src.write_bytes(b"pt")
        job = svc.start_pt_conversion(str(src), "Teste.pt", str(tmp_path), imgsz=640,
                                      task=task, imgsz_from_checkpoint=from_checkpoint)
    run_job(svc, job.job_id)
    done = svc.get_job(job.job_id)
    assert done is not None
    return done


def _sidecar(tmp_path):
    import json
    return json.loads((tmp_path / "Teste.settings.json").read_text())


class TestRunJobKeepsTheCheckpoint:
    """The .pt a conversion starts from becomes the model's weights sidecar."""

    def _run(self, monkeypatch, tmp_path, *, engine_ok=True):
        return _run_inline(monkeypatch, tmp_path, engine_ok=engine_ok)

    def test_pt_moves_into_the_weights_sidecar(self, monkeypatch, tmp_path):
        job = self._run(monkeypatch, tmp_path)
        assert job.status == ConversionStatus.DONE, job.error
        assert (tmp_path / "weights" / "Teste.pt").read_bytes() == b"pt"
        assert not (tmp_path / "Teste.pt").exists(), "no phantom .pt beside the engine"
        assert not (tmp_path / "Teste.onnx").exists()

    def test_failed_job_drops_the_pt(self, monkeypatch, tmp_path):
        job = self._run(monkeypatch, tmp_path, engine_ok=False)
        assert job.status == ConversionStatus.FAILED
        assert not (tmp_path / "Teste.pt").exists()
        assert not (tmp_path / "weights").exists()


class TestRunJobImgsz:
    """An upload that names no size exports at the checkpoint's training size."""

    def test_the_converter_is_asked_for_the_checkpoint_size(self, monkeypatch, tmp_path):
        exports = []
        job = _run_inline(monkeypatch, tmp_path, from_checkpoint=True, exports=exports)
        assert job.status == ConversionStatus.DONE, job.error
        assert exports == [(640, True)], "640 is only the fallback"

    def test_the_size_the_export_used_is_recorded(self, monkeypatch, tmp_path):
        import api.services.conversion_service as cs
        job = _run_inline(monkeypatch, tmp_path, task="classify", from_checkpoint=True,
                          exported=cs.ConverterOutput(["a", "b"], "classify", [[1, 2]], 320))
        assert job.status == ConversionStatus.DONE, job.error
        assert job.imgsz == 320
        assert _sidecar(tmp_path) == {"imgsz": 320, "task": "classify"}

    def test_a_named_size_is_not_replaced(self, monkeypatch, tmp_path):
        exports = []
        _run_inline(monkeypatch, tmp_path, exports=exports)
        assert exports == [(640, False)]


class TestRunJobTask:
    """Declare at upload, verify at conversion."""

    def test_the_declared_task_is_recorded_with_the_export_size(self, monkeypatch, tmp_path):
        job = _run_inline(monkeypatch, tmp_path, task="detect")
        assert job.status == ConversionStatus.DONE, job.error
        assert _sidecar(tmp_path) == {"imgsz": 640, "task": "detect"}

    def test_an_undeclared_task_records_what_the_converter_found(self, monkeypatch, tmp_path):
        job = _run_inline(monkeypatch, tmp_path, task=None)
        assert job.status == ConversionStatus.DONE, job.error
        assert _sidecar(tmp_path)["task"] == "detect"

    def test_a_model_of_another_task_fails_before_the_engine_build(self, monkeypatch, tmp_path):
        import api.services.conversion_service as cs
        built = []
        job = _run_inline(monkeypatch, tmp_path, task="detect", built=built,
                          exported=cs.ConverterOutput(["a", "b"], "classify", [[1, 2]]))
        assert job.status == ConversionStatus.FAILED
        assert job.error is not None
        assert "'classify'" in job.error and "'detect'" in job.error
        assert built == []
        assert not (tmp_path / "Teste.settings.json").exists()

    def test_the_graph_shape_decides_when_ultralytics_does_not_say(self, monkeypatch, tmp_path):
        import api.services.conversion_service as cs
        job = _run_inline(monkeypatch, tmp_path, task="detect",
                          exported=cs.ConverterOutput([], None, [[1, 32, 160, 160],
                                                                 [1, 300, 38]]))
        assert job.status == ConversionStatus.FAILED
        assert job.error is not None and "'segment'" in job.error

    def test_onnx_upload_is_inspected_and_its_task_recorded(self, monkeypatch, tmp_path):
        job = _run_inline(monkeypatch, tmp_path, onnx=True, task="detect",
                          onnx_shapes=[[1, 300, 6]])
        assert job.status == ConversionStatus.DONE, job.error
        assert _sidecar(tmp_path) == {"task": "detect"}

    def test_onnx_of_another_task_fails(self, monkeypatch, tmp_path):
        job = _run_inline(monkeypatch, tmp_path, onnx=True, task="detect",
                          onnx_shapes=[[1, 4]])
        assert job.status == ConversionStatus.FAILED
        assert job.error is not None and "'classify'" in job.error

    def test_uninspectable_onnx_is_left_to_activation(self, monkeypatch, tmp_path):
        job = _run_inline(monkeypatch, tmp_path, onnx=True, task="detect", onnx_shapes=None)
        assert job.status == ConversionStatus.DONE, job.error
        assert _sidecar(tmp_path) == {"task": "detect"}
