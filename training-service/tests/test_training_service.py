# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Unit tests for TrainingService job-state handling and start() validation.

The trainer subprocess itself is not exercised: ``_run`` is stubbed out so
``start()`` stops at the state transition it owns (validation, single-job
lock, dataset freezing).
"""
from types import SimpleNamespace

import pytest
from service.dataset_service import DatasetError
from service.training_service import TrainingService, build_trainer_argv


class FakeDataset:
    def __init__(self, geometry="tiles:auto", task="detect"):
        self.frozen = False
        self.validated = 0
        self.split_calls = []
        self.geometry = geometry
        self._task = task

    def task(self):
        return self._task

    def validate_for_training(self):
        self.validated += 1

    def build_split(self, job_id, **kwargs):
        from service.dataset_service import SplitResult
        from service.tile_split import TileSplitStats
        self.split_calls.append(kwargs)
        return SplitResult(f"/runs/{job_id}/dataset/data.yaml", 4, 2, self.geometry,
                           TileSplitStats())


class FakeWeightsStore:
    """Records what a federated round stashes for the hub to collect."""

    def __init__(self, tasks=None):
        self.stashed = []
        self.stashed_tasks = []
        self.tasks = dict(tasks or {})

    def stash_file(self, src_path, task=""):
        self.stashed.append(src_path)
        self.stashed_tasks.append(task)
        return "w-fed-1"

    def path(self, weights_id):
        return f"/data/training/weights/{weights_id}.pt"

    def task_of(self, weights_id):
        return self.tasks.get(weights_id, "")


class FakeRegistry:
    def __init__(self, dataset):
        self._dataset = dataset

    def get(self, dataset_id):
        return self._dataset

    def freeze(self, dataset_id):
        from service.dataset_service import DatasetError
        if self._dataset.frozen:
            raise DatasetError("Dataset is locked while a training job is running")
        self._dataset.frozen = True
        return self._dataset

    def release(self, ds):
        ds.frozen = False


@pytest.fixture
def config(tmp_path):
    return SimpleNamespace(
        DEFAULT_EPOCHS=50,
        TRAIN_BATCH=4,
        DEFAULT_PATIENCE=10,
        BASE_WEIGHTS="/assets/yolo26s.pt",
        BASE_WEIGHTS_CLS="/assets/yolo26s-cls.pt",
        IMG_SIZE=640,
        IMG_SIZE_CLS=224,
        TRAIN_WORKERS=0,
        TRAIN_AMP=True,
        TRAIN_OVERRIDES="",
        TRAIN_TILE="auto",
        TRAIN_TILE_OVERLAP=0.2,
        TRAIN_TILE_MIN_VISIBLE=0.25,
        TRAIN_TIMEOUT_SEC=0,
        TRAIN_STALL_TIMEOUT_SEC=3600,
        GATEWAY_ADDR="http://gateway.test:5000",
        runs_dir=str(tmp_path / "runs"),
        base_dir=str(tmp_path / "base"),
    )


@pytest.fixture
def dataset():
    return FakeDataset()


@pytest.fixture
def svc(config, dataset, monkeypatch):
    # Keep start() from launching the real trainer subprocess.
    monkeypatch.setattr(TrainingService, "_run", lambda self, *a, **k: None)
    return TrainingService(config, FakeRegistry(dataset))  # pyright: ignore[reportArgumentType]


class TestInitialState:
    def test_starts_idle_and_inactive(self, svc):
        assert svc.get_job().status == "idle"
        assert svc.is_active() is False

    def test_get_job_returns_a_copy(self, svc):
        job = svc.get_job()
        job.status = "training"
        assert svc.get_job().status == "idle"

    def test_cancel_and_finish_early_require_a_running_job(self, svc):
        assert svc.cancel() is False
        assert svc.finish_early() is False


class TestStartValidation:
    def test_epochs_out_of_range_is_rejected(self, svc):
        with pytest.raises(DatasetError, match="Epochs"):
            svc.start("d1", "my-model", epochs=1001)

    def test_invalid_model_name_is_rejected(self, svc):
        with pytest.raises(DatasetError):
            svc.start("d1", "")

    def test_initial_weights_require_a_weights_store(self, svc):
        with pytest.raises(DatasetError, match="Weights store"):
            svc.start("d1", "my-model", initial_weights_id="w1")

    def test_base_model_and_initial_weights_are_exclusive(self, svc):
        with pytest.raises(DatasetError, match="mutually exclusive"):
            svc.start("d1", "my-model", initial_weights_id="w1", base_model="a.pt")

    def test_base_model_must_be_a_valid_model_name(self, svc, monkeypatch):
        import service.training_service as ts
        monkeypatch.setattr(ts, "list_models_with_weights",
                            lambda gw: pytest.fail("not reached"))
        with pytest.raises(DatasetError, match="Invalid model name"):
            svc.start("d1", "my-model", base_model="../Teste.engine")

    def test_base_model_must_exist_on_the_device(self, svc, dataset, monkeypatch):
        import service.training_service as ts
        monkeypatch.setattr(ts, "list_models_with_weights", lambda gw: {"Other.engine": "detect"})
        with pytest.raises(DatasetError, match="no training checkpoint"):
            svc.start("d1", "my-model", base_model="Teste.engine")
        assert dataset.frozen is False

    def test_unreachable_gateway_fails_the_rpc(self, svc, dataset, monkeypatch):
        import service.training_service as ts

        def boom(gw):
            raise ts.requests.ConnectionError("down")

        monkeypatch.setattr(ts, "list_models_with_weights", boom)
        with pytest.raises(DatasetError, match="Could not list"):
            svc.start("d1", "my-model", base_model="Teste.engine")
        assert dataset.frozen is False


class TestStart:
    def test_successful_start_prepares_the_job(self, svc, dataset, config):
        job = svc.start("d1", "my-model")
        assert job.status == "preparing"
        assert job.model_name == "my-model"
        assert job.dataset_id == "d1"
        assert job.total_epochs == config.DEFAULT_EPOCHS
        assert job.federated is False
        assert svc.is_active() is True
        assert dataset.validated == 1
        assert dataset.frozen is True

    def test_only_one_job_at_a_time(self, svc):
        svc.start("d1", "my-model")
        with pytest.raises(DatasetError, match="already running"):
            svc.start("d1", "other-model")

    def test_federated_blank_name_becomes_a_label(self, svc):
        job = svc.start("d1", "", federated=True)
        assert job.model_name == "federated"
        assert job.federated is True

    def test_explicit_epochs_override_the_default(self, svc):
        assert svc.start("d1", "my-model", epochs=7).total_epochs == 7

    def test_base_model_is_recorded_on_the_job(self, svc, monkeypatch):
        import service.training_service as ts
        monkeypatch.setattr(ts, "list_models_with_weights", lambda gw: {"Teste.engine": "detect"})
        job = svc.start("d1", "Teste", base_model=" Teste.engine ")
        assert job.base_model == "Teste.engine"

    def test_start_unloads_sam(self, config, dataset, monkeypatch):
        monkeypatch.setattr(TrainingService, "_run", lambda self, *a, **k: None)
        unloaded = []
        sam = SimpleNamespace(unload=lambda: unloaded.append("sam"))
        svc = TrainingService(config, FakeRegistry(dataset),  # pyright: ignore[reportArgumentType]
                              sam_service=sam)
        svc.start("d1", "my-model")
        assert unloaded == ["sam"]


class TestTrainerArgv:
    def test_default_argv_matches_the_640_contract(self, config):
        cmd = build_trainer_argv(config, "/d/data.yaml", "/assets/yolo26s.pt",
                                 epochs=50, patience=10, batch=4, job_id="job1")
        assert cmd[1:3] == ["-m", "service._yolo_trainer"]
        assert cmd[cmd.index("--imgsz") + 1] == "640"
        assert "--override" not in cmd
        assert "--no-amp" not in cmd

    def test_imgsz_and_overrides_are_forwarded(self, config):
        config.IMG_SIZE = 1280
        config.TRAIN_AMP = False
        config.TRAIN_OVERRIDES = "freeze=10 lr0=0.002"
        cmd = build_trainer_argv(config, "/d/data.yaml", "/w.pt",
                                 epochs=20, patience=10, batch=4, job_id="job1")
        assert cmd[cmd.index("--imgsz") + 1] == "1280"
        assert "--no-amp" in cmd
        pairs = [cmd[i + 1] for i, tok in enumerate(cmd) if tok == "--override"]
        assert pairs == ["freeze=10", "lr0=0.002"]

    def test_start_rejects_invalid_overrides_before_freezing(self, svc, config, dataset):
        config.TRAIN_OVERRIDES = "imgsz=1280"
        with pytest.raises(DatasetError, match="TRAIN_OVERRIDES"):
            svc.start("ds", "model-a")
        assert dataset.frozen is False


class FakeProcess:
    """A trainer that prints one ``done`` line and exits 0."""

    def __init__(self, best="/runs/job/weights/best.pt", last=None):
        import json
        payload = {"done": True, "best": best}
        if last:
            payload["last"] = last
        self.stdout = iter([json.dumps(payload) + "\n"])
        self.stderr = iter([])
        self.returncode = 0
        self.pid = 4242

    def poll(self):
        return 0

    def wait(self):
        return 0


class TestRun:
    """``_run`` with the subprocess, upload and event plumbing faked."""

    @pytest.fixture
    def running(self, config, dataset, monkeypatch):
        import service.training_service as ts
        monkeypatch.setattr(ts.subprocess, "Popen", lambda *a, **k: FakeProcess())
        uploads = []
        monkeypatch.setattr(TrainingService, "_upload_best",
                            lambda self, best: uploads.append(best) or "conv-1")
        svc = TrainingService(config, FakeRegistry(dataset))  # pyright: ignore[reportArgumentType]
        # Drive the job through start() so _job_dataset and the job state are real.
        monkeypatch.setattr(TrainingService, "_run", lambda self, *a, **k: None)
        svc.start("d1", "my-model")
        monkeypatch.undo()  # restore the real _run (Popen/upload stay patched below)
        monkeypatch.setattr(ts.subprocess, "Popen", lambda *a, **k: FakeProcess())
        monkeypatch.setattr(TrainingService, "_upload_best",
                            lambda self, best: uploads.append(best) or "conv-1")
        return svc, uploads

    def test_split_uses_the_config_tile_knobs_and_records_the_geometry(
            self, running, dataset, config):
        svc, uploads = running
        job = svc.get_job()
        svc._run(job.job_id, epochs=1, batch=4, patience=10, weights_path="/w.pt")
        assert dataset.split_calls == [{"tile": "auto", "overlap": 0.2, "min_visible": 0.25}]
        done = svc.get_job()
        assert done.status == "done" and done.geometry == "tiles:auto"
        assert uploads == ["/runs/job/weights/best.pt"]

    def test_federated_round_trains_on_the_tile_split_and_stashes_last_pt(
            self, config, dataset, monkeypatch):
        # A federated round is the regular job on a shard: it must go through
        # the same tile split as a local job (the hub averages what every
        # device produced at that geometry) and hand last.pt to the weights
        # stash instead of the model-upload route.
        import service.training_service as ts
        store = FakeWeightsStore()
        svc = TrainingService(config, FakeRegistry(dataset),  # pyright: ignore[reportArgumentType]
                              weights_store=store)  # pyright: ignore[reportArgumentType]
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(TrainingService, "_run", lambda self, *a, **k: None)
            svc.start("d1", "", federated=True)
        uploads = []
        monkeypatch.setattr(ts.subprocess, "Popen",
                            lambda *a, **k: FakeProcess(last="/runs/job/weights/last.pt"))
        monkeypatch.setattr(TrainingService, "_upload_best",
                            lambda self, best: uploads.append(best) or "conv-1")

        job = svc.get_job()
        svc._run(job.job_id, epochs=1, batch=4, patience=10, weights_path="/w.pt")

        assert dataset.split_calls == [{"tile": "auto", "overlap": 0.2, "min_visible": 0.25}]
        done = svc.get_job()
        assert done.status == "done" and done.federated is True
        assert done.geometry == "tiles:auto"
        assert done.result_weights_id == "w-fed-1"
        assert store.stashed == ["/runs/job/weights/last.pt"]
        # The stash records what the checkpoint was trained for.
        assert store.stashed_tasks == ["detect"]
        assert uploads == [], "a federated round never uploads a model itself"
        assert dataset.frozen is False

    def test_base_model_is_fetched_before_the_split_and_used_as_weights(
            self, running, dataset, config, monkeypatch):
        import service.training_service as ts
        svc, _ = running
        argv = []
        monkeypatch.setattr(ts.subprocess, "Popen",
                            lambda cmd, **k: argv.append(cmd) or FakeProcess())
        fetched = []
        monkeypatch.setattr(ts, "fetch_weights",
                            lambda gw, name, dest: fetched.append((gw, name, dest))
                            or "/base/Teste.pt")
        svc._run(svc.get_job().job_id, epochs=1, batch=4, patience=10,
                 weights_path="/assets/yolo26s.pt", base_model="Teste.engine")
        assert fetched == [("http://gateway.test:5000", "Teste.engine", config.base_dir)]
        assert argv[0][argv[0].index("--weights") + 1] == "/base/Teste.pt"
        assert svc.get_job().status == "done"

    def test_a_failed_base_fetch_fails_the_job(self, running, dataset, monkeypatch):
        import service.training_service as ts

        def boom(gw, name, dest):
            raise DatasetError("Model 'Teste.engine' has no training checkpoint on the device")

        monkeypatch.setattr(ts, "fetch_weights", boom)
        svc, uploads = running
        svc._run(svc.get_job().job_id, epochs=1, batch=4, patience=10,
                 weights_path="/w.pt", base_model="Teste.engine")
        job = svc.get_job()
        assert job.status == "failed" and "no training checkpoint" in job.error
        assert uploads == [] and dataset.frozen is False

    def test_tile_off_is_forwarded(self, running, dataset, config):
        svc, _ = running
        config.TRAIN_TILE = None
        svc._run(svc.get_job().job_id, epochs=1, batch=4, patience=10, weights_path="/w.pt")
        assert dataset.split_calls[0]["tile"] is None

    def test_the_materialised_split_is_removed_after_the_run(self, running, config, tmp_path):
        svc, _ = running
        job_id = svc.get_job().job_id
        scratch = tmp_path / "runs" / job_id / "dataset" / "train" / "images"
        scratch.mkdir(parents=True)
        (scratch / "a_t0.jpg").write_bytes(b"jpg")
        (tmp_path / "runs" / job_id / "weights").mkdir(parents=True)
        (tmp_path / "runs" / job_id / "weights" / "best.pt").write_bytes(b"pt")
        svc._run(job_id, epochs=1, batch=4, patience=10, weights_path="/w.pt")
        assert not (tmp_path / "runs" / job_id / "dataset").exists()
        assert (tmp_path / "runs" / job_id / "weights" / "best.pt").exists()


class TestUploadBest:
    def test_posts_imgsz_and_the_effective_geometry(self, config, dataset, monkeypatch, tmp_path):
        import service.training_service as ts
        posted = {}

        def fake_post(url, files, data, timeout):
            posted.update(url=url, filename=files["file"][0], data=data)
            return SimpleNamespace(status_code=202, json=lambda: {"job_id": "conv-7"})

        monkeypatch.setattr(ts.requests, "post", fake_post)
        svc = TrainingService(config, FakeRegistry(dataset))  # pyright: ignore[reportArgumentType]
        svc._job.model_name = "my-model"
        svc._job.geometry = "tiles:auto"
        best = tmp_path / "best.pt"
        best.write_bytes(b"pt")
        assert svc._upload_best(str(best)) == "conv-7"
        assert posted["url"] == "http://gateway.test:5000/api/v1/model"
        assert posted["filename"] == "my-model.pt"
        # A job that predates the task field uploads as detection.
        assert posted["data"] == {"imgsz": "640", "train_geometry": "tiles:auto",
                                  "task": "detect"}


class TestClassification:
    """A classify dataset trains YOLO26s-cls on class folders at 224."""

    @staticmethod
    def _svc(config, dataset, **kwargs):
        return TrainingService(config, FakeRegistry(dataset),  # pyright: ignore[reportArgumentType]
                               **kwargs)

    def test_the_task_picks_the_weights_and_the_input_size(self, config):
        from service.config import base_weights_for, img_size_for
        assert base_weights_for(config, "classify") == "/assets/yolo26s-cls.pt"
        assert base_weights_for(config, "detect") == "/assets/yolo26s.pt"
        assert base_weights_for(config, "segment").endswith("/yolo26s-seg.pt")
        assert img_size_for(config, "segment") == img_size_for(config, "detect")
        assert (img_size_for(config, "classify"), img_size_for(config, "detect")) == (224, 640)

    def test_the_job_starts_from_the_classification_weights(self, config, monkeypatch):
        import threading
        started = threading.Event()
        weights = []

        def fake_run(self, job_id, epochs, batch, patience, weights_path, **kwargs):
            weights.append(weights_path)
            started.set()

        monkeypatch.setattr(TrainingService, "_run", fake_run)
        job = self._svc(config, FakeDataset(task="classify")).start("d1", "pets")
        assert started.wait(2.0)
        assert job.task == "classify"
        assert weights == ["/assets/yolo26s-cls.pt"]

    def test_a_base_model_of_another_task_is_refused(self, config, monkeypatch):
        import service.training_service as ts
        monkeypatch.setattr(ts, "list_models_with_weights",
                            lambda gw: {"Teste.engine": "detect"})
        dataset = FakeDataset(task="classify")
        with pytest.raises(DatasetError, match="'detect' model"):
            self._svc(config, dataset).start("d1", "pets", base_model="Teste.engine")
        assert dataset.frozen is False

    def test_initial_weights_of_another_task_are_refused(self, config):
        dataset = FakeDataset(task="classify")
        svc = self._svc(config, dataset, weights_store=FakeWeightsStore({"w1": "detect"}))
        with pytest.raises(DatasetError, match="trained for 'detect'"):
            svc.start("d1", "", federated=True, initial_weights_id="w1")
        assert dataset.frozen is False

    def test_initial_weights_of_an_unknown_task_are_accepted(self, config, monkeypatch):
        monkeypatch.setattr(TrainingService, "_run", lambda self, *a, **k: None)
        svc = self._svc(config, FakeDataset(task="classify"), weights_store=FakeWeightsStore())
        assert svc.start("d1", "", federated=True, initial_weights_id="w1").status == "preparing"

    def test_the_run_trains_whole_images_at_224(self, config, monkeypatch):
        import service.training_service as ts
        dataset = FakeDataset(geometry="frames", task="classify")
        svc = self._svc(config, dataset)
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(TrainingService, "_run", lambda self, *a, **k: None)
            svc.start("d1", "pets")
        argv, uploads = [], []
        monkeypatch.setattr(ts.subprocess, "Popen",
                            lambda cmd, **k: argv.append(cmd) or FakeProcess())
        monkeypatch.setattr(TrainingService, "_upload_best",
                            lambda self, best: uploads.append(best) or "conv-1")
        svc._run(svc.get_job().job_id, epochs=1, batch=4, patience=10,
                 weights_path="/assets/yolo26s-cls.pt")
        assert dataset.split_calls[0]["tile"] is None
        assert argv[0][argv[0].index("--imgsz") + 1] == "224"
        done = svc.get_job()
        assert done.status == "done" and done.geometry == "frames" and uploads

    def test_the_upload_declares_the_task_and_its_size(self, config, dataset, monkeypatch,
                                                       tmp_path):
        import service.training_service as ts
        posted = {}

        def fake_post(url, files, data, timeout):
            posted.update(data=data)
            return SimpleNamespace(status_code=202, json=lambda: {"job_id": "conv-8"})

        monkeypatch.setattr(ts.requests, "post", fake_post)
        svc = self._svc(config, dataset)
        svc._job.model_name = "pets"
        svc._job.geometry = "frames"
        svc._job.task = "classify"
        best = tmp_path / "best.pt"
        best.write_bytes(b"pt")
        assert svc._upload_best(str(best)) == "conv-8"
        assert posted["data"] == {"imgsz": "224", "train_geometry": "frames",
                                  "task": "classify"}


class FakeFaceDataset(FakeDataset):
    """A face dataset: no split, one enrollment package per job."""

    def __init__(self, tmp_path):
        super().__init__(geometry="", task="face")
        self._tmp_path = tmp_path
        self.packages = []

    def build_face_package(self, job_id, model_name):
        path = self._tmp_path / f"{model_name}.faces"
        path.write_bytes(b"faces")
        self.packages.append((job_id, model_name))
        return str(path), 3


class TestFace:
    """A face dataset builds a gallery: no trainer, a .faces upload instead."""

    @staticmethod
    def _svc(config, dataset, **kwargs):
        return TrainingService(config, FakeRegistry(dataset),  # pyright: ignore[reportArgumentType]
                               **kwargs)

    def test_the_face_input_size_is_the_yunet_square(self, config):
        from service.config import img_size_for
        assert img_size_for(config, "face") == 640

    def test_federated_base_model_and_initial_weights_are_refused(self, config, tmp_path):
        dataset = FakeFaceDataset(tmp_path)
        store = FakeWeightsStore()
        with pytest.raises(DatasetError, match="federated"):
            self._svc(config, dataset, weights_store=store).start("d1", "door",
                                                                  federated=True)
        with pytest.raises(DatasetError, match="base model"):
            self._svc(config, dataset).start("d1", "door", base_model="Teste.engine")
        with pytest.raises(DatasetError, match="initial weights"):
            self._svc(config, dataset, weights_store=store).start(
                "d1", "door", initial_weights_id="w1")
        assert dataset.frozen is False

    def test_start_never_resolves_base_weights(self, config, tmp_path, monkeypatch):
        import threading

        import service.training_service as ts
        monkeypatch.setattr(ts, "base_weights_for",
                            lambda *a, **k: pytest.fail("base weights resolved for face"))
        started = threading.Event()
        weights = []

        def fake_run(self, job_id, epochs, batch, patience, weights_path, **kwargs):
            weights.append(weights_path)
            started.set()

        monkeypatch.setattr(TrainingService, "_run", fake_run)
        job = self._svc(config, FakeFaceDataset(tmp_path)).start("d1", "door")
        assert started.wait(2.0)
        assert job.task == "face" and job.total_epochs == 0
        assert "photos" in job.message and weights == [""]

    def test_the_job_packages_the_photos_and_uploads_them(self, config, tmp_path, monkeypatch):
        import service.training_service as ts
        monkeypatch.setattr(ts.subprocess, "Popen",
                            lambda *a, **k: pytest.fail("a face job runs no trainer"))
        posted = {}

        def fake_post(url, files, data, timeout):
            posted.update(url=url, filename=files["file"][0], data=data,
                          body=files["file"][1].read())
            return SimpleNamespace(status_code=202, json=lambda: {"job_id": "conv-face"})

        monkeypatch.setattr(ts.requests, "post", fake_post)
        dataset = FakeFaceDataset(tmp_path)
        svc = self._svc(config, dataset)
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(TrainingService, "_run", lambda self, *a, **k: None)
            svc.start("d1", "door")
        job_id = svc.get_job().job_id
        svc._run(job_id, epochs=0, batch=4, patience=10, weights_path="")

        assert dataset.packages == [(job_id, "door")] and dataset.split_calls == []
        assert posted["url"] == "http://gateway.test:5000/api/v1/model"
        assert posted["filename"] == "door.faces"
        assert posted["data"] == {"task": "face"}
        assert posted["body"] == b"faces"
        done = svc.get_job()
        assert done.status == "done" and done.progress == 100
        assert done.conversion_job_id == "conv-face"
        # The package is job scratch and the dataset is released again.
        assert not (tmp_path / "door.faces").exists()
        assert dataset.frozen is False

    def test_a_packaging_failure_fails_the_job(self, config, tmp_path, monkeypatch):
        dataset = FakeFaceDataset(tmp_path)

        def boom(job_id, model_name):
            raise DatasetError("Assign at least one photo to a person")

        svc = self._svc(config, dataset)
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(TrainingService, "_run", lambda self, *a, **k: None)
            svc.start("d1", "door")
        monkeypatch.setattr(dataset, "build_face_package", boom)
        svc._run(svc.get_job().job_id, epochs=0, batch=4, patience=10, weights_path="")
        done = svc.get_job()
        assert done.status == "failed" and "at least one photo" in done.error
