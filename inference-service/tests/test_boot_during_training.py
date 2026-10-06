# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""A boot in the middle of a training loads nothing onto the GPU.

Only this container restarting while the trainer runs must not restore the
model's engine beside it: the boot asks the training-service first, selects
the model without loading it and takes the handover over, so leaving training
mode resumes detection.
"""
from concurrent import futures
from types import SimpleNamespace
from typing import cast

import grpc
import pytest
import training_pb2 as trn
import training_pb2_grpc as trn_grpc
from api.composition import Application
from api.services import training_probe


class _Training(trn_grpc.TrainingControlServicer):
    def __init__(self, status):
        self.status = status

    def GetTraining(self, request, context):
        return trn.TrainingJob(job_id="job-1", status=self.status)


@pytest.fixture
def training_server():
    servers = []

    def start(status):
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
        trn_grpc.add_TrainingControlServicer_to_server(_Training(status), server)
        port = server.add_insecure_port("127.0.0.1:0")
        server.start()
        servers.append(server)
        return f"127.0.0.1:{port}"

    yield start
    for server in servers:
        server.stop(None)


class TestProbe:
    @pytest.mark.parametrize("status", training_probe.ACTIVE_JOB_STATUSES)
    def test_an_active_job_is_reported(self, training_server, status):
        assert training_probe.training_job_active(training_server(status)) is True

    @pytest.mark.parametrize("status", ["idle", "done", "failed", "canceled"])
    def test_a_finished_job_is_not(self, training_server, status):
        assert training_probe.training_job_active(training_server(status)) is False

    def test_a_training_service_that_does_not_answer_is_no_job(self):
        # A whole-device boot: the training-service is still starting.
        assert training_probe.training_job_active("127.0.0.1:1", timeout=0.5) is False


class _Models:
    def __init__(self):
        self.calls = []

    def load_persisted_current_model(self):
        return "parts.engine"

    def select_model(self, name):
        self.calls.append(("select", name))
        return True, name

    def hold_for_training(self):
        self.calls.append("hold")

    def activate_model(self, name):
        self.calls.append(("activate", name))
        return True, name, False


def _boot(training_at_boot):
    """Run ``Application.initialize`` over fakes; the model calls and starts."""
    models, started = _Models(), []
    app = SimpleNamespace(
        _training_at_boot=training_at_boot,
        application_service=SimpleNamespace(task="detect"),
        model_service=models,
        detection_service=SimpleNamespace(start=lambda: started.append(True),
                                          initialize=lambda: started.append("init")),
        video_service=SimpleNamespace(wait_for_camera=lambda timeout: True),
    )
    Application.initialize(cast(Application, app))
    return models.calls, started


def test_a_boot_during_training_selects_without_loading_and_holds_the_handover():
    calls, started = _boot(training_at_boot=True)
    assert calls == [("select", "parts.engine"), "hold"]
    assert started == []


def test_a_normal_boot_restores_and_starts():
    calls, started = _boot(training_at_boot=False)
    assert calls == [("activate", "parts.engine")]
    assert started == [True]
