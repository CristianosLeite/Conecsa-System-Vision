# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Releasing TensorRT workers never reports a GPU it did not free.

A worker process that survives terminate + kill keeps its CUDA memory, so the
client keeps its handle and says so, and ``release_all_workers`` names every
port it could not free (after trying all of them).
"""
from subprocess import TimeoutExpired

import api.runtime_management.worker_client as wc
import pytest


class UndeadProcess:
    """A worker that ignores terminate and kill (e.g. stuck in the driver)."""

    def __init__(self, returncode=None):
        self.signals = []
        self.returncode = returncode

    def terminate(self):
        self.signals.append("terminate")

    def kill(self):
        self.signals.append("kill")

    def wait(self, timeout=None):
        # An unbounded wait on an unreapable process would never return.
        assert timeout is not None, "unbounded wait on a worker"
        raise TimeoutExpired("worker", timeout)

    def poll(self):
        return self.returncode


def test_an_undead_worker_is_reported_and_kept_without_hanging():
    client = wc.WorkerClient(59001)
    process = UndeadProcess()
    client._process = process  # pyright: ignore[reportAttributeAccessIssue]
    with pytest.raises(RuntimeError, match="59001 did not exit"):
        client._terminate_process(timeout=0.01)
    assert process.signals == ["terminate", "kill"]
    assert client._process is process


def test_an_exiting_worker_is_forgotten(monkeypatch):
    client = wc.WorkerClient(59002)
    client._process = UndeadProcess(returncode=0)  # pyright: ignore[reportAttributeAccessIssue]
    client._terminate_process(timeout=0.01)
    assert client._process is None


def test_release_tries_every_worker_and_names_the_failures(monkeypatch):
    closed = []

    class Client:
        def __init__(self, port, fails):
            self.port, self.fails = port, fails

        def release(self):
            closed.append(self.port)
            if self.fails:
                raise RuntimeError("did not exit")

    monkeypatch.setattr(wc, "_client_cache", {
        1: Client(5501, True), 2: Client(5502, False), 3: Client(5517, True)})
    with pytest.raises(RuntimeError, match="ports 5501, 5517"):
        wc.release_all_workers()
    assert closed == [5501, 5502, 5517]


def test_release_is_silent_when_everything_exits(monkeypatch):
    monkeypatch.setattr(wc, "_client_cache", {})
    wc.release_all_workers()


class ExitingProcess:
    """A worker that exits on terminate."""

    returncode = None

    def terminate(self):
        self.returncode = 0

    def kill(self):
        self.returncode = -9

    def wait(self, timeout=None):
        assert timeout is not None, "unbounded wait on a worker"
        return self.returncode

    def poll(self):
        return self.returncode


def test_a_release_forgets_the_engine_so_the_next_start_loads_nothing():
    client = wc.WorkerClient(59003)
    client._process = ExitingProcess()  # pyright: ignore[reportAttributeAccessIssue]
    client._model_path = "/models/old.engine"
    client.release()
    assert client._process is None
    # A conversion's engine build restarts this worker: the old engine must
    # not come back beside it.
    assert client._model_path is None


def test_a_failed_release_keeps_the_engine_with_the_worker():
    client = wc.WorkerClient(59004)
    client._process = UndeadProcess()  # pyright: ignore[reportAttributeAccessIssue]
    client._model_path = "/models/old.engine"
    with pytest.raises(RuntimeError, match="59004 did not exit"):
        client.release()
    assert client._model_path == "/models/old.engine"


def test_a_transparent_close_keeps_the_engine_for_the_reload():
    client = wc.WorkerClient(59005)
    client._process = ExitingProcess()  # pyright: ignore[reportAttributeAccessIssue]
    client._model_path = "/models/current.engine"
    client.close()
    assert client._model_path == "/models/current.engine"


def test_live_worker_ports_names_only_the_survivors(monkeypatch):
    undead, exited = wc.WorkerClient(5501), wc.WorkerClient(5517)
    undead._process = UndeadProcess()  # pyright: ignore[reportAttributeAccessIssue]
    exited._process = UndeadProcess(returncode=0)  # pyright: ignore[reportAttributeAccessIssue]
    monkeypatch.setattr(wc, "_client_cache", {1: undead, 2: exited, 3: wc.WorkerClient(5502)})
    assert wc.live_worker_ports() == [5501]
