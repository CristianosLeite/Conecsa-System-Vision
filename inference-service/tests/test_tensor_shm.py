# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Shared-memory tensor transport: the layout, the worker's
load/infer handlers over a real memfd, and the client's shared-buffer path with
its fallbacks — the worker driven in process through a fake connection."""
import os
from types import SimpleNamespace

import numpy as np
import pytest
from api.runtime_management import tensor_shm
from api.runtime_management import worker_client as wc
from api.runtime_management import worker_server as ws

needs_memfd = pytest.mark.skipif(not hasattr(os, "memfd_create"), reason="memfd_create unavailable")

INPUT = [{"name": "images", "index": 0, "shape": [1, 3, 8, 8], "dtype": np.float32}]
OUTPUTS = [{"name": "output0", "index": 0, "shape": [1, 5, 6], "dtype": np.float32},
           {"name": "output1", "index": 1, "shape": [1, 2, 4, 4], "dtype": np.float32}]


class FakeInterpreter:
    """Outputs derived from the input, so a round trip proves the data moved."""

    def __init__(self, outputs=OUTPUTS):
        self._outputs_details = outputs
        self._out: list[np.ndarray] = [np.zeros(d["shape"], np.dtype(d["dtype"])) for d in outputs]
        self.closed = False

    def get_input_details(self):
        return INPUT

    def get_output_details(self):
        return self._outputs_details

    def set_tensor(self, index, value):
        self._in = np.array(value, dtype=np.float32).reshape(INPUT[0]["shape"])

    def invoke(self):
        total = float(self._in.sum())
        for k, out in enumerate(self._out):
            out[...] = total + k

    def get_tensor(self, index):
        return self._out[index]

    def close(self):
        self.closed = True


class FakeConn:
    def __init__(self):
        self.sent = []

    def send(self, message):
        self.sent.append(message)


@pytest.fixture
def fd():
    handle = tensor_shm.create_fd("test-tensor-shm")
    if handle is None:
        pytest.skip("memfd_create unavailable")
    yield handle
    os.close(handle)


def _load(monkeypatch, shm, interpreter=None):
    interpreter = interpreter or FakeInterpreter()
    monkeypatch.setattr(ws, "_create_runtime",
                        lambda: SimpleNamespace(create_interpreter=lambda path: interpreter))
    conn = FakeConn()
    _, loaded = ws._handle_load_command(conn, {"model_path": "m.engine"}, None, None, shm)
    return conn.sent[-1], loaded


class TestLayout:
    def test_slots_are_aligned_and_the_size_is_whole_pages(self):
        layout = tensor_shm.layout_for(INPUT, OUTPUTS)
        assert layout is not None
        slots = [layout["input"], *layout["outputs"]]
        assert [s["nbytes"] for s in slots] == [768, 120, 128]
        assert all(s["offset"] % 64 == 0 for s in slots)
        assert slots[1]["offset"] >= slots[0]["offset"] + 768
        assert layout["size"] % tensor_shm.mmap.PAGESIZE == 0
        assert layout["input"]["dtype"] == np.dtype(np.float32).str

    def test_a_dynamic_shape_has_no_layout(self):
        dynamic = [{**OUTPUTS[0], "shape": [1, -1, 6]}]
        assert tensor_shm.layout_for(INPUT, dynamic) is None
        assert tensor_shm.layout_for([], OUTPUTS) is None

    @needs_memfd
    def test_two_mappings_of_one_file_share_the_tensors(self, fd):
        layout = tensor_shm.layout_for(INPUT, OUTPUTS)
        assert layout is not None
        tensor_shm.size_fd(fd, layout["size"])
        worker, client = tensor_shm.TensorBuffer(fd, layout), tensor_shm.TensorBuffer(fd, layout)
        client.input[...] = 3.0
        assert float(worker.input.sum()) == 3.0 * 192
        worker.outputs[1][...] = 7.0
        assert float(client.outputs[1].max()) == 7.0
        assert client.fits_input(np.zeros((1, 3, 8, 8), np.float32))
        assert not client.fits_input(np.zeros((1, 3, 8, 8), np.float64))
        worker.close()
        client.close()


class TestWorkerHandlers:
    def test_load_announces_the_layout_and_infer_fills_the_slots(self, fd, monkeypatch):
        shm = SimpleNamespace(fd=fd, buffer=None)
        answer, interpreter = _load(monkeypatch, shm)
        assert answer["status"] == "ok" and answer["shm_layout"] is not None
        client = tensor_shm.TensorBuffer(fd, answer["shm_layout"])
        client.input[...] = 1.0

        conn = FakeConn()
        ws._handle_infer_shm_command(conn, interpreter, shm)
        assert conn.sent == [{"status": "ok", "shm": True}]
        assert float(client.outputs[0][0, 0, 0]) == 192.0
        assert float(client.outputs[1][0, 0, 0, 0]) == 193.0
        client.close()

    def test_without_a_shared_file_the_load_has_no_layout(self, monkeypatch):
        answer, _ = _load(monkeypatch, SimpleNamespace(fd=None, buffer=None))
        assert answer["status"] == "ok" and answer["shm_layout"] is None

    def test_an_output_that_does_not_fit_its_slot_travels_in_the_answer(self, fd, monkeypatch):
        shm = SimpleNamespace(fd=fd, buffer=None)
        interpreter = FakeInterpreter()
        _, loaded = _load(monkeypatch, shm, interpreter)
        interpreter._out[0] = np.zeros((1, 7, 6), np.float32)  # the engine answered another shape
        conn = FakeConn()
        ws._handle_infer_shm_command(conn, loaded, shm)
        (answer,) = conn.sent
        assert answer["status"] == "ok" and answer["shm"] is False
        assert [o["shape"] for o in answer["outputs"]] == [[1, 7, 6], [1, 2, 4, 4]]

    def test_infer_shm_without_a_buffer_is_an_error(self):
        conn = FakeConn()
        ws._handle_infer_shm_command(conn, FakeInterpreter(), SimpleNamespace(fd=None, buffer=None))
        assert conn.sent[0]["status"] == "error"

    def test_the_file_never_shrinks_on_a_smaller_engine(self, fd, monkeypatch):
        shm = SimpleNamespace(fd=fd, buffer=None)
        tensor_shm.size_fd(fd, 1 << 20)
        _load(monkeypatch, shm)
        assert os.fstat(fd).st_size >= 1 << 20


class InProcessWorker:
    """Answers a WorkerClient's requests with the real handlers, no subprocess."""

    def __init__(self, fd, monkeypatch, interpreter=None):
        self.interpreter = interpreter or FakeInterpreter()
        monkeypatch.setattr(ws, "_create_runtime",
                            lambda: SimpleNamespace(create_interpreter=lambda p: self.interpreter))
        self.shm = SimpleNamespace(fd=fd, buffer=None)
        self.loaded = None
        self.commands = []
        self.fail_shm = False

    def request(self, payload, retry_on_fail=True, timeout=None):
        self.commands.append(payload["cmd"])
        conn = FakeConn()
        if payload["cmd"] == "load":
            _, self.loaded = ws._handle_load_command(conn, payload, None, self.loaded, self.shm)
        elif payload["cmd"] == "infer_shm":
            if self.fail_shm:
                raise TimeoutError("worker timeout")
            ws._handle_infer_shm_command(conn, self.loaded, self.shm)
        elif payload["cmd"] == "infer":
            ws._handle_infer_command(conn, payload, self.loaded)
        return conn.sent[-1]


@pytest.fixture
def client(fd, monkeypatch):
    client = wc.WorkerClient(59_997)
    client._conn = object()  # connected: no subprocess in tests
    client._shm_fd = os.dup(fd)
    worker = InProcessWorker(fd, monkeypatch)
    monkeypatch.setattr(client, "_request", worker.request)
    client.load_model("m.engine")
    yield client, worker
    client._unmap_shm()
    if client._shm_fd is not None:
        os.close(client._shm_fd)


class TestClient:
    def test_inference_goes_through_the_shared_buffer(self, client):
        client, worker = client
        outputs = client.infer(np.full((1, 3, 8, 8), 2.0, np.float32))
        assert worker.commands == ["load", "infer_shm"]
        assert [o.shape for o in outputs] == [(1, 5, 6), (1, 2, 4, 4)]
        assert float(outputs[0].max()) == 384.0 and float(outputs[1].max()) == 385.0

    def test_an_input_that_does_not_fit_goes_over_the_connection(self, client):
        client, worker = client
        outputs = client.infer(np.full((1, 3, 8, 8), 2.0, np.float64))
        assert worker.commands == ["load", "infer"]
        assert float(outputs[0].max()) == 384.0

    def test_a_failed_shared_request_falls_back_to_the_connection(self, client):
        client, worker = client
        worker.fail_shm = True
        outputs = client.infer(np.full((1, 3, 8, 8), 1.0, np.float32))
        assert worker.commands == ["load", "infer_shm", "infer"]
        assert float(outputs[0].max()) == 192.0

    def test_without_a_layout_everything_goes_over_the_connection(self, client):
        client, worker = client
        client._map_shm(None)
        client.infer(np.zeros((1, 3, 8, 8), np.float32))
        assert worker.commands == ["load", "infer"]

    def test_terminating_the_worker_closes_its_shared_file(self, client, monkeypatch):
        client, _ = client
        monkeypatch.setattr(client, "_wait_for_port_release", lambda: None)
        fd = client._shm_fd
        client._terminate_process()
        assert client._shm is None and client._shm_fd is None
        with pytest.raises(OSError):
            os.fstat(fd)
