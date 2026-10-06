# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""The shared GPU release: one task every GPU consumer goes through.

The Orin Nano's single GPU (8 GB, shared with the CPU) cannot hold two heavy
workloads at once, so before anything loads onto it the other holders must go.
:func:`acquire_gpu` is the only place that decides it. It refuses in exactly
three states — detection running, a training job in progress
(``ACTIVE_JOB_STATUSES``), a model conversion in progress
(``ACTIVE_CONVERSION_STATUSES``) — and evicts everything else: the SAM3
checkpoint (training-service), the labeling engine, idle TensorRT workers and a
training handover left behind by a closed training page (inference-service
``FreeGpu``).

Each consumer names its :class:`Purpose`; :data:`POLICIES` says which of the
three states block it, whether it frees the whole inference runtime or only
the labeling assistants, and whether an unanswered probe refuses (strict,
503) or counts as idle (fail-open). A refusal is :class:`GpuBusy`, which the
routes answer with :func:`busy_response`.

The checks and the operation they clear must not be split by another GPU
request (a conversion queued between a release and an application switch, a
Start between a release and a SAM3 load, a select between a training's release
and its start). So every GPU consumer route runs inside :func:`gpu_session`
(or :func:`gpu_lock` for one that only resumes detection): a gateway-wide lock
held from the checks through the consumer's own RPC. The gateway is the only
way in for every consumer, and waitress serves it from one process. A request
that cannot get the lock within ``LOCK_WAIT_SEC`` answers 503 instead of
pinning its thread behind a long operation such as a SAM3 cold load.

Evicting is not best-effort: an assistant whose unload fails, or that still
reports loaded afterwards, refuses the acquisition (500), and so does one
whose status cannot be read (503) — except that a fail-open purpose treats a
training-service nobody answers on (``UNAVAILABLE``) as one without SAM3: the
SAM3 worker is a child process of that container and goes down with it.
"""
import enum
import logging
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Generator, Optional

import grpc

from .grpc_clients import clients, inf, trn
from .helpers import _json

logger = logging.getLogger(__name__)

#: Training job statuses that own the GPU (the trainer subprocess is alive).
ACTIVE_JOB_STATUSES = ("preparing", "training", "uploading")
#: Conversion statuses meaning a TensorRT build may hold the GPU.
ACTIVE_CONVERSION_STATUSES = ("pending", "converting_to_onnx", "converting_to_engine")

#: The three states that keep the GPU.
DETECTION = "detection"
TRAINING = "training"
CONVERSION = "conversion"

#: One sentence per blocker, shared by every route.
BUSY_MESSAGES = {
    DETECTION: "Detection is running; stop it first.",
    TRAINING: "A model training is in progress; try again after it finishes.",
    CONVERSION: "A model conversion is in progress; try again after it finishes.",
}
LOCKED_MESSAGE = "Another GPU operation is in progress; try again in a moment."
UNVERIFIABLE_MESSAGE = ("Cannot verify that the GPU is free; try again when the training "
                        "and inference services are reachable.")

#: FreeGpuResult.busy → blocker.
_HOLDERS = {
    inf.GPU_HOLDER_DETECTION: DETECTION,
    inf.GPU_HOLDER_CONVERSION: CONVERSION,
}


class GpuProbeError(RuntimeError):
    """A strict GPU-busy probe could not reach the service it asks."""


class GpuBusy(Exception):
    """The GPU cannot be freed for the caller.

    ``holder`` is one of ``DETECTION``/``TRAINING``/``CONVERSION``, or
    ``None`` when a strict probe could not answer (``status`` 503).
    """

    def __init__(self, holder: Optional[str], message: str, status: int = 409) -> None:
        super().__init__(message)
        self.holder = holder
        self.message = message
        self.status = status


#: Serializes every GPU consumer's checks with its operation (see module doc).
_GPU_LOCK = threading.Lock()
#: How long a GPU request waits for another one to finish before a 503.
LOCK_WAIT_SEC = 15.0


@contextmanager
def gpu_lock(wait: Optional[float] = None) -> Generator[None, None, None]:
    """Hold the gateway-wide GPU lock, or raise :class:`GpuBusy` (503).

    Not reentrant: a route takes it once, around its checks and its RPC.
    """
    if not _GPU_LOCK.acquire(timeout=LOCK_WAIT_SEC if wait is None else wait):
        raise GpuBusy(None, LOCKED_MESSAGE, 503)
    try:
        yield
    finally:
        _GPU_LOCK.release()


@contextmanager
def gpu_session(purpose: "Purpose", *, upload_name: Optional[str] = None,
                upload_job: Optional[str] = None) -> Generator[None, None, None]:
    """:func:`acquire_gpu` and the caller's operation under :func:`gpu_lock`."""
    with gpu_lock():
        acquire_gpu(purpose, upload_name=upload_name, upload_job=upload_job)
        yield


def busy_response(exc: GpuBusy):
    """JSON ``{"error", "gpu_busy"}`` Response for a refusal."""
    return _json({"error": exc.message, "gpu_busy": exc.holder}, exc.status)


class Purpose(enum.Enum):
    """What the GPU is acquired for (one per consumer route)."""

    APPLICATION = "application"    # PUT /api/v1/application
    DETECTION = "detection"        # POST /api/v1/start
    TRAINING = "training"          # POST /api/v1/training/enter, /training/train
    SAM = "sam"                    # POST /api/v1/training/sam/load (and a lazy segment)
    LABEL_MODEL = "label_model"    # POST /api/v1/training/label-model/load
    MODEL_SELECT = "model_select"  # POST /api/v1/model/select
    ENGINE_UPLOAD = "engine_upload"  # POST /api/v1/model of a prebuilt .engine/.plan
    CONVERSION = "conversion"      # POST /api/v1/model of a .pt/.onnx/.faces


@dataclass(frozen=True)
class _Policy:
    #: Which of the three states refuse this purpose.
    blockers: frozenset
    #: ``"free"`` = inference FreeGpu (idle runtime and workers go),
    #: ``"handover"`` = FreeGpu stopping detection and marking the training
    #: handover in the same inference-side operation, ``None`` = only the
    #: labeling assistants go (the caller's own workload adopts the runtime).
    runtime: Optional[str]
    #: The SAM3 checkpoint stays (the caller is about to use it).
    keep_sam: bool = False
    #: An unanswered probe refuses (503) instead of counting as idle.
    strict: bool = False
    #: The training job's own result upload (``<model_name>.pt`` or
    #: ``.faces`` while the job is ``uploading``, sent by the training-service
    #: with the job's id) is not blocked by that job.
    own_result_exempt: bool = False


_ALL = frozenset({DETECTION, TRAINING, CONVERSION})

POLICIES = {
    # The switch deselects models and drops the runtime: nothing may hold the
    # GPU, and an unverifiable GPU is not a free one.
    Purpose.APPLICATION: _Policy(_ALL, "free", strict=True),
    # Detection keeps its own runtime loaded across a stop (a restart must not
    # re-deserialize the engine); the start itself ends a handover. The
    # probes fail open: a training-service that is down is not training.
    Purpose.DETECTION: _Policy(frozenset({TRAINING, CONVERSION}), None),
    # Entering training mode is a confirmed stop; a duplicate job is the
    # training-service's refusal, and a re-enter during a run is idempotent.
    Purpose.TRAINING: _Policy(frozenset({CONVERSION}), "handover"),
    Purpose.SAM: _Policy(_ALL, "free", keep_sam=True, strict=True),
    Purpose.LABEL_MODEL: _Policy(_ALL, "free", strict=True),
    # A running detection adopts the selected model: it is the owner, not a
    # blocker.
    Purpose.MODEL_SELECT: _Policy(frozenset({TRAINING, CONVERSION}), None),
    # A prebuilt engine is activated as soon as it lands, exactly like a select.
    Purpose.ENGINE_UPLOAD: _Policy(frozenset({TRAINING, CONVERSION}), None),
    # A conversion queues behind another one and may run beside a live
    # detection, but never on top of a trainer — except the finished
    # training's own upload, sent while its job is still `uploading`.
    Purpose.CONVERSION: _Policy(frozenset({TRAINING}), None, own_result_exempt=True),
}

#: What a training job uploads as its result (``training_service._upload_best``).
_RESULT_SUFFIXES = (".pt", ".faces")
#: The training job a result upload belongs to, set by the training-service on
#: the internal network (nginx clears it; hub-verified requests are ignored).
TRAINING_JOB_HEADER = "X-Conecsa-Training-Job"


def _active_job(strict: bool = False):
    """The training job while it holds the GPU (``ACTIVE_JOB_STATUSES``), else None.

    An unreachable training-service cannot be training, so by default a gRPC
    failure answers None (logged). ``strict`` callers refuse on an unanswered
    probe instead and get :class:`GpuProbeError`.
    """
    try:
        job = clients.training.GetTraining(trn.Empty())
    except grpc.RpcError as exc:
        if strict:
            raise GpuProbeError(f"training job probe failed: {exc.code()}") from exc
        logger.warning("Training job probe failed; assuming idle: %s", exc)
        return None
    return job if job.status in ACTIVE_JOB_STATUSES else None


def training_job_active(strict: bool = False) -> bool:
    """True while a training job holds the GPU (see :func:`_active_job`)."""
    return _active_job(strict) is not None


def _is_own_result(job, upload_name: Optional[str], upload_job: Optional[str]) -> bool:
    """Whether the upload is the job's own result, sent while it uploads.

    The name alone proves nothing (any model upload may pick it), so the
    upload must also carry the job's id (``TRAINING_JOB_HEADER``), which only
    the training-service sends: nginx clears the header on every request it
    relays, and the route ignores it on a hub-verified one.
    """
    return (job.status == "uploading" and bool(job.model_name) and bool(job.job_id)
            and upload_job == job.job_id and upload_name is not None
            and upload_name in {job.model_name + suffix for suffix in _RESULT_SUFFIXES})


def conversion_active(strict: bool = False) -> bool:
    """True while a model conversion (TensorRT build) may hold the GPU.

    Fail-open by default like :func:`training_job_active`; ``strict`` raises
    :class:`GpuProbeError` when the inference-service does not answer.
    """
    try:
        cl = clients.model.ListConversions(inf.Empty())
    except grpc.RpcError as exc:
        if strict:
            raise GpuProbeError(f"conversion probe failed: {exc.code()}") from exc
        logger.warning("Conversion probe failed; assuming idle: %s", exc)
        return False
    return any(j.status in ACTIVE_CONVERSION_STATUSES for j in cl.jobs)


def _code(exc: grpc.RpcError):
    code = getattr(exc, "code", None)
    return code() if callable(code) else None


def detection_running(strict: bool = False) -> bool:
    """True while detection runs. Fail-open like the other probes; ``strict``
    raises :class:`GpuProbeError` when the inference-service does not answer."""
    try:
        return bool(clients.detection.GetStatus(inf.Empty()).is_running)
    except grpc.RpcError as exc:
        if strict:
            raise GpuProbeError(f"detection probe failed: {exc.code()}") from exc
        logger.warning("Detection probe failed; assuming stopped: %s", exc)
        return False


def _unload_sam(strict: bool) -> None:
    """Drop the SAM3 checkpoint, and make sure it is gone.

    Status-probed first (no spurious event from unloading nothing). A failed
    or unconfirmed unload refuses: the worker would keep its memory. A status
    that cannot be read refuses too, unless the purpose is fail-open and
    nothing answers at all (the container is down, its SAM3 worker with it).
    """
    try:
        loaded = clients.training.GetSamStatus(trn.Empty()).loaded
    except grpc.RpcError as exc:
        if not strict and _code(exc) == grpc.StatusCode.UNAVAILABLE:
            logger.warning("training-service unreachable; no SAM3 worker to unload: %s", exc)
            return
        raise GpuBusy(None, "Cannot verify that SAM3 is unloaded; try again when the "
                            "training service answers.", 503) from exc
    if not loaded:
        return
    try:
        r = clients.training.UnloadSam(trn.Empty())
        still_loaded = r.success and clients.training.GetSamStatus(trn.Empty()).loaded
    except grpc.RpcError as exc:
        raise GpuBusy(None, "Could not unload SAM3; try again when the training service "
                            "answers.", 503) from exc
    if not r.success or still_loaded:
        detail = r.message if not r.success else "the worker is still loaded"
        raise GpuBusy(None, f"Could not unload SAM3: {detail}", 500)


def _unload_label_model() -> None:
    """Drop the labeling engine, and make sure it is gone.

    Every caller's next step runs on the inference-service, so a status that
    cannot be read refuses as well (503); a failed or unconfirmed unload 500.
    """
    try:
        if not clients.model.GetLabelModelStatus(inf.Empty()).loaded:
            return
        r = clients.model.UnloadLabelModel(inf.Empty())
        still_loaded = r.success and clients.model.GetLabelModelStatus(inf.Empty()).loaded
    except grpc.RpcError as exc:
        raise GpuBusy(None, "Cannot verify that the labeling engine is unloaded; try again "
                            "when the inference service answers.", 503) from exc
    if not r.success or still_loaded:
        detail = r.message if not r.success else "it is still loaded"
        raise GpuBusy(None, f"Could not unload the labeling engine: {detail}", 500)


def unload_assistants() -> None:
    """Unload SAM3 and the labeling engine, verified; raise :class:`GpuBusy` if not.

    For a caller about to load the detection runtime itself (leaving training
    mode with a resume). Like the fail-open purposes, a training-service
    nobody answers on counts as one without SAM3.
    """
    _unload_sam(strict=False)
    _unload_label_model()


def acquire_gpu(purpose: Purpose, *, upload_name: Optional[str] = None,
                upload_job: Optional[str] = None) -> None:
    """Free the GPU for ``purpose`` or raise :class:`GpuBusy`.

    Order: the gateway-side probes (a training job lives in the
    training-service; a conversion and a running detection are probed here
    as well as inside ``FreeGpu``, which catches a change afterwards), the
    SAM3 unload,
    then the inference-service release its policy asks for — nothing is
    stopped or evicted before every probe passed. The inference-side
    activations (select, prebuilt engine upload, Start) re-check conversions
    under the lock a conversion enqueue takes, so a build that starts after
    these probes is still refused there. Detection is never stopped except for
    ``Purpose.TRAINING`` (entering training mode is a confirmed stop), whose
    release and handover mark are one inference-side operation.
    ``upload_name`` is the uploaded file's name and ``upload_job`` the
    training job it claims to be the result of (``Purpose.CONVERSION``). A
    gRPC failure of the inference release itself propagates as
    ``grpc.RpcError`` for the route's usual mapping.
    """
    policy = POLICIES[purpose]
    try:
        if TRAINING in policy.blockers:
            job = _active_job(strict=policy.strict)
            if job is not None and not (policy.own_result_exempt
                                        and _is_own_result(job, upload_name, upload_job)):
                raise GpuBusy(TRAINING, BUSY_MESSAGES[TRAINING])
        if CONVERSION in policy.blockers and conversion_active(strict=policy.strict):
            raise GpuBusy(CONVERSION, BUSY_MESSAGES[CONVERSION])
        # Detection too, before anything is evicted (FreeGpu re-checks it
        # under the inference lock): a refused request keeps its assistant.
        if DETECTION in policy.blockers and detection_running(strict=policy.strict):
            raise GpuBusy(DETECTION, BUSY_MESSAGES[DETECTION])
    except GpuProbeError as exc:
        logger.warning("GPU acquisition for %s refused: %s", purpose.value, exc)
        raise GpuBusy(None, UNVERIFIABLE_MESSAGE, 503) from exc

    if not policy.keep_sam:
        _unload_sam(policy.strict)

    if policy.runtime is None:
        _unload_label_model()
        return
    handover = policy.runtime == "handover"
    # One inference-side critical section: with a handover, the stop, the
    # release and the mark cannot be split by a start or a conversion.
    r = clients.management.FreeGpu(inf.FreeGpuRequest(stop_detection=handover,
                                                      begin_handover=handover))
    if not r.success:
        holder = _HOLDERS.get(r.busy)
        if holder is None:
            raise GpuBusy(None, r.message or "Could not free the GPU", 500)
        raise GpuBusy(holder, BUSY_MESSAGES[holder])
