# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Builds a face recognition model from an enrollment package.

The training-service packages a ``face`` dataset as ``<name>.faces``: a ZIP
with ``manifest.json`` (``{"format": 1, "classes": [...], "images":
[{"image_id", "class_id", "file"}]}``) and the JPEGs it names. The build runs
as a conversion job on the device, through the same TensorRT runtime and
preprocessing the live pipeline uses, so enrolled and live embeddings are
comparable:

1. the shared YuNet and SFace engines are built from the bundled graphs the
   first time (``<models dir>/face/``) and reused afterwards;
2. every photo goes through YuNet; its largest face is aligned and embedded
   (a photo without a face is skipped and counted);
3. the gallery, the class names and a copy of the detector engine are written
   as ``<name>.gallery.npz``, ``<name>.txt`` and ``<name>.engine`` — the
   engine last, since it is what makes the model appear in the model list.
"""
import contextlib
import copy
import json
import logging
import os
import shutil
import struct
import threading
import zipfile
from dataclasses import dataclass, field
from typing import Any, Callable, ContextManager, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from conecsa_common.tasks import is_reserved_face_name, is_safe_class_name, person_key

from ..postprocess import _face_assets, _face_gallery, _yunet
from ..postprocess._face_align import align_crop
from ..postprocess._face_embedder import FaceEmbedder, close_worker
from ..postprocess.contract import ContractError, check
from ..postprocess.face import gallery_file_for_model
from ..views.detection_boxes import nms_indices

logger = logging.getLogger(__name__)

PACKAGE_FORMAT = 1
MANIFEST = "manifest.json"
#: A face must score above this to be enrolled (stricter than the live gate:
#: an enrollment photo is expected to show one clear face).
ENROLL_SCORE_THRESHOLD = 0.6
#: A photo is ambiguous — and skipped — when its second-largest face reaches
#: this share of the largest one's area: the name would be a coin toss.
AMBIGUOUS_AREA_RATIO = 0.65
#: IoU that collapses one face's neighbouring anchors (OpenCV's YuNet default).
ENROLL_NMS_IOU = 0.3
#: Upper bounds that keep a crafted package from exhausting the device.
MAX_IMAGES = 10_000
MAX_IMAGE_BYTES = 16 * 1024 * 1024
#: ``cv2.imdecode`` allocates the whole raster before anything can check it, and
#: a highly compressible photo passes the byte cap at any resolution — a
#: 60k×60k PNG is ~10 GB. Dimensions come from the header first, as
#: ``training-service``'s dataset import already does.
MAX_IMAGE_PIXELS = 64_000_000

#: One gallery build at a time: every build drives the same two worker ports
#: and writes the same shared engine files, so two uploads arriving together
#: would race on both.
_BUILD_LOCK = threading.Lock()

Progress = Callable[[int, str], None]
#: Wraps the publication of a model, by engine filename: what the model service
#: holds its lifecycle lock in, reloading the model when it is the active one.
PublishGuard = Callable[[str], ContextManager[Any]]


class PackageError(ValueError):
    """The enrollment package is malformed."""


@dataclass
class PackageImage:
    image_id: str
    class_id: int
    file: str


@dataclass
class BuildSummary:
    people: int
    embedded: int
    skipped: Dict[str, int] = field(default_factory=dict)

    def message(self) -> str:
        text = f"Face gallery built: {self.embedded} photos of {self.people} people"
        missing = sum(self.skipped.values())
        if missing:
            text += f"; {missing} photos without a clear face were skipped"
        return text


#: Appended to each output path while a build writes it (``_write_model``).
STAGING_SUFFIX = ".tmp"
#: Appended to a previous model's files while a build replaces them.
BACKUP_SUFFIX = ".bak"


def model_outputs(engine_out: str) -> Tuple[str, str, str, str]:
    """The files a gallery build publishes, in order: gallery, class names,
    settings sidecar (task ``face``), engine."""
    # Lazy: the model service imports the service graph.
    from .model_service import ModelService

    return (gallery_file_for_model(engine_out),
            ModelService.classes_file_for_model(engine_out),
            ModelService.settings_file_for_model(engine_out), engine_out)


def discard_staging(engine_out: str) -> None:
    """Remove whatever a build staged or backed up for ``engine_out`` and did not publish."""
    for path in model_outputs(engine_out):
        for suffix in (STAGING_SUFFIX, BACKUP_SUFFIX):
            try:
                os.remove(f"{path}{suffix}")
            except FileNotFoundError:
                pass
            except OSError as exc:
                logger.warning("Could not remove %s%s: %s", path, suffix, exc)


def publish_outputs(outputs: Sequence[str]) -> None:
    """Move each ``<path>.tmp`` into place; a failure part-way restores the
    previous set, so a model of the same name is replaced whole or not at all.

    Power loss in the middle is out of reach here (three files, no
    transactional rename); the gallery's ``labels_sha256`` stamp makes the
    strategy refuse such a torn set at activation instead of misnaming faces.
    """
    backups = []
    try:
        for path in outputs:
            if os.path.exists(path):
                shutil.copyfile(path, f"{path}{BACKUP_SUFFIX}")
                backups.append(path)
        published = []
        try:
            for path in outputs:
                os.replace(f"{path}{STAGING_SUFFIX}", path)
                published.append(path)
        except OSError:
            for path in published:
                if path in backups:
                    os.replace(f"{path}{BACKUP_SUFFIX}", path)
                else:
                    os.remove(path)
            raise
    finally:
        for path in backups:
            try:
                os.remove(f"{path}{BACKUP_SUFFIX}")
            except FileNotFoundError:
                pass


def read_manifest(archive: zipfile.ZipFile) -> Tuple[List[str], List[PackageImage]]:
    """Validate and parse a package's manifest."""
    try:
        manifest = json.loads(archive.read(MANIFEST))
    except KeyError:
        raise PackageError("the package has no manifest.json") from None
    except ValueError as exc:
        raise PackageError(f"manifest.json is not valid JSON: {exc}") from None
    if not isinstance(manifest, dict) or manifest.get("format") != PACKAGE_FORMAT:
        raise PackageError(f"unsupported package format {manifest.get('format')!r}"
                           if isinstance(manifest, dict) else "manifest.json is not an object")
    classes = manifest.get("classes")
    images = manifest.get("images")
    if not isinstance(classes, list) or not classes:
        raise PackageError("the package has no people")
    # The same rules a face dataset applies to a person's name: a package can
    # be uploaded directly, so the .txt sidecar and the gallery must never
    # carry a name the dataset layer would have refused.
    if not all(is_safe_class_name(c) for c in classes):
        raise PackageError("a person's name is empty, too long or has characters that are "
                           "not letters, digits, space, '_', '-', '.' or a colour suffix")
    if any(not person_key(c) for c in classes):
        raise PackageError("a person's name cannot be a colour alone")
    if any(is_reserved_face_name(c) for c in classes):
        raise PackageError("'unknown' is reserved for a face nobody matches, not a person")
    if len({person_key(c) for c in classes}) != len(classes):
        raise PackageError("two people share a name (letter case and colour do not tell "
                           "them apart)")
    if not isinstance(images, list) or not images:
        raise PackageError("the package has no photos")
    if len(images) > MAX_IMAGES:
        raise PackageError(f"the package has more than {MAX_IMAGES} photos")
    names = set(archive.namelist())
    parsed = []
    files_seen = set()
    for entry in images:
        if not isinstance(entry, dict):
            raise PackageError("a manifest image entry is not an object")
        class_id, file = entry.get("class_id"), entry.get("file")
        if not (isinstance(class_id, int) and not isinstance(class_id, bool)
                and 0 <= class_id < len(classes)):
            raise PackageError(f"image {entry.get('image_id')!r} has an invalid class")
        if not isinstance(file, str) or file not in names:
            raise PackageError(f"image {entry.get('image_id')!r} is missing from the package")
        if file in files_seen:
            # One photo under two people would enrol one face as both.
            raise PackageError(f"photo {file!r} is listed more than once")
        files_seen.add(file)
        parsed.append(PackageImage(str(entry.get("image_id", "")), class_id, file))
    return [c.strip() for c in classes], parsed


def image_dimensions(payload: bytes) -> Optional[Tuple[int, int]]:
    """``(width, height)`` of a PNG/JPEG/BMP from its header, without decoding.

    ``None`` when the header is unreadable — the decode then decides, bounded
    by ``MAX_IMAGE_BYTES``.
    """
    try:
        if payload[:8] == b"\x89PNG\r\n\x1a\n":
            width, height = struct.unpack(">II", payload[16:24])
            return int(width), int(height)
        if payload[:2] == b"BM":
            width, height = struct.unpack("<ii", payload[18:26])
            return abs(int(width)), abs(int(height))
        if payload[:2] == b"\xff\xd8":
            i = 2
            while i + 9 < len(payload):
                if payload[i] != 0xFF:
                    i += 1
                    continue
                marker = payload[i + 1]
                if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
                    i += 2
                    continue
                length = struct.unpack(">H", payload[i + 2:i + 4])[0]
                if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
                    height, width = struct.unpack(">HH", payload[i + 5:i + 9])
                    return int(width), int(height)
                i += 2 + length
    except (struct.error, IndexError):
        return None
    return None


def _detector_config(config: Any, engine: str) -> Any:
    cfg = copy.copy(config)
    cfg.MODEL_PATH = engine
    return cfg


class FaceGalleryBuilder:
    """One gallery build (the conversion job's worker thread drives it)."""

    def __init__(self, config: Any, model_directory: str,
                 build_engine: Callable[[str, str], None], progress: Progress,
                 publish_guard: Optional[PublishGuard] = None):
        self._config = config
        self._directory = model_directory
        self._build_engine = build_engine
        self._progress = progress
        self._publish_guard: PublishGuard = publish_guard or (
            lambda _engine: contextlib.nullcontext())

    def ensure_engines(self) -> Tuple[str, str]:
        """The shared detector and embedder engines, built when missing."""
        paths = []
        for i, name in enumerate((_face_assets.DETECTOR_ONNX, _face_assets.EMBEDDER_ONNX)):
            engine = _face_assets.engine_path(self._directory, name)
            if not os.path.isfile(engine):
                self._progress(5 + 10 * i, "Building the face engines (first gallery only; "
                                           "this may take several minutes)…")
                os.makedirs(os.path.dirname(engine), exist_ok=True)
                self._build_engine(_face_assets.onnx_path(name), engine)
            paths.append(engine)
        return paths[0], paths[1]

    def build(self, package_path: str, engine_out: str) -> BuildSummary:
        """Enroll every photo of ``package_path`` into the model ``engine_out``.

        Serialized: the workers and the shared engines are one per device.
        """
        with _BUILD_LOCK:
            return self._build(package_path, engine_out)

    def _build(self, package_path: str, engine_out: str) -> BuildSummary:
        if not _face_assets.assets_available():
            raise RuntimeError("this build does not include the face recognition models")
        with zipfile.ZipFile(package_path) as archive:
            classes, images = read_manifest(archive)
            detector_engine, embedder_engine = self.ensure_engines()
            embeddings, labels, image_ids, skipped = self._enroll(
                archive, classes, images, detector_engine, embedder_engine)
        if not embeddings:
            raise RuntimeError("no photo shows a clear face; add photos where the face is "
                               "large, frontal and well lit")
        gallery = _face_gallery.Gallery(
            embeddings=np.stack(embeddings), labels=np.asarray(labels, np.int32),
            names=classes, image_ids=image_ids,
            embedder_sha256=_face_assets.embedder_sha256())
        # Only the publication runs inside the guard: enrolling takes minutes
        # and must not hold the model lifecycle up.
        with self._publish_guard(os.path.basename(engine_out)):
            self._write_model(gallery, detector_engine, engine_out)
        return BuildSummary(people=len(set(labels)), embedded=len(embeddings), skipped=skipped)

    def _enroll(self, archive: zipfile.ZipFile, classes: List[str], images: List[PackageImage],
                detector_engine: str, embedder_engine: str):
        # Lazy: ModelManager pulls in the runtime registry.
        from ..model_manager import ModelManager

        detector_port, embedder_port = _face_assets.build_worker_ports()
        embeddings: List[np.ndarray] = []
        labels: List[int] = []
        image_ids: List[str] = []
        skipped: Dict[str, int] = {}
        try:
            detector = ModelManager(_detector_config(self._config, detector_engine),
                                    port=detector_port, task="face")
            check("face", [d["shape"] for d in detector.output_details],
                  detector.input_details[0]["shape"])
            try:
                indices = _yunet.output_indices(detector.output_details)
            except KeyError as exc:
                raise ContractError(f"the face detector has no '{exc.args[0]}' output") from None
            embedder = FaceEmbedder(self._config, embedder_engine, embedder_port)
            for n, image in enumerate(images):
                if n % 10 == 0:
                    self._progress(20 + int(75 * n / len(images)),
                                   f"Enrolling photo {n + 1} of {len(images)}…")
                vector = self._embed_one(archive, image, detector, indices, embedder)
                if vector is None:
                    name = classes[image.class_id]
                    skipped[name] = skipped.get(name, 0) + 1
                    continue
                embeddings.append(vector)
                labels.append(image.class_id)
                image_ids.append(image.image_id)
        finally:
            close_worker(detector_port)
            close_worker(embedder_port)
        return embeddings, labels, image_ids, skipped

    @staticmethod
    def _embed_one(archive: zipfile.ZipFile, image: PackageImage, detector: Any,
                   indices: Dict[str, int], embedder: FaceEmbedder) -> Optional[np.ndarray]:
        info = archive.getinfo(image.file)
        if info.file_size > MAX_IMAGE_BYTES:
            logger.warning("Skipping %s: larger than %d bytes", image.file, MAX_IMAGE_BYTES)
            return None
        payload = archive.read(info)
        dims = image_dimensions(payload)
        if dims is not None and dims[0] * dims[1] > MAX_IMAGE_PIXELS:
            logger.warning("Skipping %s: %dx%d is past the %d pixel limit",
                           image.file, dims[0], dims[1], MAX_IMAGE_PIXELS)
            return None
        frame = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            logger.warning("Skipping %s: not a decodable image", image.file)
            return None
        tensors, metas = detector.preprocess_tiles(frame)
        outputs, _ = detector.run_inference(tensors[0])
        faces = _yunet.to_frame(
            _yunet.decode(outputs, indices, int(metas[0].input_size), ENROLL_SCORE_THRESHOLD),
            metas[0])
        if len(faces.scores) == 0:
            return None
        # Suppress the neighbouring anchors of one face first: they overlap
        # almost exactly, so counting them as separate faces would make every
        # photo look ambiguous. The live strategy runs the same NMS.
        keep = nms_indices(faces.boxes.astype(np.int32), faces.scores, ENROLL_NMS_IOU)
        if keep.size == 0:
            return None
        boxes, landmarks = faces.boxes[keep], faces.landmarks[keep]
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        order = np.argsort(-areas)
        largest = int(order[0])
        # One photo carries one person's name, and the builder can only pick a
        # face by size: when a second face is nearly as big, the name could
        # land on the wrong person, so the photo is skipped instead. A
        # bystander far behind the subject stays harmless.
        if len(order) > 1 and areas[int(order[1])] >= AMBIGUOUS_AREA_RATIO * areas[largest]:
            logger.warning("Skipping %s: two faces of similar size share the photo", image.file)
            return None
        return embedder.embed([align_crop(frame, landmarks[largest])])[0]

    def _write_model(self, gallery: _face_gallery.Gallery, detector_engine: str,
                     engine_out: str) -> None:
        """Publish the gallery, the names and the engine as one set.

        All three are staged (``model_outputs(engine_out)``) and moved into
        place only once every one of them exists, so a failure part-way never
        leaves a gallery without its engine, or replaces one file of an
        existing model of the same name but not the others.
        """
        # Lazy: the repository and model service import the service graph.
        from ..repositories.class_labels_repository import ClassLabelsRepository
        from .model_settings_service import ModelSettingsService

        gallery_out, labels_out, settings_out, engine_final = outputs = model_outputs(engine_out)
        gallery.labels_sha256 = _face_gallery.labels_stamp(gallery.names)
        try:
            _face_gallery.save(f"{gallery_out}{STAGING_SUFFIX}", gallery)
            if not ClassLabelsRepository(f"{labels_out}{STAGING_SUFFIX}").save_labels(
                    list(gallery.names)):
                raise OSError("the class names could not be written")
            # The task must reach the sidecar with the engine, or the model
            # would list and activate as a detection model: a previous
            # model's settings (the operator's thresholds) are carried over.
            staged_settings = f"{settings_out}{STAGING_SUFFIX}"
            if os.path.isfile(settings_out):
                shutil.copyfile(settings_out, staged_settings)
            ModelSettingsService.record_training(staged_settings, None, task="face")
            if ModelSettingsService.task_of(staged_settings) != "face":
                raise OSError("the settings sidecar could not be written")
            shutil.copyfile(detector_engine, f"{engine_final}{STAGING_SUFFIX}")
            publish_outputs(outputs)
        finally:
            discard_staging(engine_out)
