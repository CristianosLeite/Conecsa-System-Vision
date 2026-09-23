# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""One working dataset: images, YOLO labels, classes and metadata.

On-disk layout (one instance per dataset, under the training-data volume):

    {DATA_DIR}/datasets/{dataset_id}/images/{uuid}.jpg    JPEG in the dataset's geometry (below)
    {DATA_DIR}/datasets/{dataset_id}/labels/{uuid}.txt    YOLO rows "class cx cy w h"; for a segmentation dataset
                                                          one "class x1 y1 … xn yn" row per polygon ring; for a
                                                          classification dataset one line "class"
                                                          (absent/empty = unlabeled)
    {DATA_DIR}/datasets/{dataset_id}/classes.json         ["cap", ...]
    {DATA_DIR}/datasets/{dataset_id}/meta.json            {"name", "created_at", "cover_image_id", "geometry", "task"}

``meta.json["task"]`` is the task the dataset is labeled for ("detect",
"classify", "segment", "face"), fixed at creation from the device's
application; a legacy meta.json without it means "detect" and is backfilled on
the next save, like the geometry. The task decides which label kind the
dataset accepts (``check_label_kinds``): boxes for detection, polygons for
segmentation, one image class for classification and for face recognition
(the class is the person a photo shows; ``uses_image_class``). A face dataset
is never split for training: ``build_face_package`` packages its labeled
photos for the gallery build on the inference-service.

``meta.json["geometry"]`` records how the images were stored — either
``{"letterbox": 640}`` (every image letterboxed to that square, the historical
format) or ``{"native": true}`` (the stereo-combined frame at its own
resolution). Label coordinates are always normalized to the stored image. A
legacy meta.json without the key means ``{"letterbox": 640}``; the key is
backfilled the next time the meta is saved. Capture, ZIP import and hub
ingest all produce the geometry selected by ``Config.DATASET_IMG_SIZE``, and
``check_geometry`` refuses to add images to a dataset recorded with a
different one, so a dataset never mixes formats.

Instances are created and cached by the DatasetRegistry. ``build_split``
materializes the ultralytics train/valid layout for one training job plus
the data.yaml, mirroring the Roboflow export format of the reference
dataset. By default every image is sliced into the square tile crops the
inference-service runs on (``service.tile_split``, geometry from
``conecsa_common.tiling``: tile side = the image's short side unless
``TRAIN_TILE`` pins pixels) with the labels rewritten per tile, because a
model only performs at the scale it was trained at; images the grid cannot
slice (a legacy 640×640 letterboxed dataset, a square image) and every image
when tiling is off are symlinked whole (same volume).
"""
import json
import logging
import os
import random
import threading
import time
import uuid
import zipfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

from conecsa_common import atomic_write_bytes, atomic_write_json, read_json
from conecsa_common.tasks import (
    CLASSIFY,
    FACE,
    is_reserved_face_name,
    person_key,
    task_or_default,
)

from .tile_split import TileSpec, TileSplitStats, materialize_tiles

logger = logging.getLogger(__name__)

GEOMETRY_FRAMES = "frames"

#: The one label kind each dataset task accepts.
_LABEL_KIND = {"detect": "boxes", "segment": "polygons", "classify": "an image class",
               "face": "an image class"}

#: Entry holding the class list and the photo index of a face package.
FACE_MANIFEST = "manifest.json"
#: ``manifest.json["format"]`` understood by the inference-service's gallery builder.
FACE_PACKAGE_FORMAT = 1
#: Extension of a face enrollment package (``<model name>.faces``).
FACE_PACKAGE_SUFFIX = ".faces"
#: Why a face dataset is never exported, downloaded or transferred.
FACE_EXPORT_REFUSED = "Face datasets stay on the device: the enrolment photos are biometric data"


def uses_image_class(task: str) -> bool:
    """True for a task whose images carry one image class (no shapes).

    Classification labels each image with its class; face recognition labels
    each photo with the person it shows. Both store the one-line label file
    and share every image-class code path; only classification builds a
    training split.
    """
    return task in (CLASSIFY, FACE)


def geometry_label(tile: TileSpec, stats: TileSplitStats) -> str:
    """The training geometry a split actually produced, as the upload declares it.

    ``"frames"`` when no crop was written (tiling off, or every image kept
    whole), else ``"tiles:auto"`` / ``"tiles:<px>"`` — the vocabulary the
    inference-service records in the model's settings sidecar and checks
    against ``TILING_MODE`` on activation.
    """
    if tile is None or stats.tiles == 0:
        return GEOMETRY_FRAMES
    return f"tiles:{tile}"

# Reject path tricks and characters that break ultralytics/data.yaml parsing.
_NAME_SAFE = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 _-.")
# Class names may additionally carry the "name #rrggbb" color suffix. '#' is
# safe here because data.yaml emits names as single-quoted YAML scalars, where
# '#' does not start a comment.
_CLASS_NAME_SAFE = _NAME_SAFE | {"#"}


@dataclass
class Box:
    """One YOLO label box: class id + normalized center/size (cx, cy, w, h)."""

    class_id: int
    cx: float
    cy: float
    w: float
    h: float


@dataclass
class NamedBox:
    """One pre-label carried by class name, already in letterbox YOLO space."""

    class_name: str
    cx: float
    cy: float
    w: float
    h: float


@dataclass
class Polygon:
    """One segmentation label ring: class id + normalized ``[[x, y], …]`` vertices.

    ``instance`` groups the rings of one object while a label travels; the
    label file has one row per ring and no instance column, so rings read
    back from disk get one instance each.
    """

    class_id: int
    points: List[List[float]]
    instance: int = 0


@dataclass
class NamedPolygon:
    """One segmentation pre-label ring by class name, in the stored image's space."""

    class_name: str
    points: List[List[float]]
    instance: int = 0


def points_to_pairs(flat) -> List[List[float]]:
    """A flat ``x1, y1, x2, y2, …`` list as ``[[x, y], …]`` (at least 3 vertices)."""
    values = [float(v) for v in flat]
    if len(values) % 2 or len(values) < 6:
        raise DatasetError("A polygon needs at least 3 x, y vertex pairs")
    return [[values[i], values[i + 1]] for i in range(0, len(values), 2)]


def normalize_polygons(polygons: List, width: int, height: int) -> List:
    """Normalize label rings at the stored image size, instance by instance.

    Rings of one instance (same ``instance`` and class) are rasterized
    together, so overlapping rings merge and a ring the rasterization splits
    yields several; each surviving ring keeps its instance and class and the
    input type (:class:`Polygon` or :class:`NamedPolygon`). Degenerate or
    too-small instances disappear.
    """
    from conecsa_common.polygons import normalize_rings

    groups: Dict[Tuple[int, Union[int, str]], List[List[List[float]]]] = {}
    for p in polygons:
        label = p.class_id if isinstance(p, Polygon) else p.class_name
        groups.setdefault((p.instance, label), []).append(p.points)
    out: List = []
    for (instance, label), rings in groups.items():
        for ring in normalize_rings(rings, width, height):
            out.append(Polygon(label, ring, instance) if isinstance(label, int)
                       else NamedPolygon(label, ring, instance))
    return out


def check_polygon_points(points: List[List[float]]) -> None:
    """Refuse a ring with fewer than 3 vertices or coordinates outside 0..1."""
    if len(points) < 3:
        raise DatasetError("A polygon needs at least 3 vertices")
    for point in points:
        if len(point) != 2 or not all(0.0 <= float(v) <= 1.0 for v in point):
            raise DatasetError("Polygon coordinates must be normalized (0..1)")


@dataclass
class SplitResult:
    """What ``build_split`` produced for one job."""

    yaml_path: str            # data.yaml, or the split root for classification
                              # (ultralytics trains a classifier on data=<root>)
    train_count: int          # files in train/images (crops or whole images)
    valid_count: int          # files in valid/images
    geometry: str             # "frames" | "tiles:auto" | "tiles:<px>" (effective)
    stats: TileSplitStats


@dataclass
class ImageEntry:
    """A dataset image's metadata: id, capture time, and label state."""

    image_id: str
    created_at: float
    labeled: bool
    box_count: int
    replica: bool = False
    # A classification image's class id (None: unlabeled, or another task).
    image_class: Optional[int] = None


class DatasetError(Exception):
    """Validation error surfaced to the client as Result(success=False)."""


# Geometry recorded by datasets created before the key existed: every image
# letterboxed to the 640×640 square.
LEGACY_GEOMETRY: Dict = {"letterbox": 640}


def geometry_for(dataset_img_size: int) -> Dict:
    """The ``meta.json["geometry"]`` value ``Config.DATASET_IMG_SIZE`` produces.

    ``0`` (or anything non-positive) stores frames at native resolution; any
    positive value letterboxes every image to that square.
    """
    size = int(dataset_img_size)
    return {"native": True} if size <= 0 else {"letterbox": size}


def normalize_geometry(value) -> Dict:
    """Coerce a stored ``geometry`` value into one of the two known shapes.

    Missing/unrecognized values are the legacy 640×640 letterbox format, so such
    a dataset works without a migration step.
    """
    if isinstance(value, dict):
        if value.get("native") is True:
            return {"native": True}
        try:
            size = int(value.get("letterbox", 0))
        except (TypeError, ValueError):
            size = 0
        if size > 0:
            return {"letterbox": size}
    return dict(LEGACY_GEOMETRY)


def describe_geometry(geometry: Dict) -> str:
    """Operator-readable description of a geometry dict (for error messages)."""
    if geometry.get("native") is True:
        return "native-resolution images"
    size = int(geometry["letterbox"])
    return f"{size}×{size} letterboxed images"


def _geometry_env_value(geometry: Dict) -> str:
    """The ``TRAIN_DATASET_IMG_SIZE`` value that reproduces ``geometry``."""
    return "0" if geometry.get("native") is True else str(int(geometry["letterbox"]))


class LabelKindError(DatasetError):
    """A label kind the dataset's task does not use (mapped to INVALID_ARGUMENT)."""


def check_label_kinds(task: str, boxes: int = 0, polygons: int = 0,
                      image_class: Optional[str] = None) -> None:
    """Refuse labels of a kind the dataset's task does not take.

    A detection dataset takes boxes only, a segmentation dataset polygons
    only, a classification dataset one image class only. An image without
    any label is always allowed (background for detect/segment, excluded
    from a classification split).
    """
    offered = []
    if boxes:
        offered.append("boxes")
    if polygons:
        offered.append("polygons")
    if image_class is not None:
        offered.append("an image class")
    accepted = _LABEL_KIND.get(task)
    for kind in offered:
        if kind != accepted:
            raise LabelKindError(
                f"A '{task}' dataset takes {accepted or 'no labels'}, not {kind}")


def validate_dataset_name(name: str) -> str:
    """Validate dataset name."""
    name = (name or "").strip()
    if not name:
        raise DatasetError("Dataset name must not be empty")
    if len(name) > 64 or not all(c in _NAME_SAFE for c in name):
        raise DatasetError("Dataset name has invalid characters")
    return name


class DatasetService:
    """CRUD for one dataset's images, YOLO labels and class list on disk.

    Scoped to a single ``dataset_id`` under ``{root_dir}``; enforces name/id
    validation and refuses mutations once the dataset is frozen.
    """

    def __init__(self, dataset_id: str, root_dir: str, config):
        self.dataset_id = dataset_id
        self._root = root_dir
        self._config = config
        self._lock = threading.RLock()
        # Set by TrainingService while a job runs; dataset mutations are
        # rejected so the symlinked split cannot change under the trainer.
        self.frozen = False
        os.makedirs(self._images_dir, exist_ok=True)
        os.makedirs(self._labels_dir, exist_ok=True)

    # ── paths ─────────────────────────────────────────────────────────────────

    @property
    def _images_dir(self) -> str:
        return os.path.join(self._root, "images")

    @property
    def _labels_dir(self) -> str:
        return os.path.join(self._root, "labels")

    @property
    def _classes_file(self) -> str:
        return os.path.join(self._root, "classes.json")

    @property
    def _meta_file(self) -> str:
        return os.path.join(self._root, "meta.json")

    def _image_path(self, image_id: str) -> str:
        """Image path."""
        self._check_id(image_id)
        return os.path.join(self._images_dir, f"{image_id}.jpg")

    def _label_path(self, image_id: str) -> str:
        """Label path."""
        self._check_id(image_id)
        return os.path.join(self._labels_dir, f"{image_id}.txt")

    @staticmethod
    def _check_id(image_id: str) -> None:
        """Check id."""
        if not image_id or not all(c in "0123456789abcdef-" for c in image_id):
            raise DatasetError(f"Invalid image id '{image_id}'")

    def _check_frozen(self) -> None:
        """Check frozen."""
        if self.frozen:
            raise DatasetError("Dataset is locked while a training job is running")

    # ── images ────────────────────────────────────────────────────────────────

    def add_image(self, jpeg: bytes) -> ImageEntry:
        """Add image."""
        with self._lock:
            self._check_frozen()
            image_id = str(uuid.uuid4())
            with open(self._image_path(image_id), "wb") as f:
                f.write(jpeg)
        return ImageEntry(image_id=image_id, created_at=time.time(),
                          labeled=False, box_count=0)

    def add_labeled_image(self, jpeg: bytes, boxes: List[NamedBox],
                          image_class: Optional[str] = None,
                          polygons: Optional[List[NamedPolygon]] = None) -> ImageEntry:
        """Add an externally captured image with pre-labels by class name.

        ``boxes`` pre-label a detection dataset, ``polygons`` (rings already in
        the stored image's space) a segmentation dataset, ``image_class`` (a
        class name; empty or ``None`` means no class) a classification
        dataset — the dataset's task decides which kind it accepts. Class
        names are resolved against classes.json under the lock (missing names
        are appended), so the name→id mapping stays consistent with the
        written label file. Polygons are normalized at the image's size.
        Everything is validated before any write.
        """
        image_class = (image_class or "").strip() or None
        polygons = list(polygons or [])
        with self._lock:
            self._check_frozen()
            check_label_kinds(self._task_locked(), boxes=len(boxes), polygons=len(polygons),
                              image_class=image_class)
            box_names = [self._validate_class_name(b.class_name) for b in boxes]
            polygon_names = [self._validate_class_name(p.class_name) for p in polygons]
            class_name = (self._validate_class_name_for_task(image_class)
                          if image_class is not None else None)
            for b in boxes:
                for v in (b.cx, b.cy, b.w, b.h):
                    if not 0.0 <= v <= 1.0:
                        raise DatasetError("Box coordinates must be normalized (0..1)")
            for p in polygons:
                check_polygon_points(p.points)
            classes = self._load_classes()
            for name in box_names + polygon_names + ([class_name] if class_name else []):
                if name not in classes:
                    classes.append(name)
            resolved = [
                Box(classes.index(name), b.cx, b.cy, b.w, b.h)
                for name, b in zip(box_names, boxes, strict=False)
            ]
            rings: List[Polygon] = []
            if polygons:
                width, height = self._stored_size(jpeg=jpeg)
                rings = normalize_polygons(
                    [Polygon(classes.index(name), p.points, p.instance)
                     for name, p in zip(polygon_names, polygons, strict=True)],
                    width, height)
            class_id = classes.index(class_name) if class_name is not None else None
            self._save_classes(classes)
            image_id = str(uuid.uuid4())
            with open(self._image_path(image_id), "wb") as f:
                f.write(jpeg)
            if class_id is not None:
                self._write_image_class(image_id, class_id)
            elif polygons:
                self._write_polygons(image_id, rings)
            else:
                self._write_boxes(image_id, resolved)
        return ImageEntry(image_id=image_id, created_at=time.time(),
                          labeled=bool(resolved) or bool(rings) or class_id is not None,
                          box_count=len(resolved) + len(rings), image_class=class_id)

    def list_images(self) -> List[ImageEntry]:
        """List images, newest first, with their label state.

        A detection image is labeled when it has boxes, a segmentation image
        when it has polygon rings (``box_count`` counts the rings), a
        classification image when it has a class (``ImageEntry.image_class``).
        """
        with self._lock:
            meta = self._load_meta()
            replicas = set(meta.get("replica_image_ids", []))
            task = task_or_default(meta.get("task"))
            entries = []
            for name in os.listdir(self._images_dir):
                if not name.endswith(".jpg"):
                    continue
                image_id = name[:-4]
                path = os.path.join(self._images_dir, name)
                image_class = (self._read_image_class(image_id)
                               if uses_image_class(task) else None)
                shapes = self._read_shapes(image_id, task)
                entries.append(ImageEntry(
                    image_id=image_id,
                    created_at=os.path.getmtime(path),
                    labeled=bool(shapes) or image_class is not None,
                    box_count=len(shapes),
                    replica=image_id in replicas,
                    image_class=image_class,
                ))
            entries.sort(key=lambda e: e.created_at, reverse=True)
            return entries

    def get_image_bytes(self, image_id: str) -> bytes:
        """Get image bytes."""
        path = self._image_path(image_id)
        if not os.path.exists(path):
            raise DatasetError(f"Image '{image_id}' not found")
        with open(path, "rb") as f:
            return f.read()

    def delete_image(self, image_id: str) -> None:
        """Delete image."""
        with self._lock:
            self._check_frozen()
            path = self._image_path(image_id)
            if not os.path.exists(path):
                raise DatasetError(f"Image '{image_id}' not found")
            os.remove(path)
            label = self._label_path(image_id)
            if os.path.exists(label):
                os.remove(label)
            self._forget_replica(image_id)

    def replicate_image(self, image_id: str, count: int) -> int:
        """Duplicate a labeled, non-replica image (with its labels) ``count``
        times. Each copy gets a fresh uuid and is flagged as a replica in
        meta.json. Returns the number of copies created."""
        with self._lock:
            self._check_frozen()
            try:
                count = max(1, min(int(count), 50))
            except (TypeError, ValueError):
                raise DatasetError("Replica count must be an integer") from None
            if not os.path.exists(self._image_path(image_id)):
                raise DatasetError(f"Image '{image_id}' not found")
            meta = self._load_meta()
            replicas = set(meta.get("replica_image_ids", []))
            if image_id in replicas:
                raise DatasetError("Cannot replicate a replicated image")
            task = self._task_locked()
            image_class = (self._read_image_class(image_id)
                           if uses_image_class(task) else None)
            shapes = self._read_shapes(image_id, task)
            if not shapes and image_class is None:
                raise DatasetError("Only labeled images can be replicated")
            jpeg = self.get_image_bytes(image_id)
            created: list[str] = []
            try:
                for _ in range(count):
                    new_id = str(uuid.uuid4())
                    with open(self._image_path(new_id), "wb") as f:
                        f.write(jpeg)
                    if uses_image_class(task):
                        self._write_image_class(new_id, image_class)
                    else:
                        self._write_shapes(new_id, task, shapes)
                    created.append(new_id)
                    replicas.add(new_id)
            except Exception:
                # Best-effort rollback so we don't leave untracked images behind.
                for rid in created:
                    for path in (self._image_path(rid), self._label_path(rid)):
                        try:
                            os.remove(path)
                        except FileNotFoundError:
                            # Rollback is idempotent: file may already be absent
                            # if creation failed part-way or it was removed earlier.
                            pass
                raise

            meta["replica_image_ids"] = sorted(replicas)
            self._save_meta(meta)
            return len(created)

    def _forget_replica(self, image_id: str) -> None:
        """Drop *image_id* from the persisted replica set (no-op if absent)."""
        meta = self._load_meta()
        replicas = meta.get("replica_image_ids")
        if isinstance(replicas, list) and image_id in replicas:
            meta["replica_image_ids"] = [r for r in replicas if r != image_id]
            self._save_meta(meta)

    # ── labels ────────────────────────────────────────────────────────────────

    def _read_boxes(self, image_id: str) -> List[Box]:
        """Read boxes."""
        path = self._label_path(image_id)
        if not os.path.exists(path):
            return []
        boxes: List[Box] = []
        with open(path, "r") as f:
            for line in f:
                parts = line.split()
                if len(parts) != 5:
                    continue
                try:
                    boxes.append(Box(int(parts[0]), float(parts[1]), float(parts[2]),
                                     float(parts[3]), float(parts[4])))
                except ValueError:
                    continue
        return boxes

    def _read_image_class(self, image_id: str) -> Optional[int]:
        """A classification label file's class id (its one line), ``None`` if unlabeled."""
        path = self._label_path(image_id)
        if not os.path.exists(path):
            return None
        with open(path, "r") as f:
            for line in f:
                parts = line.split()
                if not parts:
                    continue
                if len(parts) != 1:
                    return None
                try:
                    value = int(parts[0])
                except ValueError:
                    return None
                return value if value >= 0 else None
        return None

    def _write_image_class(self, image_id: str, class_id: Optional[int]) -> None:
        """Write (or, for ``None``, remove) a classification label file."""
        path = self._label_path(image_id)
        if class_id is None:
            if os.path.exists(path):
                os.remove(path)
            return
        atomic_write_bytes(path, f"{int(class_id)}\n".encode("utf-8"), mode=0o644)

    def _read_polygons(self, image_id: str) -> List[Polygon]:
        """A segmentation label file's rings, one instance per row."""
        path = self._label_path(image_id)
        if not os.path.exists(path):
            return []
        polygons: List[Polygon] = []
        with open(path, "r") as f:
            for line in f:
                parts = line.split()
                # "class x1 y1 … xn yn" with n >= 3: an odd token count of 7 or more.
                if len(parts) < 7 or len(parts) % 2 == 0:
                    continue
                try:
                    class_id = int(parts[0])
                    values = [float(v) for v in parts[1:]]
                except ValueError:
                    continue
                pairs = [[values[i], values[i + 1]] for i in range(0, len(values), 2)]
                polygons.append(Polygon(class_id, pairs, len(polygons)))
        return polygons

    def _write_polygons(self, image_id: str, polygons: List[Polygon]) -> None:
        """Write one row per ring (or, for no rings, remove the label file)."""
        path = self._label_path(image_id)
        if not polygons:
            if os.path.exists(path):
                os.remove(path)
            return
        lines = [
            f"{p.class_id} " + " ".join(f"{x:.6f} {y:.6f}" for x, y in p.points)
            for p in polygons
        ]
        atomic_write_bytes(path, ("\n".join(lines) + "\n").encode("utf-8"), mode=0o644)

    def _read_shapes(self, image_id: str, task: str) -> list:
        """The box or polygon rows of an image (``[]`` for an image-class dataset)."""
        if uses_image_class(task):
            return []
        return self._read_polygons(image_id) if task == "segment" else self._read_boxes(image_id)

    def _write_shapes(self, image_id: str, task: str, shapes: list) -> None:
        """Write the box or polygon rows of an image, by the dataset's task."""
        if task == "segment":
            self._write_polygons(image_id, shapes)
        else:
            self._write_boxes(image_id, shapes)

    def _stored_size(self, image_id: str = "", jpeg: Optional[bytes] = None) -> Tuple[int, int]:
        """``(width, height)`` of a stored image (of ``jpeg`` when given).

        A letterboxed dataset stores its recorded square; a native one each
        image at its own size, read from the JPEG header (decoded as a
        fallback). The caller holds the lock.
        """
        size = int(normalize_geometry(self._load_meta().get("geometry")).get("letterbox", 0))
        if size > 0:
            return size, size
        if jpeg is None:
            with open(self._image_path(image_id), "rb") as f:
                jpeg = f.read()
        from .dataset_import import image_dimensions_from_bytes

        dims = image_dimensions_from_bytes(jpeg)
        if dims is None:
            import cv2
            import numpy as np

            img = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise DatasetError("The image cannot be decoded")
            dims = (int(img.shape[1]), int(img.shape[0]))
        return int(dims[0]), int(dims[1])

    def get_polygons(self, image_id: str) -> List[Polygon]:
        """The rings of a segmentation image (``[]`` for another task or no label)."""
        if not os.path.exists(self._image_path(image_id)):
            raise DatasetError(f"Image '{image_id}' not found")
        with self._lock:
            if self._task_locked() != "segment":
                return []
            return self._read_polygons(image_id)

    def get_labels(self, image_id: str) -> List[Box]:
        """Get labels."""
        if not os.path.exists(self._image_path(image_id)):
            raise DatasetError(f"Image '{image_id}' not found")
        with self._lock:
            return self._read_boxes(image_id)

    def get_image_class(self, image_id: str) -> Optional[int]:
        """The image's class in a classification dataset (``None``: unlabeled or another task)."""
        if not os.path.exists(self._image_path(image_id)):
            raise DatasetError(f"Image '{image_id}' not found")
        with self._lock:
            if not uses_image_class(self._task_locked()):
                return None
            return self._read_image_class(image_id)

    def set_labels(self, image_id: str, boxes: List[Box],
                   image_class: Optional[int] = None,
                   polygons: Optional[List[Polygon]] = None) -> None:
        """Replace an image's labels: ``boxes`` for a detection dataset,
        ``polygons`` for a segmentation dataset (normalized at the stored
        image's size), the ``image_class`` id (``None`` clears it) for a
        classification dataset."""
        polygons = list(polygons or [])
        with self._lock:
            self._check_frozen()
            if not os.path.exists(self._image_path(image_id)):
                raise DatasetError(f"Image '{image_id}' not found")
            task = self._task_locked()
            check_label_kinds(task, boxes=len(boxes), polygons=len(polygons),
                              image_class=None if image_class is None else str(image_class))
            classes = self._load_classes()
            if image_class is not None and not 0 <= image_class < len(classes):
                raise DatasetError(f"Unknown class id {image_class}")
            for b in boxes:
                if not 0 <= b.class_id < len(classes):
                    raise DatasetError(f"Unknown class id {b.class_id}")
                for v in (b.cx, b.cy, b.w, b.h):
                    if not 0.0 <= v <= 1.0:
                        raise DatasetError("Box coordinates must be normalized (0..1)")
            for p in polygons:
                if not 0 <= p.class_id < len(classes):
                    raise DatasetError(f"Unknown class id {p.class_id}")
                check_polygon_points(p.points)
            if uses_image_class(task):
                self._write_image_class(image_id, image_class)
            elif task == "segment":
                width, height = self._stored_size(image_id)
                self._write_polygons(image_id, normalize_polygons(polygons, width, height))
            else:
                self._write_boxes(image_id, boxes)

    def _write_boxes(self, image_id: str, boxes: List[Box]) -> None:
        """Write boxes."""
        path = self._label_path(image_id)
        if not boxes:
            if os.path.exists(path):
                os.remove(path)
            return
        lines = [
            f"{b.class_id} {b.cx:.6f} {b.cy:.6f} {b.w:.6f} {b.h:.6f}"
            for b in boxes
        ]
        atomic_write_bytes(path, ("\n".join(lines) + "\n").encode("utf-8"),
                           mode=0o644)

    # ── classes ───────────────────────────────────────────────────────────────

    def _load_classes(self) -> List[str]:
        """Load classes."""
        if not os.path.exists(self._classes_file):
            return []
        try:
            with open(self._classes_file, "r") as f:
                data = json.load(f)
            return [str(c) for c in data] if isinstance(data, list) else []
        except (ValueError, OSError):
            logger.warning("classes.json unreadable; treating as empty")
            return []

    def _save_classes(self, classes: List[str]) -> None:
        """Save classes (atomic + fsync; power-cut safe)."""
        atomic_write_json(self._classes_file, classes, mode=0o644,
                          ensure_ascii=False)

    def get_classes(self) -> List[str]:
        """Get classes."""
        with self._lock:
            return self._load_classes()

    @staticmethod
    def _validate_class_name(name: str) -> str:
        """Validate class name."""
        name = (name or "").strip()
        if not name:
            raise DatasetError("Class name must not be empty")
        if len(name) > 64 or not all(c in _CLASS_NAME_SAFE for c in name):
            raise DatasetError("Class name has invalid characters")
        if set(name) == {"."}:
            # A classification split turns class names into folder names.
            raise DatasetError("Class name must not consist of dots only")
        return name

    def _validate_class_name_for_task(self, name: str) -> str:
        """``_validate_class_name`` plus the task's own rules; the caller holds the lock.

        A face dataset's classes are people, and ``unknown`` is what the
        device calls a face nobody matches: enrolling a person under that
        name would make them indistinguishable from a stranger.
        """
        name = self._validate_class_name(name)
        if self._task_locked() == FACE:
            if not person_key(name):
                raise DatasetError("A person's name cannot be a colour alone")
            if is_reserved_face_name(name):
                raise DatasetError("'unknown' is reserved for a face nobody matches")
        return name

    def _same_person(self, a: str, b: str) -> bool:
        """Two class entries naming the same person: a face dataset compares
        names without case (``Alice`` and ``alice`` would be two galleries
        entries nobody can tell apart); other tasks compare exactly."""
        if self._task_locked() != FACE:
            return a == b
        return person_key(a) == person_key(b)

    def add_class(self, name: str) -> List[str]:
        """Add class."""
        with self._lock:
            self._check_frozen()
            name = self._validate_class_name_for_task(name)
            classes = self._load_classes()
            if any(self._same_person(name, c) for c in classes):
                raise DatasetError(f"Class '{name}' already exists")
            classes.append(name)
            self._save_classes(classes)
            return classes

    def rename_class(self, index: int, name: str) -> List[str]:
        """Rename class."""
        with self._lock:
            self._check_frozen()
            name = self._validate_class_name_for_task(name)
            classes = self._load_classes()
            if not 0 <= index < len(classes):
                raise DatasetError(f"No class at index {index}")
            if any(self._same_person(name, c) for i, c in enumerate(classes) if i != index):
                raise DatasetError(f"Class '{name}' already exists")
            classes[index] = name
            self._save_classes(classes)
            return classes

    def remove_class(self, index: int) -> List[str]:
        """Remove a class: drop its boxes or polygons from every label file (a
        classification image of that class becomes unlabeled) and decrement
        the ids above it so the label files stay consistent with classes.json."""
        with self._lock:
            self._check_frozen()
            classes = self._load_classes()
            if not 0 <= index < len(classes):
                raise DatasetError(f"No class at index {index}")
            classes.pop(index)
            task = self._task_locked()
            image_class_task = uses_image_class(task)
            for name in os.listdir(self._labels_dir):
                if not name.endswith(".txt"):
                    continue
                image_id = name[:-4]
                if image_class_task:
                    current = self._read_image_class(image_id)
                    if current is not None:
                        self._write_image_class(
                            image_id, None if current == index
                            else current - 1 if current > index else current)
                    continue
                if task == "segment":
                    self._write_polygons(image_id, [
                        Polygon(p.class_id - 1 if p.class_id > index else p.class_id,
                                p.points, p.instance)
                        for p in self._read_polygons(image_id) if p.class_id != index
                    ])
                    continue
                boxes = self._read_boxes(image_id)
                kept = [
                    Box(b.class_id - 1 if b.class_id > index else b.class_id,
                        b.cx, b.cy, b.w, b.h)
                    for b in boxes if b.class_id != index
                ]
                self._write_boxes(image_id, kept)
            self._save_classes(classes)
            return classes

    # ── metadata (name / cover) ───────────────────────────────────────────────

    def _load_meta(self) -> Dict:
        """Load meta.

        Missing file → empty (first run); corrupt file → reported and
        quarantined by read_json, never silently treated as an empty dataset.
        """
        data = read_json(self._meta_file, {})
        return data if isinstance(data, dict) else {}

    def _save_meta(self, meta: Dict) -> None:
        """Save meta (atomic + fsync; power-cut safe).

        Backfills ``geometry`` and ``task`` for legacy datasets: a meta.json
        written before the keys existed describes 640×640 letterboxed images
        labeled for detection, and the first save (rename, cover, replicate,
        ...) persists that explicitly.
        """
        meta["geometry"] = normalize_geometry(meta.get("geometry"))
        meta["task"] = task_or_default(meta.get("task"))
        atomic_write_json(self._meta_file, meta, mode=0o644,
                          ensure_ascii=False)

    def write_meta(self, name: str, created_at: Optional[float] = None,
                   geometry: Optional[Dict] = None, task: Optional[str] = None) -> None:
        """Create or refresh the dataset's name/created_at.

        ``geometry`` and ``task`` are recorded only when the dataset has none
        yet (a dataset's storage format and label task are fixed at
        creation); ``None`` means the legacy 640×640 letterbox format and
        detection, which is what the legacy-layout migration relies on.
        """
        with self._lock:
            meta = self._load_meta()
            meta.setdefault("cover_image_id", "")
            if "geometry" not in meta:
                meta["geometry"] = normalize_geometry(geometry)
            if "task" not in meta:
                meta["task"] = task_or_default(task)
            meta["name"] = name
            meta["created_at"] = created_at or meta.get("created_at") or time.time()
            self._save_meta(meta)

    def geometry(self) -> Dict:
        """How this dataset's images are stored (see the module docstring)."""
        with self._lock:
            return normalize_geometry(self._load_meta().get("geometry"))

    def task(self) -> str:
        """The task this dataset is labeled for (``"detect"`` for legacy datasets)."""
        with self._lock:
            return self._task_locked()

    def _task_locked(self) -> str:
        """The dataset's task; the caller holds the lock."""
        return task_or_default(self._load_meta().get("task"))

    def check_geometry(self, dataset_img_size: int) -> None:
        """Refuse to add images in a geometry other than the recorded one.

        Called before capture, hub ingest and any other path that stores a new
        image; ``dataset_img_size`` is the current ``Config.DATASET_IMG_SIZE``.
        """
        stored = self.geometry()
        if stored == geometry_for(dataset_img_size):
            return
        name = str(self._load_meta().get("name") or self.dataset_id)
        raise DatasetError(
            f"Dataset '{name}' stores {describe_geometry(stored)}; set "
            f"TRAIN_DATASET_IMG_SIZE={_geometry_env_value(stored)} or create a "
            f"new dataset"
        )

    def rename(self, name: str) -> None:
        """Rename."""
        name = validate_dataset_name(name)
        with self._lock:
            meta = self._load_meta()
            meta["name"] = name
            self._save_meta(meta)

    def set_cover(self, image_id: str) -> None:
        """Set cover."""
        with self._lock:
            self._check_frozen()
            if not os.path.exists(self._image_path(image_id)):
                raise DatasetError(f"Image '{image_id}' not found")
            meta = self._load_meta()
            meta["cover_image_id"] = image_id
            self._save_meta(meta)

    def meta(self) -> Dict:
        """Registry-card metadata with the cover resolved: the explicit cover
        if that image still exists, else the first (oldest) image, else ""."""
        with self._lock:
            meta = self._load_meta()
            entries = self.list_images()
            cover = str(meta.get("cover_image_id") or "")
            if not cover or not os.path.exists(os.path.join(self._images_dir, f"{cover}.jpg")):
                # list_images() returns newest-first; default cover is the oldest image.
                cover = entries[-1].image_id if entries else ""
            return {
                "dataset_id": self.dataset_id,
                "name": str(meta.get("name") or "Unnamed"),
                "created_at": float(meta.get("created_at") or 0.0),
                "cover_image_id": cover,
                "image_count": len(entries),
                "labeled_count": sum(1 for e in entries if e.labeled),
                "class_count": len(self.get_classes()),
                "task": task_or_default(meta.get("task")),
            }

    # ── dataset info / training gate ──────────────────────────────────────────

    def info(self) -> Dict:
        """Info."""
        with self._lock:
            entries = self.list_images()
            meta = self.meta()
            return {
                "image_count": len(entries),
                "labeled_count": sum(1 for e in entries if e.labeled),
                "classes": self.get_classes(),
                # A face gallery enrolls from as little as one photo.
                "min_images": 1 if meta["task"] == FACE else self._config.MIN_IMAGES,
                "dataset_id": self.dataset_id,
                "name": meta["name"],
                "cover_image_id": meta["cover_image_id"],
                "geometry": self.geometry(),
                "task": meta["task"],
            }

    def validate_for_training(self) -> None:
        """Validate for training.

        A face dataset builds a gallery instead of training a network: one
        person with one labeled photo is enough, so the YOLO image minimum
        does not apply.
        """
        info = self.info()
        if info["task"] == FACE:
            if not info["classes"]:
                raise DatasetError("Add at least one person before building a face gallery")
            if info["labeled_count"] == 0:
                raise DatasetError(
                    "Assign at least one photo to a person before building a face gallery")
            return
        if info["image_count"] < info["min_images"]:
            raise DatasetError(
                f"At least {info['min_images']} images are required "
                f"(have {info['image_count']})"
            )
        if not info["classes"]:
            raise DatasetError("Create at least one class before training")
        if info["labeled_count"] == 0:
            raise DatasetError("Label at least one image before training")
        if info["task"] == "classify":
            present = {e.image_class for e in self.list_images() if e.image_class is not None}
            if len(present) < 2:
                raise DatasetError(
                    "Label images of at least 2 classes before training a classifier")

    # ── export ────────────────────────────────────────────────────────────────

    def export_zip(self, dest_path: str, num_shards: int = 0,
                   shard_index: int = 0, seed: str = "") -> int:
        """Write the dataset as a YOLO-format ZIP (the layout import accepts):
        images/{id}.jpg + labels/{id}.txt + data.yaml. Returns the image
        count. Holds the lock so the archive is a consistent snapshot.

        With ``num_shards`` > 0, exports one deterministic IID shard instead:
        images are shuffled with ``seed`` and assigned round-robin, so the N
        shards of one seed are disjoint, cover the full dataset and differ in
        size by at most one image. data.yaml always carries the full class
        list, so per-shard checkpoints stay averageable (federated training).

        A classification dataset is exported one folder per class instead
        (``_export_class_folders``). A face dataset is never exported: its
        photos are biometric data that stay on the device (``DatasetError``).
        """
        with self._lock:
            if self._task_locked() == FACE:
                raise DatasetError(FACE_EXPORT_REFUSED)
            entries = self.list_images()
            classes = self._load_classes()
            if num_shards > 0:
                rng = random.Random(seed)
                shuffled = sorted(entries, key=lambda e: e.image_id)
                rng.shuffle(shuffled)
                entries = shuffled[shard_index::num_shards]
            if uses_image_class(self._task_locked()):
                return self._export_class_folders(dest_path, entries, classes)
            with zipfile.ZipFile(dest_path, "w") as zf:
                for e in entries:
                    # JPEGs are already compressed — store, don't deflate.
                    zf.write(self._image_path(e.image_id),
                             f"images/{e.image_id}.jpg",
                             compress_type=zipfile.ZIP_STORED)
                    label = self._label_path(e.image_id)
                    if os.path.exists(label):
                        zf.write(label, f"labels/{e.image_id}.txt",
                                 compress_type=zipfile.ZIP_DEFLATED)
                names = ", ".join(f"'{c}'" for c in classes)
                zf.writestr(
                    "data.yaml",
                    f"train: images\nval: images\n\nnc: {len(classes)}\nnames: [{names}]\n",
                    compress_type=zipfile.ZIP_DEFLATED,
                )
            return len(entries)

    def _export_class_folders(self, dest_path: str, entries: List[ImageEntry],
                              classes: List[str]) -> int:
        """Classification export: ``train/<class>/<id>.jpg`` plus ``classes.txt``.

        The folder-per-class layout the import accepts (and ultralytics trains
        on). ``classes.txt`` keeps the full class list in its order, classes
        without images included, so every federated shard carries the same
        classes. Unlabeled images have no folder and are left out. Returns the
        number of images written.
        """
        count = 0
        with zipfile.ZipFile(dest_path, "w") as zf:
            for e in entries:
                if e.image_class is None or not 0 <= e.image_class < len(classes):
                    continue
                zf.write(self._image_path(e.image_id),
                         f"train/{classes[e.image_class]}/{e.image_id}.jpg",
                         compress_type=zipfile.ZIP_STORED)
                count += 1
            zf.writestr("classes.txt", "".join(f"{c}\n" for c in classes),
                        compress_type=zipfile.ZIP_DEFLATED)
        return count

    # ── split builder ─────────────────────────────────────────────────────────

    def build_split(self, job_id: str, val_fraction: float = 0.2, *,
                    tile: TileSpec = None, overlap: float = 0.2,
                    min_visible: float = 0.25) -> SplitResult:
        """Create the train/valid layout + data.yaml for one job.

        Only labeled images participate — ultralytics treats label-less
        images as background, which is rarely what an operator collecting 20
        captures intends. With ``tile`` (``"auto"`` = the image's short side,
        or pixels) each image of both splits is written as its tile crops with
        per-tile labels (``service.tile_split``), so the validation metrics
        describe the geometry the model is deployed in; ``tile=None`` and
        images the grid cannot slice are symlinked whole.

        A classification dataset gets the folder-per-class layout instead
        (``_build_classify_split``; ``tile`` does not apply).
        """
        with self._lock:
            task = self._task_locked()
            if task == FACE:
                raise DatasetError(
                    "A face dataset builds a gallery (build_face_package), not a training split")
            if task == "classify":
                return self._build_classify_split(job_id, val_fraction)
            entries = [e for e in self.list_images() if e.labeled]
            if not entries:
                raise DatasetError("No labeled images to train on")
            classes = self._load_classes()

            root = os.path.join(self._config.runs_dir, job_id, "dataset")
            for split in ("train", "valid"):
                os.makedirs(os.path.join(root, split, "images"), exist_ok=True)
                os.makedirs(os.path.join(root, split, "labels"), exist_ok=True)

            rng = random.Random(job_id)
            shuffled = entries[:]
            rng.shuffle(shuffled)
            # At least 1 validation image, at least 1 training image.
            n_val = min(max(1, int(round(len(shuffled) * val_fraction))),
                        len(shuffled) - 1)
            splits = {"valid": shuffled[:n_val], "train": shuffled[n_val:]}

            stats = TileSplitStats()
            counts: Dict[str, int] = {}
            for split, items in splits.items():
                images_dir = os.path.join(root, split, "images")
                labels_dir = os.path.join(root, split, "labels")
                counts[split] = 0
                for e in items:
                    if tile is not None:
                        # A segmentation image's rings go through the same tile
                        # crops as detection boxes.
                        polygons = ([(p.class_id, p.points)
                                     for p in self._read_polygons(e.image_id)]
                                    if task == "segment" else None)
                        rows = ([] if polygons is not None else
                                [(b.class_id, b.cx, b.cy, b.w, b.h)
                                 for b in self._read_boxes(e.image_id)])
                        written, image_stats = materialize_tiles(
                            self._image_path(e.image_id), rows, images_dir, labels_dir,
                            e.image_id, tile=tile, overlap=overlap, min_visible=min_visible,
                            polygons=polygons)
                        stats.add(image_stats)
                        if written:
                            counts[split] += len(written)
                            continue
                    os.symlink(
                        self._image_path(e.image_id),
                        os.path.join(images_dir, f"{e.image_id}.jpg"),
                    )
                    os.symlink(
                        self._label_path(e.image_id),
                        os.path.join(labels_dir, f"{e.image_id}.txt"),
                    )
                    counts[split] += 1

            yaml_path = os.path.join(root, "data.yaml")
            names = ", ".join(f"'{c}'" for c in classes)
            with open(yaml_path, "w") as f:
                f.write(
                    f"train: {os.path.join(root, 'train', 'images')}\n"
                    f"val: {os.path.join(root, 'valid', 'images')}\n"
                    f"\n"
                    f"nc: {len(classes)}\n"
                    f"names: [{names}]\n"
                )
            geometry = geometry_label(tile, stats)
            logger.info(
                "Built split for job %s: %d train / %d valid images, %d classes, "
                "geometry %s (%d tiles, %d labels, %d background, %d fragment-only "
                "tiles skipped, %d images kept whole)",
                job_id, len(splits["train"]), len(splits["valid"]), len(classes),
                geometry, stats.tiles, stats.boxes, stats.background, stats.skipped,
                stats.whole,
            )
            return SplitResult(yaml_path, counts["train"], counts["valid"], geometry, stats)

    def build_face_package(self, job_id: str, model_name: str) -> Tuple[str, int]:
        """Package a face dataset for the on-device gallery build.

        Writes ``<runs dir>/<job_id>/<model_name>.faces``: a ZIP holding
        ``manifest.json`` — ``{"format": 1, "classes": [names by class id],
        "images": [{"image_id", "class_id", "file"}]}`` — and every labeled
        photo as ``images/<image_id>.jpg`` (the ``file`` each entry names).
        Unlabeled photos are left out; the full class list is kept in order,
        people without photos included, so class ids match classes.json. The
        inference-service's ``face_gallery_builder.read_manifest`` parses this
        exact layout. Returns ``(package path, photo count)``.
        """
        validate_model_name(model_name)
        with self._lock:
            if self._task_locked() != FACE:
                raise DatasetError("Only a face dataset can be packaged for a face gallery")
            classes = self._load_classes()
            if not classes:
                raise DatasetError("Add at least one person before building a face gallery")
            images = sorted(
                (e for e in self.list_images()
                 if e.image_class is not None and 0 <= e.image_class < len(classes)),
                key=lambda e: e.image_id)
            if not images:
                raise DatasetError(
                    "Assign at least one photo to a person before building a face gallery")
            job_dir = os.path.join(self._config.runs_dir, job_id)
            os.makedirs(job_dir, exist_ok=True)
            path = os.path.join(job_dir, f"{model_name}{FACE_PACKAGE_SUFFIX}")
            manifest = {
                "format": FACE_PACKAGE_FORMAT,
                "classes": classes,
                "images": [
                    {"image_id": e.image_id, "class_id": e.image_class,
                     "file": f"images/{e.image_id}.jpg"}
                    for e in images
                ],
            }
            try:
                with zipfile.ZipFile(path, "w") as zf:
                    zf.writestr(FACE_MANIFEST, json.dumps(manifest, ensure_ascii=False),
                                compress_type=zipfile.ZIP_DEFLATED)
                    for e in images:
                        # JPEGs are already compressed — store, don't deflate.
                        zf.write(self._image_path(e.image_id), f"images/{e.image_id}.jpg",
                                 compress_type=zipfile.ZIP_STORED)
            except Exception:
                if os.path.exists(path):
                    os.remove(path)
                raise
            logger.info("Packaged face dataset %s for job %s: %d photos of %d people",
                        self.dataset_id, job_id, len(images), len(classes))
            return path, len(images)

    def _build_classify_split(self, job_id: str, val_fraction: float) -> SplitResult:
        """Folder-per-class layout for a classification job.

        ``<root>/train/<class>/<id>.jpg`` and ``<root>/val/<class>/<id>.jpg``
        (symlinks, same volume); ultralytics trains on ``data=<root>``. Every
        class of the dataset gets a folder in both splits, even an empty one
        (ultralytics loads them with ``allow_empty``), so the class indices —
        the sorted folder names — are the same in both splits and in every
        federated shard. Each class is split on its own, so every class with
        two or more labeled images is in both; unlabeled images are left out
        and there is no tiling (a classifier sees the whole frame).
        """
        classes = self._load_classes()
        by_class: Dict[int, List[str]] = {}
        for e in self.list_images():
            if e.image_class is not None and 0 <= e.image_class < len(classes):
                by_class.setdefault(e.image_class, []).append(e.image_id)

        rng = random.Random(job_id)
        splits: Dict[str, Dict[int, List[str]]] = {"train": {}, "val": {}}
        for class_id in sorted(by_class):
            ids = sorted(by_class[class_id])
            rng.shuffle(ids)
            n_val = 0
            if len(ids) >= 2:
                # At least 1 validation image and at least 1 training image.
                n_val = min(max(1, int(round(len(ids) * val_fraction))), len(ids) - 1)
            splits["val"][class_id] = ids[:n_val]
            splits["train"][class_id] = ids[n_val:]
        in_both = [c for c in by_class if splits["train"][c] and splits["val"][c]]
        if len(in_both) < 2:
            raise DatasetError(
                "A classifier needs at least 2 classes with 2 or more labeled images each")

        root = os.path.join(self._config.runs_dir, job_id, "dataset")
        counts: Dict[str, int] = {}
        for split, per_class in splits.items():
            counts[split] = 0
            for class_id, name in enumerate(classes):
                class_dir = os.path.join(root, split, name)
                os.makedirs(class_dir, exist_ok=True)
                for image_id in per_class.get(class_id, []):
                    os.symlink(self._image_path(image_id),
                               os.path.join(class_dir, f"{image_id}.jpg"))
                    counts[split] += 1
        logger.info(
            "Built classification split for job %s: %d train / %d val images, "
            "%d classes (%d in both splits)",
            job_id, counts["train"], counts["val"], len(classes), len(in_both))
        return SplitResult(root, counts["train"], counts["val"], GEOMETRY_FRAMES,
                           TileSplitStats())


def validate_model_name(name: str) -> str:
    """Mandatory, filesystem-safe model name (becomes {name}.pt → {name}.engine)."""
    name = (name or "").strip()
    if not name:
        raise DatasetError("Model name is required")
    safe = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")
    if len(name) > 64 or not all(c in safe for c in name):
        raise DatasetError(
            "Model name may only contain letters, digits, '_' and '-' (max 64)"
        )
    return name
