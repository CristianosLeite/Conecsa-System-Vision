# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Import of a pre-existing YOLO-format dataset uploaded as a ZIP.

Validates the archive (structure, classes, label syntax) and normalizes it
into the internal dataset layout in the requested geometry (``img_size``,
i.e. ``Config.DATASET_IMG_SIZE``):

* ``img_size > 0`` — every image is re-encoded as an ``img_size`` square
  letterboxed JPEG and the label coordinates (normalized on the original
  W×H) are transformed onto that square with the exact same rounding as
  ``letterbox_square``.
* ``img_size == 0`` — every image is decoded (so undecodable files are still
  rejected and the pixel cap still applies) and re-encoded as a JPEG at its own
  resolution; label rows are written unchanged, since YOLO coordinates are
  already normalized to the source image.

Accepted layouts inside the ZIP (Roboflow / ultralytics exports):

    data.yaml + train|valid|test/images/*.jpg + .../labels/*.txt
    data.yaml + images/*.jpg + labels/*.txt
    classes.txt instead of data.yaml; sibling .txt next to each image as a
    last-resort pairing when no images/ directory exists.

The archive must be laid out for the new dataset's task:

* ``detect`` — label rows are "class cx cy w h"; an archive with
  segmentation rows ("class x1 y1 x2 y2 ...") or a classification layout is
  refused (silently turning polygons into boxes would hide that the archive
  is for another task).
* ``classify`` — one folder of images per class (the ultralytics layout,
  ``[train|val|test/]<class>/*.jpg``, optionally inside one wrapper folder);
  ``classes.txt``, when present, fixes the class order and may list classes
  without images (the dataset export writes one). A folder whose name begins
  with a dot is a class only when it holds images (hidden tool folders hold
  none). An archive with YOLO label files or an ``images/`` directory is a
  detection export and is refused.
* ``face`` — the classification layout: one folder of photos per person
  (the folder name is the person's name).
* ``segment`` — the detection layout with polygon rows ("class x1 y1 … xn
  yn", normalized, at least 3 vertices); box rows or a classification
  layout are refused. Every ring is letterbox-transformed like a box and
  then normalized at the stored image size (``conecsa_common.polygons``).

All failures raise DatasetImportError with an operator-readable message.
"""
import io
import json
import logging
import os
import shutil
import stat
import struct
import uuid
import zipfile
from typing import BinaryIO, Iterator, List, Optional, Tuple

import cv2
import yaml
from conecsa_common.tasks import FACE, is_reserved_face_name, person_key

from .capture_service import letterbox_square
from .dataset_service import DatasetError

logger = logging.getLogger(__name__)

_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
# Same character policy as class names in dataset_service (which admits the
# "name #rrggbb" color suffix, so an exported dataset re-imports).
_NAME_SAFE = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 _-.#")
# The tasks an archive can be imported for, as named in refusal messages.
_TASK_NAMES = {"detect": "object detection", "classify": "classification",
               "segment": "segmentation", "face": "face recognition"}
# Tasks whose archives hold one folder of images per class (per person for face).
_CLASS_FOLDER_TASKS = ("classify", "face")
# Split folders of a classification archive (ultralytics accepts all of them).
_CLASS_SPLITS = ("train", "val", "valid", "validation", "test")

# Hard caps on what an uploaded archive may expand into. The total budget
# counts bytes actually written (the ZIP's declared file_size is attacker-
# controlled, so a header lying about its size cannot bypass it). Same
# import-time env pattern as service/config.py.
_MAX_ZIP_ENTRIES = int(os.environ.get("TRAINING_MAX_ZIP_ENTRIES", "20000"))
_MAX_ZIP_ENTRY_BYTES = int(
    os.environ.get("TRAINING_MAX_ZIP_ENTRY_MB", "256")) * 1024 * 1024
_MAX_ZIP_RATIO = int(os.environ.get("TRAINING_MAX_ZIP_RATIO", "100"))
# cv2.imread allocates the full raster before any other check can run — a
# 60k×60k PNG is ~10 GB — so dimensions are read from the file header first.
_MAX_IMAGE_PIXELS = int(
    os.environ.get("TRAINING_MAX_IMAGE_PIXELS", str(64_000_000)))


class DatasetImportError(DatasetError):
    """Validation failure surfaced verbatim to the user."""


def import_dataset_zip(zip_path: str, dest_dir: str, img_size: int = 640,
                       max_total_mb: int = 512,
                       task: str = "detect") -> Tuple[List[str], int]:
    """Validate ``zip_path`` and materialize it into ``dest_dir`` (staging).

    ``img_size`` selects the storage geometry (letterbox square, or ``0`` for
    native resolution — see the module docstring); ``task`` is the new
    dataset's task, which the archive's layout must match. Returns
    (classes, imported_image_count). The caller owns cleanup of ``dest_dir``
    on failure and the atomic rename into place on success.
    """
    if task not in _TASK_NAMES:
        raise DatasetImportError(f"Cannot import datasets for the '{task}' task")
    extract_dir = os.path.join(dest_dir, ".extract")
    os.makedirs(os.path.join(dest_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(dest_dir, "labels"), exist_ok=True)
    try:
        _extract(zip_path, extract_dir, max_total_mb)
        if task in _CLASS_FOLDER_TASKS:
            if _looks_like_detection(extract_dir):
                unit = "person" if task == "face" else "class"
                raise DatasetImportError(
                    "This ZIP is an object-detection dataset (images/ + labels/); this "
                    f"device runs {_TASK_NAMES[task]}, which imports one folder of "
                    f"images per {unit}")
            classes, count = _import_class_folders(extract_dir, dest_dir, img_size, task)
        else:
            if _looks_like_classification(extract_dir):
                raise DatasetImportError(
                    "This ZIP is a classification dataset (one folder of images per "
                    f"class); this device runs {_TASK_NAMES[task]}")
            classes = _find_classes(extract_dir)
            pairs = _collect_pairs(extract_dir)
            count = _normalize(pairs, classes, dest_dir, img_size, task)
    finally:
        shutil.rmtree(extract_dir, ignore_errors=True)

    with open(os.path.join(dest_dir, "classes.json"), "w") as f:
        json.dump(classes, f, ensure_ascii=False)
    logger.info("Imported dataset: %d images, %d classes", count, len(classes))
    return classes, count


# ── extraction ────────────────────────────────────────────────────────────────

def _extract(zip_path: str, extract_dir: str, max_total_mb: int) -> None:
    """Extract, enforcing real (written-byte) limits — see the caps above."""
    os.makedirs(extract_dir, exist_ok=True)
    budget = max_total_mb * 1024 * 1024
    total = 0
    entries = 0
    try:
        archive = zipfile.ZipFile(zip_path)
    except zipfile.BadZipFile:
        raise DatasetImportError("The uploaded file is not a valid ZIP archive") from None
    with archive:
        for info in archive.infolist():
            name = info.filename
            if info.is_dir() or name.startswith("__MACOSX/"):
                continue
            if os.path.basename(name).startswith("."):
                continue
            entries += 1
            if entries > _MAX_ZIP_ENTRIES:
                raise DatasetImportError(
                    f"Archive has more than {_MAX_ZIP_ENTRIES} entries")
            if stat.S_ISLNK(info.external_attr >> 16):
                raise DatasetImportError(f"ZIP entry '{name}' is a symlink")
            target = os.path.normpath(os.path.join(extract_dir, name))
            if not target.startswith(os.path.abspath(extract_dir) + os.sep) \
                    and target != os.path.abspath(extract_dir):
                raise DatasetImportError(f"ZIP entry escapes the archive: '{name}'")
            os.makedirs(os.path.dirname(target), exist_ok=True)
            written = 0
            with archive.open(info) as src, open(target, "wb") as dst:
                while True:
                    chunk = src.read(1 << 20)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > _MAX_ZIP_ENTRY_BYTES:
                        raise DatasetImportError(
                            f"ZIP entry '{name}' exceeds "
                            f"{_MAX_ZIP_ENTRY_BYTES // (1024 * 1024)} MB")
                    if total + written > budget:
                        raise DatasetImportError(
                            f"Uncompressed dataset exceeds the {max_total_mb} MB limit"
                        )
                    dst.write(chunk)
            total += written
            # A wildly implausible ratio is a decompression bomb even when it
            # fits the budget.
            compressed = max(info.compress_size, 1)
            if written > (1 << 20) and written // compressed > _MAX_ZIP_RATIO:
                raise DatasetImportError(
                    f"ZIP entry '{name}' has an implausible compression ratio")


# ── classes ───────────────────────────────────────────────────────────────────

def _find_classes(extract_dir: str) -> List[str]:
    """Find classes."""
    yaml_path = _find_file(extract_dir, ("data.yaml", "dataset.yaml"), max_depth=2)
    if yaml_path is not None:
        return _classes_from_yaml(yaml_path)
    txt_path = _find_file(extract_dir, ("classes.txt",), max_depth=2)
    if txt_path is not None:
        with open(txt_path, "r") as f:
            names = [line.strip() for line in f if line.strip()]
        return _validate_classes(names, txt_path)
    raise DatasetImportError(
        "No data.yaml or classes.txt found — the ZIP must be a YOLO "
        "dataset export (images/ + labels/ + data.yaml)"
    )


def _find_file(root: str, names: Tuple[str, ...], max_depth: int) -> Optional[str]:
    """Find file."""
    base_depth = root.rstrip(os.sep).count(os.sep)
    for dirpath, dirnames, filenames in os.walk(root):
        if dirpath.count(os.sep) - base_depth >= max_depth:
            dirnames[:] = []
            continue
        for name in names:
            if name in filenames:
                return os.path.join(dirpath, name)
    return None


def _classes_from_yaml(yaml_path: str) -> List[str]:
    """Classes from yaml."""
    try:
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise DatasetImportError(f"Could not parse data.yaml: {exc}") from exc
    names = (data or {}).get("names")
    if isinstance(names, dict):
        # id->name mapping: order by integer id when possible, else by sorted
        # key order. Keep the dict in its own variable so the except branch still
        # sees a dict (names is rebound to a list above).
        names_map = names
        try:
            names = [str(names_map[k]) for k in sorted(names_map, key=int)]
        except (ValueError, TypeError):
            names = [str(v) for _, v in sorted(names_map.items())]
    if not isinstance(names, list) or not names:
        raise DatasetImportError("data.yaml has no 'names' class list")
    return _validate_classes([str(n) for n in names], yaml_path)


def _validate_classes(names: List[str], source: str, task: str = "detect") -> List[str]:
    """Validate classes; for ``face`` the reserved ``unknown`` is refused as a person."""
    if not names:
        raise DatasetImportError(f"No class names found in {os.path.basename(source)}")
    for name in names:
        if not name or len(name) > 64 or not all(c in _NAME_SAFE for c in name):
            raise DatasetImportError(f"Invalid class name '{name}'")
        if task == FACE and not person_key(name):
            raise DatasetImportError(f"'{name}' cannot be a person: a name cannot be a colour alone")
        if task == FACE and is_reserved_face_name(name):
            raise DatasetImportError(
                f"'{name}' cannot be a person: 'unknown' is reserved for a face nobody matches")
    # Two people whose names differ only in case or colour could not be told apart.
    keys = [person_key(n) for n in names] if task == FACE else names
    if len(set(keys)) != len(names):
        raise DatasetImportError("Duplicate class names in the dataset")
    return names


# ── layout ────────────────────────────────────────────────────────────────────

def _looks_like_classification(extract_dir: str) -> bool:
    """True for the ultralytics classification layout: ``<split>/<class>/*.jpg``.

    Images grouped in at least two folders, no ``images/`` directory and no
    label files at all (a ``classes.txt`` does not count). A detection export
    always has label files beside or parallel to its images.
    """
    image_dirs = set()
    for dirpath, _dirnames, filenames in os.walk(extract_dir):
        rel = os.path.relpath(dirpath, extract_dir)
        for name in filenames:
            lower = name.lower()
            if lower.endswith(".txt") and lower != "classes.txt":
                return False
            if lower.endswith(_IMAGE_EXTS):
                if rel == "." or "images" in rel.split(os.sep):
                    return False
                image_dirs.add(rel)
    return len(image_dirs) >= 2


def _looks_like_detection(extract_dir: str) -> bool:
    """True for a YOLO detection export: label files or an ``images/`` directory."""
    for dirpath, _dirnames, filenames in os.walk(extract_dir):
        if "images" in os.path.relpath(dirpath, extract_dir).split(os.sep):
            return True
        for name in filenames:
            lower = name.lower()
            if lower.endswith(".txt") and lower != "classes.txt":
                return True
    return False


# ── classification (folder per class) ─────────────────────────────────────────

def _holds_images(folder: str) -> bool:
    """True when ``folder`` holds an image at any depth."""
    for _dirpath, _dirnames, filenames in os.walk(folder):
        if any(name.lower().endswith(_IMAGE_EXTS) for name in filenames):
            return True
    return False


def _is_class_folder(path: str, name: str) -> bool:
    """Whether the entry ``name`` at ``path`` is a class folder.

    A hidden folder counts only when it holds images: a class name may begin
    with a dot (``.defective`` exports as ``train/.defective/``), while hidden
    tool folders (``.git``, ``.ipynb_checkpoints``) hold none.
    """
    return os.path.isdir(path) and (not name.startswith(".") or _holds_images(path))


def _unwrap(root: str) -> str:
    """Descend through lone wrapper folders (``pets.zip`` → ``pets/train/cat``).

    Stops at a split folder, at a folder holding images directly (a
    one-class archive's class folder) or where more than one entry lives.
    """
    while True:
        entries = [e for e in os.listdir(root)
                   if not e.startswith(".") or _is_class_folder(os.path.join(root, e), e)]
        if len(entries) != 1 or entries[0] in _CLASS_SPLITS:
            return root
        child = os.path.join(root, entries[0])
        if not os.path.isdir(child):
            return root
        if any(name.lower().endswith(_IMAGE_EXTS) for name in os.listdir(child)):
            return root
        root = child


def _import_class_folders(extract_dir: str, dest_dir: str, img_size: int,
                          task: str) -> Tuple[List[str], int]:
    """Import a folder-per-class archive; returns (classes, image count).

    Class folders sit at the archive root or inside split folders (all
    splits are merged — the device builds its own split per job). Each image
    is stored like a detection import (same geometry, same caps) and labeled
    with one line holding its class id.
    """
    root = _unwrap(extract_dir)
    split_roots = [os.path.join(root, s) for s in _CLASS_SPLITS
                   if os.path.isdir(os.path.join(root, s))]
    images: List[Tuple[str, str]] = []
    folders = set()
    for class_root in split_roots or [root]:
        for entry in sorted(os.listdir(class_root)):
            class_dir = os.path.join(class_root, entry)
            if not _is_class_folder(class_dir, entry):
                continue
            folders.add(entry)
            for dirpath, _dirnames, filenames in os.walk(class_dir):
                for name in sorted(filenames):
                    if name.lower().endswith(_IMAGE_EXTS):
                        images.append((os.path.join(dirpath, name), entry))
    if not images:
        raise DatasetImportError(
            "No images found in class folders — a classification ZIP holds one "
            "folder of images per class")

    listed = _find_file(root, ("classes.txt",), max_depth=1)
    if listed is not None:
        with open(listed, "r") as f:
            classes = _validate_classes([line.strip() for line in f if line.strip()], listed,
                                        task)
        unknown = sorted(folders - set(classes))
        if unknown:
            raise DatasetImportError(
                f"Class folder '{unknown[0]}' is not listed in classes.txt")
    else:
        classes = _validate_classes(sorted(folders), "the class folders", task)

    for image_path, class_name in images:
        image_id = _store_image(image_path, dest_dir, img_size)
        with open(os.path.join(dest_dir, "labels", f"{image_id}.txt"), "w") as f:
            f.write(f"{classes.index(class_name)}\n")
    return classes, len(images)


# ── image/label pairing ───────────────────────────────────────────────────────

def _label_for(image_path: str) -> str:
    """YOLO pairing: swap the innermost 'images' path component for 'labels'
    (or look for a sibling .txt when there is no images/ directory)."""
    parts = image_path.split(os.sep)
    for i in range(len(parts) - 2, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    parts[-1] = os.path.splitext(parts[-1])[0] + ".txt"
    return os.sep.join(parts)


def _collect_pairs(extract_dir: str) -> List[Tuple[str, Optional[str]]]:
    """(image_path, label_path|None) pairs; labels resolved YOLO-style."""
    all_images: List[str] = []
    for dirpath, _dirnames, filenames in os.walk(extract_dir):
        for name in sorted(filenames):
            if name.lower().endswith(_IMAGE_EXTS):
                all_images.append(os.path.join(dirpath, name))
    in_images_dirs = [
        p for p in all_images
        if "images" in os.path.relpath(p, extract_dir).split(os.sep)[:-1]
    ]
    # YOLO layout present: ignore stray images outside images/ directories.
    images = in_images_dirs or all_images
    if not images:
        raise DatasetImportError("No images found in the ZIP")
    pairs: List[Tuple[str, Optional[str]]] = []
    for image_path in images:
        label_path = _label_for(image_path)
        pairs.append((image_path,
                      label_path if os.path.isfile(label_path) else None))
    return pairs


# ── normalization ─────────────────────────────────────────────────────────────

def _label_rows(path: str, n_classes: int) -> Iterator[Tuple[int, int, List[float]]]:
    """Yield ``(lineno, class_id, coords)`` per non-blank row of a YOLO label file.

    Numbers, the class range and the 0..1 coordinate range are validated
    here; the row's shape (box or polygon) is the caller's to check.
    """
    with open(path, "r") as f:
        for lineno, line in enumerate(f, 1):
            parts = line.split()
            if not parts:
                continue
            try:
                cls = int(parts[0])
                coords = [float(v) for v in parts[1:]]
            except ValueError:
                raise DatasetImportError(
                    f"Invalid label file '{os.path.basename(path)}' line {lineno}: "
                    f"not numeric"
                ) from None
            if not 0 <= cls < n_classes:
                raise DatasetImportError(
                    f"Invalid label file '{os.path.basename(path)}' line {lineno}: "
                    f"class id {cls} out of range (dataset has {n_classes} classes)"
                )
            if not all(0.0 <= v <= 1.0 for v in coords):
                raise DatasetImportError(
                    f"Invalid label file '{os.path.basename(path)}' line {lineno}: "
                    f"coordinates must be normalized (0..1)"
                )
            yield lineno, cls, coords


def _is_polygon(coords: List[float]) -> bool:
    return len(coords) >= 6 and len(coords) % 2 == 0


def _parse_label_file(path: str, n_classes: int) -> List[Tuple[int, float, float, float, float]]:
    """Parse one YOLO detection label file into (class, cx, cy, w, h) boxes.

    A segmentation row ("class x1 y1 x2 y2 ..." polygon) is refused: the
    archive belongs to a segmentation dataset, not to this detection one.
    """
    boxes: List[Tuple[int, float, float, float, float]] = []
    for lineno, cls, coords in _label_rows(path, n_classes):
        if len(coords) == 4:
            boxes.append((cls, coords[0], coords[1], coords[2], coords[3]))
        elif _is_polygon(coords):
            raise DatasetImportError(
                f"Label file '{os.path.basename(path)}' line {lineno} is a "
                f"segmentation polygon; this device runs object detection"
            )
        else:
            raise DatasetImportError(
                f"Invalid label file '{os.path.basename(path)}' line {lineno}: "
                f"expected 'class cx cy w h', got {len(coords) + 1} values"
            )
    return boxes


def _parse_polygon_file(path: str, n_classes: int) -> List[Tuple[int, List[List[float]]]]:
    """Parse one YOLO segmentation label file into (class, [[x, y], …]) rings.

    A detection row ("class cx cy w h") is refused: the archive belongs to a
    detection dataset, not to this segmentation one.
    """
    rings: List[Tuple[int, List[List[float]]]] = []
    for lineno, cls, coords in _label_rows(path, n_classes):
        if _is_polygon(coords):
            rings.append((cls, [[coords[i], coords[i + 1]] for i in range(0, len(coords), 2)]))
        elif len(coords) == 4:
            raise DatasetImportError(
                f"Label file '{os.path.basename(path)}' line {lineno} is a "
                f"detection box; this device runs segmentation"
            )
        else:
            raise DatasetImportError(
                f"Invalid label file '{os.path.basename(path)}' line {lineno}: "
                f"expected 'class x1 y1 x2 y2 x3 y3 ...', got {len(coords) + 1} values"
            )
    return rings


def _image_dimensions(path: str) -> Optional[Tuple[int, int]]:
    """(width, height) from a PNG/JPEG/BMP header, without decoding.

    Returns None when the format is unrecognized or the header is malformed —
    the decoder then decides. Stdlib-only on purpose.
    """
    try:
        with open(path, "rb") as f:
            return _dimensions_from_stream(f)
    except OSError:
        return None


def image_dimensions_from_bytes(data: bytes) -> Optional[Tuple[int, int]]:
    """(width, height) of an in-memory PNG/JPEG/BMP, header-only (see above)."""
    return _dimensions_from_stream(io.BytesIO(data))


def _dimensions_from_stream(f: BinaryIO) -> Optional[Tuple[int, int]]:
    """Header-only dimension parse shared by the path and bytes entry points."""
    head = f.read(32)
    if head[:8] == b"\x89PNG\r\n\x1a\n" and len(head) >= 24:
        width, height = struct.unpack(">II", head[16:24])
        return width, height
    if head[:2] == b"BM" and len(head) >= 26:
        width, height = struct.unpack("<ii", head[18:26])
        return abs(width), abs(height)
    if head[:2] == b"\xff\xd8":
        f.seek(2)
        while True:
            marker = f.read(2)
            if len(marker) < 2 or marker[0] != 0xFF:
                return None
            code = marker[1]
            if code == 0x01 or 0xD0 <= code <= 0xD8:
                continue
            length_raw = f.read(2)
            if len(length_raw) < 2:
                return None
            (seg_len,) = struct.unpack(">H", length_raw)
            if 0xC0 <= code <= 0xCF and code not in (0xC4, 0xC8, 0xCC):
                body = f.read(5)
                if len(body) < 5:
                    return None
                height, width = struct.unpack(">HH", body[1:5])
                return width, height
            f.seek(seg_len - 2, 1)
    return None


def _decode_and_store(image_path: str, dest_dir: str,
                      img_size: int) -> Tuple[str, int, int]:
    """Decode one archive image, store it in the ``img_size`` geometry.

    Returns ``(image_id, source width, source height)``.
    """
    # Bound the decode before cv2.imread allocates the full raster.
    dims = _image_dimensions(image_path)
    if dims is not None and dims[0] * dims[1] > _MAX_IMAGE_PIXELS:
        raise DatasetImportError(
            f"Image '{os.path.basename(image_path)}' is too large "
            f"({dims[0]}x{dims[1]}; limit {_MAX_IMAGE_PIXELS} pixels)"
        )
    img = cv2.imread(image_path)
    if img is None:
        raise DatasetImportError(
            f"Could not decode image '{os.path.basename(image_path)}'"
        )
    h, w = img.shape[:2]
    stored = letterbox_square(img, img_size) if img_size > 0 else img
    ok, buf = cv2.imencode(".jpg", stored, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise DatasetImportError(
            f"Could not re-encode image '{os.path.basename(image_path)}'"
        )
    image_id = str(uuid.uuid4())
    with open(os.path.join(dest_dir, "images", f"{image_id}.jpg"), "wb") as f:
        f.write(buf.tobytes())
    return image_id, w, h


def _store_image(image_path: str, dest_dir: str, img_size: int) -> str:
    """Store one archive image in the ``img_size`` geometry; returns its id."""
    return _decode_and_store(image_path, dest_dir, img_size)[0]


def _polygon_lines(rings: List[Tuple[int, List[List[float]]]], w: int, h: int,
                   img_size: int) -> List[str]:
    """Segmentation rows for one stored image: letterbox-mapped, then normalized.

    Each archive row is its own instance (the YOLO file has no instance
    column); normalization (rasterize and re-extract) may drop a tiny ring
    or split one.
    """
    from conecsa_common.polygons import normalize_rings

    if img_size > 0:
        # Same rounding as letterbox_square, like the box transform below.
        scale = min(img_size / w, img_size / h)
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        left, top = (img_size - nw) // 2, (img_size - nh) // 2
        size_w = size_h = img_size
    else:
        nw, nh, left, top, size_w, size_h = w, h, 0, 0, w, h
    lines = []
    for cls, points in rings:
        mapped = [[(x * nw + left) / size_w, (y * nh + top) / size_h] for x, y in points]
        for ring in normalize_rings([mapped], size_w, size_h):
            lines.append(f"{cls} " + " ".join(f"{x:.6f} {y:.6f}" for x, y in ring))
    return lines


def _normalize(pairs, classes: List[str], dest_dir: str, img_size: int,
               task: str = "detect") -> int:
    """Re-encode every image into ``dest_dir`` in the ``img_size`` geometry and
    write its label file (letterbox-transformed, or verbatim for native;
    polygon rows are also normalized for a segmentation dataset)."""
    count = 0
    letterbox = img_size > 0
    for image_path, label_path in pairs:
        if task == "segment":
            rings = _parse_polygon_file(label_path, len(classes)) if label_path else []
            image_id, w, h = _decode_and_store(image_path, dest_dir, img_size)
            lines = _polygon_lines(rings, w, h, img_size) if rings else []
            if lines:
                with open(os.path.join(dest_dir, "labels", f"{image_id}.txt"), "w") as f:
                    f.write("\n".join(lines) + "\n")
            count += 1
            continue
        boxes = _parse_label_file(label_path, len(classes)) if label_path else []
        image_id, w, h = _decode_and_store(image_path, dest_dir, img_size)

        if boxes:
            if letterbox:
                # Same rounding as letterbox_square so boxes land exactly on
                # the letterboxed pixels.
                scale = min(img_size / w, img_size / h)
                nw = max(1, int(round(w * scale)))
                nh = max(1, int(round(h * scale)))
                left = (img_size - nw) // 2
                top = (img_size - nh) // 2
                lines = []
                for cls, cx, cy, bw, bh in boxes:
                    ncx = min(max((cx * nw + left) / img_size, 0.0), 1.0)
                    ncy = min(max((cy * nh + top) / img_size, 0.0), 1.0)
                    nbw = min(max(bw * nw / img_size, 0.0), 1.0)
                    nbh = min(max(bh * nh / img_size, 0.0), 1.0)
                    lines.append(f"{cls} {ncx:.6f} {ncy:.6f} {nbw:.6f} {nbh:.6f}")
            else:
                # Native storage keeps the source pixels, so the normalized
                # coordinates already describe the stored image.
                lines = [
                    f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
                    for cls, cx, cy, bw, bh in boxes
                ]
            with open(os.path.join(dest_dir, "labels", f"{image_id}.txt"), "w") as f:
                f.write("\n".join(lines) + "\n")
        count += 1
    return count
