# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: Apache-2.0

"""Shared plain-Python helpers for every service built on `conecsa-os-base:base`.

The package root has no third-party dependencies (unlike ``conecsa_shm``, which
pulls numpy/cv2 at import), so lightweight consumers such as the api-gateway's
audit trail can import it without growing their footprint.

Modules:

- ``atomic``: power-cut-safe file persistence (write-fsync-rename) and JSON
  loading that reports corruption instead of silently defaulting.
- ``bounded_sqlite``: the bounded SQLite ring queue shared by the detection
  buffer and the audit trail.
- ``events``: the in-process event bus shared by the inference and training
  services.
- ``tasks``: the application task ids (detect / classify / segment).
- ``tiling``: SAHI-style tile grid + cross-tile merge for small-object
  detection (needs numpy).
- ``polygons``: segmentation polygon normalization (needs numpy/cv2).

Only ``atomic`` and ``bounded_sqlite`` are re-exported here; import the other
modules explicitly so the package root stays dependency-free.
"""

from .atomic import atomic_write_bytes, atomic_write_json, fsync_dir, read_json
from .bounded_sqlite import BoundedSqliteQueue

__all__ = ["BoundedSqliteQueue", "atomic_write_bytes", "atomic_write_json", "fsync_dir",
           "read_json"]
