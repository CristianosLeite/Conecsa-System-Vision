# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""
Device-level capture source, persisted beside the models.

Which camera a device inspects — its local V4L2 camera or a remote camera's stream —
belongs to the device, not to a model: it must survive a model switch and apply
at boot with no model selected. So it lives in its own file,
``camera_source.json`` in the models directory, and never in a model's
``*.settings.json``.

The file holds the stream token, a secret: it is written owner-only (0600) and
nothing here logs its content.
"""
import logging
import os
from typing import Any, Dict, Optional

from conecsa_common import atomic_write_json, read_json

from .config_validation import ConfigValidationError, validate_source_state

logger = logging.getLogger(__name__)


class CameraSourceStore:
    """Reads and writes ``camera_source.json`` (schema version 1)."""

    FILENAME = "camera_source.json"
    VERSION = 1

    def __init__(self, directory: str):
        self._path = os.path.join(directory, self.FILENAME)

    def load(self) -> Optional[Dict[str, Any]]:
        """Return the stored source, or ``None`` when none was ever saved.

        A file that does not parse is quarantined by ``read_json``; one that
        parses but does not validate is ignored. Either way the device comes up
        on its default source rather than on half a configuration.
        """
        data = read_json(self._path, None)
        if data is None:
            return None
        if not isinstance(data, dict) or data.get("version") != self.VERSION:
            logger.error("Ignoring %s: unsupported format", self._path)
            return None
        try:
            return validate_source_state(data)
        except ConfigValidationError as exc:
            # The message names the offending field, never its value.
            logger.error("Ignoring %s: %s", self._path, exc)
            return None

    def save(self, state: Dict[str, Any]) -> None:
        """Atomically persist a validated source state. Raises ``OSError``."""
        payload = {"version": self.VERSION, **state}
        atomic_write_json(self._path, payload, mode=0o600, indent=2)

    def restore(self, previous: Optional[Dict[str, Any]]) -> None:
        """Best-effort return to ``previous`` after a failed publish."""
        try:
            if previous is None:
                if os.path.exists(self._path):
                    os.unlink(self._path)
            else:
                self.save(previous)
        except OSError as exc:
            logger.error("Could not restore %s: %s", self._path, exc)
