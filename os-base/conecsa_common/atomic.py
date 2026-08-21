"""Power-cut-safe file persistence.

The device has no battery: a power cut can land between any two writes. A
plain ``open(path, "w").write(...)`` truncates first, so the crash window
leaves an empty or half-written file — which for settings, model selection,
or dataset metadata silently erases configuration. Every durable state file
should be written through :func:`atomic_write_bytes`:

    temp file in the same directory  →  write  →  fsync  →  os.replace
    →  fsync of the parent directory

``os.replace`` is atomic on POSIX, and the two fsyncs make both the content
and the directory entry durable before the call returns.

On load, :func:`read_json` distinguishes *missing* (first boot — default
silently) from *corrupt* (something was lost — log an error, quarantine the
file as ``<name>.corrupt`` for diagnosis, then default) instead of treating
both as "use defaults".
"""
import json
import logging
import os
import tempfile
from typing import Any

logger = logging.getLogger(__name__)


def atomic_write_bytes(path: str, data: bytes, mode: int = 0o600) -> None:
    """Durably replace ``path`` with ``data`` (see the module docstring)."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp-", dir=directory)
    try:
        try:
            os.fchmod(fd, mode)
            # os.write may write fewer bytes than asked (signals, large
            # buffers); loop so the temp file is complete before the fsync.
            view = memoryview(data)
            while view:
                written = os.write(fd, view)
                view = view[written:]
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    _fsync_dir(directory)


def atomic_write_json(path: str, value: Any, mode: int = 0o600, **dumps_kwargs) -> None:
    """Durably replace ``path`` with ``value`` serialized as JSON."""
    data = json.dumps(value, **dumps_kwargs).encode("utf-8")
    atomic_write_bytes(path, data, mode=mode)


def read_json(path: str, default: Any) -> Any:
    """Load JSON state, reporting corruption instead of hiding it.

    Missing file: return ``default`` silently (first boot). Unreadable or
    unparseable file: log an error, rename it to ``<path>.corrupt`` so the
    evidence survives for diagnosis, and return ``default``.
    """
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        return default
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        logger.error("corrupt state file %s (%s); quarantining as .corrupt", path, exc)
        try:
            os.replace(path, path + ".corrupt")
        except OSError:
            logger.exception("could not quarantine %s", path)
        return default


def _fsync_dir(directory: str) -> None:
    """Make the rename itself durable. Best-effort on filesystems that refuse
    to open a directory (the data fsync already happened)."""
    try:
        dir_fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(dir_fd)
    except OSError:
        pass
    finally:
        os.close(dir_fd)
