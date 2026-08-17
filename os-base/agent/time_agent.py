"""System-clock control for the hardware agent.

The Jetson has no RTC battery, so on a LAN with no internet it boots with a
clock that predates the hub CA's `not_before` (2020-01-01). nginx then rejects
the hub's client certificate with "certificate is not yet valid" and the device
looks permanently offline right after pairing. The hub therefore relays its own
wall clock (at pairing, and on every status poll) and the api-gateway forwards
it here — setting the clock is a host operation, and this privileged agent is
the only container that owns host state.

Two invariants keep a wrong clock from doing damage in the other direction:

* **Monotonic floor** — the time is never stepped below the value persisted in
  ``FLOOR_PATH`` (the same file the host's `conecsa-fake-hwclock` units read at
  boot and write at shutdown). A hub whose own clock lost time — the kiosk runs
  on a Jetson too — can therefore never drag a device backwards.
* Every accepted step rewrites that floor immediately, so the acquired time
  survives an abrupt power cut without waiting for the 15-minute save timer.

The clock is set with ``clock_settime(CLOCK_REALTIME)`` rather than logind's
sibling ``org.freedesktop.timedate1.SetTime``: timedated refuses while
systemd-timesyncd is active, and the image keeps timesyncd enabled (pinned to
public servers) for the sites that do have internet.
"""
import ctypes
import logging
import os
import subprocess
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# Shared with conecsa-fake-hwclock.sh on the host (bind-mounted into this
# container). One line, UTC, in FLOOR_FORMAT — a fixed-width format so the
# shell side can compare two stamps by sorting them.
FLOOR_PATH = os.environ.get("CONECSA_CLOCK_FLOOR", "/var/lib/conecsa/fake-hwclock")
FLOOR_FORMAT = "%Y-%m-%d %H:%M:%S"

CLOCK_REALTIME = 0


class _Timespec(ctypes.Structure):
    _fields_ = [("tv_sec", ctypes.c_long), ("tv_nsec", ctypes.c_long)]


class TimeAgent:
    """Sets the host wall clock on behalf of the hub."""

    @staticmethod
    def read_floor() -> Optional[datetime]:
        """The last persisted time, or ``None`` when unreadable/absent."""
        try:
            with open(FLOOR_PATH, "r", encoding="utf-8") as fh:
                raw = fh.read().strip()
        except OSError:
            return None
        if not raw:
            return None
        try:
            return datetime.strptime(raw, FLOOR_FORMAT).replace(tzinfo=timezone.utc)
        except ValueError:
            logger.warning("clock: ignoring unparseable floor %r in %s", raw, FLOOR_PATH)
            return None

    @staticmethod
    def write_floor(moment: datetime) -> None:
        """Persist *moment* as the new floor (best-effort; never raises).

        Flushed all the way to disk: the whole point of the floor is to survive
        an abrupt power cut, which a write sitting in the page cache would not.
        """
        directory = os.path.dirname(FLOOR_PATH)
        try:
            os.makedirs(directory, exist_ok=True)
            tmp = FLOOR_PATH + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                fh.write(moment.astimezone(timezone.utc).strftime(FLOOR_FORMAT) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, FLOOR_PATH)
            # Persist the rename itself, so the new name survives too.
            fd = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except OSError as exc:
            logger.warning("clock: could not persist floor to %s (%s)", FLOOR_PATH, exc)

    @staticmethod
    def _apply(moment: datetime) -> None:
        """Step CLOCK_REALTIME to *moment*. Raises OSError on failure."""
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        epoch = moment.timestamp()
        spec = _Timespec(int(epoch), int((epoch % 1) * 1_000_000_000))
        if libc.clock_settime(CLOCK_REALTIME, ctypes.byref(spec)) != 0:
            err = ctypes.get_errno()
            raise OSError(err, os.strerror(err))

    @staticmethod
    def _write_rtc() -> None:
        """Push the system clock into the RTC (survives a reboot, not a power cut)."""
        try:
            subprocess.run(
                ["hwclock", "--systohc", "--utc"],
                check=False, capture_output=True, timeout=10,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            logger.debug("clock: hwclock --systohc unavailable (%s)", exc)

    @staticmethod
    def set_system_time(epoch_millis: int, source: str = "") -> dict:
        """Set the host clock to *epoch_millis* (UTC). Returns ``{success, message}``.

        Refuses anything older than the persisted floor — see the module docstring.
        """
        if epoch_millis <= 0:
            return {"success": False, "message": f"invalid timestamp {epoch_millis}"}
        try:
            moment = datetime.fromtimestamp(epoch_millis / 1000.0, timezone.utc)
        except (OverflowError, OSError, ValueError) as exc:
            return {"success": False, "message": f"invalid timestamp {epoch_millis}: {exc}"}

        floor = TimeAgent.read_floor()
        if floor is not None and moment < floor:
            message = (
                f"refused {moment:%Y-%m-%d %H:%M:%S}Z from {source or 'hub'}: "
                f"older than the persisted floor {floor:%Y-%m-%d %H:%M:%S}Z"
            )
            logger.warning("clock: %s", message)
            return {"success": False, "message": message}

        before = datetime.now(timezone.utc)
        try:
            TimeAgent._apply(moment)
        except OSError as exc:
            logger.error("clock: could not set the system time (%s)", exc)
            return {"success": False, "message": f"clock_settime failed: {exc}"}

        TimeAgent.write_floor(moment)
        TimeAgent._write_rtc()
        drift = (moment - before).total_seconds()
        message = (
            f"clock set to {moment:%Y-%m-%d %H:%M:%S}Z from {source or 'hub'} "
            f"({drift:+.1f}s)"
        )
        logger.info("clock: %s", message)
        return {"success": True, "message": message}
