#!/bin/sh
# Persist the wall clock across reboots on a board with no RTC battery.
#
# The Jetson loses the time whenever it loses power, and the sites this image
# ships to have no reachable NTP server (the .network files drop the DHCP NTP
# option and /etc/systemd/timesyncd.conf.d pins public servers). A clock that
# comes back before the hub CA's not_before makes nginx reject the hub's client
# certificate as "certificate is not yet valid" — the device pairs and then
# looks permanently offline. So: save the time periodically and at shutdown,
# and put it back at boot.
#
# The clock is only ever moved FORWARD. The saved stamp is a floor, not an
# authority: whatever real time source appears later (the hub, over the pairing
# handshake or the status poll — see gateway/clock.py — or timesyncd where
# there is internet) refines it. The `os` hardware agent writes the same file
# whenever it accepts a time from the hub, so an acquired time survives an
# abrupt power cut without waiting for the save timer.
#
# Usage: conecsa-fake-hwclock save|restore
set -eu

STATE_DIR=/var/lib/conecsa
STATE_FILE="$STATE_DIR/fake-hwclock"
# Fixed-width UTC — the format os-base/agent/time_agent.py reads and writes.
FMT="+%Y-%m-%d %H:%M:%S"

# The later of two stamps. Both are fixed-width UTC, so sorting them
# lexicographically orders them chronologically — no `date -d @epoch`, which
# not every date(1) implementation on this image is guaranteed to provide.
newest() {
    printf '%s\n%s\n' "$1" "$2" | sort | tail -n 1
}

save() {
    mkdir -p "$STATE_DIR"
    now="$(date -u "$FMT" 2>/dev/null || true)"
    if [ -z "$now" ]; then
        return 0
    fi

    # Keep the floor monotonic. A clock that came back wrong — restore ran
    # before /var was mounted, or someone moved it back by hand — must not
    # lower it, or a later hub time below the old floor would be accepted.
    # To reset the floor deliberately, delete $STATE_FILE.
    floor="$(cat "$STATE_FILE" 2>/dev/null || true)"
    if [ -n "$floor" ] && [ "$(newest "$floor" "$now")" = "$floor" ]; then
        return 0
    fi

    tmp="$STATE_FILE.tmp"
    printf '%s\n' "$now" > "$tmp"
    mv "$tmp" "$STATE_FILE"
    # The floor exists to survive a power cut, so get it out of the page cache.
    sync "$STATE_FILE" 2>/dev/null || sync
}

restore() {
    if [ ! -r "$STATE_FILE" ]; then
        return 0
    fi
    saved="$(cat "$STATE_FILE" 2>/dev/null || true)"
    if [ -z "$saved" ]; then
        return 0
    fi
    now="$(date -u "$FMT")"
    if [ "$(newest "$saved" "$now")" = "$now" ]; then
        return 0   # the clock is already at or ahead of the floor
    fi

    if date -u -s "$saved" >/dev/null 2>&1; then
        # Also push it into the RTC: that survives a reboot (just not a power
        # cut), so the next boot starts from the right time even before this
        # unit runs.
        hwclock --systohc --utc >/dev/null 2>&1 || true
        echo "fake-hwclock: clock restored to ${saved} UTC (was ${now} UTC)"
    else
        echo "fake-hwclock: could not set the clock to ${saved} UTC" >&2
    fi
}

case "${1:-}" in
    save)    save ;;
    restore) restore ;;
    *)       echo "usage: ${0##*/} save|restore" >&2; exit 2 ;;
esac
