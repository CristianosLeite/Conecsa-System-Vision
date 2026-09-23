# Conecsa kiosk: replace the stock weston.ini with a kiosk-shell config that
# autolaunches the hub kiosk wrapper (see conecsa-hub-kiosk), and make weston
# work without PAM — this distro has no `pam` in DISTRO_FEATURES, so the
# stock unit's PAMName=weston-autologin is silently ignored and no logind
# session ever creates XDG_RUNTIME_DIR or grants seat access.
FILESEXTRAPATHS:prepend := "${THISDIR}/files:"

# Upstream requires 'pam' when the init manager is systemd, because the stock
# weston.service relies on PAMName= (logind session → XDG_RUNTIME_DIR + seat).
# The 10-conecsa-kiosk.conf drop-in replaces that mechanism entirely
# (RuntimeDirectory= + the seatd daemon), so the requirement does not apply on
# this distro. Same removal in the weston bbappend.
REQUIRED_DISTRO_FEATURES:remove = "pam"

# DISTRO_FEATURES has x11, which would pull weston-xwayland and sed
# `xwayland=true` into weston.ini — the kiosk is Wayland-only, and with no
# /tmp/.X11-unix the xwayland module crashes weston at startup.
PACKAGECONFIG:remove = "xwayland"

SRC_URI += "file://10-conecsa-kiosk.conf file://99-conecsa-kiosk-hotplug.rules"

do_install:append() {
    install -Dm0644 ${WORKDIR}/10-conecsa-kiosk.conf \
        ${D}${systemd_system_unitdir}/weston.service.d/10-conecsa-kiosk.conf

    # The drop-in skips weston while no display is connected; this rule starts
    # it when one is plugged in.
    install -Dm0644 ${WORKDIR}/99-conecsa-kiosk-hotplug.rules \
        ${D}${nonarch_base_libdir}/udev/rules.d/99-conecsa-kiosk-hotplug.rules

    # The stock unit Requires/After systemd-user-sessions.service, which does
    # not exist when systemd is built without PAM — a Requires= on a missing
    # unit makes the whole service fail to start.
    sed -i '/systemd-user-sessions.service/d' \
        ${D}${systemd_system_unitdir}/weston.service
}

FILES:${PN} += "${systemd_system_unitdir}/weston.service.d/10-conecsa-kiosk.conf \
                ${nonarch_base_libdir}/udev/rules.d/99-conecsa-kiosk-hotplug.rules"
