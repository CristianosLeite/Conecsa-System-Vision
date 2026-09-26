# NVIDIA's rtl8822_setting.bin (package tegra-firmware-rtl8822) is the only
# source of channel plans for the RTL8822CE vendor driver, and it carries just
# the world-wide plan 0x7F, in which every 5 GHz channel is passive (NO_IR):
# the radio may not start an access point on 36-48 until it has heard a beacon
# there. The driver has no country table, so wpa_supplicant's country= cannot
# change that. rtl8822-chplan.py adds plan 0x62 (Realtek's id for Brazil) with
# band 1 (36-48) allowed to initiate; bands 2-3 stay passive + DFS and the
# power-limit records are byte-identical. conecsa-bootstrap's
# /etc/modprobe.d/rtl8822ce.conf selects the plan at module load.
#
# Scoped to PV 36.5.2 like the other meta-tegra overrides: an L4T bump must
# re-run the script against the new blob (it refuses a file it does not
# recognise) before this bbappend is renamed.
FILESEXTRAPATHS:prepend := "${THISDIR}/files:"

# python3native: run the script with the native sysroot interpreter rather
# than whatever python3 the build host puts on PATH (the class exports
# PYTHON; there is no PYTHON3 variable in Scarthgap).
inherit python3native

SRC_URI += "file://rtl8822-chplan.py"

do_install:append() {
    ${PYTHON} ${WORKDIR}/rtl8822-chplan.py \
        ${D}${nonarch_base_libdir}/firmware/rtl8822_setting.bin \
        ${D}${nonarch_base_libdir}/firmware/rtl8822_setting.bin
}
