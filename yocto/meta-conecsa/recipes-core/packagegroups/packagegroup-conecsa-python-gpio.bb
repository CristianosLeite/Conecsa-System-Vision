SUMMARY = "Python 3 + Jetson.GPIO for bind-mount into the os-base hardware agent"
DESCRIPTION = "The os-base hardware agent imports Jetson.GPIO. The module is \
mounted read-only from the host at /usr/lib/python3/dist-packages/Jetson — \
so it must exist in the Jetson rootfs."
LICENSE = "MIT"

inherit packagegroup

RDEPENDS:${PN} = " \
    python3 \
    python3-core \
    python3-jetson-gpio \
    "
