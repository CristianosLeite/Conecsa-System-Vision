SUMMARY = "Conecsa System Vision minimal image for NVIDIA Jetson Orin Nano"
DESCRIPTION = "Minimal Yocto image for hosting the Conecsa System Vision \
app in Docker containers. Bundles the L4T drivers, the CUDA/TRT/cuDNN runtime, \
Docker + nvidia-container-runtime, V4L2, GPIO and a Weston kiosk session on the \
DisplayPort."
LICENSE = "MIT"

inherit core-image

# SSH for administration; package-management keeps rpm/dnf on the device.
# debug-tweaks leaves root with an empty password for the serial console (the
# provisioning and recovery channel). It also sets PermitRootLogin yes /
# PermitEmptyPasswords yes in sshd_config, but conecsa-bootstrap's sshd
# drop-in is read first and makes SSH key-only.
IMAGE_FEATURES += "ssh-server-openssh package-management debug-tweaks"

IMAGE_INSTALL = " \
    packagegroup-core-boot \
    packagegroup-conecsa-nvidia \
    packagegroup-conecsa-runtime \
    packagegroup-conecsa-camera \
    packagegroup-conecsa-python-gpio \
    packagegroup-conecsa-wayland \
    conecsa-bootstrap \
    kernel-modules \
    "

# The hub kiosk recipe exists only in the private monorepo (the public mirror
# exported by scripts/export-mirror.sh ships this layer without it), so it is
# installed only when its recipe directory is present in the layer.
IMAGE_INSTALL += "${@'conecsa-hub-kiosk' if os.path.isdir(os.path.join(os.path.dirname(d.getVar('FILE')), '../recipes-conecsa/conecsa-hub-kiosk')) else ''}"

# weston.service is WantedBy=graphical.target, but rootfs-postcommands
# defaults this image to multi-user.target (no x11-base/weston in
# IMAGE_FEATURES). graphical.target pulls multi-user.target in, so the
# Docker stack is unaffected.
SYSTEMD_DEFAULT_TARGET = "graphical.target"

# tegraflash produces a self-contained tarball (initrd-flash and the signed
# bootloader binaries) that writes the image to the Orin Nano NVMe in
# recovery mode.
IMAGE_FSTYPES = "tegraflash"

# Room for the Conecsa container images in /var/lib/docker
# (conecsa-os-base:base ~6 GB; others ~2 GB each).
IMAGE_ROOTFS_EXTRA_SPACE = "30000000"
