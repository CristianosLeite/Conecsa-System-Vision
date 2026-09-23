# yocto

The lean host image for the Jetson Orin Nano. It replaces JetPack/Ubuntu with a Yocto rootfs built
by kas from the `scarthgap` branches of poky and OE4T meta-tegra: distro `conecsa`, image
`conecsa-image`, machine `jetson-orin-nano-devkit-nvme`, flashed to the NVMe as a tegraflash
tarball. The image carries the L4T BSP, the CUDA/TensorRT/cuDNN runtime and Docker with the NVIDIA
container runtime, and hosts the Docker stack from `docker-compose.yml`.

Full description: [docs/yocto-build.md](../docs/yocto-build.md).

## Layout

| Path | What it is |
|---|---|
| `kas-config.yml` | kas entry point: layer repositories, machine, distro and the `local.conf` additions (NVMe partition sizes, kernel arguments) |
| `meta-conecsa/conf/` | Layer configuration and the `conecsa` distro |
| `meta-conecsa/recipes-images/conecsa-image.bb` | The image: packagegroups, `graphical.target`, tegraflash output |
| `meta-conecsa/recipes-core/packagegroups/` | NVIDIA BSP and drivers, container runtime, camera, Wayland, Python GPIO |
| `meta-conecsa/recipes-conecsa/conecsa-bootstrap/` | Host configuration: Docker daemon, systemd-networkd and Wi-Fi template, SSH hardening, mDNS advertising, unique hostname, clock floor, zram and sysctl |
| `meta-conecsa/recipes-conecsa/jetson-gpio/` | Jetson.GPIO, bind-mounted into the `os-base` hardware agent |
| `meta-conecsa/recipes-poky-overrides/` | weston, weston-init and seatd changes for the display kiosk session without PAM |
| `meta-conecsa/recipes-oe-overrides/` | webkitgtk build fix for tegra |
| `meta-conecsa/recipes-tegra-overrides/` | Version-scoped fixes for meta-tegra pins |
| `build/`, `poky/`, `meta-*/` (other than `meta-conecsa`), `jetson-flash/` | Build output, layers cloned by kas and the extracted flash tree; gitignored |

## Build and flash

```bash
cd yocto
KAS_IMAGE_VERSION=4.7 kas-container build kas-config.yml   # as a regular user, never sudo
```

The first build takes hours and needs about 200 GB of disk. See
[Build](../docs/yocto-build.md#build), the
[post-build validations](../docs/yocto-build.md#post-build-validations-before-flashing) and
[Flashing](../docs/yocto-build.md#flashing-the-jetson-orin-nano).

## Constraints

- `docker-compose.yml` bind-mounts host paths from this image (TensorRT, cuDNN and cuDLA libraries,
  `/usr/lib/python3/dist-packages/Jetson`) and grants GID 999 (`render`) for GPIO. Change the image
  and the compose file together.
- No credentials are baked in: Wi-Fi and SSH keys are provisioned on the device after flashing.
- Root access is by design: SSH is key-only through the bootstrap sshd drop-in, and the serial
  console is the recovery channel. See
  [SSH hardening](../docs/yocto-build.md#ssh-hardening-key-only-permitted-hosts).
- kas follows the floating `scarthgap` branches. Name an override of a meta-tegra pin after the
  recipe version (`<recipe>_<PV>.bbappend`) and delete it once upstream is fixed; a bbappend for a
  recipe whose filename has no version takes no `_%`.
- Fixes belong in the recipes, followed by a rebuild, not in hand edits on a device.
- `yocto/` is excluded from pyright and ruff (the cloned layers are huge).

## Reference

- [Yocto build](../docs/yocto-build.md), including
  [Troubleshooting](../docs/yocto-build.md#troubleshooting) and
  [Updating NVIDIA versions](../docs/yocto-build.md#updating-nvidia-versions)
- [Running on the custom Yocto image](../docs/troubleshooting.md#running-on-the-custom-yocto-image-jetson-orin-nano)
