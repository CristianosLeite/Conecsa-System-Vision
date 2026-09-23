# Yocto build of the minimal Jetson Orin Nano image

Custom image to host the Conecsa System Vision application in Docker
containers, replacing the stock NVIDIA JetPack 6.2.2 (Ubuntu 22.04, ~2245
packages) with a lean Yocto rootfs. It follows the `scarthgap` branches of
poky and OE4T meta-tegra (Yocto 5.0, L4T R36.5.x, the JetPack 6.2 line with
the CUDA 12.6 / TensorRT 10.3 / cuDNN 9 versions that `docker-compose.yml`
expects).

## Architecture

```
yocto/
├── kas-config.yml                # kas entry point: layers, machine, local.conf additions
└── meta-conecsa/                 # custom Yocto layer
    ├── conf/
    │   ├── layer.conf
    │   └── distro/               # DISTRO = conecsa (conecsa.conf + include/conecsa.inc)
    ├── recipes-images/
    │   └── conecsa-image.bb      # final image, IMAGE_FSTYPES=tegraflash,
    │                             #   SYSTEMD_DEFAULT_TARGET=graphical.target
    ├── recipes-core/packagegroups/
    │   ├── packagegroup-conecsa-nvidia.bb       # L4T BSP + CUDA + TRT + cuDNN + cuDLA + NIC/Wi-Fi drivers
    │   ├── packagegroup-conecsa-runtime.bb      # docker-moby + nvidia-container-toolkit + zram
    │   ├── packagegroup-conecsa-camera.bb       # V4L2 + libv4l + v4l-utils
    │   ├── packagegroup-conecsa-wayland.bb      # Weston + Mesa + egl-wayland (kiosk)
    │   └── packagegroup-conecsa-python-gpio.bb  # Python 3 + Jetson.GPIO
    ├── recipes-conecsa/
    │   ├── conecsa-bootstrap/    # host config: docker daemon, networkd, SSH hardening,
    │   │                         #   mDNS, hostname, clock floor, zram, sysctl
    │   ├── jetson-gpio/python3-jetson-gpio_2.1.9.bb
    │   └── conecsa-hub-kiosk/    # hub kiosk wrapper (not in the public mirror)
    ├── recipes-poky-overrides/
    │   ├── weston-init/          # kiosk weston.ini + no-PAM systemd drop-in
    │   ├── weston/               # REQUIRED_DISTRO_FEATURES pam removal
    │   └── seatd/                # seatd.service (poky ships none for systemd)
    ├── recipes-oe-overrides/
    │   └── webkitgtk3/           # DEPENDS += virtual/libgbm (tegra)
    └── recipes-tegra-overrides/  # version-scoped fixes for meta-tegra pins
```

## Build host prerequisites

- Linux x86_64 (Ubuntu 22.04+ recommended; ARM works for building but not for flashing)
- Docker engine
- ≥ 200 GB free disk (cloned layers, downloads, sstate and tmp)
- ≥ 16 GB RAM (32 GB recommended for decent parallelism)
- USB-C → USB-A/C cable to put the Jetson into recovery mode at flash time

Install kas-container (preferably inside a virtual environment) and pull the
kas image version the build uses:

```bash
python -m pip install kas
docker pull ghcr.io/siemens/kas/kas:4.7
```

## Build

```bash
# IMPORTANT: pin the kas image to 4.7 (Debian 12 / gcc 12). Version 5.x uses
# Debian 13 / gcc 14, which is incompatible with Yocto scarthgap: gcc-cross
# do_compile fails with "undefined reference to `main`" inside native
# fixincludes.
export KAS_IMAGE_VERSION=4.7

# Build in the background under nohup, redirecting everything (stdout +
# stderr) to a log file, so the build survives a closed terminal and can be
# tailed from any other shell. Run as a regular user, never via sudo.
mkdir -p /tmp/kas-container
cd yocto && \
    nohup kas-container build kas-config.yml > /tmp/kas-container/build.log 2>&1 &

# Follow progress live from another shell (or the same one):
tail -f /tmp/kas-container/build.log

# Useful greps while tailing:
#   grep -E 'ERROR|FAILED'  /tmp/kas-container/build.log
#   grep -E 'Currently.*tasks running' /tmp/kas-container/build.log | tail -1
```

First build: **2–4h** (meta-tegra downloads are ~25 GB; CUDA build is expensive).
Incremental builds: **30–60 min** (depending on your hardware settings).
Downloads and shared state live in `yocto/build/downloads` and
`yocto/build/sstate-cache`; `yocto/build/tmp` holds the work directories and
the deploy output.

> Drop `nohup` and the trailing `&` if you want the command to stay in the
> foreground.

Final artifact (a self-contained tarball that bundles the flasher):
```
yocto/build/tmp/deploy/images/jetson-orin-nano-devkit-nvme/
└── conecsa-image-jetson-orin-nano-devkit-nvme.rootfs.tegraflash.tar.gz
```

The `.rootfs.tegraflash.tar.gz` unpacks to ~596 files, including the
`initrd-flash` script (the flash entry point), the `initrd-flash.img` image
(cboot initramfs booted via RCM), `doflash.sh`, `.env.initrd-flash` (board
variables) and every signed bootloader binary. There is no standalone
`doflash.sh`/`tegra-flash-helper.sh` in the deploy directory; everything comes
inside the tarball.

The `jetson-orin-nano-devkit-nvme` variant (vs. `jetson-orin-nano-devkit`,
which targets microSD) makes meta-tegra emit `external-flash.xml.in` plus
the `flash_l4t_t234_nvme.xml` layout, with `EXTERNAL_ROOTFS_DRIVE=1` and
`ROOTFS_DEVICE=nvme0n1`. Without this the rootfs ends up on the microSD slot.

## NVMe partition layout (critical)

`conecsa-image` reserves `IMAGE_ROOTFS_EXTRA_SPACE = "30000000"` (30 GiB)
for Docker images in `/var/lib/docker`, which makes the final ext4 ~36 GiB
base — **larger than the default APP partition**. The orin-nano default is
`ROOTFSPART_SIZE_DEFAULT = 30064771072` (~28 GiB) and, with A/B redundancy
(the distro default in `meta-conecsa/conf/distro/include/conecsa.inc`), that
is split in half → ~14 GiB per slot. The flash aborts with
`failed to write conecsa-image.ext4` because the image does not fit.

`kas-config.yml` (`local_conf_header.conecsa`) therefore targets the
**128 GB** NVMe of this hardware — single ~115 GB rootfs, no A/B:

```text
USE_REDUNDANT_FLASH_LAYOUT = "0"
ROOTFSPART_SIZE_DEFAULT    = "115003392000"   # 28_077_000 * 4096 (~115 GB, 4 KiB aligned)
TEGRA_EXTERNAL_DEVICE_SECTORS = "240000000"   # 240M * 512 = 122.88 GB (~96% of the SSD)
```

- `USE_REDUNDANT_FLASH_LAYOUT="0"` drops the
  `L4TConfiguration-RootfsRedundancyLevelABEnable.dtbo` overlay, switches the
  partition template to the non-redundant variant, and gives the full 115 GB
  to `APP` (no `APP_b`). The **kernel/DTB/recovery/ESP** partitions remain
  A/B-duplicated — that is hard-wired in the L4T BCT and is not affected by
  this flag.
- `ROOTFSPART_SIZE_DEFAULT` must be ≥ the ext4 size and a multiple of 4096.
- `TEGRA_EXTERNAL_DEVICE_SECTORS` must fit in the physical SSD (leave margin
  for the NVMe controller's overprovisioning).

For a different SSD size, scale both values proportionally. To keep A/B
(automatic OTA rollback), use `ROOTFSPART_SIZE_DEFAULT` ~2× the ext4 size
(each slot takes half).

## Post-build validations (before flashing)

> **IMPORTANT — Yocto vs Debian library layout.** JetPack/Ubuntu puts the
> libraries in `/usr/lib/aarch64-linux-gnu/` (Debian multiarch). **Yocto
> puts them in `/usr/lib/`** (single-arch). cuDLA is the exception: it
> lives under the CUDA toolkit prefix (`/usr/local/cuda-12.6/lib/`).
> `docker-compose.yml` must bind-mount from source `/usr/lib/...` (host)
> to destination `/usr/lib/aarch64-linux-gnu/...` (where the container
> looks). See the Troubleshooting section and the comments in
> `docker-compose.yml`.

Before flashing the Jetson, confirm the rootfs contains what
`docker-compose.yml` expects to bind-mount. Inspect the image work directory
or the manifest (`*.rootfs.manifest`):

```bash
cd yocto/build/tmp/work/jetson_orin_nano_devkit_nvme-conecsa-linux/conecsa-image/*/rootfs
ls -la usr/lib/libnvinfer.so.10.3.0
ls -la usr/lib/libcudnn.so.9
ls -la usr/lib/libcudnn_cnn.so.9
ls -la usr/lib/libcudnn_ops.so.9
ls -la usr/lib/libnvinfer_plugin.so.10.3.0
ls -la usr/lib/libnvonnxparser.so.10.3.0
ls -la usr/local/cuda-12.6/lib/libcudla.so.1.0.0
ls -la usr/lib/python3/dist-packages/Jetson         # symlink
cat etc/docker/daemon.json                          # nvidia runtime
grep '^render:' etc/group                           # GID 999 (GPIO access)

# Realtek PCIe NIC driver (Orin Nano on-board GbE) and network config:
ls -la usr/lib/modules/*/updates/drivers/net/ethernet/realtek/r8168/r8168.ko
cat etc/systemd/network/20-wired.network
cat etc/systemd/timesyncd.conf.d/00-conecsa-ntp.conf

# Wi-Fi (Realtek RTL8822CE): driver, TX-power-limit blob, regdb, supplicant:
ls -la usr/lib/modules/*/updates/drivers/net/wireless/realtek/rtl8822ce/rtl8822ce.ko
ls -la usr/lib/firmware/rtl8822_setting.bin                  # MANDATORY — power init
ls -la usr/lib/firmware/regulatory.db
ls usr/sbin/wpa_supplicant
cat etc/systemd/network/30-wireless.network
ls -la etc/wpa_supplicant/wpa_supplicant.conf.example
```

GID 999 belongs to the `render` group on this image; the GPIO udev rules use
it and `docker-compose.yml` grants it to the `os-base` hardware agent with
`group_add: "999"`.

Also confirm in the manifest that the NIC and Wi-Fi packages were pulled in:

```bash
grep -E 'nv-kernel-module-r816|nv-kernel-module-r812|nv-kernel-module-rtl8822|tegra-firmware-rtl8822|wireless-regdb|wpa-supplicant' \
  yocto/build/tmp/deploy/images/jetson-orin-nano-devkit-nvme/*.rootfs.manifest
```

## Flashing the Jetson Orin Nano

The flash artifact is the tegraflash tarball from the [Build](#build) step.
Run the [post-build validations](#post-build-validations-before-flashing)
first.

### Flash host prerequisites

Flashing requires a **Linux x86_64 host with udev running** (`initrd-flash`
calls `udevadm`; macOS and Windows do not work).

```bash
sudo apt-get install -y device-tree-compiler bmap-tools
```

- `device-tree-compiler` (`dtc`) — used by `tegra-flash-helper.sh` to sign
  the bootloader. **Without it the flash aborts immediately**
  (`ERR: 'dtc' command not found`).
- `sgdisk`, `udisksctl`, `tar`, `lsusb` — baseline requirements.
- `bmap-tools` — optional but ~4× faster when writing the rootfs (`dd` is
  the fallback). For the ~115 GB mostly-zero ext4: ~3–7 min with bmap,
  ~20 min without.

### Enter recovery mode

1. Hold **FRC + power** for 2 s, release power.
2. Connect the Jetson to the flash host via USB-C — the **side port**
   (data), not the bottom one (power only).
3. Confirm enumeration:

```bash
lsusb | grep -i "0955:7523"      # NVIDIA Corp APX
```

If the Jetson does not enumerate, re-check the cable and port and repeat the
FRC sequence.

### Flash

`initrd-flash` reads `./.env.initrd-flash` from the current directory, so it
must run **from inside** the extracted tree:

```bash
mkdir -p yocto/jetson-flash && cd yocto/jetson-flash
tar xzf ../build/tmp/deploy/images/jetson-orin-nano-devkit-nvme/conecsa-image-jetson-orin-nano-devkit-nvme.rootfs.tegraflash.tar.gz
sudo ./initrd-flash --erase-nvme
```

`initrd-flash` boots a cboot initramfs on the Jetson via RCM, which
re-exports the NVMe as USB mass storage; the host then writes the partitions
with `sgdisk` + `dd`/`bmaptool`. `--erase-nvme` wipes the old partition
table — recommended on a clean flash or whenever the partition layout
changed. A successful run ends with `Final status: SUCCESS`.

### Rebuilding after a failed `do_image_tegraflash`

A pseudo abort or `Permission denied` in `tegraflash/signed` comes from
root-owned leftovers of a build that was run as root. Clean the image work
directory, its stamps and its sstate entries, then rebuild as a regular user
(`kas-container build kas-config.yml`):

```bash
sudo rm -rf yocto/build/tmp/work/jetson_orin_nano_devkit_nvme-conecsa-linux/conecsa-image
rm -rf yocto/build/tmp/stamps/*/conecsa-image
find yocto/build/sstate-cache yocto/build/tmp/sstate-control -name '*conecsa-image*' -delete 2>/dev/null
```

### First boot

Disconnect the USB-C cable, power the Jetson normally, and wait for boot:

- The DisplayPort shows the **hub kiosk** (Weston kiosk-shell) once boot
  reaches `graphical.target`. Until the hub binary is deployed (see
  [Hub kiosk](#hub-kiosk-weston-hub-vision)) the screen stays on the empty
  compositor background; there is no getty on `tty0`.
- The on-board NIC (`enP8p1s0`, Realtek PCIe) picks up DHCP automatically
  via `systemd-networkd` (`20-wired.network`).
- SSH comes up via **socket activation** (`sshd.socket` on `[::]:22`); there
  is no long-running `sshd.service`. SSH is **key-only**: until a key is
  provisioned over the serial console, SSH refuses logins. The serial
  console (debug header) is the provisioning and recovery channel. See
  [SSH hardening](#ssh-hardening-key-only-permitted-hosts).
- Wi-Fi credentials are not in the image; see
  [Wi-Fi connection](#wi-fi-connection-realtek-rtl8822ce).
- `conecsa-bootstrap` **enables `docker.service`**, so with the containers'
  `restart: unless-stopped` policy the stack comes back on its own after
  every reboot. The `docker-moby` recipe alone only enables `docker.socket`
  (on-demand activation), which does not start the containers at boot.

### Flashing failure modes

| Symptom | Cause / fix |
|---|---|
| `ERR: 'dtc' command not found` | Install `device-tree-compiler` ([prerequisites](#flash-host-prerequisites)) |
| Flash aborts, host has no `udevadm` | Flash from a Linux x86_64 host with udev |
| Jetson does not enumerate after FRC | USB-C on the side (data) port, not the power-only one |
| `failed to write conecsa-image.ext4` | APP partition smaller than the ext4 — adjust `ROOTFSPART_SIZE_DEFAULT` + `TEGRA_EXTERNAL_DEVICE_SECTORS` ([NVMe partition layout](#nvme-partition-layout-critical)) |
| pseudo abort / `Permission denied` in `tegraflash/signed` | Root-owned build leftovers ([rebuilding](#rebuilding-after-a-failed-do_image_tegraflash)) |
| Stack doesn't come back after reboot | `docker.service` not enabled ([smoke test](#post-flash-smoke-test) step 3) |

## Post-flash smoke test

```bash
ssh root@<jetson-ip>                        # key-only — see SSH hardening section
                                            # (or use the serial console for root)

# 1. Expected versions
uname -a                                   # 5.15.x-tegra
cat /etc/nv_tegra_release                  # R36 (release), REVISION: 5.x
                                           #   (the L4T point release meta-tegra ships)

# 2. Network (on-board Realtek PCIe NIC)
ip a                                       # enP8p1s0 with DHCP lease
ip route                                   # default via gateway
timedatectl status                         # System clock synchronized: yes

# 3. Docker comes up at boot (enabled by conecsa-bootstrap)
systemctl is-enabled docker.service        # enabled
systemctl is-active docker.service         # active
docker info | grep -i runtime              # shows "nvidia"

# 4. Devices
ls -la /dev/gpiochip0 /dev/gpiochip1 /dev/video0 /dev/nvmap
getent group render                        # GID 999

# 5. Kiosk session (Weston + seatd, DisplayPort output)
systemctl get-default                      # graphical.target
systemctl is-active seatd weston           # active / active
journalctl -u weston -b | grep "DP-1"      # DRM head found + output enabled

# 6. Bring up the Conecsa app, from the directory holding docker-compose.yml
docker compose up -d
docker compose ps                          # all Up/healthy
curl http://localhost:5000/api/v1/health   # 200 OK
```

## Hub kiosk (Weston + hub-vision)

The image boots straight into a **Wayland kiosk** on the DisplayPort running
the [fleet hub](services/hub-vision.md) — the device can be both a managed
device and the hub of its own fleet. What ships in the image:

| Piece | Recipe | Notes |
|---|---|---|
| Weston 13 (kiosk-shell) | `packagegroup-conecsa-wayland` | DRM backend on the NVIDIA EGL stack (`egl-wayland`) |
| Kiosk config | `recipes-poky-overrides/weston-init` | `weston.ini`: `shell=kiosk-shell.so`, `idle-time=0`, `[autolaunch] path=/usr/bin/hub-kiosk` + `watch=true` |
| seatd daemon | `recipes-poky-overrides/seatd` | poky ships no systemd unit; `seatd -g video` mediates VT/DRM/input for the unprivileged `weston` user |
| webkit2gtk-4.1 runtime | `conecsa-hub-kiosk` RDEPENDS (`webkitgtk3`) | The runtime the hub links against; plus `liberation-fonts` |
| Launch wrapper | `conecsa-hub-kiosk` → `/usr/bin/hub-kiosk` | Sets the hub's runtime environment, then execs the hub binary |

`conecsa-image.bb` installs `conecsa-hub-kiosk` only when that recipe
directory is present in the layer. The **hub binary is not packaged**:
deploy it after flashing as described in
[Jetson kiosk deployment](services/hub-vision.md#jetson-kiosk-deployment),
which also covers the wrapper's environment and the hub's state on the
device. Until then the wrapper logs a hint and retries every 30 s, so the
image boots cleanly.

**No-PAM session design** (this distro has no `pam` in `DISTRO_FEATURES`,
so the stock `weston.service` autologin mechanism cannot work):

- The `weston-init` bbappend drops the stock unit's
  `Requires=systemd-user-sessions.service` (the unit does not exist without
  PAM, and a `Requires=` on a missing unit fails the service) and removes the
  `xwayland` PACKAGECONFIG (the x11 distro feature would inject
  `xwayland=true`, which crashes Weston with no `/tmp/.X11-unix`).
- A systemd drop-in provides `XDG_RUNTIME_DIR` via `RuntimeDirectory=weston`
  and points libseat at the seatd daemon (`LIBSEAT_BACKEND=seatd`) — the
  builtin backend cannot open `/dev/tty0` as non-root.
- `conecsa-image` sets `SYSTEMD_DEFAULT_TARGET = "graphical.target"`
  (weston.service is `WantedBy=graphical.target`; graphical pulls
  `multi-user.target` in, so the Docker stack is unaffected).
- Crash recovery: the hub exiting ends Weston (`watch=true`), and
  `Restart=always` recycles the whole session in a few seconds.

## Wi-Fi connection (Realtek RTL8822CE)

The image bakes the full Wi-Fi stack — driver, firmware blobs, regulatory
database, supplicant, and `systemd-networkd` config. Credentials are **not**
baked in (so SSID/PSK don't leak into image artifacts or git); they are
provisioned at runtime and persist across reboots.

What is shipped (all in `packagegroup-conecsa-nvidia.bb` /
`conecsa-bootstrap`):

| Component | Package / file | Purpose |
|---|---|---|
| MAC driver | `nv-kernel-module-rtl8822ce` | NVIDIA OoT Realtek driver (registers as PCI driver `rtl88x2ce`) |
| TX power limit blob | `tegra-firmware-rtl8822` → `/lib/firmware/rtl8822_setting.bin` | **Mandatory** — without it the driver's probe aborts with `power init fail` (see Troubleshooting) |
| Standard Realtek firmware | `linux-firmware-rtl8822` → `/lib/firmware/{rtw88,rtlwifi,rtl_bt}/*` | rtw88/rtlwifi firmware + Bluetooth firmware for the combo chip |
| Regulatory database | `wireless-regdb-static` → `/lib/firmware/regulatory.db` | Removes the `cfg80211: failed to load regulatory.db` warning and unlocks regulatory channels |
| Supplicant | `wpa-supplicant` | WPA1/2/3 authentication |
| systemd-networkd config | `/etc/systemd/network/30-wireless.network` (`Match Type=wlan`) | DHCPs any wireless interface, regardless of name (`wlP*`, `wlan0`, etc.) |
| Credentials template | `/etc/wpa_supplicant/wpa_supplicant.conf.example` | Placeholder + inline instructions for the operator |

Once the device is reachable over wired Ethernet, Wi-Fi is normally
configured from the device UI, through the `os-base` hardware agent (see
[Network / Wi-Fi configuration](services/os-hardware-agent.md#network-wi-fi-configuration)).
The shell procedure below is the fallback over the serial console or SSH.

### Shell procedure

On the Jetson, run these **one line at a time, in a single shell session**
(the `$IFACE` variable must be set before the lines that use it, and no
command below uses a backslash line-continuation, so a sloppy paste cannot
merge two commands into one):

```bash
# Find the wireless interface name. The kernel names the rtl8822ce on PCIe
# wlP<bus>p<slot>s<func> (for example wlP1p1s0).
IFACE=$(ip -o link show | awk -F': ' '/wl/ {print $2; exit}')
echo "Wireless interface: $IFACE"     # must be non-empty before continuing
```
```bash
# Seed wpa_supplicant config from the shipped template (one line).
install -m 0600 /etc/wpa_supplicant/wpa_supplicant.conf.example /etc/wpa_supplicant/wpa_supplicant-$IFACE.conf
```
```bash
# Append the network block via wpa_passphrase (writes a hashed PSK, so the
# cleartext PSK never lands on disk). Quote both args — SSIDs/PSKs often
# contain shell-special chars like % or $.
wpa_passphrase "YOUR_SSID" "YOUR_PSK" >> /etc/wpa_supplicant/wpa_supplicant-$IFACE.conf
```
```bash
# Enable + start the supplicant for that interface — survives reboots.
systemctl enable --now wpa_supplicant@$IFACE
```
```bash
# Verify
sleep 6
iw dev $IFACE link        # → "Connected to <BSSID>" with your SSID
ip a show dev $IFACE      # → an inet address from DHCP
ip route                  # → default route via the wireless interface
```

> If `echo "$IFACE"` prints nothing, stop — the wireless driver isn't binding;
> see the `power init fail` / `Driver: NONE` entries in Troubleshooting. Running
> the `install`/`wpa_passphrase` lines with an empty `$IFACE` is what produces
> the `wpa_supplicant-.conf: No such file or directory` error.

Every subsequent boot: `wpa_supplicant@$IFACE` authenticates automatically,
then `systemd-networkd` DHCPs via the `Type=wlan` match. No manual steps.

To change networks later, edit `/etc/wpa_supplicant/wpa_supplicant-$IFACE.conf`
(add another `network={...}` block, or replace the existing one), then run
`systemctl restart wpa_supplicant@$IFACE`. `wpa_supplicant` scans, picks the
highest-priority reachable network, and re-associates.

## SSH hardening (key-only, permitted hosts)

The image ships a hardening drop-in at
`/etc/ssh/sshd_config.d/10-conecsa-sshd-hardening.conf` (from
`conecsa-bootstrap`). Because `sshd_config` does
`Include /etc/ssh/sshd_config.d/*.conf` near its top — before the
`debug-tweaks` lines that set `PermitRootLogin yes` / `PermitEmptyPasswords
yes` — and sshd uses the **first** value seen per keyword, the drop-in wins
without editing the generated `sshd_config` or dropping the `debug-tweaks`
image feature. It sets:

```
PasswordAuthentication no
PermitEmptyPasswords no
KbdInteractiveAuthentication no
ChallengeResponseAuthentication no
PubkeyAuthentication yes
PermitRootLogin prohibit-password
```

Net effect: **SSH is key-only**; root may log in over SSH only with a public
key; passwords and empty passwords are refused. The **serial console is
unaffected** and remains the provisioning and recovery channel: root with an
empty password, from the `debug-tweaks` image feature.

> Consequence: a freshly flashed device has **no SSH access** until a key is
> provisioned. This is intentional — provision over serial (below). Don't
> lock yourself out by assuming SSH works before the key is in place.

### Provisioning the root key over serial

`/root/.ssh` is pre-created with mode `0700` so sshd won't reject the key for
loose permissions. On the Jetson serial console, paste your public key into
`authorized_keys` — and pin the **permitted hosts** with the `from="..."`
option on the same line (this is where "only permitted hosts" is enforced,
since the key is what grants access):

```bash
# Restrict to a subnet (and/or specific IPs), then the key material:
cat >> /root/.ssh/authorized_keys <<'EOF'
from="192.0.2.0/24,198.51.100.5" ssh-ed25519 AAAAC3Nza...your-key... admin@station
EOF
chmod 600 /root/.ssh/authorized_keys
```

`from=` accepts comma-separated CIDR ranges and IPs (OpenSSH 9.6). Multiple
keys / hosts: add more lines. Verify from an allowed host:

```bash
ssh root@<jetson-ip>          # succeeds with the key, from an allowed host
ssh root@<jetson-ip>          # from a non-listed host → "Permission denied"
```

### Enforcing permitted hosts at the daemon level (optional)

If you prefer the allowed network baked into the image (enforced for every
key, regardless of `from=`), uncomment and edit the `AllowUsers` line in
`meta-conecsa/recipes-conecsa/conecsa-bootstrap/files/10-conecsa-sshd-hardening.conf`:

```
AllowUsers root@192.0.2.0/24
```

then rebuild + reflash. `AllowUsers` takes space-separated `user@CIDR`
entries. The per-key `from=` and daemon-level `AllowUsers` can be combined
(both must pass).

## Troubleshooting

Flashing failures are in [Flashing failure modes](#flashing-failure-modes);
hub app failures in the kiosk are in
[Jetson kiosk deployment](services/hub-vision.md#jetson-kiosk-deployment).

- **`bitbake-layers show-recipes 'nvidia-*'` returns recipes different from
  the ones listed in `packagegroup-conecsa-nvidia.bb`** — meta-tegra
  renames recipes between minor releases. Update the packagegroup with the
  real names and run `bitbake -c cleansstate conecsa-image && bitbake
  conecsa-image`.

- **`do_fetch` of a tegra recipe fails with "Checksum mismatch" or "Unable
  to find revision"** — an override in `meta-conecsa/recipes-tegra-overrides/`
  no longer matches meta-tegra. See
  [Updating NVIDIA versions](#updating-nvidia-versions).

- **The app comes up but `inference-service` fails with `libnvinfer.so.10:
  cannot open shared object file`** — the bind-mount in `docker-compose.yml`
  points to the wrong path. On Yocto the libraries live in `/usr/lib/`
  (not `/usr/lib/aarch64-linux-gnu/` like on JetPack). The mount source is
  the Yocto path (`/usr/lib/libnvinfer.so.10.3.0`); the destination is
  where the container looks
  (`/usr/lib/aarch64-linux-gnu/libnvinfer.so.10`). cuDLA is the exception:
  `/usr/local/cuda-12.6/lib/libcudla.so.1.0.0`.

- **`cannot read file data: Is a directory`** on a `.so` bind-mount — the
  source path does not exist on the host, so Docker creates an empty
  directory at the destination. Check the source path (`ls -l` on the host).

- **`error mounting … read-only file system` in the container init** — a
  directory-wide bind-mount (`/usr/lib/aarch64-linux-gnu/nvidia:ro`)
  collides with the per-file injection from `tegra-container-passthrough`.
  Drop the `nvidia`/`tegra` directory mounts from compose; let
  `runtime: nvidia` + passthrough handle them (only TensorRT/cuDNN/cuDLA,
  which passthrough does NOT cover, need an explicit mount).

- **No `eth0` / no physical NIC** — the Orin Nano on-board GbE is a Realtek
  **PCIe** (not the SoC MGBE). `nvethernet.ko` does not work. You need
  `nv-kernel-module-r8168` (+ `nv-kernel-module-r8126`) in
  `packagegroup-conecsa-nvidia.bb`. The in-tree `kernel-module-r8169` and
  `kernel-module-realtek` packages **do not exist** in this L4T kernel
  (`# CONFIG_R8169 is not set`, `CONFIG_REALTEK_PHY=y` builtin).

- **Interface exists but has no IP** — `systemd-networkd` is enabled but
  has no `.network` file. `conecsa-bootstrap` installs
  `/etc/systemd/network/20-wired.network` (DHCP on `eth*`/`en*`).

- **Clock stuck in 1970 / `System clock synchronized: no`** — RTC without a
  battery + missing NTP client. `conecsa-bootstrap` installs
  `systemd-timesyncd` and pins public NTP servers in
  `timesyncd.conf.d/00-conecsa-ntp.conf`, and the `.network` sets
  `UseNTP=no` so an unreachable NTP server advertised over DHCP is ignored.
  On a site with no internet the hub sets the clock (see
  [Clock synchronization](services/hub-vision.md#clock-synchronization)).

- **`rtl8822ce` module loaded but no `wl*` interface, dmesg shows
  `power init fail`** — the NVIDIA OoT driver's probe is gated on loading
  `/lib/firmware/rtl8822_setting.bin` (the chip's TX-power-limit blob,
  required by the driver's `CONFIG_HEXFILE_POWER_LIMIT` build flag). The standard
  `linux-firmware-rtl8822` package does **not** ship this file — it lives
  in the NVIDIA BSP, packaged separately as `tegra-firmware-rtl8822`
  (sub-package of `tegra-firmware`). Add `tegra-firmware-rtl8822` to
  `packagegroup-conecsa-nvidia.bb` and re-flash. For an immediate test on
  the running device: `scp` the file from
  `tmp/work/.../tegra-firmware/.../packages-split/tegra-firmware-rtl8822/usr/lib/firmware/rtl8822_setting.bin`
  to `/lib/firmware/` on the Jetson, then `rmmod rtl8822ce && modprobe
  rtl8822ce`.

- **Wi-Fi card driver registered with PCI subsystem (`/sys/bus/pci/drivers/rtl88x2ce/`
  exists) but PCI device shows `Driver: NONE`** — first check
  `dmesg | grep -i 'power init fail'`. If present, see the entry above.
  If absent, try forcing a manual bind:
  `echo 0001:01:00.0 > /sys/bus/pci/drivers/rtl88x2ce/bind`. A `File exists`
  reply from `echo "10ec c822" > .../new_id` means the PCI ID **is** in
  the driver's match table — the binding was rejected later in probe, not
  at the match step.

- **`install: target '/etc/wpa_supplicant/wpa_supplicant-.conf': No such file
  or directory`** — `$IFACE` was empty when the shell procedure was run.
  Either the wireless driver isn't binding (`ip link` has no `wl*` line —
  see entries above) or the variable was assigned in a different shell. Run
  the whole procedure in **one** shell session so `IFACE` is set when the
  `install` runs.

- **Boot takes ~2 minutes before docker/the stack starts** — check that the
  `systemd-networkd-wait-online` drop-in really **replaces** the stock
  command: it must contain an empty `ExecStart=` line before the new one.
  Without the reset, the drop-in *appends* a second command to the oneshot
  unit and the stock all-links/120 s wait still runs first (paid in full
  whenever a managed link is down, e.g. unplugged Ethernet). With the fix,
  `journalctl -b -u systemd-networkd-wait-online -o short-monotonic` shows
  the unit finishing within ~10 s and docker's API up at ~20 s.

- **`weston.service` fails to start: `Unit systemd-user-sessions.service not
  found`** — the stock unit `Requires=` it, and systemd built without PAM
  does not ship it. The `weston-init` bbappend seds those lines out; on a
  live device, copy the unit to `/etc/systemd/system/` and delete the
  `systemd-user-sessions.service` references.

- **Weston dies with `Could not open target tty: Permission denied` /
  `no drm device found`** — libseat's builtin backend cannot open the VT as
  the non-root `weston` user. The kiosk uses the root `seatd` daemon instead:
  `systemctl is-active seatd` and confirm the weston unit has
  `LIBSEAT_BACKEND=seatd` (drop-in from the `weston-init` bbappend).

- **Weston core-dumps right after `failed to bind to /tmp/.X11-unix/X0`** —
  `xwayland=true` was injected into `weston.ini` (x11 is in
  `DISTRO_FEATURES`). The kiosk is Wayland-only: the `weston-init` bbappend
  removes the `xwayland` PACKAGECONFIG; on a live device delete the line
  from `/etc/xdg/weston/weston.ini`.

- **`webkitgtk3` `do_configure` fails with `GBM is required for USE_GBM`** —
  webkit 2.44 enables `USE_GBM` but on tegra `virtual/egl` is libglvnd, so
  mesa's gbm never reaches the sysroot. The
  `recipes-oe-overrides/webkitgtk3` bbappend adds `DEPENDS +=
  "virtual/libgbm"`.

## Updating NVIDIA versions

`kas-config.yml` follows the `scarthgap` branch of meta-tegra, so a new L4T
point release arrives with `kas-container update`. For a new release (e.g.
R36.6 / JetPack 6.3):

1. Update the branch in `kas-config.yml` if needed.
2. Run `kas-container update kas-config.yml`.
3. Confirm recipe names: `kas-container shell kas-config.yml -c
   "bitbake-layers show-recipes 'nvidia-*'"`.
4. Update `packagegroup-conecsa-nvidia.bb` if the names changed.
5. Compare each bbappend in `meta-conecsa/recipes-tegra-overrides/` with the
   new meta-tegra recipe and delete the ones upstream has fixed. A new
   override is named after the recipe version (`<recipe>_<PV>.bbappend`,
   never `_%`), so it stops applying at the next release.
6. Fetch everything before the multi-hour build:
   `kas-container shell kas-config.yml -c "bitbake --runall=fetch conecsa-image"`.
7. Rebuild: `kas-container build kas-config.yml`.
8. On a JetPack/L4T bump, update the version line in `README.md`,
   `docs/index.md` and `docs/getting-started.md`.
