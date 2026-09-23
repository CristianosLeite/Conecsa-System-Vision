# `os-base` — base image + hardware agent

The `os-base` service plays two roles.

## 1. The `conecsa-os-base:base` image

It builds `conecsa-os-base:base` (CUDA 12.6 + Python 3.10 + the ML stack: torch,
torchvision, tensorrt, pycuda, opencv, ultralytics, numpy, …), which is
inherited via `FROM base` by every Python service (inference-service,
api-gateway and training-service). `docker-compose.yml` maps
`base=service:os-base` in `additional_contexts`, so compose builds this image first.
The `os-base` service also owns the shared volumes (`/data/models`, `/data/runs`).
The shared ML/CUDA requirements are the single source of truth in
`os-base/requirements-common.txt`.

## 2. The privileged hardware agent

Beyond the base image, the `os-base` container runs the **privileged hardware
agent** (`python3 -m agent`): a gRPC `HardwareService` on `:50051`
(`proto/hardware.proto`) that owns all host hardware access. Its gRPC client
is the api-gateway (`GetNetworkConfig`, `SetIpConfig`, `ScanWifi`,
`ConnectWifi`, `ForgetWifi`, `GetApStatus`, `StartAp`, `StopAp`, `GetGpioStatus`,
`SetGpioTrigger`, `SetGpioPin`, `GetSystemStatus`, `SystemPower` and `SetSystemTime`); the inference-service
reads the GPIO trigger from the GPIO SHM channel instead.

### Network / Wi-Fi configuration

The device uses **systemd-networkd + wpa_supplicant** (not NetworkManager /
`nmcli`). The agent reads and writes the wired + Wi-Fi configuration, scans for
networks, and connects/forgets saved networks.

!!! warning "Wi-Fi connect must not strand the device"
    A failed connect must never persist the new config (no `SAVE_CONFIG` on
    failure): the agent rolls back with a wpa `RECONFIGURE` so a bad password
    cannot lock the device out of its network.

!!! warning "Static IP changes are validated and rolled back"
    A static configuration is accepted only when the address, the gateway and
    every DNS server parse as IPv4 addresses and the prefix length is 1–32; the
    managed `10-conecsa-<iface>.network` file is replaced atomically. If
    networkd refuses to reload or to reconfigure the link, the previous file is
    put back and applied again, so a bad request cannot leave the device with a
    configuration it never confirmed.

### Wi-Fi access point

For a [remote camera](../remote-camera.md) on a site with no router, the agent
can turn the device's single radio into a WPA2 access point (`StartAp`,
`StopAp`, `GetApStatus`; `agent/ap_agent.py`). The SSID is the device id, the
channel is 36, 40, 44 or 48 (the non-DFS block) or automatic, and the device
takes `AP_ADDRESS_CIDR` with a networkd DHCP server for the remote cameras.

Which of those channels the radio may *start* a network on is not fixed: the
regulatory `NO_IR` flag is lifted on a channel once the driver has heard a
beacon there and comes back later, so the set of usable channels changes over
time, even while the station link stays on one channel. The agent reads
the live set from `GET_CAPABILITY freq` on every status (`channels`) and at
start; a channel flagged `DISABLED` or `RADAR` is left out like a `NO_IR` one.
Automatic (channel 0, the UI default) takes the channel the station
link uses when it is usable, otherwise the lowest usable one; an explicit
channel that is blocked at the moment is refused before the radio is touched,
naming the usable ones, instead of costing the full wait on wpa_supplicant's
silent "Failed to start AP functionality". When the daemon cannot list its
frequencies at all, the status reports no `channels`, automatic falls back to
36 and an explicit channel is passed on unchecked. The passphrase must be
printable ASCII (8–63 characters) or a 64-digit hex PSK.

!!! warning "The access point takes the device off its Wi-Fi network"
    A start is refused unless the wired port has a link and an IPv4 address,
    because the hub will reach the device through it alone while the access
    point is up. A link-local address (169.254/16) counts: a hub on a direct
    cable, with no DHCP server between them, reaches the device by exactly
    that. Wi-Fi connect, forget and Wi-Fi IP changes are refused meanwhile
    with `FAILED_PRECONDITION` (a `409` through the gateway), and a Wi-Fi
    change already under way holds the radio until it is done, so a start
    that arrives meanwhile waits for it instead of reconfiguring the radio
    at the same time.

Nothing about the access point is persisted. The wpa block is created unsaved
(`save_config()` drops every `mode=2` block before `SAVE_CONFIG`, and
`enable_all()` skips them) and the networkd file,
`/run/systemd/network/05-conecsa-ap-<iface>.network`, is matched on
`WLANInterfaceType=ap` so networkd takes it only while the radio is an access
point and drops it by itself afterwards. The file sets the address from
`AP_ADDRESS_CIDR`, `DHCPServer=yes` with a pool of 20 leases starting at the
tenth address (`.11`–`.30` of the default `/24`), `EmitDNS=no` so the remote
camera is offered no resolver it could not reach, and `LinkLocalAddressing=no`
and `IPv6AcceptRA=no`; the container needs the `/run/systemd/network` bind
mount `docker-compose.yml` gives it, and refuses a start without it.
The file is written world-readable: networkd runs as the `systemd-network`
user and skips, without a word in its journal, a file it cannot open.
networkd does not re-run that match on its own when the link's type flips to
AP, so once wpa_supplicant reports the access point the agent asks networkd to
reconfigure the link, and only then waits for the address.
Before touching the radio, a start checks that the wired port is ready, that
`AP_ADDRESS_CIDR` overlaps no active subnet (a link whose addresses cannot be
read counts as an overlap: the start never assumes the hub's subnet is clear)
and that wpa_supplicant runs under
a regulatory country (`GET country`; the image sets it globally, and the agent
never sets one itself). Start is then a state machine whose every failed step —
radio not becoming an AP within 15 s, no address or no DHCP server within 10 s
each, a networkd or wpa error — removes every `mode=2` block and every
`05-conecsa-ap-*` file, reloads networkd, reconfigures the link and
`RECONFIGURE`s wpa_supplicant back to the persisted station configuration. The DHCP server is verified through `DescribeLink` before the
start reports success, so a remote camera that joins will get a lease, which
means a start can take over half a minute to answer; the gateway waits 60 s
for it. Stop does the same and is idempotent; the
agent does it again at start-up, so an agent restart or a reboot always returns
the radio to station mode. An access point no remote camera joins within five minutes
stops by itself; once a remote camera has joined, a later disconnect does not.
The agent polls the associated stations (`ALL_STA`) every 5 s, which is the
granularity of the join deadline and of the station list in the status.

`GetApStatus` reports the joined remote cameras by the address the DHCP server leased
them (matched to the association through the lease's client id) and never
returns or logs a hardware address or the passphrase.

### GPIO

GPIO uses **BOARD numbering** on the Jetson Orin Nano 40-pin header: pin 7 is
the trigger input; pins 29/31/33 are freely-controllable digital outputs (driven
from Node-RED). The agent configures the pinmux + `Jetson.GPIO` and runs a small
poll loop.

The only per-frame GPIO read is the trigger gate: it does **not** cross
gRPC but a small mmap **GPIO SHM channel** (`/run/conecsa-gpio/state`,
overridable via `GPIO_SHM_PATH`), so the
inference-service can read the trigger pin level every frame without RPC
overhead. Output pins are event-driven, not per-frame, so they are driven on
demand over gRPC (`SetGpioPin`); availability, trigger mode and current pin
levels are read with `GetGpioStatus`.

### System metrics

Host CPU/RAM/disk/temperature/GPU metrics (via `psutil` + Jetson sysfs) are
exposed over the same gRPC service and surfaced by the gateway on
`/api/system/status`.

### System power

The agent also owns host shutdown/restart via the `SystemPower` RPC, surfaced
by the gateway on `POST /api/v1/system/power`.

### System clock

The hub's wall clock is the device's time source (see
[Clock synchronization](hub-vision.md#clock-synchronization)).
`SetSystemTime` (`epoch_millis`, plus a `source` used only for the log line)
steps `CLOCK_REALTIME`. The api-gateway calls it when a hub poll or the pairing
request carries the hub's time and the drift exceeds
`CLOCK_SYNC_THRESHOLD_SEC` (pairing always steps). The agent refuses
a time older than the persisted floor (`CONECSA_CLOCK_FLOOR`, default
`/var/lib/conecsa/fake-hwclock`, shared with the host's fake-hwclock units)
and, after a successful step, rewrites the floor and pushes the time to the
RTC so it survives a reboot.

### Performance-clock pinning

At startup the agent **pins the Jetson performance clocks** (GPU
`min_freq = max_freq`, CPU cores → `performance` governor — the core of
`jetson_clocks`). Without it the dynamic governors keep the GPU at its minimum
for the bursty TensorRT workload, roughly doubling inference latency. Opt out
with `PIN_PERFORMANCE_CLOCKS=0`.

## Reference

- Python API: [`agent` and `conecsa_shm` packages](../reference/python-api/index.md)
- gRPC contract: [`proto/hardware.proto`](../reference/proto.md)
