# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""The `os-base` hardware agent.

Runs inside the privileged `os-base` container and owns all host hardware
access: network/Wi-Fi, GPIO, system metrics and the system clock. Exposes the
gRPC `HardwareService` (see proto/hardware.proto) consumed by the api-gateway.
"""
