#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: Apache-2.0
"""Add a channel plan to a Realtek ``rtl8822_setting.bin``.

NVIDIA's vendor driver for the RTL8822CE (``rtl8822ce``, built with
``CONFIG_HEXFILE_CHANNEL_PLAN``) reads every channel plan it knows from
``/lib/firmware/rtl8822_setting.bin`` and has no country table at all, so
``country=`` in wpa_supplicant never reaches the radio. The file NVIDIA ships
defines a single plan, ``0x7F`` (world-wide), in which every 5 GHz channel is
passive (``NO_IR``): the radio may not start an access point there until it
has heard a beacon on that channel.

This script adds plan ``0x62`` (Realtek's id for Brazil in its certified
map): the same channel lists as ``0x7F`` with band 1 (36-48) allowed to
initiate. Bands 2-3 stay passive + DFS, band 4 stays passive and the
power-limit records at the end of the file are copied byte for byte (the
driver tags them ``WW`` whatever the plan). The plan is selected at module
load with ``options rtl8822ce rtw_channel_plan=0x62``.

File layout (driver ``core/rtw_chplan.c::rtw_get_channel_plan_from_file`` and
``hal/hal_com_phycfg.c``): ``u16 crc`` (little endian, over everything after
it), counts of 2.4 GHz list slots, 5 GHz list slots and plan slots, then the
lists (``len, channels..., attr``; an empty slot is one zero byte), the plans
(``chd_2g, chd_5g, regd_2g, regd_5g``; valid when ``chd_2g != 8 and
chd_5g != 0``) and 6-byte power-limit records to the end.

Usage: ``rtl8822-chplan.py IN OUT [--dump]``. IN and OUT may be the same
path. Running it on its own output is a no-op.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

PLAN_ID = 0x62
WORLD_WIDE_PLAN = 0x7F
PLAN_EMPTY_2G = 8  # a plan with chd_2g == 8 is "empty" for the driver
REGD_WW = 10  # enum rtw_regd: RTW_REGD_WW
CLA_5G_B1_PASSIVE = 0x01
CLA_5G_B2_PASSIVE = 0x02
CLA_5G_B3_PASSIVE = 0x04
CLA_5G_B2_DFS = 0x10
CLA_5G_B3_DFS = 0x20
BAND1 = (36, 40, 44, 48)
HEXFILE_MAX = 3072  # RTW_HEXFILE_LEN in the driver
USAGE = "usage: rtl8822-chplan.py IN OUT [--dump]"


class SettingError(Exception):
    """The input is not the file this script knows how to patch."""


def crc16(data: bytes) -> int:
    """Port of the driver's ``rtw_calc_crc`` (bit-serial ``rtw_cal_crc16``)."""
    crc = 0xFFFF
    for byte in data:
        for i in range(8):
            shift_in = ((crc >> 15) & 1) ^ ((byte >> i) & 1)
            result = (crc << 1) & 0xFFFF
            result = result | 1 if shift_in else result & ~1
            if ((crc >> 11) & 1) ^ shift_in:
                result |= 1 << 12
            else:
                result &= ~(1 << 12)
            if ((crc >> 4) & 1) ^ shift_in:
                result |= 1 << 5
            else:
                result &= ~(1 << 5)
            crc = result
    return (~crc) & 0xFFFF


@dataclass
class ChannelList:
    channels: list[int] = field(default_factory=list)
    attr: int = 0

    @property
    def empty(self) -> bool:
        return not self.channels

    def encode(self) -> bytes:
        if self.empty:
            return b"\x00"
        return bytes([len(self.channels), *self.channels, self.attr])


@dataclass
class Setting:
    lists_2g: list[ChannelList]
    lists_5g: list[ChannelList]
    plans: list[bytes]  # 4 bytes each
    tail: bytes  # power-limit records, opaque

    def plan_valid(self, plan_id: int) -> bool:
        chd_2g, chd_5g = self.plans[plan_id][0], self.plans[plan_id][1]
        return chd_2g != PLAN_EMPTY_2G and chd_5g != 0

    def encode(self) -> bytes:
        body = bytes([len(self.lists_2g), len(self.lists_5g), len(self.plans)])
        body += b"".join(c.encode() for c in self.lists_2g)
        body += b"".join(c.encode() for c in self.lists_5g)
        body += b"".join(self.plans)
        body += self.tail
        return crc16(body).to_bytes(2, "little") + body


def parse(data: bytes) -> Setting:
    if len(data) < 5:
        raise SettingError("file too short")
    stored = int.from_bytes(data[:2], "little")
    computed = crc16(data[2:])
    if stored != computed:
        raise SettingError(f"checksum mismatch: stored {stored:#06x}, computed {computed:#06x}")
    n_2g, n_5g, n_plans = data[2], data[3], data[4]
    i = 5
    lists: list[ChannelList] = []
    for _ in range(n_2g + n_5g):
        if i >= len(data):
            raise SettingError("truncated channel list")
        n = data[i]
        if n == 0:
            lists.append(ChannelList())
            i += 1
            continue
        if i + n + 2 > len(data):
            raise SettingError("truncated channel list")
        lists.append(ChannelList(list(data[i + 1:i + 1 + n]), data[i + 1 + n]))
        i += n + 2
    if i + 4 * n_plans > len(data):
        raise SettingError("truncated plan table")
    plans = [bytes(data[i + 4 * k:i + 4 * k + 4]) for k in range(n_plans)]
    i += 4 * n_plans
    return Setting(lists[:n_2g], lists[n_2g:], plans, bytes(data[i:]))


def expected_band1_list(setting: Setting) -> ChannelList:
    """The 5 GHz list of plan 0x7F with band 1 no longer passive."""
    if len(setting.plans) <= WORLD_WIDE_PLAN or not setting.plan_valid(WORLD_WIDE_PLAN):
        raise SettingError("plan 0x7F (world-wide) is missing; not the file this script expects")
    source = setting.lists_5g[setting.plans[WORLD_WIDE_PLAN][1]]
    if any(ch not in source.channels for ch in BAND1):
        raise SettingError("plan 0x7F does not list channels 36-48")
    if not source.attr & CLA_5G_B1_PASSIVE:
        raise SettingError("band 1 is already allowed to initiate in plan 0x7F; nothing to do")
    return ChannelList(list(source.channels), source.attr & ~CLA_5G_B1_PASSIVE)


def add_plan(setting: Setting) -> bool:
    """Add plan ``PLAN_ID``. Returns False when it is already there."""
    wanted = expected_band1_list(setting)
    if len(setting.plans) <= PLAN_ID:
        raise SettingError(f"the plan table has no slot {PLAN_ID:#04x}")
    if setting.plan_valid(PLAN_ID):
        current = setting.plans[PLAN_ID]
        if (
            setting.lists_5g[current[1]] == wanted
            and current[0] == setting.plans[WORLD_WIDE_PLAN][0]
            and current[2:] == bytes([REGD_WW, REGD_WW])
        ):
            return False
        raise SettingError(f"plan {PLAN_ID:#04x} already exists with different content")
    source_slot = setting.plans[WORLD_WIDE_PLAN][1]
    try:
        slot = next(k for k, c in enumerate(setting.lists_5g) if k > source_slot and c.empty)
    except StopIteration as exc:
        raise SettingError("no empty 5 GHz list slot after the world-wide one") from exc
    setting.lists_5g[slot] = wanted
    chd_2g = setting.plans[WORLD_WIDE_PLAN][0]
    setting.plans[PLAN_ID] = bytes([chd_2g, slot, REGD_WW, REGD_WW])
    return True


def verify(original: Setting, patched: bytes) -> Setting:
    """Re-parse the output and prove only the intended bytes changed."""
    result = parse(patched)
    if len(patched) > HEXFILE_MAX:
        raise SettingError(f"output is {len(patched)} bytes, over the driver's {HEXFILE_MAX}")
    if result.tail != original.tail:
        raise SettingError("power-limit records changed")
    if result.lists_2g != original.lists_2g:
        raise SettingError("2.4 GHz lists changed")
    if result.plans[WORLD_WIDE_PLAN] != original.plans[WORLD_WIDE_PLAN]:
        raise SettingError("plan 0x7F changed")
    world = result.lists_5g[result.plans[WORLD_WIDE_PLAN][1]]
    if world != original.lists_5g[original.plans[WORLD_WIDE_PLAN][1]]:
        raise SettingError("the world-wide 5 GHz list changed")
    valid_before = sum(original.plan_valid(k) for k in range(len(original.plans)))
    valid_after = sum(result.plan_valid(k) for k in range(len(result.plans)))
    if valid_after != valid_before + 1 or not result.plan_valid(PLAN_ID):
        raise SettingError("the plan table does not hold exactly one new plan")
    new = result.lists_5g[result.plans[PLAN_ID][1]]
    if new.channels != world.channels or new.attr != world.attr & ~CLA_5G_B1_PASSIVE:
        raise SettingError("the new 5 GHz list is not the world-wide one with band 1 active")
    if new.attr & (CLA_5G_B2_PASSIVE | CLA_5G_B3_PASSIVE | CLA_5G_B2_DFS | CLA_5G_B3_DFS) != (
        CLA_5G_B2_PASSIVE | CLA_5G_B3_PASSIVE | CLA_5G_B2_DFS | CLA_5G_B3_DFS
    ):
        raise SettingError("bands 2-3 must stay passive and DFS")
    return result


def flags(attr: int, ch: int) -> str:
    out = []
    if 36 <= ch <= 48 and attr & CLA_5G_B1_PASSIVE:
        out.append("NO_IR")
    if 52 <= ch <= 64:
        out += ["NO_IR"] * bool(attr & CLA_5G_B2_PASSIVE) + ["DFS"] * bool(attr & CLA_5G_B2_DFS)
    if 100 <= ch <= 144:
        out += ["NO_IR"] * bool(attr & CLA_5G_B3_PASSIVE) + ["DFS"] * bool(attr & CLA_5G_B3_DFS)
    if 149 <= ch <= 177:
        out += ["NO_IR"] * bool(attr & 0x08) + ["DFS"] * bool(attr & 0x40)
    return " ".join(out)


def dump(setting: Setting) -> None:
    for plan_id, plan in enumerate(setting.plans):
        if not setting.plan_valid(plan_id):
            continue
        chd_2g, chd_5g, regd_2g, regd_5g = plan
        print(f"plan 0x{plan_id:02X}: chd_2g={chd_2g} chd_5g={chd_5g} regd_2g={regd_2g} regd_5g={regd_5g}")
        ch_2g = setting.lists_2g[chd_2g]
        print(f"  2.4 GHz {ch_2g.channels} attr={ch_2g.attr:#04x}")
        ch_5g = setting.lists_5g[chd_5g]
        print(f"  5 GHz attr={ch_5g.attr:#04x}")
        for ch in ch_5g.channels:
            print(f"    {ch:3d} {flags(ch_5g.attr, ch)}")
    print(f"power-limit records: {len(setting.tail) // 6}")


def main(argv: list[str]) -> int:
    args = [a for a in argv[1:] if not a.startswith("--")]
    opts = {a for a in argv[1:] if a.startswith("--")}
    if len(args) != 2 or opts - {"--dump"}:
        print(USAGE, file=sys.stderr)
        return 2
    src, dst = args
    try:
        with open(src, "rb") as fh:
            data = fh.read()
        original = parse(data)
        setting = parse(data)
        if add_plan(setting):
            patched = setting.encode()
            result = verify(original, patched)
            tmp = f"{dst}.tmp"
            with open(tmp, "wb") as fh:
                fh.write(patched)
            os.replace(tmp, dst)
            print(f"{dst}: added plan 0x{PLAN_ID:02X} ({len(data)} -> {len(patched)} bytes)")
        else:
            result = setting
            if os.path.abspath(src) != os.path.abspath(dst):
                with open(dst, "wb") as fh:
                    fh.write(data)
            print(f"{dst}: plan 0x{PLAN_ID:02X} already present; unchanged")
    except (OSError, SettingError) as exc:
        print(f"rtl8822-chplan: {exc}", file=sys.stderr)
        return 1
    if "--dump" in opts:
        dump(result)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
