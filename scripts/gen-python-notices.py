#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: Apache-2.0

"""Generate the third-party notices of the Python service images.

Reads the installed package metadata of the dev virtualenv (which holds every
production requirement) and writes one table per image:

- ``os-base/THIRD-PARTY-NOTICES.md``: the base image, from
  ``os-base/requirements-common.txt`` and ``os-base/agent/requirements.txt``;
  inference-service and training-service add no pip packages on top of it.
- ``api-gateway/THIRD-PARTY-NOTICES.md``: what the gateway adds on top.

Dependencies are followed transitively with environment markers evaluated for
the device (Linux aarch64, Python 3.10), so the output does not depend on the
machine that runs this. Packages the NVIDIA base image supplies (torch and its
CUDA stack, TensorRT) are not pip-installed by these images and are left out.
Versions are omitted on purpose: only direct requirements are pinned.

Usage:
    .venv/bin/python scripts/gen-python-notices.py           # write the files
    .venv/bin/python scripts/gen-python-notices.py --check   # exit 1 when stale

Either mode exits 1 when a package carries a GPL-family license that is not a
known, reviewed exception.
"""
import argparse
import difflib
import re
import sys
from importlib import metadata
from pathlib import Path
from typing import Optional

from packaging.markers import Marker
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parent.parent
DEVICE_ENV = {
    "python_version": "3.10",
    "python_full_version": "3.10.12",
    "platform_system": "Linux",
    "sys_platform": "linux",
    "platform_machine": "aarch64",
    "os_name": "posix",
    "implementation_name": "cpython",
    "platform_python_implementation": "CPython",
    "extra": "",
}
# Supplied by the NVIDIA L4T base image, not by pip in these images.
BASE_IMAGE = re.compile(r"^(torch|torchvision|torchaudio|triton|tensorrt.*|nvidia-.*|pycuda)$")
# Packaging tools come with the interpreter or venv, at a version the venv tool
# picks (not our requirements); their metadata drifts between environments.
BOOTSTRAP = {"pip", "setuptools", "wheel"}
# Copyleft licenses reviewed in the licensing plan (ultralytics-thop is
# ultralytics' own AGPL dependency); any other GPL-family license fails.
REVIEWED_COPYLEFT = {"ultralytics", "ultralytics-thop", "zeroconf"}
COPYLEFT = re.compile(r"\b(A|L)?GPL|General Public License", re.IGNORECASE)

IMAGES = [
    ("os-base/THIRD-PARTY-NOTICES.md", "os-base image (inference-service, training-service)",
     ["os-base/requirements-common.txt", "os-base/agent/requirements.txt"], []),
    ("api-gateway/THIRD-PARTY-NOTICES.md", "api-gateway image, on top of os-base",
     ["api-gateway/requirements.txt"],
     ["os-base/requirements-common.txt", "os-base/agent/requirements.txt"]),
]


def requirement_names(path: Path) -> list[str]:
    names = []
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or line.startswith("--"):
            continue
        if line.startswith("-r "):
            names += requirement_names(path.parent / line[3:].strip())
            continue
        names.append(Requirement(line).name)
    return names


def closure(roots: list[str]) -> set[str]:
    seen: set[str] = set()
    stack = [canonicalize_name(r) for r in roots]
    while stack:
        name = stack.pop()
        if name in seen or name in BOOTSTRAP or BASE_IMAGE.match(name):
            continue
        try:
            requires = metadata.requires(name) or []
        except metadata.PackageNotFoundError:
            sys.exit(f"gen-python-notices: {name} is not installed; run scripts/init.sh first")
        seen.add(name)
        for spec in requires:
            req = Requirement(spec)
            if req.marker is None or Marker(str(req.marker)).evaluate(DEVICE_ENV):
                stack.append(canonicalize_name(req.name))
    return seen


def field(dist: metadata.Distribution, name: str) -> Optional[str]:
    """First value of a metadata field, or None."""
    values = dist.metadata.get_all(name)
    return values[0] if values else None


def license_of(dist: metadata.Distribution) -> str:
    expression = field(dist, "License-Expression")
    if expression:
        return expression.strip()
    classifiers = [c.split(" :: ")[-1] for c in dist.metadata.get_all("Classifier") or []
                   if c.startswith("License :: ") and c != "License :: OSI Approved"]
    if classifiers:
        return " / ".join(sorted(set(classifiers)))
    text = (field(dist, "License") or "").strip()
    return text if text and "\n" not in text and len(text) <= 80 else "see package"


def url_of(dist: metadata.Distribution) -> str:
    for entry in dist.metadata.get_all("Project-URL") or []:
        label, _, url = entry.partition(",")
        if label.strip().lower() in {"homepage", "home", "source", "repository", "source code"}:
            return url.strip()
    home = field(dist, "Home-page")
    return (home or f"https://pypi.org/project/{field(dist, 'Name')}/").strip()


def render(title: str, names: set[str]) -> tuple[str, list[str]]:
    rows, problems = [], []
    for name in sorted(names):
        dist = metadata.distribution(name)
        display = field(dist, "Name") or name
        lic = license_of(dist)
        if COPYLEFT.search(lic) and name not in REVIEWED_COPYLEFT:
            problems.append(f"{display}: {lic}")
        rows.append(f"| {display} | {lic} | {url_of(dist)} |")
    text = "\n".join([
        f"# Third-party notices: {title}",
        "",
        "Generated by `scripts/gen-python-notices.py`; do not edit by hand.",
        "Python packages installed with pip into this image, with their licenses as declared",
        "in each package's metadata. Packages supplied by the NVIDIA base image (torch and",
        "its CUDA stack, TensorRT) are not listed. Each package's full license text ships",
        "with the installed package.",
        "",
        "| Package | License | Project |",
        "|---|---|---|",
        *rows,
        "",
    ])
    return text, problems


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate the third-party notices of the Python service images.")
    parser.add_argument("--check", action="store_true", help="fail when a file is stale")
    args = parser.parse_args()

    status = 0
    for out, title, reqs, base in IMAGES:
        roots = [n for r in reqs for n in requirement_names(ROOT / r)]
        base_roots = [n for r in base for n in requirement_names(ROOT / r)]
        names = closure(roots) - (closure(base_roots) if base_roots else set())
        text, problems = render(title, names)
        for problem in problems:
            print(f"gen-python-notices: unreviewed copyleft license: {problem}", file=sys.stderr)
            status = 1
        target = ROOT / out
        if args.check:
            current = target.read_text() if target.exists() else ""
            if current != text:
                print(f"gen-python-notices: {out} is stale; run scripts/gen-python-notices.py",
                      file=sys.stderr)
                sys.stderr.writelines(difflib.unified_diff(
                    current.splitlines(keepends=True), text.splitlines(keepends=True),
                    fromfile=f"{out} (committed)", tofile=f"{out} (generated)"))
                status = 1
        else:
            target.write_text(text)
            print(f"gen-python-notices: wrote {out} ({len(names)} packages)")
    return status


if __name__ == "__main__":
    sys.exit(main())
