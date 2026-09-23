# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Entrypoint: `python3 -m agent` starts the `os-base` hardware agent's gRPC server."""
from .server import serve

if __name__ == "__main__":
    serve()
