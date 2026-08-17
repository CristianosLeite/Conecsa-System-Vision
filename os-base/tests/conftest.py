"""Shared pytest import setup for the os-base (hardware agent) suite.

The suite runs from inside ``os-base``, so the ``agent`` package is already
importable — but ``agent.server`` pulls in the generated proto stubs, which are
only built inside the image. The tests therefore import the individual agent
modules (``agent.time_agent`` and friends), never the gRPC server.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_OS_BASE = os.path.abspath(os.path.join(_HERE, os.pardir))

if _OS_BASE not in sys.path:
    sys.path.insert(0, _OS_BASE)
