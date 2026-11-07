# Copyright 2025-present the zvec project
# SPDX-License-Identifier: Apache-2.0
"""Preserve bounded build/test diagnostics without dumping runner credentials."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def collect(build, output, extra_logs=None):
    output.mkdir(parents=True, exist_ok=True)
    environment = {
        "os": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version,
        "cpu_count": os.cpu_count(),
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "packages": {},
    }
    for name in ("cmake", "pytest", "pybind11", "psutil", "numpy", "zvec"):
        try:
            environment["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            environment["packages"][name] = "not installed"
    (output / "environment.json").write_text(json.dumps(environment, indent=2))
    temporary = build / "Testing/Temporary"
    for name in ("LastTest.log", "LastTestsFailed.log", "CTestCostData.txt"):
        source = temporary / name
        if source.is_file():
            shutil.copy2(source, output / name)
    cache = build / "CMakeCache.txt"
    if cache.is_file():
        allowed = (
            "CMAKE_BUILD_TYPE:",
            "CMAKE_C_COMPILER:",
            "CMAKE_CXX_COMPILER:",
            "CMAKE_C_FLAGS:",
            "CMAKE_CXX_FLAGS:",
            "CMAKE_GENERATOR:",
            "BUILD_",
            "DISKANN_SUPPORTED:",
            "RABITQ_SUPPORTED:",
        )
        (output / "build-config.txt").write_text(
            "\n".join(
                line
                for line in cache.read_text().splitlines()
                if line.startswith(allowed)
            )
            + "\n"
        )
    if extra_logs and extra_logs.is_dir():
        destination = output / "device-logs"
        destination.mkdir(exist_ok=True)
        for source in extra_logs.glob("*_test.log"):
            shutil.copy2(source, destination / source.name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extra-logs", type=Path)
    args = parser.parse_args()
    collect(args.build, args.output, args.extra_logs)
