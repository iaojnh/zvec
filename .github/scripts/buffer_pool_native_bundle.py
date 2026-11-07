# Copyright 2025-present the zvec project
# SPDX-License-Identifier: Apache-2.0

"""Bundle native tests with their build-local transitive shared libraries."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path


def dependencies(binary, env=None):
    result = subprocess.run(
        ["ldd", str(binary)],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )
    if "not found" in result.stdout:
        raise RuntimeError(result.stdout)
    return [
        Path(value)
        for value in re.findall(r"(?:=>\s+|^\s*)(/\S+)\s+\(", result.stdout, re.M)
    ]


def bundle(build, output, targets):
    build = build.resolve()
    manifest = {}
    with tempfile.TemporaryDirectory(prefix="zvec-native-bundle-") as temporary:
        stage = Path(temporary)
        (stage / "bin").mkdir()
        (stage / "lib").mkdir()
        for target in targets:
            binary = build / "bin" / target
            shutil.copy2(binary, stage / "bin" / target)
            for dependency in dependencies(binary):
                if not dependency.resolve().is_relative_to(build):
                    continue
                destination = stage / "lib" / dependency.name
                digest = hashlib.sha256(dependency.read_bytes()).hexdigest()
                if (
                    destination.exists()
                    and hashlib.sha256(destination.read_bytes()).hexdigest() != digest
                ):
                    raise RuntimeError(f"conflicting shared library: {dependency.name}")
                shutil.copy2(dependency, destination)
                manifest[dependency.name] = digest
        env = {**os.environ, "LD_LIBRARY_PATH": str(stage / "lib")}
        for target in targets:
            binary = stage / "bin" / target
            resolved = dependencies(binary, env)
            if any(path.resolve().is_relative_to(build) for path in resolved):
                raise RuntimeError(
                    f"{target} still depends on the original build directory"
                )
            subprocess.run(
                [str(binary), "--gtest_list_tests"],
                env=env,
                cwd=stage,
                stdout=subprocess.DEVNULL,
                check=True,
                timeout=30,
            )
        (stage / "native-libraries.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        with tarfile.open(output, "w:gz") as archive:
            for name in ("bin", "lib", "native-libraries.json"):
                archive.add(stage / name, arcname=name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("targets", nargs="+")
    args = parser.parse_args()
    bundle(args.build, args.output, args.targets)
