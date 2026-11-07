# Copyright 2025-present the zvec project
# SPDX-License-Identifier: Apache-2.0
"""Run the required pressure lifecycle contract on an explicitly selected device."""

from __future__ import annotations

import argparse
import json
import os
import plistlib
import re
import subprocess
import uuid
from pathlib import Path

TARGET = "mobile_pressure_lifecycle_test"
REQUIRED_PHASES = {"background_reclaimed", "critical_denied", "normal_retry", "closed"}


def validate_output(output, platform):
    assert not re.search(r"^\[  FAILED  \]", output, re.M), "device tests failed"
    assert re.search(r"^\[  PASSED  \]\s+3 tests\.\s*$", output, re.M), (
        "required 3/3 device tests did not pass"
    )
    assert re.search(
        r"^\[==========\]\s+3 tests from 1 test suite ran\.", output, re.M
    ), "device suite incomplete"
    samples = [
        json.loads(line.split("MOBILE_RSS ", 1)[1])
        for line in output.splitlines()
        if "MOBILE_RSS " in line
    ]
    assert {row["phase"] for row in samples} >= REQUIRED_PHASES, (
        "missing pressure checkpoint"
    )
    assert sum(row["phase"] == "closed" for row in samples) == 36, (
        "missing lifecycle rounds"
    )
    for sample in samples:
        assert 0 < sample["rss_bytes"] <= sample["target_bytes"] == 128 * 1024**2
        if platform == "ios":
            assert sample["footprint_bytes"] > 0, "not a native iOS application sample"
    return samples


def invoke(command, log, env=None, timeout=900):
    with log.open("w") as output:
        try:
            result = subprocess.run(
                command,
                check=False,
                stdout=output,
                stderr=subprocess.STDOUT,
                env=env,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"device command timed out; see {log}") from None
    assert result.returncode == 0, (
        f"device command exited {result.returncode}; see {log}"
    )
    return log.read_text()


def run_android(args):
    adb = ["adb", "-s", args.device]
    emulator = invoke(
        [*adb, "shell", "getprop", "ro.kernel.qemu"], args.evidence / "device-kind.log"
    ).strip()
    assert emulator != "1", "physical Android acceptance cannot use an emulator"
    abi = invoke(
        [*adb, "shell", "getprop", "ro.product.cpu.abilist"],
        args.evidence / "device-abi.log",
    ).strip()
    assert "arm64-v8a" in abi.split(","), "an ARM64 physical device is required"
    binary = args.build / "bin" / TARGET
    assert binary.is_file(), f"missing {binary}"
    remote = "/data/local/tmp/zvec-pressure-" + uuid.uuid4().hex
    try:
        invoke([*adb, "shell", "mkdir", "-p", remote], args.evidence / "prepare.log")
        invoke(
            [*adb, "push", str(binary), remote + "/" + TARGET],
            args.evidence / "install.log",
        )
        invoke(
            [*adb, "shell", "chmod", "755", remote + "/" + TARGET],
            args.evidence / "chmod.log",
        )
        return invoke(
            [
                *adb,
                "shell",
                f"cd {remote} && ZVEC_REQUIRE_PHYSICAL_DEVICE=1 ./{TARGET} --gtest_color=no",
            ],
            args.evidence / "device-tests.log",
        )
    finally:
        # Delete only the uniquely named test directory this run created.
        subprocess.run([*adb, "shell", "rm", "-rf", remote], check=False, timeout=30)


def run_ios(args):
    apps = list(args.build.rglob(TARGET + ".app"))
    assert len(apps) == 1, "exactly one signed device pressure app is required"
    app = apps[0]
    with (app / "Info.plist").open("rb") as source:
        identifier = plistlib.load(source)["CFBundleIdentifier"]
    invoke(
        ["codesign", "--verify", "--deep", "--strict", str(app)],
        args.evidence / "signature.log",
    )
    invoke(
        [
            "xcrun",
            "devicectl",
            "device",
            "install",
            "app",
            "--device",
            args.device,
            "--json-output",
            str(args.evidence / "install.json"),
            str(app),
        ],
        args.evidence / "install.log",
    )
    env = {
        **os.environ,
        "DEVICECTL_CHILD_ZVEC_REQUIRE_PHYSICAL_DEVICE": "1",
        "DEVICECTL_CHILD_GTEST_COLOR": "no",
    }
    output = invoke(
        [
            "xcrun",
            "devicectl",
            "device",
            "process",
            "launch",
            "--device",
            args.device,
            "--console",
            "--terminate-existing",
            "--json-output",
            str(args.evidence / "launch.json"),
            identifier,
            "--gtest_color=no",
        ],
        args.evidence / "device-tests.log",
        env=env,
    )
    result = json.loads((args.evidence / "launch.json").read_text())
    assert result["info"]["outcome"] == "success", "device launch failed"
    termination = result.get("result", {}).get("terminationResult")
    if termination is not None:
        assert termination.get("exitCode") == 0, "device app did not exit normally"
    return output


def run(args):
    args.evidence = args.evidence.resolve()
    args.evidence.mkdir(parents=True, exist_ok=True)
    record = {
        "platform": args.platform,
        "execution": "physical_device",
        "pressure_source": "injected_host_notifications",
        "passed": False,
        "commit": os.environ.get("GITHUB_SHA", "local"),
    }
    destination = args.evidence / "device-result.json"
    destination.write_text(json.dumps(record, indent=2))
    output = run_android(args) if args.platform == "android" else run_ios(args)
    samples = validate_output(output, args.platform)
    record.update(
        passed=True,
        tests=3,
        lifecycle_rounds=36,
        peak_rss_bytes=max(row["rss_bytes"] for row in samples),
    )
    destination.write_text(json.dumps(record, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=("android", "ios"), required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    run(parser.parse_args())
