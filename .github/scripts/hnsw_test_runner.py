"""Validate HNSW coverage reports, or run the suite on a connected device.

Examples:
  python .github/scripts/hnsw_test_runner.py verify-xml build/test_tmp/hnsw_platform_test/results.xml
  python .github/scripts/hnsw_test_runner.py android --device SERIAL --binary build_android_arm64-v8a/bin/hnsw_platform_test
  python .github/scripts/hnsw_test_runner.py ios --device UDID --app build_ios/bin/Release-iphoneos/hnsw_platform_test.app

Device commands require an unlocked, trusted device. The iOS app must already
be signed for that device. Reports/logs are retained under --output.
"""

from __future__ import annotations

import argparse
import json
import logging
import plistlib
import re
import shlex
import subprocess
import sys
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path

# Three formats, three metrics, two layouts, four storage modes, plus dispatch.
MINIMUM_TESTS = 3 * 3 * 2 * 4 + 1


def verify_architectures(value: str, expected_arch: str | None) -> None:
    flags = dict(part.split("=", 1) for part in value.split(";") if part)
    if flags.get("Scalar") != "1":
        raise ValueError("Missing scalar FP16 coverage")
    if expected_arch and flags.get(expected_arch) != "1":
        raise ValueError(f"Expected {expected_arch} FP16 coverage: {value}")


def verify_xml(path: Path, expected_arch: str | None = None) -> int:
    root = ET.parse(path).getroot()
    cases = root.findall(".//testcase")
    if len(cases) < MINIMUM_TESTS or int(root.get("tests", "0")) != len(cases):
        raise ValueError(f"Expected at least {MINIMUM_TESTS} HNSW tests in {path}")
    if any(int(root.get(key, "0")) for key in ("failures", "errors", "disabled")):
        raise ValueError("HNSW report contains failures, errors, or disabled tests")
    for case in cases:
        if (
            case.get("status") != "run"
            or case.get("result", "completed") != "completed"
            or case.find("skipped") is not None
            or case.find("failure") is not None
            or case.find("error") is not None
        ):
            raise ValueError(f"HNSW test did not pass: {case.get('name')}")
    features = root.find(".//property[@name='fp16_cpu_architectures']")
    if features is None:
        raise ValueError("Missing FP16 architecture execution report")
    verify_architectures(features.get("value", ""), expected_arch)
    return len(cases)


def verify_console(output: str, expected_arch: str = "NEON") -> int:
    totals = re.findall(r"^\[==========\]\s+(\d+) tests? from .+ ran\.", output, re.M)
    passed = re.findall(r"^\[  PASSED  \]\s+(\d+) tests?\.", output, re.M)
    if (
        not totals
        or not passed
        or int(totals[-1]) < MINIMUM_TESTS
        or totals[-1] != passed[-1]
        or re.search(r"^\[\s*(?:FAILED|SKIPPED)\s*\]", output, re.M)
    ):
        raise ValueError("HNSW suite was incomplete, failed, or skipped tests")
    features = re.search(r"^FP16 architecture coverage: (.+)$", output, re.M)
    if not features:
        raise ValueError("Missing FP16 architecture execution report")
    verify_architectures(features[1], expected_arch)
    return int(totals[-1])


def run(command: list[str], log: Path, timeout: int = 600) -> str:
    log.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        output = error.stdout or b""
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        log.write_text(output + "\nHNSW device command timed out.\n", encoding="utf-8")
        raise
    log.write_text(result.stdout, encoding="utf-8")
    sys.stdout.write(result.stdout)
    sys.stdout.flush()
    result.check_returncode()
    return result.stdout


def android(device: str, binary: Path, output: Path) -> None:
    if not binary.is_file():
        raise ValueError(f"Missing test binary: {binary}")
    adb = ["adb", "-s", device]
    abi = run([*adb, "shell", "getprop", "ro.product.cpu.abi"], output / "abi.log", 30)
    if abi.strip() != "arm64-v8a":
        raise ValueError(f"Expected an Android ARM64 device, got {abi.strip()}")
    emulator = run(
        [*adb, "shell", "getprop", "ro.kernel.qemu"], output / "device.log", 30
    )
    if emulator.strip() == "1":
        raise ValueError("This entry point requires a physical Android ARM64 device")
    remote = f"/data/local/tmp/zvec-hnsw-{uuid.uuid4().hex}"
    try:
        run([*adb, "shell", "mkdir", "-p", remote], output / "mkdir.log", 30)
        run(
            [*adb, "push", str(binary.resolve()), f"{remote}/hnsw_platform_test"],
            output / "push.log",
            120,
        )
        command = (
            f"cd {shlex.quote(remote)} && chmod 700 hnsw_platform_test && "
            "./hnsw_platform_test --gtest_color=no --gtest_output=xml:results.xml"
        )
        run([*adb, "shell", command], output / "run.log")
        run(
            [*adb, "pull", f"{remote}/results.xml", str(output / "results.xml")],
            output / "pull.log",
            60,
        )
        count = verify_xml(output / "results.xml", "NEON")
        logging.info("Android ARM64 HNSW: %s passed, no skipped tests", count)
    finally:
        # Only remove this invocation's generated directory.
        try:
            subprocess.run(
                [*adb, "shell", "rm", "-rf", remote], check=False, timeout=30
            )
        except subprocess.TimeoutExpired:
            logging.warning(
                "Device cleanup timed out; test results are retained locally"
            )


def ios(device: str, app: Path, output: Path) -> None:
    if sys.platform != "darwin":
        raise ValueError("The iOS device runner requires macOS and Xcode")
    with (app / "Info.plist").open("rb") as stream:
        bundle = plistlib.load(stream)["CFBundleIdentifier"]
    run(
        [
            "xcrun",
            "devicectl",
            "device",
            "install",
            "app",
            "--device",
            device,
            str(app.resolve()),
        ],
        output / "install.log",
        180,
    )
    result = run(
        [
            "xcrun",
            "devicectl",
            "device",
            "process",
            "launch",
            "--device",
            device,
            "--terminate-existing",
            "--console",
            "--environment-variables",
            json.dumps({"GTEST_COLOR": "no"}),
            bundle,
            "--",
            "--gtest_color=no",
        ],
        output / "run.log",
    )
    count = verify_console(result)
    logging.info("iOS device HNSW: %s passed, no skipped tests", count)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    verify = commands.add_parser("verify-xml")
    verify.add_argument("path", type=Path)
    verify.add_argument("--expect-arch", choices=["SSE2", "NEON"])
    console = commands.add_parser("verify-console")
    console.add_argument("path", type=Path, help="Log path, or - for stdin")
    console.add_argument("--expect-arch", choices=["SSE2", "NEON"], required=True)
    for name, artifact in [("android", "binary"), ("ios", "app")]:
        command = commands.add_parser(name)
        command.add_argument("--device", required=True)
        command.add_argument(f"--{artifact}", required=True, type=Path)
        command.add_argument(
            "--output", type=Path, default=Path("build/hnsw-device-results")
        )
    args = parser.parse_args()
    if args.command == "verify-xml":
        count = verify_xml(args.path, args.expect_arch)
        logging.info("HNSW coverage: %s passed, no skipped tests", count)
    elif args.command == "verify-console":
        output = (
            sys.stdin.read()
            if args.path == Path("-")
            else args.path.read_text(encoding="utf-8")
        )
        count = verify_console(output, args.expect_arch)
        logging.info("HNSW coverage: %s passed, no skipped tests", count)
    elif args.command == "android":
        android(args.device, args.binary, args.output.resolve())
    else:
        ios(args.device, args.app, args.output.resolve())


if __name__ == "__main__":
    main()
