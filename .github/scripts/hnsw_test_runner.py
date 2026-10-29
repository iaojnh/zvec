"""Validate FP16 coverage reports, or run both suites on a connected device.

Examples:
  python .github/scripts/hnsw_test_runner.py verify-xml build/test_tmp/hnsw_platform_test/results.xml
  python .github/scripts/hnsw_test_runner.py verify-build build --require-arch AVX2
  python .github/scripts/hnsw_test_runner.py android --device SERIAL --build-dir build_android_arm64-v8a
  python .github/scripts/hnsw_test_runner.py ios --device UDID --build-dir build_ios

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
HNSW_CASES = {
    f"{encoding}_{metric}_{layout}_{storage}"
    for encoding in ("Fp32", "NativeFp16", "QuantizedFp16")
    for metric in ("L2", "IP", "Cosine")
    for layout in ("Regular", "Contiguous")
    for storage in ("Mmap", "ReadOnly", "CopyOnWrite", "BufferPool")
} | {"Fp16ExercisesEveryAvailableArchitecture"}
FP16_CASES = {
    f"{index}_{metric}_{encoding}Fp16"
    for index in ("Flat", "IVF", "DiskAnn")
    for metric in ("L2", "IP", "Cosine")
    for encoding in ("Native", "Quantized")
    if index != "DiskAnn" or encoding == "Native"
}
ARCHITECTURES = (
    "Scalar",
    "SSE2",
    "AVX2",
    "AVX512",
    "NEON",
    "NEON_BASELINE",
    "NEON_FP16",
)


def index_cases(diskann: str | None, require_diskann: bool = False) -> set[str]:
    if diskann not in ("0", "1") or (require_diskann and diskann != "1"):
        raise ValueError("Missing or unexpected DiskANN platform support record")
    return {
        name for name in FP16_CASES if diskann == "1" or not name.startswith("DiskAnn_")
    }


def verify_architectures(
    value: str, expected_arch: str | None, required_arches: tuple[str, ...] = ()
) -> None:
    flags = dict(part.split("=", 1) for part in value.split(";") if part)
    if flags.get("Scalar") != "1":
        raise ValueError("Missing scalar FP16 coverage")
    if expected_arch and flags.get(expected_arch) != "1":
        raise ValueError(f"Expected {expected_arch} FP16 coverage: {value}")
    for arch in required_arches:
        if arch not in ARCHITECTURES or flags.get(arch) != "1":
            raise ValueError(f"Required {arch} FP16 coverage was not executed: {value}")


def verify_names(names: set[str], expected: set[str]) -> None:
    names = {name.split("/")[-1].split(".")[-1] for name in names}
    if not expected <= names:
        raise ValueError(f"Missing FP16 test cases: {sorted(expected - names)}")


def xml_property(root: ET.Element, name: str) -> str | None:
    # GTest 1.10 writes suite properties as attributes and case properties as
    # child elements. Accept both representations.
    feature = root.find(f".//property[@name='{name}']")
    if feature is not None:
        return feature.get("value")
    return next(
        (element.get(name) for element in root.iter() if name in element.attrib), None
    )


def verify_xml(
    path: Path,
    expected_arch: str | None = None,
    suite: str = "hnsw",
    required_arches: tuple[str, ...] = (),
    require_diskann: bool = False,
) -> int:
    root = ET.parse(path).getroot()
    cases = root.findall(".//testcase")
    minimum = MINIMUM_TESTS if suite == "hnsw" else 12
    if len(cases) < minimum or int(root.get("tests", "0")) != len(cases):
        raise ValueError(f"Expected at least {minimum} {suite} tests in {path}")
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
    if suite == "fp16":
        expected = index_cases(
            xml_property(root, "fp16_diskann_supported"), require_diskann
        )
        names = {case.get("name", "").split("/")[-1] for case in cases}
        verify_names(names, expected)
        return len(cases)
    verify_names({case.get("name", "") for case in cases}, HNSW_CASES)
    features = xml_property(root, "fp16_cpu_architectures")
    if features is None:
        raise ValueError("Missing FP16 architecture execution report")
    verify_architectures(features, expected_arch, required_arches)
    return len(cases)


def verify_console(
    output: str,
    expected_arch: str = "NEON",
    suite: str = "hnsw",
    required_arches: tuple[str, ...] = (),
    require_diskann: bool = False,
) -> int:
    totals = re.findall(r"^\[==========\]\s+(\d+) tests? from .+ ran\.", output, re.M)
    passed = re.findall(r"^\[  PASSED  \]\s+(\d+) tests?\.", output, re.M)
    if (
        not totals
        or not passed
        or int(totals[-1]) < (MINIMUM_TESTS if suite == "hnsw" else 12)
        or totals[-1] != passed[-1]
        or re.search(r"^\[\s*(?:FAILED|SKIPPED)\s*\]", output, re.M)
    ):
        raise ValueError("HNSW suite was incomplete, failed, or skipped tests")
    if suite == "fp16":
        feature = re.search(r"^FP16 index coverage: DiskAnn=([01]);", output, re.M)
        expected = index_cases(feature[1] if feature else None, require_diskann)
        names = set(re.findall(r"^\[\s*OK\s*\]\s+(\S+) \(", output, re.M))
        verify_names(names, expected)
        return int(totals[-1])
    verify_names(set(re.findall(r"^\[\s*OK\s*\]\s+(\S+) \(", output, re.M)), HNSW_CASES)
    features = re.search(r"^FP16 architecture coverage: (.+)$", output, re.M)
    if not features:
        raise ValueError("Missing FP16 architecture execution report")
    verify_architectures(features[1], expected_arch, required_arches)
    return int(totals[-1])


def verify_build(
    build: Path, expected_arch: str | None, required_arches: tuple[str, ...]
) -> None:
    counts = {}
    for suite in ("hnsw", "fp16"):
        report = build / "test_tmp" / f"{suite}_platform_test" / "results.xml"
        counts[suite] = verify_xml(report, expected_arch, suite, required_arches)
    report = build / "test_tmp/hnsw_platform_test/results.xml"
    features = xml_property(ET.parse(report).getroot(), "fp16_cpu_architectures")
    summary = {"tests": counts, "architectures": features}
    index_report = build / "test_tmp/fp16_platform_test/results.xml"
    summary["diskann_supported"] = (
        xml_property(ET.parse(index_report).getroot(), "fp16_diskann_supported") == "1"
    )
    (build / "test_tmp/fp16-coverage.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    logging.info("FP16 coverage: %s", json.dumps(summary))


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


def android(
    device: str,
    binary: Path,
    output: Path,
    suite: str = "hnsw",
    required_arches: tuple[str, ...] = (),
) -> None:
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
    remote = f"/data/local/tmp/zvec-fp16-{uuid.uuid4().hex}"
    executable = f"{suite}_platform_test"
    try:
        run([*adb, "shell", "mkdir", "-p", remote], output / "mkdir.log", 30)
        run(
            [*adb, "push", str(binary.resolve()), f"{remote}/{executable}"],
            output / "push.log",
            120,
        )
        command = (
            f"cd {shlex.quote(remote)} && chmod 700 {executable} && "
            f"./{executable} --gtest_color=no --gtest_output=xml:results.xml"
        )
        run([*adb, "shell", command], output / "run.log")
        run(
            [*adb, "pull", f"{remote}/results.xml", str(output / "results.xml")],
            output / "pull.log",
            60,
        )
        count = verify_xml(
            output / "results.xml",
            "NEON",
            suite,
            required_arches,
            require_diskann=suite == "fp16",
        )
        logging.info("Android ARM64 %s: %s passed, no skipped tests", suite, count)
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


def ios(
    device: str,
    app: Path,
    output: Path,
    suite: str = "hnsw",
    required_arches: tuple[str, ...] = (),
) -> None:
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
    count = verify_console(
        result, "NEON", suite, required_arches, require_diskann=suite == "fp16"
    )
    logging.info("iOS device %s: %s passed, no skipped tests", suite, count)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    verify = commands.add_parser("verify-xml")
    verify.add_argument("path", type=Path)
    verify.add_argument("--expect-arch", choices=["SSE2", "NEON"])
    verify.add_argument("--suite", choices=["hnsw", "fp16"], default="hnsw")
    verify.add_argument(
        "--require-arch", choices=ARCHITECTURES, action="append", default=[]
    )
    build = commands.add_parser("verify-build")
    build.add_argument("path", type=Path)
    build.add_argument("--expect-arch", choices=["Scalar", "SSE2", "NEON"])
    build.add_argument(
        "--require-arch", choices=ARCHITECTURES, action="append", default=[]
    )
    console = commands.add_parser("verify-console")
    console.add_argument("path", type=Path, help="Log path, or - for stdin")
    console.add_argument("--expect-arch", choices=["SSE2", "NEON"], required=True)
    console.add_argument("--suite", choices=["hnsw", "fp16"], default="hnsw")
    console.add_argument(
        "--require-arch", choices=ARCHITECTURES, action="append", default=[]
    )
    for name, artifact in [("android", "binary"), ("ios", "app")]:
        command = commands.add_parser(name)
        command.add_argument("--device", required=True)
        artifacts = command.add_mutually_exclusive_group(required=True)
        artifacts.add_argument(f"--{artifact}", type=Path)
        artifacts.add_argument("--build-dir", type=Path)
        command.add_argument("--suite", choices=["hnsw", "fp16"], default="hnsw")
        command.add_argument(
            "--require-arch", choices=ARCHITECTURES, action="append", default=[]
        )
        command.add_argument(
            "--output", type=Path, default=Path("build/hnsw-device-results")
        )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    if args.command == "verify-build":
        verify_build(args.path, args.expect_arch, tuple(args.require_arch))
    elif args.command == "verify-xml":
        count = verify_xml(
            args.path, args.expect_arch, args.suite, tuple(args.require_arch)
        )
        logging.info("HNSW coverage: %s passed, no skipped tests", count)
    elif args.command == "verify-console":
        output = (
            sys.stdin.read()
            if args.path == Path("-")
            else args.path.read_text(encoding="utf-8")
        )
        count = verify_console(
            output, args.expect_arch, args.suite, tuple(args.require_arch)
        )
        logging.info("HNSW coverage: %s passed, no skipped tests", count)
    else:
        suites = ("hnsw", "fp16") if args.build_dir else (args.suite,)
        artifacts = []
        for suite in suites:
            if args.command == "android":
                binary = (
                    args.build_dir / "bin" / f"{suite}_platform_test"
                    if args.build_dir
                    else args.binary
                )
                if not binary.is_file():
                    raise ValueError(f"Missing test binary: {binary}")
                artifacts.append((suite, binary))
            else:
                if args.build_dir:
                    apps = list(
                        (args.build_dir / "bin").rglob(f"{suite}_platform_test.app")
                    )
                    if len(apps) != 1:
                        raise ValueError(f"Expected one {suite} test app, got {apps}")
                    app = apps[0]
                else:
                    app = args.app
                if not (app / "Info.plist").is_file():
                    raise ValueError(f"Missing test app: {app}")
                artifacts.append((suite, app))
        for suite, artifact in artifacts:
            output = args.output.resolve() / suite
            execute = android if args.command == "android" else ios
            execute(args.device, artifact, output, suite, tuple(args.require_arch))


if __name__ == "__main__":
    main()
