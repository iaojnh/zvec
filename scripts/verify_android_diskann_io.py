"""Verify production Android direct-I/O fallback and actual page-cache behavior."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import benchmark_diskann_platform as benchmark


def run_test(binary, output, probe=False):
    remote = "/data/local/tmp/zvec-android-io"
    benchmark.run(["adb", "shell", "mkdir", "-p", remote])
    target = remote + "/" + binary.name
    benchmark.run(["adb", "push", str(binary), target])
    benchmark.run(["adb", "shell", "chmod", "755", target])
    # Both names/paths are fixed CI-generated values without shell metacharacters.
    args = ["adb", "shell", f"cd {remote} && env ZVEC_ANDROID_IO_PROBE=1 {target}"]
    with output.open("w") as log:
        result = subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, text=True, timeout=600)
    text = output.read_text(errors="replace")
    if result.returncode or "[  FAILED  ]" in text or not re.search(r"\[  PASSED  \]\s+\d+ tests?\.", text):
        print(text[-18000:])
        raise RuntimeError("Android regression tests failed")
    if not probe:
        return None
    matches = re.findall(r"ZVEC_ANDROID_IO_RESULT (\{[^\r\n]+\})", text)
    if len(matches) != 1:
        raise ValueError("Missing production-reader cache probe")
    return json.loads(matches[0])


def evaluate(report):
    cases = {c["state"]: c for c in report["cases"]}
    if not report.get("payload_validated") or len(cases) != 4:
        raise ValueError("Incomplete cache verification")
    warm = cases["production-warm"]
    cold = cases["production-cold"]
    buffered = cases["buffered-cold"]
    if any(c["operations"] != 4096 for c in cases.values()):
        raise ValueError("Insufficient completed reads")
    # Warm reads must still issue storage I/O; cold reads must not populate the
    # file cache. The buffered control proves that cache residency/counters work.
    control_valid = (cases["buffered-warm"]["resident_before"] >= report["fixture_bytes"] * .9
                     and cold["resident_before"] <= report["fixture_bytes"] * .01
                     and buffered["resident_before"] <= report["fixture_bytes"] * .01
                     and buffered["resident_after"] >= buffered["requested_bytes"] * .9
                     and buffered["read_bytes"] >= buffered["requested_bytes"] * .9)
    verified = (control_valid and warm["direct_requested"] and cold["direct_requested"]
                and warm["read_bytes"] >= warm["requested_bytes"] * .9
                and cold["read_bytes"] >= cold["requested_bytes"] * .9
                and cold["resident_after"] <= report["fixture_bytes"] * .01)
    fallback = not warm["direct_requested"] or not cold["direct_requested"]
    report["verification"] = {"controls_valid": control_valid,
                              "cache_bypass_verified": verified,
                              "classification": "verified-direct" if verified else
                              "explicit-buffered-fallback" if fallback else
                              "direct-requested-but-cache-bypass-not-verified",
                              "scope": "Android guest file cache and storage accounting, not host cache or phone flash"}
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binaries", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {}
    for name, command in {
        "mounts": ["adb", "shell", "cat", "/proc/mounts"],
        "kernel": ["adb", "shell", "uname", "-a"],
        "filesystem": ["adb", "shell", "stat", "-f", "/data/local/tmp"],
        "model": ["adb", "shell", "getprop", "ro.product.model"],
    }.items():
        metadata[name] = benchmark.optional_info(command)
    (args.output / "environment.json").write_text(json.dumps(metadata, indent=2) + "\n")
    run_test(args.binaries / "diskann_mobile_compat_test", args.output / "mobile-compat.log")
    report = evaluate(run_test(args.binaries / "diskann_android_io_test", args.output / "android-io.log", probe=True))
    (args.output / "cache-verification.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# Android DiskANN direct-I/O verification", "",
             "Production reader, 64 MiB nonzero fixture, 4096 aligned 4 KiB reads per case; every byte checked.",
             "Cache residency uses mincore; storage reads use /proc/thread-self/io. These are guest-level checks.", "",
             "| Case | O_DIRECT requested | Fallback errno | Read MiB | Cached MiB before | Cached MiB after |",
             "| --- | --- | --- | --- | --- | --- |"]
    for c in report["cases"]:
        lines.append(f'| {c["state"]} | {c["direct_requested"]} | {c["fallback_errno"]} | '
                     f'{c["read_bytes"] / 1048576:.3f} | {c["resident_before"] / 1048576:.3f} | {c["resident_after"] / 1048576:.3f} |')
    lines += ["", "Classification: **" + report["verification"]["classification"] + "**", ""]
    (args.output / "verification.md").write_text("\n".join(lines))
    print("\n".join(lines), flush=True)
    # End-to-end search must still work and retain measured IO/RSS/recall.
    benchmark.run([sys.executable, "scripts/benchmark_diskann_platform.py",
                   "--binary", str(args.binaries / "diskann_platform_benchmark_test"),
                   "--data", str(args.data), "--output", str(args.output / "queries"),
                   "--label", "android-x86_64-direct-io", "--transport", "android", "--cache-comparison"])
    if not report["verification"]["controls_valid"]:
        raise RuntimeError("Cache controls were not established; no bypass claim is valid")
    if not report["verification"]["cache_bypass_verified"] and report["verification"]["classification"] != "explicit-buffered-fallback":
        raise RuntimeError("O_DIRECT was accepted but cache bypass was not verified; inspect the filesystem/counters")


if __name__ == "__main__":
    main()
