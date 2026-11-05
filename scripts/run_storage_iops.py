"""Run the same standalone native 4K random-read probe on CI and simulators."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
from pathlib import Path

import benchmark_diskann_platform as existing


def execute(command, **kwargs):
    return subprocess.run(command, check=True, text=True, **kwargs)


def setup(args):
    if args.transport == "native":
        args.data.mkdir(parents=True, exist_ok=True)
        return str(args.data.resolve() / "random-4k.data")
    if args.transport == "android":
        directory = "/data/local/tmp/zvec-storage-iops"
        execute(["adb", "shell", "mkdir", "-p", directory])
        execute(["adb", "push", str(args.binary), directory + "/storage_iops"])
        execute(["adb", "shell", "chmod", "755", directory + "/storage_iops"])
        return directory + "/random-4k.data"
    execute(["xcrun", "simctl", "install", args.device, str(args.binary)])
    directory = execute(["xcrun", "simctl", "get_app_container", args.device,
                         "com.zvec.storage_iops", "data"], capture_output=True).stdout.strip()
    return str(Path(directory) / "Documents" / "random-4k.data")


def purge_ci_cache(args, mode):
    if mode != "uncached" or platform.system() != "Darwin":
        return {"attempted": False}
    # Never purge a user's computer. Each GitHub-hosted job has its own VM.
    if (os.environ.get("GITHUB_ACTIONS") != "true"
            or os.environ.get("RUNNER_ENVIRONMENT") != "github-hosted"):
        return {"attempted": False, "reason": "Not a GitHub-hosted VM"}
    binary = shutil.which("purge")
    if not binary:
        return {"attempted": False, "reason": "purge not installed"}
    completed = subprocess.run(["sudo", "-n", binary], capture_output=True, text=True, timeout=90)
    return {"attempted": True, "exit": completed.returncode,
            "stdout": completed.stdout, "stderr": completed.stderr}


def probe(args, filename, mode, threads=1):
    arguments = ["--file", filename, "--mode", mode, "--threads", str(threads),
                 "--seconds", str(args.seconds), "--size-mib", str(args.size_mib)]
    if args.transport == "native":
        command = [str(args.binary.resolve()), *arguments]
    elif args.transport == "android":
        command = ["adb", "shell", "/data/local/tmp/zvec-storage-iops/storage_iops", *arguments]
    else:
        command = ["xcrun", "simctl", "launch", "--console", "--terminate-running-process",
                   args.device, "com.zvec.storage_iops", *arguments]
    cache = purge_ci_cache(args, mode)
    path = args.output / f"{mode}-{threads}.log"
    with path.open("w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                text=True, timeout=max(180, args.seconds + 90))
    text = path.read_text(errors="replace")
    matches = re.findall(r"IOPS_RESULT (\{[^\r\n]+\})", text)
    race = (args.transport == "ios" and result.returncode == 3
            and "FBSOpenApplicationServiceErrorDomain" in text and len(matches) == 1)
    if (result.returncode and not race) or len(matches) != 1 or "IOPS_ERROR" in text:
        print(text[-12000:])
        raise RuntimeError(f"Storage probe failed: {mode}, {threads}, exit={result.returncode}")
    record = json.loads(matches[0])
    record.update({"mode": mode, "threads": threads, "cache_purge": cache})
    if mode != "prepare" and record.get("supported"):
        if record["operations"] <= 0 or record["elapsed_seconds"] < args.seconds * 0.9:
            raise ValueError("Insufficient I/O measurements")
        if not record.get("payload_validated"):
            raise ValueError("Read payload was not validated")
    (args.output / f"{mode}-{threads}.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)
    return record


def summarize(source, output):
    reports = [json.loads(path.read_text()) for path in sorted(source.rglob("result.json"))]
    lines = ["# Cross-platform storage IOPS", "",
             "4 KiB random reads; deterministic 256 MiB nonzero fixture; 10 seconds per case; native synchronous calls with 1 or 8 worker threads.",
             "Each worker owns a separate file descriptor/handle. Eight threads mean up to eight concurrent reads, not one asynchronous queue of depth eight.", "",
             "| Environment | Uncached policy | Buffered 1-thread IOPS | Buffered 8-thread IOPS | Uncached 1-thread IOPS | Uncached 8-thread IOPS | Uncached 1-thread P95 ms |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for report in sorted(reports, key=lambda item: item["platform"]):
        cases = {(c["mode"], c["threads"]): c for c in report["cases"]}
        def metric(mode, threads, name, digits=0):
            case = cases.get((mode, threads), {})
            value = case.get(name)
            return f"{value:,.{digits}f}" if case.get("supported") and value is not None else "unsupported/missing"
        uncached = cases.get(("uncached", 1), {})
        label = report["platform"] + (" (SIMULATOR)" if report["transport"] != "native" else "")
        lines.append(f'| {label} | {uncached.get("policy", uncached.get("reason", "missing"))} | '
                     f'{metric("buffered",1,"iops")} | {metric("buffered",8,"iops")} | '
                     f'{metric("uncached",1,"iops")} | {metric("uncached",8,"iops")} | '
                     f'{metric("uncached",1,"p95_ms",3)} |')
    lines.extend(["", "Apple F_NOCACHE is an advisory per-file policy; the cache-purge command status and available OS disk-byte accounting are retained for checking cache effects.",
                  "Android O_DIRECT bypasses the guest cache, not necessarily QEMU/host caches. Android and iOS simulator numbers are not physical-phone flash benchmarks.",
                  "Windows process I/O-byte counters are logical; Linux read_bytes and macOS proc_pid_rusage diskio counters have OS-specific accounting semantics.", ""])
    output.mkdir(parents=True, exist_ok=True)
    (output / "iops.md").write_text("\n".join(lines))
    (output / "results.json").write_text(json.dumps(reports, indent=2) + "\n")
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--data", type=Path, default=Path("iops-data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label")
    parser.add_argument("--transport", choices=["native", "android", "ios"], default="native")
    parser.add_argument("--device", default="booted")
    parser.add_argument("--seconds", type=int, default=10)
    parser.add_argument("--size-mib", type=int, default=256)
    parser.add_argument("--summarize", type=Path)
    args = parser.parse_args()
    if args.summarize:
        summarize(args.summarize, args.output)
        return
    if not args.binary or not args.label:
        parser.error("--binary and --label are required for measurements")
    args.output.mkdir(parents=True, exist_ok=True)
    filename = setup(args)
    host = existing.metadata()
    host["data_file"] = filename
    if args.transport == "native":
        host["data_disk_usage"] = str(shutil.disk_usage(args.data))
        host["data_filesystem_device"] = args.data.stat().st_dev
        if platform.system() == "Linux":
            host["data_mount"] = existing.optional_info(["findmnt", "-T", str(args.data.resolve())])
        elif platform.system() == "Darwin":
            host["data_df"] = existing.optional_info(["df", "-P", str(args.data.resolve())])
    elif args.transport == "android":
        host["guest_kernel"] = existing.optional_info(["adb", "shell", "uname", "-a"])
        host["guest_storage"] = existing.optional_info(["adb", "shell", "df", "-h", "/data/local/tmp"])
        host["guest_model"] = existing.optional_info(["adb", "shell", "getprop", "ro.product.model"])
    else:
        host["simulator_devices"] = existing.optional_info(["xcrun", "simctl", "list", "devices", "booted"])
    report = {"platform": args.label, "transport": args.transport, "host": host,
              "fixture": {"bytes": args.size_mib * 1048576, "seed": 20261105,
                          "pattern": "SplitMix64(seed + word index); identical little-endian bytes on all tested architectures"},
              "method": {"block_bytes": 4096, "seconds_per_case": args.seconds,
                         "concurrency": [1, 8], "api": "native synchronous reads; separate handle per thread",
                         "buffered_cache": "entire fixture read sequentially before each buffered case",
                         "uncached": "platform-specific no-cache flags; Apple flags are advisory; CI-host purge status recorded",
                         "validation": "first 64-bit word of every read checked against deterministic generator"},
              "cases": []}
    report["prepare"] = probe(args, filename, "prepare")
    for mode in ("buffered", "uncached"):
        for threads in (1, 8):
            report["cases"].append(probe(args, filename, mode, threads))
            (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
