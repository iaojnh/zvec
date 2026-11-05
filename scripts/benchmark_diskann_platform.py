"""Run isolated native/mobile benchmark processes and write publication metrics."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path


def run(command, **kwargs):
    return subprocess.run(command, check=True, text=True, **kwargs)


def optional_info(command):
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=30)
        return completed.stdout.strip() if completed.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def metadata():
    info = {
        "os": platform.platform(),
        "architecture": platform.machine(),
        "logical_cpus": os.cpu_count(),
        "runner": os.environ.get("RUNNER_NAME"),
        "runner_image": os.environ.get("ImageOS"),
        "runner_image_version": os.environ.get("ImageVersion"),
        "commit": os.environ.get("GITHUB_SHA"),
        "run_url": "https://github.com/" + os.environ.get("GITHUB_REPOSITORY", "")
        + "/actions/runs/" + os.environ.get("GITHUB_RUN_ID", ""),
    }
    if sys.platform == "darwin":
        info["cpu"] = optional_info(["sysctl", "-n", "machdep.cpu.brand_string"])
        memory = optional_info(["sysctl", "-n", "hw.memsize"])
        info["memory_bytes"] = int(memory) if memory else None
        info["storage"] = optional_info(["diskutil", "info", "/"])
    elif sys.platform == "win32":
        info["cpu"] = platform.processor()
        info["storage"] = optional_info([
            "powershell", "-NoProfile", "-Command",
            "Get-PhysicalDisk | Select-Object FriendlyName,MediaType,Size | ConvertTo-Json",
        ])
    elif Path("/proc/cpuinfo").exists():
        cpuinfo = Path("/proc/cpuinfo").read_text()
        info["cpu"] = next((line.split(":", 1)[1].strip() for line in cpuinfo.splitlines()
                            if line.startswith("model name")), platform.processor())
        info["meminfo"] = Path("/proc/meminfo").read_text().splitlines()[:3]
        info["storage"] = optional_info(["lsblk", "-o", "NAME,MODEL,SIZE,ROTA,TYPE,MOUNTPOINTS"])
    info["filesystem"] = str(shutil.disk_usage("."))
    return info


def setup_transport(args):
    if args.transport == "native":
        return str(args.data.resolve())
    if args.transport == "android":
        remote = "/data/local/tmp/zvec-platform-bench"
        run(["adb", "shell", "mkdir", "-p", remote])
        run(["adb", "push", str(args.binary), remote + "/benchmark"])
        run(["adb", "shell", "chmod", "755", remote + "/benchmark"])
        for name in ("base.fbin", "queries.fbin", "neighbors.ibin"):
            run(["adb", "push", str(args.data / name), remote + "/" + name])
        return remote
    run(["xcrun", "simctl", "install", args.device, str(args.binary)])
    container = run(["xcrun", "simctl", "get_app_container", args.device,
                     "com.zvec.diskann_platform_benchmark_test", "data"], capture_output=True).stdout.strip()
    root = Path(container) / "Documents" / "benchmark"
    root.mkdir(parents=True, exist_ok=True)
    for name in ("base.fbin", "queries.fbin", "neighbors.ibin"):
        shutil.copy2(args.data / name, root / name)
    return str(root)


def measure(args, remote, algorithm, phase):
    variables = {
        "ZVEC_BENCH_DATA": remote,
        "ZVEC_BENCH_INDEX": algorithm,
        "ZVEC_BENCH_MODE": phase,
        "ZVEC_BENCH_REPETITIONS": str(args.repetitions),
        "ZVEC_BENCH_TARGET_RECALL": str(args.target_recall),
    }
    env = os.environ.copy()
    if args.transport == "native":
        env.update(variables)
        command = [str(args.binary.resolve()), "--gtest_filter=DiskAnnPlatformBenchmark.Measure"]
    elif args.transport == "android":
        assignments = [f"{key}={value}" for key, value in variables.items()]
        command = ["adb", "shell", "env", *assignments, remote + "/benchmark",
                   "--gtest_filter=DiskAnnPlatformBenchmark.Measure"]
    else:
        env.update({"SIMCTL_CHILD_" + key: value for key, value in variables.items()})
        command = ["xcrun", "simctl", "launch", "--console", args.device,
                   "com.zvec.diskann_platform_benchmark_test",
                   "--gtest_filter=DiskAnnPlatformBenchmark.Measure"]
    log_path = args.output / f"{algorithm}-{phase}.log"
    print(f"Running {args.label}: {algorithm} {phase}", flush=True)
    with log_path.open("w") as log:
        completed = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   text=True, timeout=5400)
    log_text = log_path.read_text(errors="replace")
    matches = re.findall(r"ZVEC_BENCH_RESULT (\{[^\r\n]+\})", log_text)
    # simctl may report FrontBoard code 3 after a short app has already exited.
    # Accept only that documented race with a complete, passing benchmark log.
    launch_race = (args.transport == "ios" and completed.returncode == 3
                   and "FBSOpenApplicationServiceErrorDomain" in log_text
                   and re.search(r"\[  PASSED  \]\s+1 test\.", log_text))
    if (completed.returncode != 0 and not launch_race) or len(matches) != 1 or "[  FAILED  ]" in log_text:
        print(log_text[-18000:])
        raise RuntimeError(f"{algorithm} {phase} failed, exit={completed.returncode}")
    result = json.loads(matches[0])
    if phase == "query" and not (0 <= result["recall_at_10"] <= 1):
        raise ValueError("Invalid recall")
    (args.output / f"{algorithm}-{phase}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--transport", choices=["native", "android", "ios"], default="native")
    parser.add_argument("--device", default="booted")
    parser.add_argument("--target-recall", type=float, default=0.95)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "platform": args.label,
        "environment": "native CI host" if args.transport == "native" else "SIMULATOR; not physical device",
        "host": metadata(),
        "dataset": json.loads((args.data / "dataset.json").read_text()),
        "measurement": {
            "api": "C++ core index interface; not Python or full Collection overhead",
            "memory": "warm query process RSS/working set, decimal MB; build process excluded",
            "concurrency": 1,
            "cache": "warm; entire evaluation set searched once before measurement",
            "qps": "single-thread search-only throughput, excludes harness/recall/RSS sampling",
            "target_calibration_recall": args.target_recall,
            "held_out_recall_required": args.target_recall - 0.02,
            "recall_pair_tolerance": 0.02,
            "storage": "MMAP; same setting for both indexes",
            "diskann_build": {"max_degree": 64, "list_size": 100, "pq_chunk_num": "dimension/4"},
            "hnsw_build": {"m": 32, "ef_construction": 200},
        },
    }
    remote = setup_transport(args)
    for algorithm in ("hnsw", "diskann"):
        report[algorithm] = {"build": measure(args, remote, algorithm, "build"),
                             "query": measure(args, remote, algorithm, "query")}
        (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    hnsw = report["hnsw"]["query"]
    diskann = report["diskann"]["query"]
    report["comparable_recall"] = (
        min(hnsw["recall_at_10"], diskann["recall_at_10"]) >= args.target_recall - 0.02
        and abs(hnsw["recall_at_10"] - diskann["recall_at_10"]) <= 0.02
    )
    report["rss_saving_percent"] = 100 * (1 - diskann["steady_rss_bytes"] / hnsw["steady_rss_bytes"])
    report["index_rss_delta_saving_percent"] = (
        100 * (1 - diskann["index_rss_delta_bytes"] / hnsw["index_rss_delta_bytes"])
        if hnsw["index_rss_delta_bytes"] else None
    )
    (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["comparable_recall"]:
        raise RuntimeError("Recall differs too much; result retained but not eligible for publication table")


if __name__ == "__main__":
    main()
