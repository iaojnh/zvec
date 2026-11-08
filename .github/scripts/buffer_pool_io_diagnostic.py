# Copyright 2025-present the zvec project
# SPDX-License-Identifier: Apache-2.0
"""Compare disk locations, backends and queue depth without replacing acceptance."""

# Standalone diagnostic output is intentional.
# ruff: noqa: T201
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

FILE_BYTES = 400_000 * 4096
FIO = "/usr/bin/fio"


def invoke(arguments, output):
    command = [FIO, *arguments, "--output-format=json", f"--output={output}"]
    result = subprocess.run(
        command, check=False, capture_output=True, text=True, timeout=180
    )
    output.with_suffix(".stderr").write_text(result.stderr)
    assert result.returncode == 0, (command, result.stdout, result.stderr)
    data = json.loads(output.read_text())
    assert data["jobs"]
    assert all(job["error"] == 0 for job in data["jobs"])
    return data


def run(root, mounted, evidence):
    evidence.mkdir(exist_ok=True)
    report = {
        "diagnostic_only": True,
        "file_bytes": FILE_BYTES,
        "logical_pages": 400_000,
        "page_bytes": 4096,
        "workers": 4,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "paths": [],
        "cases": [],
    }
    devices = set()
    for label, directory in (("runner_temp", root), ("mnt", mounted)):
        device = directory.stat().st_dev
        location = {"label": label, "path": str(directory), "device": device}
        report["paths"].append(location)
        if device in devices:
            location["same_device_as_previous"] = True
            continue
        devices.add(device)
        available = os.statvfs(directory)
        assert available.f_bavail * available.f_frsize > FILE_BYTES + 1024**3
        filename = directory / "vectors.dat"
        assert not filename.exists()
        try:
            invoke(
                [
                    "--name=prepare",
                    f"--filename={filename}",
                    f"--size={FILE_BYTES}",
                    "--rw=write",
                    "--bs=262144",
                    "--ioengine=psync",
                    "--direct=1",
                    "--refill_buffers=1",
                    "--end_fsync=1",
                ],
                evidence / f"{label}-prepare.json",
            )
            assert filename.stat().st_size == FILE_BYTES
            for direct in (0, 1):
                for engine, depth in (
                    ("psync", 1),
                    ("io_uring", 1),
                    ("io_uring", 16),
                    ("libaio", 1),
                    ("libaio", 16),
                ):
                    name = f"{label}-{engine}-direct{direct}-depth{depth}"
                    data = invoke(
                        [
                            f"--name={name}",
                            f"--filename={filename}",
                            f"--size={FILE_BYTES}",
                            "--rw=randread",
                            "--bs=4096",
                            f"--ioengine={engine}",
                            f"--iodepth={depth}",
                            f"--direct={direct}",
                            "--numjobs=4",
                            "--time_based",
                            "--runtime=15",
                            "--ramp_time=5",
                            "--group_reporting",
                            "--readonly",
                            "--allow_file_create=0",
                            "--randrepeat=1",
                            "--randseed=2026",
                            "--invalidate=1",
                        ],
                        evidence / f"{name}.json",
                    )
                    job = data["jobs"][0]
                    read = job["read"]
                    assert read["total_ios"] > 0
                    case = {
                        "name": name,
                        "iops": read["iops"],
                        "bandwidth_kib_s": read["bw"],
                        "mean_latency_ns": read["lat_ns"]["mean"],
                        "p99_completion_ns": read["clat_ns"]["percentile"]["99.000000"],
                        "user_cpu": job["usr_cpu"],
                        "system_cpu": job["sys_cpu"],
                        "actual_depth": job["iodepth_level"],
                    }
                    report["cases"].append(case)
                    (evidence / "summary.json").write_text(json.dumps(report, indent=2))
                    print(json.dumps(case), flush=True)
        finally:
            filename.unlink(missing_ok=True)
    (evidence / "summary.json").write_text(json.dumps(report, indent=2))
    print("IO_DIAGNOSTIC_COMPLETE", flush=True)


if __name__ == "__main__":
    run(*(Path(value).resolve() for value in sys.argv[1:]))
