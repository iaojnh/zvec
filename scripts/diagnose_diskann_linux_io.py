"""Same-index Linux I/O diagnosis; all extra runs are diagnostic, not publication data."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import benchmark_diskann_platform as benchmark


def command_info(command):
    result = subprocess.run(command, capture_output=True, text=True)
    return {"command": command, "exit": result.returncode,
            "stdout": result.stdout, "stderr": result.stderr}


def snapshot(args, name):
    commands = [
        ["uname", "-a"], ["lscpu"], ["lsblk", "-o", "NAME,MODEL,SIZE,ROTA,TYPE,MOUNTPOINTS"],
        ["findmnt", "-T", str(args.data)], ["df", "-h", str(args.data), "/dev/shm", "/mnt"],
        ["sysctl", "kernel.io_uring_disabled"], ["sysctl", "kernel.io_uring_group"],
        ["sh", "-c", "ldconfig -p | grep libaio || true"],
        ["sh", "-c", "cat /sys/block/sda/queue/scheduler /sys/block/sda/queue/nr_requests /sys/block/sda/device/queue_depth 2>/dev/null || true"],
        ["cat", "/proc/diskstats"], ["cat", "/proc/meminfo"],
    ]
    (args.output / f"{name}-environment.json").write_text(
        json.dumps([command_info(command) for command in commands], indent=2) + "\n")


def probe(args, name):
    env = os.environ.copy()
    env["ZVEC_BENCH_DATA"] = str(args.data.resolve())
    path = args.output / f"{name}-io-probe.log"
    with path.open("w") as log:
        result = subprocess.run([str(args.binary.resolve()), "--gtest_filter=DiskAnnLinuxIO.RandomReads"],
                                env=env, stdout=log, stderr=subprocess.STDOUT, timeout=120)
    text = path.read_text(errors="replace")
    results = [json.loads(value) for value in re.findall(r"ZVEC_IO_PROBE_RESULT (\{[^\n]+\})", text)]
    if result.returncode or len(results) != 2:
        print(text[-12000:])
        raise RuntimeError("Native I/O probe failed")
    return results


def query_case(args, name, expected=None):
    snapshot(args, name)
    probes = probe(args, name)
    if expected and any(value["backend"] != expected for value in probes):
        report = {"case": name, "skipped": True, "reason": "Requested backend unavailable",
                  "requested_backend": expected, "io_probes": probes}
    else:
        case_args = argparse.Namespace(**vars(args))
        case_args.output = args.output / name
        case_args.output.mkdir(parents=True, exist_ok=True)
        result = benchmark.measure(case_args, str(args.data.resolve()), "diskann", "query")
        text = (case_args.output / "diskann-query.log").read_text(errors="replace")
        usage = [json.loads(value) for value in re.findall(r"ZVEC_IO_PROCESS_RESULT (\{[^\n]+\})", text)]
        report = {"case": name, "io_probes": probes, "query": result, "process": usage}
    (args.output / f"{name}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return report


def fio_case(args, engine, direct, depth):
    name = f"fio-{engine}-direct{direct}-depth{depth}"
    filename = args.data.resolve() / "diskann.index"
    size = (filename.stat().st_size // 4096) * 4096
    command = ["/usr/bin/fio", f"--name={name}", f"--filename={filename}", "--readonly",
               "--rw=randread", "--bs=4k", f"--size={size}", f"--ioengine={engine}",
               f"--direct={direct}", f"--iodepth={depth}", "--numjobs=1", "--runtime=10",
               "--time_based=1", "--group_reporting=1", "--output-format=json"]
    result = command_info(command)
    (args.output / f"{name}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(name, "exit", result["exit"], flush=True)


def kernel_control(value):
    result = command_info(["sudo", "sysctl", "-w", f"kernel.io_uring_disabled={value}"])
    print(json.dumps(result), flush=True)
    if result["exit"]:
        raise RuntimeError("Cannot control io_uring on this CI host")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["initial", "async", "pread", "summary"], required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.transport = "native"
    args.repetitions = 3
    args.target_recall = 0.95
    if args.stage == "initial":
        snapshot(args, "initial")
        subprocess.run([
            sys.executable, str(Path(__file__).with_name("benchmark_diskann_platform.py")),
            "--binary", str(args.binary), "--data", str(args.data),
            "--output", str(args.output / "baseline"), "--label", args.label,
        ], check=True)
        query_case(args, "default")
    elif args.stage == "async":
        kernel_control(0)
        query_case(args, "io_uring", "io_uring")
        for engine, direct, depth in [("sync", 0, 1), ("sync", 1, 1),
                                      ("io_uring", 1, 1), ("io_uring", 1, 8)]:
            fio_case(args, engine, direct, depth)
        # Trace in a separate invocation; traced timings are never publication data.
        env = os.environ.copy()
        env.update({"ZVEC_BENCH_MODE": "query", "ZVEC_BENCH_INDEX": "diskann",
                    "ZVEC_BENCH_DATA": str(args.data.resolve()), "ZVEC_BENCH_REPETITIONS": "1"})
        with (args.output / "strace-query.log").open("w") as log:
            result = subprocess.run([
                "strace", "-f", "-c", "-e", "trace=io_uring_setup,io_uring_enter,pread64,io_submit,io_getevents",
                "-o", str(args.output / "strace-summary.txt"), str(args.binary.resolve()),
                "--gtest_filter=DiskAnnPlatformBenchmark.Measure",
            ], env=env, stdout=log, stderr=subprocess.STDOUT, timeout=300)
        print("Separate syscall trace exit", result.returncode, flush=True)
        kernel_control(2)
        query_case(args, "libaio", "libaio")
    elif args.stage == "pread":
        kernel_control(2)
        query_case(args, "pread", "pread")
        ram = Path("/dev/shm") / ("zvec-io-diagnostic-" + args.label)
        if ram.exists():
            shutil.rmtree(ram)
        shutil.copytree(args.data, ram)
        args.data = ram
        query_case(args, "pread-tmpfs", "pread")
    else:
        cases = [json.loads(path.read_text()) for path in args.output.glob("*.json")
                 if not path.name.endswith("environment.json") and not path.name.startswith("fio-")]
        text = ["# Linux DiskANN I/O diagnostics", "",
                "Same built index, same query set. All A/B and traced runs are diagnostic only.", "",
                "| Case | Backend | 4K QD1 P95 ms | 8-read batch P95 ms | Query P95 ms | QPS | Recall@10 | Wall s | User CPU s | System CPU s |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for case in cases:
            if case.get("skipped"):
                text.append(f'| {case["case"]} | unavailable | | | | | | | | |')
                continue
            probes, q = case["io_probes"], case["query"]
            process = case["process"][-1] if case["process"] else {}
            text.append(f'| {case["case"]} | {probes[0]["backend"]} | {probes[0]["p95_ms"]:.3f} | '
                        f'{probes[1]["p95_ms"]:.3f} | {q["p95_ms"]:.3f} | {q["qps"]:.1f} | '
                        f'{q["recall_at_10"]:.4f} | {process.get("wall_seconds")} | '
                        f'{process.get("user_cpu_seconds")} | {process.get("system_cpu_seconds")} |')
        text.extend(["", "Fio outputs, native backend logs, CPU/I/O counters, syscall trace, kernel settings and disk metadata are included alongside this report.",
                     "tmpfs is a diagnostic RAM-backed filesystem; it is excluded from publication results.", ""])
        report = "\n".join(text)
        (args.output / "diagnosis.md").write_text(report)
        print(report)


if __name__ == "__main__":
    main()
