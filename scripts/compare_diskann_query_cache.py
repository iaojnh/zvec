"""Same-index first-pass/steady-query controls; never equate warmup with cache hits."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import platform
import statistics
import subprocess
from pathlib import Path

import benchmark_diskann_platform as benchmark


def command(argv):
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=120)
        return {"command": argv, "exit": result.returncode,
                "stdout": result.stdout, "stderr": result.stderr}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"command": argv, "exit": None, "error": str(error)}


def reset_cache(args):
    # No cache purging on a developer's computer or a physical phone.
    if (os.environ.get("GITHUB_ACTIONS") != "true"
            or os.environ.get("RUNNER_ENVIRONMENT") != "github-hosted"):
        return {"reset": False, "scope": "none", "reason": "Not an ephemeral GitHub-hosted VM"}
    if args.transport == "android":
        steps = [command(["adb", "root"]), command(["adb", "wait-for-device"]),
                 command(["adb", "shell", "sync"]),
                 command(["adb", "shell", "sysctl", "-w", "vm.drop_caches=3"])]
        return {"reset": steps[-1]["exit"] == 0, "scope": "Android guest only; host cache uncontrolled",
                "steps": steps}
    if platform.system() == "Darwin":
        result = command(["sudo", "-n", "/usr/sbin/purge"])
        return {"reset": result["exit"] == 0, "scope": "Apple CI host", "steps": [result]}
    if platform.system() == "Linux" and args.transport == "native":
        steps = [command(["sync"]), command(["sudo", "-n", "sysctl", "-w", "vm.drop_caches=3"])]
        return {"reset": all(s["exit"] == 0 for s in steps), "scope": "Linux CI host", "steps": steps}
    return {"reset": False, "scope": "none",
            "reason": "No verified Windows file-cache reset; first-pass cache state is uncontrolled"}


def hashes(args, remote):
    result = {}
    for name in ("hnsw.index", "diskann.index", "queries.fbin", "neighbors.ibin"):
        if args.transport == "android":
            output = benchmark.run(["adb", "shell", "sha256sum", remote + "/" + name],
                                   capture_output=True).stdout
            value = output.split()[0]
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("Invalid Android SHA-256")
        else:
            with (Path(remote) / name).open("rb") as source:
                value = hashlib.file_digest(source, "sha256").hexdigest()
        result[name] = value
    return result


def compare(args, remote, baseline):
    report = {"platform": args.label, "environment": baseline["environment"],
              "host": baseline["host"], "dataset": baseline["dataset"],
              "same_index_sha256": hashes(args, remote), "trials_per_case": 3,
              "method": "Fresh process per trial; fixed calibrated budget; concurrency 1. "
                        "First pass has no calibration or workload warmup, one held-out pass, "
                        "after an attempted host/guest cache reset before opening the index. "
                        "This includes warming during that first pass, not a cold reset per query. "
                        "Steady trials search the held-out set once, then time three passes. "
                        "Report process RSS, not total memory including kernel page cache. "
                        "Use read-byte counters to assess actual I/O; reset success alone is not proof.",
              "cases": []}
    output = args.output / "cache-comparison.json"
    for algorithm in ("hnsw", "diskann"):
        original = baseline[algorithm]["query"]
        for trial in range(1, 4):
            for state in ("first-pass", "steady"):
                cache = reset_cache(args) if state == "first-pass" else {"reset": False, "scope": "workload warmup"}
                current = copy.copy(args)
                current.output = args.output / "cache-cases" / algorithm / f"{state}-{trial}"
                current.output.mkdir(parents=True, exist_ok=True)
                current.repetitions = 1 if state == "first-pass" else 3
                current.query_env = {"ZVEC_BENCH_FIXED_BUDGET": str(original["budget"]),
                                     "ZVEC_BENCH_WARMUP": "0" if state == "first-pass" else "1"}
                query = benchmark.measure(current, remote, algorithm, "query")
                if query["budget"] != original["budget"] or not query.get("fixed_budget"):
                    raise ValueError("Cache experiment changed calibrated query parameters")
                if abs(query["recall_at_10"] - original["recall_at_10"]) > 0.0001:
                    raise ValueError("Cache experiment changed recall for the same index and queries")
                report["cases"].append({"algorithm": algorithm, "state": state,
                                        "trial": trial, "cache_control": cache, "query": query})
                output.write_text(json.dumps(report, indent=2) + "\n")
    report["unchanged_index_and_query_hashes"] = hashes(args, remote) == report["same_index_sha256"]
    output.write_text(json.dumps(report, indent=2) + "\n")
    if not report["unchanged_index_and_query_hashes"]:
        raise ValueError("Index/query bytes changed during cache comparison")
    summarize([report], args.output / "cache-comparison.md")


def summarize(reports, path):
    lines = ["# Same-index query cache comparison", "",
             "Medians of three fresh-process trials. First pass: one held-out pass without calibration/warmup; reset before index open when supported. It warms naturally during the pass, not cold per query.",
             "Steady: one workload warmup, then three timed passes. Query budgets and index/query hashes must remain identical.", "",
             "| Environment | Index | State | Successful resets / trials | Query RSS MB | Recall@10 | P95 ms | QPS | Timed read kB / query | I/O counter |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for report in sorted(reports, key=lambda r: r["platform"]):
        for algorithm in ("hnsw", "diskann"):
            for state in ("first-pass", "steady"):
                cases = [c for c in report["cases"] if c["algorithm"] == algorithm and c["state"] == state]
                if not cases:
                    continue
                def median(key):
                    return statistics.median(c["query"][key] for c in cases)
                reads = [c["query"].get("io", {}).get("host_timed_read_bytes",
                         c["query"].get("io", {}).get("timed_read_bytes")) for c in cases]
                read_kb = "unavailable"
                if all(value is not None for value in reads):
                    read_kb = f'{statistics.median(value / (c["query"]["queries"] * c["query"]["repetitions"]) / 1000 for c, value in zip(cases, reads)):.3f}'
                resets = sum(c["cache_control"]["reset"] for c in cases)
                reset_text = f"{resets}/{len(cases)}" if state == "first-pass" else "workload warmup"
                label = report["platform"] + (" (SIMULATOR)" if report["environment"] != "native CI host" else "")
                lines.append(f'| {label} | {algorithm} | {state} | {reset_text} | '
                             f'{median("steady_rss_bytes") / 1e6:.2f} | {median("recall_at_10"):.4f} | '
                             f'{median("p95_ms"):.3f} | {median("qps"):.1f} | {read_kb} | '
                             f'{cases[0]["query"].get("io", {}).get("host_counter_source", cases[0]["query"].get("io", {}).get("source", "unavailable"))} |')
    lines.extend(["", "Windows transfer counters are logical, not physical-device bytes; its first-pass cache state is uncontrolled. Android reset/counters cover the guest only. Optional iOS disk-byte observations cover the simulator process on its macOS host, not phone flash; without the host observer, the counter is unavailable. No-cache flags and successful reset commands do not exclude hypervisor/storage caches.",
                  "RSS is query-process memory, not total memory including kernel file cache. Index-open time/bytes, warmup bytes, timed bytes, CPU and page faults are retained in JSON. These CI results do not rank operating systems.", ""])
    path.write_text("\n".join(lines))
