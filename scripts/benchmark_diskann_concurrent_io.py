"""Measure 2/4/8 query workers sharing one process-wide native I/O budget."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import benchmark_diskann_controlled_io as fixture_tools
import benchmark_diskann_platform as native
from benchmark_diskann_paced_io import aggregate, evaluate_probe, prepare_android, validate_query


def validate_concurrent_query(query, algorithm, fixture, rate, label, workers, bps, depth):
    validate_query(query, algorithm, fixture, rate, label, bps, depth, allow_idle_limiter=True)
    if (query.get("query_threads") != workers or query.get("workers_alive_at_rss_sample") != workers or
            len(query.get("worker_queries", [])) != workers):
        raise ValueError("Requested query workers or live-worker RSS sample not verified")
    count = query["queries"] * query["repetitions"]
    if (sum(query["worker_queries"]) != count or min(query["worker_queries"]) <= 0 or
            max(query["worker_queries"]) - min(query["worker_queries"]) > 1):
        raise ValueError("Workers did not partition the same fixed total query workload")
    wall = query["timed_block_wall_ms"]
    if wall <= 0 or not math.isclose(count * 1000 / wall, query["qps"], rel_tol=1e-9):
        raise ValueError("Concurrent QPS does not use full process wall time")
    if not 1 <= query["max_active_queries"] <= workers:
        raise ValueError("Invalid observed query concurrency")
    if algorithm == "diskann":
        pace = query["native_pacing"]
        if (pace.get("scope") != "process-wide" or
                not 1 <= pace.get("max_reserved_reads", 0) <= depth or
                pace["bytes"] != pace["reads"] * 4096 or
                pace["actual_wait_ns"] > wall * 1e6 * workers * 1.001):
            raise ValueError("I/O budget/depth was not shared by every worker")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--transport", choices=["native", "android", "ios"], default="native")
    parser.add_argument("--device", default="booted")
    parser.add_argument("--profiles", default="2000,10000,20000,0")
    parser.add_argument("--threads", default="2,4,8")
    parser.add_argument("--common-iops", type=int, default=2000)
    parser.add_argument("--queue-depth", type=int, default=8)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--target-recall", type=float, default=0.95)
    parser.add_argument("--baseline", type=Path, default=Path(
        "benchmarks/diskann-controlled-io/results/2026-11-05/all-platform-results.json"))
    args = parser.parse_args()
    rates = list(dict.fromkeys(int(value) for value in args.profiles.split(",")))
    threads = list(dict.fromkeys(int(value) for value in args.threads.split(",")))
    threads.sort()
    if (set(threads) != {2, 4, 8} or args.common_iops not in rates or 0 not in rates or
            any(rate < 0 for rate in rates) or args.queue_depth != 8 or
            args.trials != 3 or args.repetitions != 3):
        parser.error("Need only 2/4/8 new workers, fixed total depth 8, three trials/repetitions and common+native tiers")
    args.data, args.output = args.data.resolve(), args.output.resolve()
    if args.data == args.output or args.data in args.output.parents or args.output in args.data.parents:
        parser.error("Data/output must be separate non-nested directories")
    args.output.mkdir(parents=True, exist_ok=True)
    args.monitor_ios_io = args.transport == "ios"
    fixture = fixture_tools.verify_fixture(args.data)
    if "synthetic" in fixture["dataset"]["dataset"].lower():
        raise ValueError("Synthetic data cannot supply publication measurements")
    baseline_bytes = args.baseline.read_bytes()
    baselines = [r for r in json.loads(baseline_bytes) if r["label"] == args.label]
    if len(baselines) != 1 or baselines[0]["fixture"] != fixture or not baselines[0]["publication_ready"]:
        raise ValueError("No verified same-index one-worker baseline for this platform")
    baseline = baselines[0]
    prepare_android(args)
    remote = native.setup_transport(args)
    fixture_tools.copy_indexes(args, remote, fixture)
    report = {
        "schema_version": 1, "kind": "concurrent-paced-native-file-io", "label": args.label,
        "host": native.metadata(), "fixture": fixture, "data_path": remote,
        "new_query_threads": threads, "baseline_reused": True,
        "baseline": baseline, "baseline_file_sha256": hashlib.sha256(baseline_bytes).hexdigest(),
        "common_iops": args.common_iops, "queue_depth": args.queue_depth,
        "trials": args.trials, "repetitions": args.repetitions, "requested_profiles": rates,
        "probes": {}, "profiles": {},
        "measurement": {
            "api": "closed-loop concurrent C++ core index search; one shared index per process",
            "workload": "same 1000 held-out queries repeated three times total, partitioned across workers",
            "qps": "total completed queries / full process timed-block wall time",
            "p95": "pooled actual search-call latency, including global rate/depth queue waits",
            "memory": "total process RSS/Windows working set while all worker TLS contexts remain alive",
            "io_budget": "one shared IOPS bucket, byte bucket and total depth 8; no per-worker multiplication",
            "baseline": "one-worker data reused from earlier CI hosts; same indexes/parameters, different runner instances",
        },
    }
    def save():
        (args.output / "concurrent-results.json").write_text(json.dumps(report, indent=2) + "\n")
    for rate in (value for value in rates if value):
        worker = copy.copy(args)
        worker.output = args.output / f"probe-{rate}"
        worker.output.mkdir()
        worker.gtest_filter = "DiskAnnPacedIOProbe.Measure"
        worker.query_env = {"ZVEC_BENCH_IOPS": str(rate), "ZVEC_BENCH_PACE_DEPTH": str(args.queue_depth),
                            "ZVEC_BENCH_PACE_BPS": str(max(16777216, rate * 8192))}
        probe = native.measure(worker, remote, "diskann", "probe")
        report["probes"][str(rate)] = {"raw": probe, "verification": evaluate_probe(probe, args.label, rate)}
        save()
    for rate in rates:
        qualification = report["probes"].get(str(rate), {}).get("verification")
        profile = {"iops_limit": rate, "bytes_per_second_limit": max(16777216, rate * 8192) if rate else 0,
                   "qualification": qualification, "threads": {"1": copy.deepcopy(baseline["profiles"].get(str(rate), {}))}}
        profile["threads"]["1"]["reused"] = True
        report["profiles"][str(rate)] = profile
        # Always collect the common cap, recording host qualification separately.
        # Optional higher tiers require the pre-existing 90% capacity check.
        if rate and rate != args.common_iops and not qualification["eligible"]:
            for count in threads:
                profile["threads"][str(count)] = {"eligible": False, "reasons": qualification["reasons"]}
            save()
            continue
        if rate and not qualification["cache_verified"]:
            raise ValueError("Common native cache policy failed verification")
        for count in threads:
            entry = {"eligible": True, "query_threads": count, "raw_trials": {"hnsw": [], "diskann": []},
                     "medians": {}, "host_qualified": not rate or qualification["eligible"]}
            profile["threads"][str(count)] = entry
            for trial in range(args.trials):
                worker = copy.copy(args)
                worker.output = args.output / f"rate-{rate}-threads-{count}-trial-{trial+1}"
                worker.output.mkdir()
                worker.gtest_filter = "DiskAnnConcurrentBenchmark.Measure"
                for algorithm in ("hnsw", "diskann"):
                    print(f"Measure {args.label}: total_IOPS={rate}, query_threads={count}, trial={trial+1}, {algorithm}", flush=True)
                    worker.query_env = {
                        "ZVEC_BENCH_PACED_IO": "1", "ZVEC_BENCH_THREADS": str(count),
                        "ZVEC_BENCH_FIXED_BUDGET": str(fixture["budgets"][algorithm]),
                        "ZVEC_BENCH_IOPS": str(rate), "ZVEC_BENCH_PACE_DEPTH": str(args.queue_depth),
                        "ZVEC_BENCH_PACE_BPS": str(profile["bytes_per_second_limit"])}
                    query = native.measure(worker, remote, algorithm, "query")
                    validate_concurrent_query(query, algorithm, fixture, rate, args.label, count,
                                              profile["bytes_per_second_limit"], args.queue_depth)
                    entry["raw_trials"][algorithm].append(query)
                    save()
            entry["medians"] = {algorithm: aggregate(rows) for algorithm, rows in entry["raw_trials"].items()}
            fixture_tools.verify_fixture(args.data)
            save()
    report["measurement_complete"] = True
    report["common_host_qualified"] = report["probes"][str(args.common_iops)]["verification"]["eligible"]
    save()


if __name__ == "__main__":
    main()
