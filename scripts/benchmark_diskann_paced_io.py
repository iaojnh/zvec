"""Measure real, rate-limited native file I/O and complete query wall time."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import statistics

import benchmark_diskann_controlled_io as fixture_tools
import benchmark_diskann_platform as native


def observed_probe(case, probe):
    host = probe.get("host_probe_io", {}).get(case["state"])
    return host["read_bytes"] if host else case.get("read_bytes")


def evaluate_probe(probe, label, rate):
    cases = {c["state"]: c for c in probe["cases"]}
    expected = {"buffered-warm", "native-unlimited", "paced-target", "paced-half"}
    if set(cases) != expected:
        raise ValueError("Incomplete native pacing probe")
    if len({c["checksum"] for c in cases.values()}) != 1 or cases["buffered-warm"]["checksum"] == "0":
        raise ValueError("Probe payload differs across native reads and pacing rates")
    windows = label.startswith("windows-")
    cache_valid = True
    reasons = []
    for name, case in cases.items():
        if (case["operations"] != 8192 or case["requested_bytes"] != 8192 * 4096 or
                case.get("warmup_operations") != 8192):
            raise ValueError("Probe I/O volume changed")
        physical = observed_probe(case, probe)
        case["observed_read_bytes"] = physical
        if name == "buffered-warm":
            if not windows and (physical is None or physical > case["requested_bytes"] * 0.05):
                cache_valid = False
                reasons.append("Buffered warm control is not demonstrably cached")
        elif windows:
            if "FILE_FLAG_NO_BUFFERING" not in case["cache_policy"]:
                cache_valid = False
                reasons.append("Windows unbuffered policy not active")
        elif physical is None or physical < case["requested_bytes"] * 0.95 or "fallback" in case["cache_policy"]:
            cache_valid = False
            reasons.append(name + ": warm native reads did not verify cache bypass")
    target, half = cases["paced-target"], cases["paced-half"]
    if target["iops_limit"] != rate or half["iops_limit"] != max(1, rate // 2):
        raise ValueError("Probe did not apply the requested IOPS tiers")
    ceiling = all(c["achieved_iops"] <= c["iops_limit"] * 1.05 and c["actual_wait_ns"] > 0
                  for c in (target, half))
    capacity = all(c["achieved_iops"] >= c["iops_limit"] * 0.90 for c in (target, half))
    scale = target["achieved_iops"] / half["achieved_iops"]
    capacity = capacity and 1.7 <= scale <= 2.3
    if not ceiling:
        reasons.append("Real-time IOPS ceiling or actual wait not verified")
    if not capacity:
        reasons.append("Native path cannot sustain this IOPS tier within 10%")
    return {
        "requested_iops": rate, "cache_verified": cache_valid,
        "ceiling_verified": ceiling, "tier_attainable": capacity,
        "eligible": cache_valid and ceiling and capacity,
        "observed_iops": target["achieved_iops"], "half_rate_iops": half["achieved_iops"],
        "cache_evidence": "Windows unbuffered API; physical-device I/O not verified" if windows else
                          "OS storage-read accounting versus buffered warm control",
        "reasons": reasons,
    }


def validate_query(query, algorithm, fixture, rate, label, expected_bps=None, expected_depth=None):
    if not query["fixed_budget"] or query["budget"] != fixture["budgets"][algorithm]:
        raise ValueError("Frozen query budget changed")
    if abs(query["recall_at_10"] - fixture["recall_at_10"][algorithm]) > 0.0001:
        raise ValueError("Recall differs from canonical index")
    if algorithm == "hnsw":
        return
    pacing = query.get("native_pacing")
    if not pacing or pacing["kind"] != "real-time-native-file-io" or pacing["resident_backing_bytes"] != 0:
        raise ValueError("Real native reader not active, or index preloaded into RAM")
    if pacing["iops_limit"] != rate or pacing["reads"] <= 0 or pacing["bytes"] <= 0:
        raise ValueError("Native pacing configuration/read count mismatch")
    if ((expected_bps is not None and pacing["bytes_per_second_limit"] != expected_bps) or
            (expected_depth is not None and pacing["max_in_flight"] != expected_depth)):
        raise ValueError("Native bandwidth/depth configuration mismatch")
    if rate and pacing["achieved_iops"] > rate * 1.03:
        raise ValueError("Timed query exceeds configured IOPS ceiling")
    if rate and pacing["actual_wait_ns"] <= 0:
        raise ValueError("No actual pacing wait during timed queries")
    observed = query["io"].get("host_timed_read_bytes", query["io"].get("timed_read_bytes"))
    if not label.startswith("windows-") and (observed is None or observed < pacing["bytes"] * 0.90):
        raise ValueError("Timed native queries did not sustain verified storage reads")


def aggregate(trials):
    keys = ("steady_rss_bytes", "baseline_rss_bytes", "index_rss_delta_bytes",
            "sampled_peak_query_rss_bytes", "recall_at_10", "p50_ms", "p95_ms", "p99_ms", "qps")
    result = {key: statistics.median(t[key] for t in trials) for key in keys}
    if "native_pacing" in trials[0]:
        result["achieved_iops"] = statistics.median(t["native_pacing"]["achieved_iops"] for t in trials)
        result["cache_policy"] = trials[0]["native_pacing"]["cache_policy"]
        result["reads_per_query"] = statistics.median(
            t["native_pacing"]["reads"] / (t["queries"] * t["repetitions"]) for t in trials)
    return result


def prepare_android(args):
    if args.transport != "android":
        return
    if (os.environ.get("GITHUB_ACTIONS") != "true" or
            os.environ.get("RUNNER_ENVIRONMENT") != "github-hosted" or
            native.optional_info(["adb", "shell", "getprop", "ro.kernel.qemu"]) != "1"):
        raise ValueError("The Android unencrypted control is restricted to the ephemeral CI emulator")
    native.run(["adb", "root"])
    native.run(["adb", "wait-for-device"])
    native.run(["adb", "shell", "test", "-d", "/data/unencrypted"])
    args.android_data_root = "/data/unencrypted/zvec-paced-io"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--transport", choices=["native", "android", "ios"], default="native")
    parser.add_argument("--device", default="booted")
    parser.add_argument("--profiles", default="2000,10000,20000,0")
    parser.add_argument("--common-iops", type=int, default=2000)
    parser.add_argument("--queue-depth", type=int, default=8)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--target-recall", type=float, default=0.95)
    args = parser.parse_args()
    rates = list(dict.fromkeys(int(value) for value in args.profiles.split(",")))
    if (not 1 <= args.queue_depth <= 128 or args.trials < 1 or args.repetitions < 1 or
            args.common_iops not in rates or 0 not in rates or any(rate < 0 for rate in rates)):
        parser.error("Need common tier + unlimited reference, valid rates/depth/trials")
    args.data, args.output = args.data.resolve(), args.output.resolve()
    if args.data == args.output or args.data in args.output.parents or args.output in args.data.parents:
        parser.error("Data and output must be separate non-nested directories")
    args.output.mkdir(parents=True, exist_ok=True)
    args.monitor_ios_io = args.transport == "ios"
    fixture = fixture_tools.verify_fixture(args.data)
    if "synthetic" in fixture["dataset"]["dataset"].lower():
        raise ValueError("Synthetic smoke data cannot produce publication measurements")
    prepare_android(args)
    remote = native.setup_transport(args)
    fixture_tools.copy_indexes(args, remote, fixture)
    report = {
        "schema_version": 1, "kind": "paced-native-file-io", "label": args.label,
        "host": native.metadata(), "data_path": remote, "fixture": fixture,
        "common_iops": args.common_iops, "queue_depth": args.queue_depth,
        "probe_operations": 8192, "probe_warmup_operations": 8192,
        "trials": args.trials, "repetitions": args.repetitions,
        "requested_profiles": rates, "profiles": {}, "probes": {},
        "measurement": {
            "api": "serial C++ core index search; not Python/full Collection",
            "qps": "queries / actual complete timed-block wall time, including native I/O and pacing waits",
            "p95": "median of trial P95 values for real search-call wall latency",
            "memory": "steady query-process RSS; Windows working set; decimal MB; kernel file cache excluded",
            "cache": "held-out workload warmed once; HNSW resident; DiskANN native cache policy verified separately",
            "hardware": "GitHub runner CPU/RAM/storage differ; IOPS limits are verified ceilings, not a hardware match",
        },
    }
    def save():
        (args.output / "paced-results.json").write_text(json.dumps(report, indent=2) + "\n")
    for rate in (rate for rate in rates if rate):
        worker = copy.copy(args)
        worker.output = args.output / ("probe-" + str(rate))
        worker.output.mkdir()
        worker.gtest_filter = "DiskAnnPacedIOProbe.Measure"
        worker.query_env = {
            "ZVEC_BENCH_IOPS": str(rate), "ZVEC_BENCH_PACE_DEPTH": str(args.queue_depth),
            "ZVEC_BENCH_PACE_BPS": str(max(16777216, rate * 8192)),
        }
        probe = native.measure(worker, remote, "diskann", "probe")
        verification = evaluate_probe(probe, args.label, rate)
        report["probes"][str(rate)] = {"raw": probe, "verification": verification}
        save()
    for rate in rates:
        if rate and not report["probes"][str(rate)]["verification"]["eligible"]:
            report["profiles"][str(rate)] = {
                "eligible": False, "reasons": report["probes"][str(rate)]["verification"]["reasons"]}
            save()
            continue
        profile = {"eligible": True, "iops_limit": rate,
                   "bytes_per_second_limit": max(16777216, rate * 8192) if rate else 0,
                   "raw_trials": {"hnsw": [], "diskann": []}, "medians": {}}
        report["profiles"][str(rate)] = profile
        for trial in range(args.trials):
            worker = copy.copy(args)
            worker.output = args.output / (f"rate-{rate}-trial-{trial + 1}")
            worker.output.mkdir()
            for algorithm in ("hnsw", "diskann"):
                worker.query_env = {
                    "ZVEC_BENCH_PACED_IO": "1",
                    "ZVEC_BENCH_FIXED_BUDGET": str(fixture["budgets"][algorithm]),
                    "ZVEC_BENCH_IOPS": str(rate),
                    "ZVEC_BENCH_PACE_DEPTH": str(args.queue_depth),
                    "ZVEC_BENCH_PACE_BPS": str(profile["bytes_per_second_limit"]),
                }
                query = native.measure(worker, remote, algorithm, "query")
                validate_query(query, algorithm, fixture, rate, args.label,
                               profile["bytes_per_second_limit"], args.queue_depth)
                profile["raw_trials"][algorithm].append(query)
                save()
        profile["medians"] = {algorithm: aggregate(trials) for algorithm, trials in profile["raw_trials"].items()}
        fixture_tools.verify_fixture(args.data)
        save()
    common_valid = report["profiles"][str(args.common_iops)]["eligible"]
    report["publication_ready"] = common_valid
    save()
    if not common_valid:
        raise SystemExit("Common IOPS tier failed verification; raw results retained, publication row withheld")


if __name__ == "__main__":
    main()
