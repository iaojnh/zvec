# Copyright 2025-present the zvec project
# SPDX-License-Identifier: Apache-2.0
"""Observe the public IVF example and export a qualified profile only on success."""

# Standalone evidence assertions and JSON output are the command-line contract.
# ruff: noqa: T201, PT018

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import runpy
import statistics
import subprocess
import sys
from itertools import pairwise
from pathlib import Path

from buffer_pool_rss_monitor import ProcessRssMonitor

STAGES = ("build", "mixed", "verify")
STORAGES = ("buffer", "mmap")
REPOSITORY = Path(__file__).resolve().parents[2]


def observe(args):
    config = json.loads(args.profile.read_text())
    target_mb = (
        config["memory_limit_mb"]
        if args.storage == "buffer"
        else config[
            "mmap_reference_memory_limit_mb"
            if args.stage == "query"
            else "mmap_preparation_memory_limit_mb"
        ]
    )
    result_path = args.evidence / f"{args.storage}-{args.stage}-{args.trial}.json"
    monitor_path = result_path.with_suffix(".monitor.jsonl")
    argv = [
        str(REPOSITORY / "examples/python/ivf_memory_profile.py"),
        "--profile",
        str(args.profile.resolve()),
        "--root",
        str(args.root.resolve()),
        "--storage",
        args.storage,
        "--stage",
        args.stage,
        "--result",
        str(result_path.resolve()),
        "--trial",
        str(args.trial),
    ]
    args.evidence.mkdir(parents=True, exist_ok=True)
    original_argv = sys.argv
    try:
        with (
            monitor_path.open("x") as output,
            ProcessRssMonitor(output, target_mb * 1024, enforce_soft_limit=False),
        ):
            sys.argv = argv
            runpy.run_path(argv[0], run_name="__main__")
    finally:
        sys.argv = original_argv
        if result_path.is_file():
            result = json.loads(result_path.read_text())
            result["source_commit"] = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True
            ).strip()
            result["wheel_sha256"] = hashlib.sha256(args.wheel.read_bytes()).hexdigest()
            result_path.write_text(json.dumps(result, indent=2) + "\n")


def rss_window(samples, start, end, maximum_gap):
    """Time-weighted whole-process RSS; no baseline subtraction or peak divisor."""
    assert math.isfinite(start) and math.isfinite(end) and end > start
    times = [row["monotonic_seconds"] for row in samples]
    assert all(math.isfinite(value) for value in times)
    assert all(a < b for a, b in pairwise(times))
    first = bisect.bisect_right(times, start) - 1
    last = bisect.bisect_left(times, end)
    assert first >= 0 and last < len(samples), "RSS samples do not bracket query window"
    assert last > first, "missing independent RSS intervals"
    area = peak = 0.0
    for i in range(first, last):
        left, right = times[i], times[i + 1]
        assert right - left <= maximum_gap, "RSS sampling stalled during queries"
        rss = samples[i]["rss_kib"]
        assert math.isfinite(rss) and rss > 0
        elapsed = min(right, end) - max(left, start)
        area += rss * max(0.0, elapsed)
        peak = max(peak, rss)
    return {"average_rss_mib": area / (end - start) / 1024, "peak_rss_mib": peak / 1024}


def read_case(root, config, digest, storage, stage, trial):
    path = root / f"{storage}-{stage}-{trial}.json"
    value = json.loads(path.read_text())
    assert value["passed"] is True, (
        f"{storage} {stage} failed: {value.get('error', path)}"
    )
    assert value["profile_sha256"] == digest and value["profile"] == config
    assert (value["storage"], value["stage"], value["trial"]) == (storage, stage, trial)
    assert value["environment"]["system"] == "Linux"
    assert value["environment"]["machine"] == "x86_64"
    monitor = [
        json.loads(line)
        for line in path.with_suffix(".monitor.jsonl").read_text().splitlines()
    ]
    assert len(monitor) == 1 and monitor[0]["observer"] == "independent_process"
    assert monitor[0]["samples"] >= 2
    target_mb = (
        config["memory_limit_mb"]
        if storage == "buffer"
        else config[
            "mmap_reference_memory_limit_mb"
            if stage == "query"
            else "mmap_preparation_memory_limit_mb"
        ]
    )
    assert monitor[0]["target_rss_kib"] == target_mb * 1024
    assert (
        monitor[0]["sampling_max_gap_seconds"]
        <= config["acceptance"]["maximum_sampling_gap_seconds"]
    )
    group = [
        json.loads(line)
        for line in path.with_suffix(".cgroup.jsonl").read_text().splitlines()
    ]
    assert len(group) == 2
    before, after = group
    assert before["phase"] == "cgroup_start" and after["phase"] == "cgroup_complete"
    limit_mib = config[
        "mmap_preparation_system_limit_mib"
        if storage == "mmap" and stage != "query"
        else "system_limit_mib"
    ]
    assert before["memory.max"] == after["memory.max"] == limit_mib * 1024**2
    assert (
        before["memory.swap.max"]
        == after["memory.swap.max"]
        == after["memory.swap.current"]
        == 0
    )
    assert after["returncode"] == 0
    for event in ("oom", "oom_kill", "oom_group_kill"):
        assert int(before["memory.events"].get(event, 0)) == int(
            after["memory.events"].get(event, 0)
        )
    value["rss_observation"] = monitor[0]
    return value


def evaluate(profile, evidence):
    config = json.loads(profile.read_text())
    digest = hashlib.sha256(profile.read_bytes()).hexdigest()
    assert (
        config["rows"] * config["dimension"] * 4 > config["memory_limit_mb"] * 1024**2
    ), "qualification requires a larger-than-budget corpus"
    expected_rows = config["rows"] + config["mixed"]["batches"] * config["batch_size"]
    values, queries = [], {storage: {} for storage in STORAGES}
    for storage in STORAGES:
        for stage in STAGES:
            value = read_case(evidence, config, digest, storage, stage, 1)
            values.append(value)
            assert value["rows"] == (
                config["rows"] if stage == "build" else expected_rows
            )
            if stage == "build":
                assert value["post_build_query_verified"] is True
            if stage == "mixed":
                assert value["seconds"] >= config["mixed"]["minimum_seconds"]
                assert (
                    value["write_phase_seconds"] >= config["mixed"]["minimum_seconds"]
                )
                assert (
                    value["cycles"] == config["mixed"]["cycles"]
                    and value["write_operations"]
                    == (expected_rows - config["rows"]) * config["mixed"]["cycles"]
                )
                assert value["reader_queries"] > 0
                assert value["reader_queries_during_writes"] > 0
            if stage == "verify":
                assert value["fields_and_vectors_verified"] is True
                assert value["recall"] >= config["acceptance"]["minimum_recall"]
    for storage in STORAGES:
        for trial in range(1, config["query"]["trials"] + 1):
            value = read_case(evidence, config, digest, storage, "query", trial)
            values.append(value)
            assert value["rows"] == expected_rows
            assert len(value["cases"]) == len(config["query"]["distributions"])
            samples_path = evidence / f"{storage}-query-{trial}.monitor.samples.jsonl"
            samples = [
                json.loads(line) for line in samples_path.read_text().splitlines()
            ]
            seen = set()
            for case in value["cases"]:
                distribution = case["distribution"]
                assert (
                    distribution in config["query"]["distributions"]
                    and distribution not in seen
                )
                seen.add(distribution)
                assert case["trial"] == trial and case["cache"] == "warm"
                assert case["end"] - case["start"] >= config["query"]["minimum_seconds"]
                assert case["queries"] >= config["query"]["minimum_queries"]
                for metric in ("qps", "p99_ms", "recall"):
                    assert math.isfinite(case[metric]) and case[metric] > 0
                assert math.isclose(
                    case["qps"],
                    case["queries"] / (case["end"] - case["start"]),
                    rel_tol=1e-9,
                )
                assert case["recall"] >= config["acceptance"]["minimum_recall"]
                case.update(
                    rss_window(
                        samples,
                        case["start"],
                        case["end"],
                        config["acceptance"]["maximum_sampling_gap_seconds"],
                    )
                )
                case["qps_per_rss_mib"] = case["qps"] / case["average_rss_mib"]
                queries[storage].setdefault(distribution, []).append(case)
    commits = {value["source_commit"] for value in values}
    wheels = {value["wheel_sha256"] for value in values}
    environments = {
        json.dumps(value["environment"], sort_keys=True) for value in values
    }
    assert len(commits) == len(wheels) == len(environments) == 1, (
        "mixed versions or environments"
    )
    assert len(next(iter(commits))) == 40 and len(next(iter(wheels))) == 64
    comparisons = []
    acceptance = config["acceptance"]
    for distribution in config["query"]["distributions"]:
        stats = {
            storage: {
                metric: statistics.median(
                    case[metric] for case in queries[storage][distribution]
                )
                for metric in (
                    "qps",
                    "p99_ms",
                    "recall",
                    "average_rss_mib",
                    "qps_per_rss_mib",
                )
            }
            for storage in STORAGES
        }
        buffer, mmap = stats["buffer"], stats["mmap"]
        ratio = buffer["qps_per_rss_mib"] / mmap["qps_per_rss_mib"]
        checks = {
            "qps_per_rss_higher_than_mmap": ratio
            > acceptance["minimum_qps_per_rss_ratio"],
            "qps_floor": buffer["qps"] >= mmap["qps"] * acceptance["minimum_qps_ratio"],
            "p99_guard": buffer["p99_ms"]
            <= mmap["p99_ms"] * acceptance["maximum_p99_ratio"],
            "matched_recall": abs(buffer["recall"] - mmap["recall"])
            <= acceptance["maximum_recall_difference"],
        }
        comparisons.append(
            {
                "distribution": distribution,
                **stats,
                "qps_per_rss_ratio": ratio,
                "checks": checks,
                "passed": all(checks.values()),
            }
        )
    return {
        "schema_version": 1,
        "passed": all(case["passed"] for case in comparisons),
        "profile": config,
        "profile_sha256": digest,
        "source_commit": commits.pop(),
        "wheel_sha256": wheels.pop(),
        "environment": json.loads(environments.pop()),
        "metric": "median of per-trial QPS / time-weighted query-window whole-process RSS MiB",
        "comparisons": comparisons,
        "lifecycle": [
            {k: value[k] for k in ("storage", "stage", "rows", "rss_observation")}
            for value in values
            if value["stage"] in STAGES
        ],
        "trials": queries,
    }


def markdown_report(result):
    lines = [
        "# IVF profile acceptance",
        "",
        "**PASS**" if result["passed"] else "**FAIL — no qualified profile generated**",
        "",
    ]
    if "error" in result:
        return "\n".join([*lines, result["error"], ""])
    config = result["profile"]
    lines += [
        f"Candidate: `{result['source_commit']}`",
        f"Profile: `{config['name']}`; external limit {config['system_limit_mib']} MiB, swap disabled.",
        f"mmap baseline preparation uses {config['mmap_preparation_system_limit_mib']} MiB; "
        "both query modes use the same external limit above.",
        "",
        "Ratio uses median per-trial QPS / time-weighted whole-process RSS MiB.",
        "QPS and RSS columns show individual trial medians. All declared cases must pass.",
        "",
        "| Distribution | Buffer QPS | Buffer RSS MiB | mmap QPS | mmap RSS MiB | QPS/RSS ratio | Result |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for case in result["comparisons"]:
        buffer, mmap = case["buffer"], case["mmap"]
        lines.append(
            f"| {case['distribution']} | {buffer['qps']:.3f} | {buffer['average_rss_mib']:.1f} "
            f"| {mmap['qps']:.3f} | {mmap['average_rss_mib']:.1f} "
            f"| {case['qps_per_rss_ratio']:.3f} | {'PASS' if case['passed'] else 'FAIL'} |"
        )
    lines += [
        "",
        "| Distribution | Buffer recall | mmap recall | Buffer p99 ms | mmap p99 ms |",
        "|---|---:|---:|---:|---:|",
    ]
    for case in result["comparisons"]:
        buffer, mmap = case["buffer"], case["mmap"]
        lines.append(
            f"| {case['distribution']} | {buffer['recall']:.4f} | {mmap['recall']:.4f} "
            f"| {buffer['p99_ms']:.3f} | {mmap['p99_ms']:.3f} |"
        )
    lines.append("")
    for case in result["comparisons"]:
        failed = [name for name, passed in case["checks"].items() if not passed]
        if failed:
            lines.append(
                f"Failed checks for {case['distribution']}: {', '.join(failed)}."
            )
    lines += ["", "A PASS qualifies only this recorded workload and environment.", ""]
    return "\n".join(lines)


def export(args):
    args.output.mkdir(parents=True, exist_ok=True)
    qualified = args.output / "ivf-profile-qualified.json"
    # Never leave a stale qualified configuration after a failed rerun.
    qualified.unlink(missing_ok=True)
    try:
        result = evaluate(args.profile, args.evidence)
    except (AssertionError, KeyError, ValueError, OSError) as error:
        result = {"passed": False, "error": f"{type(error).__name__}: {error}"}
    (args.output / "ivf-profile-report.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    (args.output / "ivf-profile-report.md").write_text(markdown_report(result))
    if result["passed"]:
        qualification = {
            key: result[key]
            for key in (
                "passed",
                "source_commit",
                "wheel_sha256",
                "profile_sha256",
                "environment",
                "metric",
                "comparisons",
            )
        }
        # Keep controls at the top level so the exported profile can be passed
        # directly to the public example's --profile argument.
        qualified.write_text(
            json.dumps({**result["profile"], "qualification": qualification}, indent=2)
            + "\n"
        )
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("observe")
    run.add_argument("--profile", type=Path, required=True)
    run.add_argument("--root", type=Path, required=True)
    run.add_argument("--evidence", type=Path, required=True)
    run.add_argument("--wheel", type=Path, required=True)
    run.add_argument("--storage", choices=STORAGES, required=True)
    run.add_argument("--stage", choices=(*STAGES, "query"), required=True)
    run.add_argument("--trial", type=int, default=1)
    report = sub.add_parser("export")
    report.add_argument("--profile", type=Path, required=True)
    report.add_argument("--evidence", type=Path, required=True)
    report.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "observe":
        observe(args)
        return 0
    return export(args)


if __name__ == "__main__":
    sys.exit(main())
