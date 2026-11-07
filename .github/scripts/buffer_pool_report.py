# Copyright 2025-present the zvec project
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed acceptance evidence validation and performance gates."""

# Standalone report assertions and printed JSON are the command-line contract.
# ruff: noqa: T201, PT018
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from buffer_pool_backend_tests import (
    BACKENDS,
    NATIVE_TARGETS,
    PYTHON_TESTS,
    verify_page_cycles,
)
from buffer_pool_metrics import POLICY
from buffer_pool_rss_monitor import validate_soft_rss


def rows(path):
    values = [json.loads(line) for line in path.read_text().splitlines()]
    assert sum(row["phase"] == "complete" for row in values) == 1, (
        f"incomplete workload: {path}"
    )
    assert any(
        row["phase"] == "calibration" and len(row["controls"]) == 3 for row in values
    ), f"missing calibration: {path}"
    reference = any(
        row.get("role") == "baseline" or row.get("storage") == "mmap" for row in values
    )
    if not reference:
        traces = [row for row in values if row["phase"] == "rss_soft_limit"]
        assert len(traces) == 1, f"missing independent RSS observation: {path}"
        validate_soft_rss(traces[0])
        starts = [row for row in values if row["phase"] == "start"]
        if starts:
            assert traces[0]["target_rss_kib"] == starts[0]["budget_mib"] * 1024
    return values


def performance(baseline, candidate):
    def cases(path):
        result = {}
        for row in rows(path):
            if row["phase"] != "query_trial":
                continue
            key = (row["distribution"], row["cache"])
            assert row["trial"] not in {r["trial"] for r in result.get(key, [])}, (
                "duplicate trial"
            )
            for metric in ("qps", "p99_ms", "recall_at_10", "read_bytes"):
                assert math.isfinite(row[metric]) and row[metric] >= 0, (
                    f"invalid {metric}"
                )
            assert row["qps"] > 0 and row["p99_ms"] > 0
            assert row["recall_at_10"] >= POLICY["minimum_recall_at_10"]
            if row["cache"] == "cold":
                assert row["read_bytes"] > 0, "cold cache has no physical-read evidence"
            result.setdefault(key, []).append(row)
        expected = {
            (d, c)
            for d in ("uniform", "hot_80_20", "zipf", "repeat_90_10")
            for c in ("cold", "hot")
        }
        assert set(result) == expected, "missing distribution/cache case"
        for trial_rows in result.values():
            assert {r["trial"] for r in trial_rows} == set(
                range(1, POLICY["performance_trials"] + 1)
            ), "missing trials"
        return result

    old, new = cases(baseline), cases(candidate)
    reports = []
    for key in sorted(old):
        medians = [
            {
                metric: statistics.median(r[metric] for r in data[key])
                for metric in ("qps", "p99_ms", "read_bytes")
            }
            for data in (old, new)
        ]
        a, b = medians
        checks = {
            "qps": b["qps"] >= a["qps"] * POLICY["minimum_qps_ratio"],
            "p99": b["p99_ms"] <= a["p99_ms"] * POLICY["maximum_p99_ratio"],
            "physical_reads": b["read_bytes"]
            <= a["read_bytes"] * POLICY["maximum_read_bytes_ratio"]
            + POLICY["read_bytes_noise_allowance"],
        }
        reports.append(
            {
                "distribution": key[0],
                "cache": key[1],
                "baseline": a,
                "candidate": b,
                "qps_ratio": b["qps"] / a["qps"],
                "p99_ratio": b["p99_ms"] / a["p99_ms"],
                "checks": checks,
                "passed": all(checks.values()),
            }
        )
    return {
        "baseline_commit": POLICY["baseline_commit"],
        "cases": reports,
        "passed": all(row["passed"] for row in reports),
    }


def build_memory_results(root):
    results = []
    config = POLICY["index_build"]
    for index in ("ivf", "diskann"):
        directory = root / f"buffer-pool-evidence-build-{index}"
        values = rows(directory / "build.jsonl")
        phases = {row["phase"]: row for row in values}
        for phase in (
            "start",
            "insert_finished",
            "flush_finished",
            "build_started",
            "build_finished",
            "closed_after_build",
            "reopened_verified",
            "complete",
        ):
            assert sum(row["phase"] == phase for row in values) == 1, (
                f"missing/duplicate index build phase: {index}/{phase}"
            )
        complete = phases["complete"]
        assert complete["index"] == index
        for key in ("rows", "dimension", "budget_mib", "build_workers"):
            assert complete[key] == config[key], f"wrong build control: {index}/{key}"
        vector_bytes = complete["rows"] * complete["dimension"] * 4
        assert complete["input_vector_bytes"] == vector_bytes
        assert vector_bytes > complete["budget_mib"] * 1024**2, (
            "build corpus must exceed memory budget"
        )
        assert phases["start"]["max_buffer_mib"] == 1
        assert phases["build_finished"]["indexed_rows"] == complete["rows"]
        assert phases["reopened_verified"]["indexed_rows"] == complete["rows"]
        trace = next(row for row in values if row["phase"] == "rss_soft_limit")
        assert trace["target_rss_kib"] == complete["budget_mib"] * 1024
        start = phases["build_started"]["monotonic_seconds"]
        end = phases["build_finished"]["monotonic_seconds"]
        assert end > start
        samples = [
            json.loads(line)
            for line in (directory / "build.samples.jsonl").read_text().splitlines()
        ]
        build_samples = [
            row for row in samples if start <= row["monotonic_seconds"] <= end
        ]
        assert len(build_samples) >= 2, "missing independently observed build RSS"
        tests = list(ET.parse(directory / "native.xml").getroot().iter("testcase"))
        prefix = "Ivf" if index == "ivf" else "Diskann"
        for precision in ("Fp16", "Fp32"):
            matching = [
                test
                for test in tests
                if test.get("name", "").endswith("/" + prefix + precision)
            ]
            assert len(matching) == 1 and all(
                matching[0].find(tag) is None for tag in ("skipped", "failure", "error")
            ), f"missing successful build denial/retry: {index}/{precision}"
        cgroup = [
            json.loads(line)
            for line in (directory / "cgroup.jsonl").read_text().splitlines()
        ]
        assert len(cgroup) == 2 and cgroup[0]["phase"] == "cgroup_start"
        assert (
            cgroup[-1]["phase"] == "cgroup_complete" and cgroup[-1]["returncode"] == 0
        )
        for row in cgroup:
            assert row["memory.max"] == config["system_limit_mib"] * 1024**2
            assert row["memory.swap.max"] == row["memory.swap.current"] == 0
        for event in ("oom", "oom_kill", "oom_group_kill"):
            assert int(cgroup[-1]["memory.events"].get(event, 0)) == int(
                cgroup[0]["memory.events"].get(event, 0)
            ), f"index build OOM: {index}/{event}"
        results.append(
            {
                "index": index,
                "budget_mib": complete["budget_mib"],
                "input_vector_bytes": vector_bytes,
                "build_workers": complete["build_workers"],
                "build_seconds": end - start,
                "build_peak_rss_kib": max(row["rss_kib"] for row in build_samples),
                "overall_peak_rss_kib": trace["peak_rss_kib"],
            }
        )
    return results


def native_results(root):
    verified = []
    unsupported = []
    for shard in ("budget", "storage", "ivf", "diskann"):
        directory = root / f"buffer-pool-evidence-{shard}"
        xml_files = list(directory.glob("*.xml"))
        assert xml_files, f"missing native/public results: {shard}"
        tests = []
        for path in xml_files:
            tests += list(ET.parse(path).getroot().iter("testcase"))
        assert tests and not any(
            t.find("failure") is not None or t.find("error") is not None for t in tests
        )
        for test in tests:
            skipped = test.find("skipped")
            if skipped is not None:
                reason = skipped.get("message", skipped.text or "")
                assert not re.search(
                    r"libaio|no asynchronous|AIO backend|Native AIO", reason, re.I
                ), f"required Linux AIO runtime unavailable: {reason}"
                unsupported.append(
                    {"suite": shard, "test": test.get("name"), "reason": reason}
                )
        if shard == "budget":
            verify_page_cycles(tests)
        if shard == "storage":
            for name in (
                "WritableOpenReclaimsOtherStoresCleanPages",
                "WritableOpenWaitsForOtherStoresDirtyPages",
            ):
                executed = [
                    t
                    for t in tests
                    if t.get("name") == name and t.find("skipped") is None
                ]
                assert len(executed) == 1, f"missing actual eviction witness: {name}"
        verified.append(f"native-{shard}")
    return verified, unsupported


def backend_results(root):
    verified = []
    candidate_commit = (
        (root / "buffer-pool-candidate" / "commit.txt").read_text().strip()
    )
    assert candidate_commit, "missing candidate revision"
    for platform in ("linux-x64", "linux-arm64"):
        for backend, expected in BACKENDS.items():
            directory = root / f"buffer-pool-backends-{platform}-{backend}"
            record = json.loads((directory / "backend.json").read_text())
            assert record["commit"] == candidate_commit, (
                "backend evidence is from another revision"
            )
            assert record["passed"] and record["actual_backend"] == expected
            assert (
                record["expected_backend"] == backend and record["platform"] == platform
            )
            assert set(record["native_targets"]) == set(NATIVE_TARGETS)
            assert set(record["python_tests"]) == set(PYTHON_TESTS)
            for name in ("native.xml", "python.xml"):
                cases = list(ET.parse(directory / name).getroot().iter("testcase"))
                assert cases and any(case.find("skipped") is None for case in cases)
                assert not any(
                    case.find("failure") is not None or case.find("error") is not None
                    for case in cases
                )
                if name == "native.xml":
                    verify_page_cycles(cases)
            verified.append(f"{platform}-{backend}-shared-paths")
    return verified


def summary(root, needs):
    assert set(needs) == set(POLICY["required_stages"]), (
        "a required CI stage was omitted"
    )
    assert needs and all(job["result"] == "success" for job in needs.values()), (
        "some CI stages failed, skipped or were cancelled"
    )
    verified, unsupported = native_results(root)
    builds = build_memory_results(root)
    verified.extend(f"build-memory-{row['index']}" for row in builds)
    verified.extend(backend_results(root))
    benchmarks = []
    suite = root / "buffer-pool-public-suite" / "public.xml"
    testcases = list(ET.parse(suite).getroot().iter("testcase"))
    assert testcases and not any(
        t.find("failure") is not None or t.find("error") is not None for t in testcases
    )
    assert any(
        t.get("name") == "test_optimize_rebuilds_fts" and t.find("skipped") is None
        for t in testcases
    ), "missing FTS compaction contract"
    verified.append("public-mutations-FTS-sparse-quantized-WAL")
    for name in (
        "mixed-100k",
        "ivf-400k",
        "diskann-200k",
        "boundary-128-1",
        "boundary-512-4",
        "boundary-1024-8",
    ):
        values = rows(root / f"buffer-pool-evidence-{name}" / "rss.jsonl")
        complete = next(r for r in values if r["phase"] == "complete")
        assert complete["rounds"] >= POLICY["minimum_lifecycle_rounds"]
        if name == "boundary-128-1":
            log = (root / f"buffer-pool-evidence-{name}" / "stress.log").read_text()
            assert any(int(n) > 0 for n in re.findall(r"\bevict=(\d+)", log)), (
                "working-set boundary did not trigger actual eviction"
            )
        if name == "mixed-100k":
            log = (root / f"buffer-pool-evidence-{name}" / "stress.log").read_text()
            assert "Compacted vector index" in log and "Compacted fts index" in log, (
                "missing physical compaction witness"
            )
            assert any(r["phase"] == "compaction_verified" for r in values)
        verified.append(name)
    for index in ("ivf", "diskann"):
        soak = rows(root / f"buffer-pool-evidence-soak-{index}" / "rss.jsonl")
        complete = next(r for r in soak if r["phase"] == "complete")
        assert complete["elapsed_seconds"] >= POLICY["soak_seconds"]
        assert any(r["phase"] == "wal_recovered" for r in soak)
        external = root / f"buffer-pool-evidence-cgroup-{index}"
        rows(external / "rss.jsonl")
        cgroup = [
            json.loads(line)
            for line in (external / "cgroup.jsonl").read_text().splitlines()
        ]
        assert (
            len(cgroup) == 2
            and cgroup[-1]["phase"] == "cgroup_complete"
            and cgroup[-1]["returncode"] == 0
        )
        assert cgroup[-1]["memory.swap.current"] == 0
        assert cgroup[0]["memory.max"] == cgroup[-1]["memory.max"] == 1536 * 1024**2
        assert cgroup[-1]["memory.swap.max"] == 0
        for event in ("oom", "oom_kill", "oom_group_kill"):
            assert int(cgroup[-1]["memory.events"].get(event, 0)) == int(
                cgroup[0]["memory.events"].get(event, 0)
            ), "OOM event"
        for storage in ("buffer", "mmap"):
            directory = root / f"buffer-pool-evidence-cohere-{index}-{storage}"
            result = performance(
                directory / "baseline.jsonl", directory / "candidate.jsonl"
            )
            assert result["passed"], f"performance regression: {index}/{storage}"
            benchmarks.append({"index": index, "storage": storage, **result})
            assert any(
                r["phase"] == "million_mutations_recovered"
                for r in rows(directory / "candidate.jsonl")
            )
            verified.append(f"cohere-{index}-{storage}")
        verified += [f"soak-{index}", f"cgroup-{index}"]
    return {
        "passed": True,
        "verified": verified,
        "unsupported": unsupported,
        "benchmarks": benchmarks,
        "builds": builds,
        "scope": POLICY["scope"],
        "policy": POLICY,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    compare = sub.add_parser("performance")
    compare.add_argument("baseline", type=Path)
    compare.add_argument("candidate", type=Path)
    compare.add_argument("output", type=Path)
    report = sub.add_parser("summary")
    report.add_argument("artifacts", type=Path)
    report.add_argument("needs", type=Path)
    report.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        result = (
            performance(args.baseline, args.candidate)
            if args.command == "performance"
            else summary(args.artifacts, json.loads(args.needs.read_text()))
        )
    except (AssertionError, OSError, ValueError, KeyError, ET.ParseError) as error:
        result = {"passed": False, "reason": str(error)}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["passed"] else 1)
