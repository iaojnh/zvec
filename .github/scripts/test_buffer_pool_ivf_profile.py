# Copyright 2025-present the zvec project
# SPDX-License-Identifier: Apache-2.0
"""The profile certificate must fail closed on missing or incomparable evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pytest
from buffer_pool_ivf_profile import evaluate, export, rss_window


def test_rss_is_time_weighted_with_clipped_window():
    samples = [
        {"monotonic_seconds": t, "rss_kib": rss * 1024}
        for t, rss in [(0, 100), (1, 300), (3, 500)]
    ]
    assert rss_window(samples, 0.5, 2.5, 2)["average_rss_mib"] == 250


@pytest.mark.parametrize(
    "damage", ["missing_start", "missing_end", "gap", "zero_rss", "duplicate_time"]
)
def test_incomplete_or_stalled_rss_observation_is_not_accepted(damage):
    samples = [{"monotonic_seconds": i, "rss_kib": 1024} for i in range(5)]
    if damage == "missing_start":
        samples = samples[2:]
    elif damage == "missing_end":
        samples = samples[:3]
    elif damage == "gap":
        samples.pop(2)
    elif damage == "zero_rss":
        samples[2]["rss_kib"] = 0
    else:
        samples[2]["monotonic_seconds"] = 1
    with pytest.raises(AssertionError):
        rss_window(samples, 0.5, 3.5, 1)


@pytest.fixture
def evidence(tmp_path):
    source = (
        Path(__file__).resolve().parents[2] / "examples/python/ivf_memory_profile.json"
    )
    profile = tmp_path / "profile.json"
    profile.write_bytes(source.read_bytes())
    config = json.loads(profile.read_text())
    digest = hashlib.sha256(profile.read_bytes()).hexdigest()
    root = tmp_path / "evidence"
    root.mkdir()
    for storage in ("buffer", "mmap"):
        for stage, trial in [
            *((name, 1) for name in ("build", "mixed", "verify")),
            *(("query", i) for i in range(1, 4)),
        ]:
            value = {
                "passed": True,
                "profile_sha256": digest,
                "profile": config,
                "storage": storage,
                "stage": stage,
                "trial": trial,
                "source_commit": "a" * 40,
                "wheel_sha256": "b" * 64,
                "environment": {
                    "system": "Linux",
                    "machine": "x86_64",
                    "python": "3.10.0",
                    "zvec": "test",
                },
                "rows": config["rows"] if stage == "build" else config["rows"] + 1536,
                "seconds": 60,
                "cycles": 1,
                "write_operations": 1536,
                "reader_queries": 12,
                "reader_queries_during_writes": 12,
                "fields_and_vectors_verified": True,
                "recall": 1.0,
            }
            if stage == "query":
                value["cases"] = [
                    {
                        "distribution": distribution,
                        "cache": "warm",
                        "trial": trial,
                        "start": 0.5,
                        "end": 3.5,
                        "queries": 30,
                        "qps": 10,
                        "p99_ms": 100,
                        "recall": 1.0,
                    }
                    for distribution in config["query"]["distributions"]
                ]
                samples = [
                    {
                        "monotonic_seconds": i / 2,
                        "rss_kib": (200 if storage == "buffer" else 400) * 1024,
                    }
                    for i in range(9)
                ]
                (root / f"{storage}-query-{trial}.monitor.samples.jsonl").write_text(
                    "\n".join(json.dumps(row) for row in samples)
                )
            (root / f"{storage}-{stage}-{trial}.json").write_text(json.dumps(value))
            (root / f"{storage}-{stage}-{trial}.monitor.jsonl").write_text(
                json.dumps(
                    {
                        "observer": "independent_process",
                        "samples": 9,
                        "sampling_max_gap_seconds": 0.5,
                    }
                )
            )
            group = {
                "memory.max": 768 * 1024**2,
                "memory.swap.max": 0,
                "memory.swap.current": 0,
                "memory.events": {"oom": "0", "oom_kill": "0"},
            }
            (root / f"{storage}-{stage}-{trial}.cgroup.jsonl").write_text(
                json.dumps({**group, "phase": "cgroup_start"})
                + "\n"
                + json.dumps({**group, "phase": "cgroup_complete", "returncode": 0})
            )
    return profile, root


def change(path, mutate):
    value = json.loads(path.read_text())
    mutate(value)
    path.write_text(json.dumps(value))


def test_complete_profile_uses_candidate_buffer_vs_candidate_mmap(evidence):
    result = evaluate(*evidence)
    assert result["passed"] is True
    assert all(case["qps_per_rss_ratio"] == 2 for case in result["comparisons"])


@pytest.mark.parametrize(
    "damage",
    [
        "missing_stage",
        "different_profile",
        "different_wheel",
        "different_commit",
        "oom",
        "swap",
        "failed_write",
        "no_concurrent_reads",
        "short_write_run",
        "missing_trial",
        "low_recall",
        "wrong_qps",
        "duplicate_distribution",
    ],
)
def test_profile_cannot_be_qualified_from_incomplete_or_mixed_results(evidence, damage):
    profile, root = evidence
    if damage == "missing_stage":
        (root / "buffer-mixed-1.json").unlink()
    elif damage == "missing_trial":
        (root / "mmap-query-3.json").unlink()
    elif damage in {"oom", "swap"}:
        path = root / "buffer-build-1.cgroup.jsonl"
        groups = [json.loads(line) for line in path.read_text().splitlines()]
        if damage == "oom":
            groups[-1]["memory.events"]["oom"] = "1"
        else:
            groups[-1]["memory.swap.current"] = 1
        path.write_text("\n".join(json.dumps(row) for row in groups))
    else:
        path = root / "buffer-mixed-1.json"
        if damage in {"low_recall", "wrong_qps", "duplicate_distribution"}:
            path = root / "buffer-query-1.json"

        def mutate(value):
            if damage == "different_profile":
                value["profile"]["max_buffer_size"] *= 2
            elif damage == "different_wheel":
                value["wheel_sha256"] = "c" * 64
            elif damage == "different_commit":
                value["source_commit"] = "d" * 40
            elif damage == "failed_write":
                value["passed"] = False
            elif damage == "no_concurrent_reads":
                value["reader_queries_during_writes"] = 0
            elif damage == "short_write_run":
                value["seconds"] = 1
            elif damage == "low_recall":
                value["cases"][0]["recall"] = 0.5
            elif damage == "wrong_qps":
                value["cases"][0]["qps"] = 1000
            else:
                value["cases"][1]["distribution"] = value["cases"][0]["distribution"]

        change(path, mutate)
    with pytest.raises((AssertionError, FileNotFoundError)):
        evaluate(profile, root)


def test_equal_qps_per_rss_does_not_meet_higher_than_mmap_target(evidence):
    profile, root = evidence
    for path in root.glob("buffer-query-*.monitor.samples.jsonl"):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            row["rss_kib"] = 400 * 1024
        path.write_text("\n".join(json.dumps(row) for row in rows))
    result = evaluate(profile, root)
    assert result["passed"] is False
    assert not any(
        case["checks"]["qps_per_rss_higher_than_mmap"] for case in result["comparisons"]
    )


def test_failed_export_removes_stale_qualified_profile(evidence, tmp_path):
    profile, root = evidence
    output = tmp_path / "export"
    args = argparse.Namespace(profile=profile, evidence=root, output=output)
    assert export(args) == 0
    assert (output / "ivf-profile-qualified.json").exists()
    (root / "buffer-verify-1.json").unlink()
    assert export(args) == 1
    assert not (output / "ivf-profile-qualified.json").exists()
    assert (
        json.loads((output / "ivf-profile-report.json").read_text())["passed"] is False
    )
