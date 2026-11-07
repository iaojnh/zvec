# Copyright 2025-present the zvec project
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for acceptance gates, including their failure paths."""

from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys
import tarfile

import pytest
from buffer_pool_metrics import POLICY, stability
from buffer_pool_native_bundle import bundle
from buffer_pool_report import performance, rows, summary
from buffer_pool_system_limit import snapshot


def write_rows(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in values))
    return path


def benchmark(qps=100, p99=10, reads=1024**2):
    values = [{"phase": "calibration", "controls": [{}, {}, {}]}]
    for distribution in ("uniform", "hot_80_20", "zipf", "repeat_90_10"):
        for cache in ("cold", "hot"):
            for trial in (1, 2, 3):
                values.append(
                    {
                        "phase": "query_trial",
                        "distribution": distribution,
                        "cache": cache,
                        "trial": trial,
                        "qps": qps,
                        "p99_ms": p99,
                        "recall_at_10": 0.95,
                        "read_bytes": reads,
                    }
                )
    return [*values, {"phase": "million_mutations_recovered"}, {"phase": "complete"}]


@pytest.mark.parametrize(
    ("qps", "p99", "passed"),
    [(80, 12.5, True), (79.99, 10, False), (100, 12.51, False)],
)
def test_user_selected_thresholds(tmp_path, qps, p99, passed):
    a = write_rows(tmp_path / "old", benchmark())
    b = write_rows(tmp_path / "new", benchmark(qps=qps, p99=p99))
    assert performance(a, b)["passed"] is passed


@pytest.mark.parametrize("damage", ["missing", "duplicate", "nan", "recall", "cold"])
def test_invalid_performance_evidence_fails_closed(tmp_path, damage):
    values = benchmark()
    if damage == "missing":
        del values[1]
    elif damage == "duplicate":
        values.append(copy.deepcopy(values[1]))
    elif damage == "nan":
        values[1]["qps"] = float("nan")
    elif damage == "recall":
        values[1]["recall_at_10"] = 0.89
    else:
        values[1]["read_bytes"] = 0
    with pytest.raises(AssertionError):
        performance(
            write_rows(tmp_path / "old", benchmark()),
            write_rows(tmp_path / "new", values),
        )


def test_io_regression_fails_even_when_latency_passes(tmp_path):
    assert not performance(
        write_rows(tmp_path / "old", benchmark()),
        write_rows(tmp_path / "new", benchmark(reads=10 * 1024**2)),
    )["passed"]


@pytest.mark.parametrize(
    "values",
    [
        [],
        [{"phase": "complete"}],
        benchmark()[:-1],
        [*benchmark(), {"phase": "complete"}],
    ],
)
def test_incomplete_workload_is_not_a_pass(tmp_path, values):
    with pytest.raises(AssertionError):
        rows(write_rows(tmp_path / "rss", values))


def test_hour_duration_and_slow_leak():
    stable = [(i * 100, 100_000) for i in range(37)]
    assert stability(stable, 3600)["elapsed_seconds"] == 3600
    with pytest.raises(AssertionError, match="soak too short"):
        stability(stable[:-1], 3600)
    with pytest.raises(AssertionError, match="RSS slope"):
        stability([(t, rss + t * 10_000 / 3600) for t, rss in stable], 3600)


def test_cgroup_requires_numeric_applied_limits(tmp_path):
    files = {
        "memory.max": "1610612736",
        "memory.current": "100",
        "memory.peak": "200",
        "memory.swap.max": "0",
        "memory.swap.current": "0",
        "memory.events": "oom 0\noom_kill 0\n",
    }
    for name, value in files.items():
        (tmp_path / name).write_text(value)
    assert snapshot(tmp_path)["memory.max"] == 1610612736
    (tmp_path / "memory.max").write_text("max")
    with pytest.raises(ValueError, match="invalid literal"):
        snapshot(tmp_path)


@pytest.fixture
def complete_artifacts(tmp_path):
    complete = [
        {"phase": "calibration", "controls": [{}, {}, {}]},
        {"phase": "wal_recovered"},
        {"phase": "compaction_verified"},
        {"phase": "complete", "rounds": 36, "elapsed_seconds": POLICY["soak_seconds"]},
    ]
    for shard in ("budget", "storage", "ivf", "diskann"):
        path = tmp_path / f"buffer-pool-evidence-{shard}" / "native.xml"
        path.parent.mkdir()
        path.write_text(
            '<testsuite><testcase name="WritableOpenReclaimsOtherStoresCleanPages"/>'
            '<testcase name="WritableOpenWaitsForOtherStoresDirtyPages"/></testsuite>'
        )
    public = tmp_path / "buffer-pool-public-suite" / "public.xml"
    public.parent.mkdir()
    public.write_text(
        '<testsuite><testcase name="test_optimize_rebuilds_fts"/></testsuite>'
    )
    names = [
        "mixed-100k",
        "ivf-400k",
        "diskann-200k",
        "boundary-128-1",
        "boundary-512-4",
        "boundary-1024-8",
    ]
    names += [
        f"{kind}-{index}" for kind in ("soak", "cgroup") for index in ("ivf", "diskann")
    ]
    for name in names:
        directory = tmp_path / f"buffer-pool-evidence-{name}"
        write_rows(directory / "rss.jsonl", complete)
        (directory / "stress.log").write_text(
            "evict=12 Compacted vector index Compacted fts index"
        )
        write_rows(
            directory / "cgroup.jsonl",
            [
                {
                    "phase": phase,
                    "returncode": 0,
                    "memory.max": 1536 * 1024**2,
                    "memory.swap.max": 0,
                    "memory.swap.current": 0,
                    "memory.events": {"oom": "0"},
                }
                for phase in ("cgroup_start", "cgroup_complete")
            ],
        )
    for index in ("ivf", "diskann"):
        for storage in ("buffer", "mmap"):
            directory = tmp_path / f"buffer-pool-evidence-cohere-{index}-{storage}"
            for role in ("baseline", "candidate"):
                write_rows(directory / f"{role}.jsonl", benchmark())
    return tmp_path


def test_summary_requires_all_evidence(complete_artifacts):
    needs = {stage: {"result": "success"} for stage in POLICY["required_stages"]}
    assert summary(complete_artifacts, needs)["passed"]
    (
        complete_artifacts
        / "buffer-pool-evidence-cohere-diskann-mmap"
        / "candidate.jsonl"
    ).unlink()
    with pytest.raises(FileNotFoundError):
        summary(complete_artifacts, needs)


@pytest.mark.parametrize("result", ["failure", "skipped", "cancelled"])
def test_summary_never_accepts_an_unexecuted_stage(tmp_path, result):
    needs = {stage: {"result": "success"} for stage in POLICY["required_stages"]}
    needs["build"]["result"] = result
    with pytest.raises(AssertionError):
        summary(tmp_path, needs)


def test_missing_required_stage_is_not_implicitly_waived(tmp_path):
    with pytest.raises(AssertionError, match="omitted"):
        summary(tmp_path, {"build": {"result": "success"}})


def test_required_aio_skip_blocks_acceptance(complete_artifacts):
    native = complete_artifacts / "buffer-pool-evidence-budget" / "native.xml"
    native.write_text(
        '<testsuite><testcase name="required"><skipped message="libaio is unavailable"/></testcase></testsuite>'
    )
    needs = {stage: {"result": "success"} for stage in POLICY["required_stages"]}
    with pytest.raises(AssertionError, match="required Linux AIO"):
        summary(complete_artifacts, needs)


def test_optional_native_skip_is_reported_as_unsupported(complete_artifacts):
    native = complete_artifacts / "buffer-pool-evidence-budget" / "native.xml"
    native.write_text(
        '<testsuite><testcase name="optional"><skipped message="kernel rejects io_uring"/></testcase></testsuite>'
    )
    needs = {stage: {"result": "success"} for stage in POLICY["required_stages"]}
    result = summary(complete_artifacts, needs)
    assert result["unsupported"] == [
        {"suite": "budget", "test": "optional", "reason": "kernel rejects io_uring"}
    ]


@pytest.mark.skipif(sys.platform != "linux", reason="Linux native relocation contract")
def test_native_archive_includes_transitive_runtime_libraries(tmp_path):
    build = tmp_path / "build"
    (build / "bin").mkdir(parents=True)
    (build / "lib").mkdir()
    (tmp_path / "base.c").write_text("int base(void) { return 42; }\n")
    (tmp_path / "front.c").write_text(
        "int base(void); int answer(void) { return base(); }\n"
    )
    (tmp_path / "main.c").write_text(
        "int answer(void); int main(void) { return answer() == 42 ? 0 : 1; }\n"
    )
    subprocess.run(
        [
            "cc",
            "-shared",
            "-fPIC",
            str(tmp_path / "base.c"),
            "-o",
            str(build / "lib/libbase.so"),
        ],
        check=True,
    )
    subprocess.run(
        [
            "cc",
            "-shared",
            "-fPIC",
            str(tmp_path / "front.c"),
            "-L" + str(build / "lib"),
            "-lbase",
            "-Wl,-rpath,$ORIGIN",
            "-o",
            str(build / "lib/libfront.so"),
        ],
        check=True,
    )
    subprocess.run(
        [
            "cc",
            str(tmp_path / "main.c"),
            "-L" + str(build / "lib"),
            "-lfront",
            "-Wl,-rpath,$ORIGIN/../lib",
            "-o",
            str(build / "bin/native_smoke"),
        ],
        check=True,
    )
    archive = tmp_path / "native.tar.gz"
    bundle(build, archive, ["native_smoke"])
    build.rename(tmp_path / "original-build-unavailable")
    relocated = tmp_path / "relocated"
    with tarfile.open(archive) as packed:
        packed.extractall(relocated, filter="data")
    env = {**os.environ, "LD_LIBRARY_PATH": str(relocated / "lib")}
    subprocess.run([str(relocated / "bin/native_smoke")], env=env, check=True)
    assert set(json.loads((relocated / "native-libraries.json").read_text())) == {
        "libfront.so",
        "libbase.so",
    }
    bare = tmp_path / "bare/bin"
    bare.mkdir(parents=True)
    shutil.copy2(relocated / "bin/native_smoke", bare / "native_smoke")
    rejected = subprocess.run(
        [str(bare / "native_smoke")],
        env={**os.environ, "LD_LIBRARY_PATH": ""},
        capture_output=True,
        check=False,
    )
    assert rejected.returncode != 0
    assert b"libfront.so" in rejected.stderr
