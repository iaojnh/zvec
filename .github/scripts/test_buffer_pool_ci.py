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
import time
from pathlib import Path

import pytest
from buffer_pool_backend_tests import BACKENDS, NATIVE_TARGETS, PYTHON_TESTS
from buffer_pool_metrics import POLICY, stability
from buffer_pool_native_bundle import bundle
from buffer_pool_report import build_memory_results, performance, rows, summary
from buffer_pool_rss_monitor import ProcessRssMonitor, RssTrace, validate_soft_rss
from buffer_pool_system_limit import snapshot


def write_rows(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in values))
    return path


def soft_rss(budget_mib=128):
    trace = RssTrace(budget_mib * 1024)
    trace.add(0, 64 * 1024)
    trace.add(1, 64 * 1024)
    return trace.result(0)


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
    return [
        *values,
        soft_rss(),
        {"phase": "million_mutations_recovered"},
        {"phase": "complete"},
    ]


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
        soft_rss(),
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
    (tmp_path / "buffer-pool-candidate").mkdir()
    (tmp_path / "buffer-pool-candidate" / "commit.txt").write_text("test-revision")
    config = POLICY["index_build"]
    for index in ("ivf", "diskann"):
        directory = tmp_path / f"buffer-pool-evidence-build-{index}"
        phases = [
            {"phase": "calibration", "controls": [{}, {}, {}]},
            {"phase": "start", **config, "max_buffer_mib": 1},
            {"phase": "insert_finished"},
            {"phase": "flush_finished"},
            {"phase": "build_started", "monotonic_seconds": 4},
            {
                "phase": "build_finished",
                "monotonic_seconds": 5,
                "indexed_rows": config["rows"],
            },
            {"phase": "closed_after_build"},
            {"phase": "reopened_verified", "indexed_rows": config["rows"]},
            {
                "phase": "complete",
                "index": index,
                **config,
                "input_vector_bytes": config["rows"] * config["dimension"] * 4,
            },
            soft_rss(config["budget_mib"]),
        ]
        write_rows(directory / "build.jsonl", phases)
        write_rows(
            directory / "build.samples.jsonl",
            [{"monotonic_seconds": t, "rss_kib": 64 * 1024} for t in (4.1, 4.5, 4.9)],
        )
        prefix = "Ivf" if index == "ivf" else "Diskann"
        (directory / "native.xml").write_text(
            "<testsuite>"
            + "".join(
                f'<testcase name="RejectsExhaustedBudgetAndRetries/{prefix}{precision}"/>'
                for precision in ("Fp16", "Fp32")
            )
            + "</testsuite>"
        )
        write_rows(
            directory / "cgroup.jsonl",
            [
                {
                    "phase": phase,
                    "returncode": 0,
                    "memory.max": config["system_limit_mib"] * 1024**2,
                    "memory.swap.max": 0,
                    "memory.swap.current": 0,
                    "memory.events": {"oom": "0", "oom_kill": "0"},
                }
                for phase in ("cgroup_start", "cgroup_complete")
            ],
        )
    for platform in ("linux-x64", "linux-arm64"):
        for backend, expected in BACKENDS.items():
            directory = tmp_path / f"buffer-pool-backends-{platform}-{backend}"
            directory.mkdir()
            (directory / "backend.json").write_text(
                json.dumps(
                    {
                        "platform": platform,
                        "expected_backend": backend,
                        "actual_backend": expected,
                        "passed": True,
                        "commit": "test-revision",
                        "native_targets": list(NATIVE_TARGETS),
                        "python_tests": list(PYTHON_TESTS),
                    }
                )
            )
            for name in ("native.xml", "python.xml"):
                (directory / name).write_text(
                    '<testsuite><testcase name="executed"/></testsuite>'
                )
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


@pytest.mark.parametrize("damage", ["phase", "samples", "denial", "oom", "size"])
def test_build_memory_evidence_rejects_missing_or_invalid_proof(
    complete_artifacts, damage
):
    directory = complete_artifacts / "buffer-pool-evidence-build-ivf"
    if damage == "phase":
        path = directory / "build.jsonl"
        values = [json.loads(line) for line in path.read_text().splitlines()]
        write_rows(path, [r for r in values if r["phase"] != "build_finished"])
    elif damage == "samples":
        (directory / "build.samples.jsonl").write_text("")
    elif damage == "denial":
        path = directory / "native.xml"
        path.write_text(path.read_text().replace("/>", "><skipped/></testcase>", 1))
    elif damage == "oom":
        path = directory / "cgroup.jsonl"
        values = [json.loads(line) for line in path.read_text().splitlines()]
        values[-1]["memory.events"]["oom_kill"] = "1"
        write_rows(path, values)
    else:
        path = directory / "build.jsonl"
        values = [json.loads(line) for line in path.read_text().splitlines()]
        next(r for r in values if r["phase"] == "complete")["input_vector_bytes"] = 1
        write_rows(path, values)
    with pytest.raises(AssertionError):
        build_memory_results(complete_artifacts)


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


@pytest.mark.skipif(sys.platform == "win32", reason="standalone Unix compiler smoke")
def test_compile_gate_builds_without_running_the_test(tmp_path):
    project = Path(__file__).resolve().parents[2]
    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.16)\n"
        "project(compile_gate_smoke LANGUAGES CXX)\n"
        f'set(PROJECT_ROOT_DIR "{project}")\n'
        'include("${PROJECT_ROOT_DIR}/cmake/bazel.cmake")\n'
        "cc_test(NAME compile_probe SRCS probe.cc)\n"
    )
    (tmp_path / "probe.cc").write_text(
        "#include <cstdio>\n"
        'int main() { auto *f = std::fopen("executed.txt", "w"); '
        "if (!f) return 1; return std::fclose(f); }\n"
    )
    build = tmp_path / "build"
    subprocess.run(
        ["cmake", "-S", str(tmp_path), "-B", str(build), "-G", "Unix Makefiles"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["cmake", "--build", str(build), "--target", "unittest.build"],
        check=True,
        capture_output=True,
    )
    assert not list(build.rglob("executed.txt"))
    subprocess.run(
        ["cmake", "--build", str(build), "--target", "unittest"],
        check=True,
        capture_output=True,
    )
    assert len(list(build.rglob("executed.txt"))) == 1


@pytest.mark.parametrize(
    "damage", ["missing", "target", "duration", "fraction", "gap", "recovery"]
)
def test_soft_rss_observation_fails_closed(tmp_path, damage):
    values = benchmark()
    trace = next(row for row in values if row["phase"] == "rss_soft_limit")
    if damage == "missing":
        values.remove(trace)
    else:
        key = {
            "target": "target_rss_kib",
            "duration": "longest_over_target_seconds",
            "fraction": "over_target_fraction",
            "gap": "sampling_max_gap_seconds",
            "recovery": "recovery_seconds",
        }[damage]
        trace[key] = -1 if damage == "target" else 100
    with pytest.raises(AssertionError):
        rows(write_rows(tmp_path / "rss.jsonl", values))


def test_soft_rss_checks_time_and_continuous_excess():
    trace = RssTrace(100)
    trace.add(0, 90)
    trace.add(99, 101)
    trace.add(100, 90)
    result = trace.result(0)
    assert result["over_target_fraction"] == pytest.approx(0.01)
    assert result["longest_over_target_seconds"] == 1
    # Sparse sampling must not masquerade as continuous observation.
    with pytest.raises(AssertionError, match="sampling stalled"):
        validate_soft_rss(result)


def test_rss_observer_is_an_independent_process(tmp_path):
    path = tmp_path / "observed.jsonl"
    with path.open("w") as evidence, ProcessRssMonitor(evidence, 4 * 1024**2):
        time.sleep(0.15)
    observed = json.loads(path.read_text())
    assert observed["observer"] == "independent_process"
    assert observed["samples"] >= 2
    validate_soft_rss(observed)


@pytest.mark.parametrize("damage", ["backend", "missing_path", "empty", "failure"])
def test_backend_shared_path_evidence_is_required(complete_artifacts, damage):
    directory = complete_artifacts / "buffer-pool-backends-linux-arm64-libaio"
    if damage == "backend":
        record = json.loads((directory / "backend.json").read_text())
        record["actual_backend"] = 0
        (directory / "backend.json").write_text(json.dumps(record))
    elif damage == "missing_path":
        record = json.loads((directory / "backend.json").read_text())
        record["native_targets"].remove("buffer_storage_write_test")
        (directory / "backend.json").write_text(json.dumps(record))
    else:
        content = (
            "<testsuite/>"
            if damage == "empty"
            else '<testsuite><testcase name="failed"><failure/></testcase></testsuite>'
        )
        (directory / "native.xml").write_text(content)
    needs = {stage: {"result": "success"} for stage in POLICY["required_stages"]}
    with pytest.raises(AssertionError):
        summary(complete_artifacts, needs)
