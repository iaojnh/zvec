# Copyright 2025-present the zvec project
# SPDX-License-Identifier: Apache-2.0

"""Independent empty-process controls and bounded workload accounting."""

# Delayed imports keep report-only jobs free from native package dependencies.
# ruff: noqa: T201, PLC0415
from __future__ import annotations

import hashlib
import importlib
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

POLICY = json.loads(Path(__file__).with_name("buffer_pool_policy.json").read_text())


def control(module_name, budget, workers, path, prepare=False):
    importlib.import_module(module_name)  # same imports as the measured program
    from buffer_pool_acceptance import rss_kib

    import zvec

    imported = rss_kib()[0]
    zvec.init(
        memory_limit_mb=512 if prepare else budget,
        query_threads=workers,
        optimize_threads=1,
    )
    initialized = rss_kib()[0]
    if prepare:
        collection = zvec.create_and_open(
            str(path),
            zvec.CollectionSchema(
                name="empty_control",
                fields=[zvec.FieldSchema("value", zvec.DataType.INT64)],
            ),
            zvec.CollectionOption(enable_mmap=False),
        )
    else:
        collection = zvec.open(
            str(path), zvec.CollectionOption(read_only=True, enable_mmap=False)
        )
    opened = rss_kib()[0]
    collection.close()
    print(
        json.dumps(
            {
                "import_kib": imported,
                "initialized_kib": initialized,
                "empty_open_kib": opened,
            }
        ),
        flush=True,
    )


def calibration(
    module,
    budget,
    workers,
    dimension,
    batch_size,
    payload_bytes=0,
    decoder_bytes=0,
    fixture_bytes=0,
):
    # Measure before the workload. Initialization/open allocations are reported,
    # but excluded from the baseline because their managed part is in the budget.
    script = str(Path(__file__).resolve())
    env = dict(os.environ)
    env["PYTHONPATH"] = (
        str(Path(script).parent) + os.pathsep + env.get("PYTHONPATH", "")
    )
    with tempfile.TemporaryDirectory(prefix="zvec-empty-control-") as directory:
        path = str(Path(directory) / "collection")

        def invoke(prepare=False):
            command = [sys.executable, script, module, str(budget), str(workers), path]
            if prepare:
                command.append("prepare")
            result = subprocess.run(
                command,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                check=True,
                timeout=120,
            )
            return json.loads(result.stdout.splitlines()[-1])

        invoke(True)
        controls = [invoke() for _ in range(3)]
    imports = [row["import_kib"] for row in controls]
    # One live input batch plus its conversion copy; result copies are managed
    # and are deliberately not added again. Sizes use this interpreter's objects.
    vector_bytes = sys.getsizeof([0.0] * dimension) + dimension * sys.getsizeof(0.0)
    document_bytes = (
        vector_bytes + payload_bytes + 2 * sys.getsizeof({"embedding": None}) + 512
    )
    input_bytes = 2 * batch_size * document_bytes
    # Python request workers and native query workers, plus optimizer, eviction,
    # writeback and Arrow I/O; each stack
    # is bounded at 8 MiB by this CI contract (not a budget percentage).
    stack_bytes = (2 * workers + 4) * 8 * 1024**2
    outside_bytes = input_bytes + stack_bytes + decoder_bytes + fixture_bytes
    baseline = max(imports)
    noise = max(imports) - min(imports)
    import zvec

    extension = next(Path(zvec.__file__).parent.glob("_zvec*.so"), None)
    if extension is None:
        extension = next(Path(zvec.__file__).parent.glob("_zvec*.pyd"))
    return {
        "controls": controls,
        "baseline_kib": baseline,
        "measurement_error_kib": noise,
        "outside_bytes": {
            "input_batch": input_bytes,
            "bounded_thread_stacks": stack_bytes,
            "parquet_decoder": decoder_bytes,
            "query_fixture": fixture_bytes,
        },
        "rss_limit_kib": baseline
        + budget * 1024
        + (outside_bytes + 1023) // 1024
        + noise,
        "budget_mib": budget,
        "workers": workers,
        "os": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version,
        "page_bytes": os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096,
        "io_backend": zvec.io_backend_description(),
        "extension_sha256": hashlib.sha256(extension.read_bytes()).hexdigest(),
        "allocator": "package default; no allocator overrides or trim",
    }


def _steady_trend(samples):
    # Discard the first third only for the trend; the RSS ceiling includes it.
    steady = samples[len(samples) // 3 :]
    middle = len(steady) // 2
    growth = statistics.median(v for _, v in steady[middle:]) - statistics.median(
        v for _, v in steady[:middle]
    )
    xs, ys = zip(*steady, strict=True)
    x_mean, y_mean = statistics.mean(xs), statistics.mean(ys)
    denominator = sum((x - x_mean) ** 2 for x in xs)
    slope = (
        sum((x - x_mean) * (y - y_mean) for x, y in steady) / denominator * 3600
        if denominator
        else 0
    )
    return {"steady_growth_kib": growth, "slope_kib_per_hour": slope}


def stability(samples, minimum_seconds=0, workloads=None):
    assert len(samples) >= POLICY["minimum_lifecycle_rounds"], (
        "not enough lifecycle rounds"
    )
    elapsed = samples[-1][0] - samples[0][0]
    assert elapsed >= minimum_seconds, f"soak too short: {elapsed:.1f}s"
    trends = {"all": _steady_trend(samples)}
    mixed_trend = trends["all"]
    if workloads is not None:
        assert len(workloads) == len(samples), "missing lifecycle workload labels"
        groups = {}
        for sample, workload in zip(samples, workloads, strict=True):
            groups.setdefault(workload, []).append(sample)
        assert 1 <= len(groups) <= 2, "unexpected lifecycle workloads"
        counts = [len(group) for group in groups.values()]
        assert max(counts) - min(counts) <= 1, "unbalanced lifecycle workloads"
        assert min(counts) >= POLICY["minimum_lifecycle_rounds"] // len(groups), (
            "not enough rounds per lifecycle workload"
        )
        # Top-k 1 and 32 exercise different query workspaces. Compare each
        # workload with itself: medians across their two RSS levels can move
        # merely because the middle pair straddles different allocator states.
        # Apply the unchanged growth AND hourly slope gates to every workload.
        trends = {str(key): _steady_trend(group) for key, group in groups.items()}
    growth = max(trend["steady_growth_kib"] for trend in trends.values())
    slope = max(trend["slope_kib_per_hour"] for trend in trends.values())
    assert growth <= POLICY["maximum_steady_growth_kib"], (
        f"steady RSS grew {growth:.0f} KiB"
    )
    # An hourly slope is meaningful for the hour-long job, not a seconds-long smoke.
    if minimum_seconds >= POLICY["soak_seconds"]:
        assert slope <= POLICY["maximum_steady_slope_kib_per_hour"], (
            f"RSS slope {slope:.0f} KiB/hour"
        )
    return {
        "rounds": len(samples),
        "elapsed_seconds": elapsed,
        "steady_growth_kib": growth,
        "slope_kib_per_hour": slope,
        "workload_trends": trends,
        "mixed_workload_trend": mixed_trend,
    }


if __name__ == "__main__":
    control(
        sys.argv[1],
        int(sys.argv[2]),
        int(sys.argv[3]),
        Path(sys.argv[4]),
        len(sys.argv) > 5,
    )
