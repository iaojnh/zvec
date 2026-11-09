# Copyright 2025-present the zvec project
# SPDX-License-Identifier: Apache-2.0

"""Observe the workload from another process, independently of its Python GIL."""

# Command-line evidence and explicit acceptance assertions are intentional.
# ruff: noqa: PT018
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

from buffer_pool_metrics import POLICY


class RssTrace:
    def __init__(self, target_kib):
        assert target_kib > 0
        self.target = target_kib
        self.samples = 0
        self.histogram = Counter()
        self.duration = self.over_time = self.longest_over = self.max_gap = 0.0
        self.over_run = 0.0
        self.previous = None
        self.peak = 0

    def add(self, timestamp, rss_kib):
        assert math.isfinite(timestamp) and math.isfinite(rss_kib) and rss_kib > 0
        if self.previous is not None:
            previous_time, previous_rss = self.previous
            elapsed = timestamp - previous_time
            assert elapsed > 0, "RSS timestamps must increase"
            self.max_gap = max(self.max_gap, elapsed)
            self.duration += elapsed
            if previous_rss > self.target:
                self.over_time += elapsed
                self.over_run += elapsed
                self.longest_over = max(self.longest_over, self.over_run)
            else:
                self.over_run = 0.0
        self.previous = timestamp, rss_kib
        self.samples += 1
        self.peak = max(self.peak, rss_kib)
        # MiB bins bound observer memory independently of the number of samples.
        self.histogram[math.ceil(rss_kib / 1024) * 1024] += 1

    def percentile(self, fraction):
        needed = math.ceil(self.samples * fraction)
        count = 0
        for rss, frequency in sorted(self.histogram.items()):
            count += frequency
            if count >= needed:
                return rss
        raise AssertionError("empty RSS trace")

    def result(self, recovery_seconds):
        assert self.samples >= 2 and self.duration > 0, "incomplete RSS observation"
        return {
            "phase": "rss_soft_limit",
            "target_rss_kib": self.target,
            "samples": self.samples,
            "observed_seconds": self.duration,
            "over_target_fraction": self.over_time / self.duration,
            "longest_over_target_seconds": self.longest_over,
            "recovery_seconds": recovery_seconds,
            "sampling_max_gap_seconds": self.max_gap,
            "p95_rss_kib": self.percentile(0.95),
            "p99_rss_kib": self.percentile(0.99),
            "peak_rss_kib": self.peak,
            "observer": "independent_process",
        }


def validate_soft_rss(result):
    assert result["observer"] == "independent_process"
    for key in (
        "target_rss_kib",
        "samples",
        "observed_seconds",
        "over_target_fraction",
        "longest_over_target_seconds",
        "recovery_seconds",
        "sampling_max_gap_seconds",
        "p95_rss_kib",
        "p99_rss_kib",
        "peak_rss_kib",
    ):
        assert math.isfinite(result[key]) and result[key] >= 0, f"invalid RSS {key}"
    assert result["target_rss_kib"] > 0 and result["samples"] >= 2
    assert result["observed_seconds"] > 0
    assert (
        result["over_target_fraction"] <= POLICY["maximum_rss_over_target_fraction"]
    ), "RSS exceeds the configured target too often"
    assert (
        result["longest_over_target_seconds"]
        <= POLICY["maximum_rss_over_target_seconds"]
    ), "RSS stayed above the configured target too long"
    assert result["recovery_seconds"] <= POLICY["maximum_rss_recovery_seconds"], (
        "RSS did not recover after the workload released its resources"
    )
    assert (
        result["sampling_max_gap_seconds"] <= POLICY["maximum_rss_sampling_gap_seconds"]
    ), "RSS sampling stalled"


def read_rss(pid):
    if sys.platform == "linux":
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
        raise RuntimeError("workload has no readable RSS")
    import psutil  # noqa: PLC0415

    return psutil.Process(pid).memory_info().rss // 1024


def observe(args):
    trace = RssTrace(args.target_kib)
    recovery_start = None
    recovery = None
    resources = None
    state_path = os.environ.get("ZVEC_BUILD_RESOURCE_STATE")
    if state_path:
        from buffer_pool_build_resources import ResourceCollector

        resources = ResourceCollector(
            args.pid, state_path, args.samples.with_suffix(".resources.jsonl")
        )
    with args.samples.open("x") as output:
        while True:
            now = time.monotonic()
            rss = read_rss(args.pid)
            trace.add(now, rss)
            if resources:
                resources.sample(now, force=args.stop.exists())
            print(
                json.dumps({"monotonic_seconds": now, "rss_kib": rss}),
                file=output,
                flush=True,
            )
            if not args.ready.exists():
                args.ready.touch()
            if args.stop.exists():
                if recovery_start is None:
                    recovery_start = now
                recovery = now - recovery_start
                if rss <= args.target_kib and trace.samples >= 2:
                    break
                if recovery > POLICY["maximum_rss_recovery_seconds"]:
                    break
            time.sleep(POLICY["rss_sampling_interval_seconds"])
    args.output.write_text(json.dumps(trace.result(recovery)))
    if resources:
        resources.close()


class ProcessRssMonitor:
    def __init__(self, evidence, target_kib):
        self.evidence = evidence
        self.target = target_kib
        self.process = None
        self.directory = None

    def __enter__(self):
        self.directory = tempfile.TemporaryDirectory(prefix="zvec-rss-monitor-")
        folder = Path(self.directory.name)
        self.stop = folder / "stop"
        self.ready = folder / "ready"
        self.output = folder / "summary.json"
        samples = Path(self.evidence.name).with_suffix(".samples.jsonl").resolve()
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--pid",
            str(os.getpid()),
            "--target-kib",
            str(self.target),
            "--stop",
            str(self.stop),
            "--ready",
            str(self.ready),
            "--output",
            str(self.output),
            "--samples",
            str(samples),
        ]
        self.process = subprocess.Popen(
            command,
            stdout=None if os.environ.get("ZVEC_BUILD_RESOURCE_STATE") else subprocess.DEVNULL,
        )
        deadline = time.monotonic() + 15
        try:
            while not self.ready.exists():
                if self.process.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError("RSS observer did not become ready")
                time.sleep(0.02)
        except BaseException:
            self.process.terminate()
            self.process.wait(timeout=5)
            self.directory.cleanup()
            raise
        return self

    def __exit__(self, exception_type, _exception, _traceback):
        try:
            self.stop.touch()
            try:
                self.process.wait(timeout=POLICY["maximum_rss_recovery_seconds"] + 10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
                raise RuntimeError("RSS observer did not stop") from None
            assert self.process.returncode == 0 and self.output.is_file(), (
                "RSS observer failed"
            )
            result = json.loads(self.output.read_text())
            print(json.dumps(result), file=self.evidence, flush=True)
            if exception_type is None:
                validate_soft_rss(result)
        finally:
            self.directory.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--target-kib", type=int, required=True)
    for name in ("stop", "ready", "output", "samples"):
        parser.add_argument("--" + name, type=Path, required=True)
    observe(parser.parse_args())
