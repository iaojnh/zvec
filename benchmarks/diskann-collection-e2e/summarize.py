#!/usr/bin/env python3
"""Summarize comparable Collection benchmark artifacts by platform."""

import argparse
import json
import os
from pathlib import Path
from statistics import median


PLATFORMS = (
    "linux-x86_64", "linux-arm64", "windows-x86_64", "windows-arm64",
    "macos-x86_64", "macos-arm64", "android-x86_64-emulator", "ios-arm64-simulator",
)


def format_pair(hnsw, diskann, field, divisor=1, decimals=2):
    return " / ".join(f"{row[field] / divisor:.{decimals}f}" for row in (hnsw, diskann))


def observed_io_row(label, trials):
    windows = [trial.get("storage_observation") for trial in trials]
    if not windows or not all(window and trial.get("query_window_ns")
                              for trial, window in zip(trials, windows)):
        return None
    seconds = sum(window.get("counter_window_ns", trial["query_window_ns"])
                  for trial, window in zip(trials, windows)) / 1e9
    queries = sum(trial["queries_per_pass"] * trial["passes"] for trial in trials)

    def rate(key, divisor):
        values = [window.get(key) for window in windows]
        return f"{sum(values) / divisor:.1f}" if all(value is not None for value in values) else "—"

    columns = [
        label, windows[0]["scope"], f"{queries / seconds:.1f}",
        f"{median(trial['p95_ms'] for trial in trials):.3f}",
        rate("device_read_ios", seconds), rate("process_logical_read_ios", seconds),
        rate("process_storage_read_bytes", seconds * 4096),
        rate("process_storage_read_bytes", queries * 1024), windows[0]["cache_policy"],
    ]
    return "| " + " | ".join(columns) + " |"


def cold_storage_row(label, trials):
    measurements = [trial["cold_storage"] for trial in trials]
    queries = sum(item["verified_queries"] for item in measurements)
    seconds = sum(item["query_ns"] for item in measurements) / 1e9
    read_bytes = sum(item["storage_read_bytes"] for item in measurements)
    p95 = median(trial["p95_ms"] for trial in trials)
    recall = median(trial["recall_at_10"] for trial in trials)
    equivalents = (f"{read_bytes / 4096 / seconds:.1f}"
                   if all("min_storage_read_bytes" in item for item in measurements) else "—")
    return (f"| {label} | {queries:,} | {queries / seconds:.1f} | {p95:.3f} | {recall:.2%} | "
            f"{read_bytes / queries / 1024:.1f} | {equivalents} |")


def summarize(input_dir):
    yield "## Zvec Collection API benchmark"
    yield ""
    yield "SIFT-100K · 128 dimensions · Recall@10 · Three fresh query processes per index"
    yield ""
    yield (
        "Pairs read HNSW / DiskANN. RSS is the process resident set. "
        "The query increment is post-query RSS minus the pre-open baseline. MB = 1,000,000 bytes."
    )
    yield ""
    yield (
        "| Platform | SDK | Baseline RSS (MB) | Post-query RSS (MB) | Query increment (MB) | "
        "DiskANN saves (MB) | Recall@10 (%) | P95 (ms) | QPS |"
    )
    yield "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"

    fixtures = None
    io_rows = []
    cold_rows = []
    for label in PLATFORMS:
        path = input_dir / f"collection-e2e-result-{label}" / "summary.json"
        if not path.is_file():
            yield f"| {label} | — | No result | — | — | — | — | — | — |"
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        revision = os.environ.get("GITHUB_SHA")
        if report["label"] != label or (revision and report["revision"] != revision):
            raise ValueError(f"Report provenance mismatch: {path}")
        dataset = report["dataset"]
        if (dataset["dataset"], dataset["vectors"], dataset["dimension"]) != (
            "SIFT-128 first-N subset", 100_000, 128
        ):
            raise ValueError(f"Unexpected fixture: {path}")
        if fixtures is None:
            fixtures = dataset["sha256"]
        elif fixtures != dataset["sha256"]:
            raise ValueError(f"Dataset checksums differ across platforms: {path}")
        if not report["comparable_recall"]:
            yield f"| {label} | {report['sdk']} | Recall check failed | — | — | — | — | — | — |"
            continue

        if report.get("cold_storage_requested"):
            cold_rows.append(
                f"| {label} | Unverified | — | — | — | — | — |"
                if report.get("cold_storage_error") else cold_storage_row(label, report["cold_storage_trials"])
            )
        if row := observed_io_row(label, report.get("query_trials", {}).get("diskann", [])):
            io_rows.append(row)
        elif label not in ("linux-x86_64", "linux-arm64") and report.get("storage_observation_requested"):
            io_rows.append(f"| {label} | Incomplete | — | — | — | — | — | — | — |")
        hnsw, diskann = (report["query_medians"][name] for name in ("hnsw", "diskann"))
        columns = (
            label,
            report["sdk"],
            format_pair(hnsw, diskann, "baseline_rss_bytes", 1e6),
            format_pair(hnsw, diskann, "after_query_rss_bytes", 1e6),
            format_pair(hnsw, diskann, "query_rss_delta_bytes", 1e6),
            f"{report['diskann_query_rss_delta_saved_bytes'] / 1e6:.2f}",
            format_pair(hnsw, diskann, "recall_at_10", 0.01),
            format_pair(hnsw, diskann, "p95_ms", decimals=3),
            format_pair(hnsw, diskann, "qps", decimals=1),
        )
        yield "| " + " | ".join(columns) + " |"

    for label in ("linux-x86_64", "linux-arm64"):
        path = input_dir / f"collection-e2e-result-{label}" / "io-sweep" / "io-sweep.md"
        if path.is_file():
            yield ""
            yield path.read_text(encoding="utf-8").rstrip()

    if io_rows:
        yield ""
        yield "## DiskANN query-window storage observations"
        yield ""
        yield ("Device read IOPS include activity from other processes. Process 4 KiB equivalents/s "
               "divide OS-accounted process storage bytes by 4 KiB and elapsed seconds. "
               "A dash means unavailable.")
        yield ""
        yield ("| Platform | Device scope | Paired QPS | Median P95 (ms) | Device read IOPS | "
               "Process logical reads/s | Process 4 KiB equivalents/s | Process storage KiB/query | Cache policy |")
        yield "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"
        yield from io_rows

    if cold_rows:
        yield ""
        yield "## DiskANN queries with verified index-page eviction"
        yield ""
        yield ("The persisted index files have zero resident pages immediately before each timed query. "
               "QPS and P95 time only Collection queries. The Android emulator verifies its guest page cache. "
               "Buffered reads can refill that cache within a query.")
        yield ""
        yield ("macOS and Android count process storage bytes across each query. "
               "The iOS simulator reports host-accounted process bytes across the complete observed phase.")
        yield ""
        yield ("| Platform | Verified queries | Query QPS | Median P95 (ms) | Recall@10 | "
               "Process storage KiB/query | Process 4 KiB equivalents/s |")
        yield "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"
        yield from cold_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text("\n".join(summarize(args.input)) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
