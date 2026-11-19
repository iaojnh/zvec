#!/usr/bin/env python3
"""Summarize comparable Collection benchmark artifacts by platform."""

import argparse
import json
import os
from pathlib import Path


PLATFORMS = (
    "linux-x86_64", "linux-arm64", "windows-x86_64", "windows-arm64",
    "macos-x86_64", "macos-arm64", "android-x86_64-emulator", "ios-arm64-simulator",
)


def format_pair(hnsw, diskann, field, divisor=1, decimals=2):
    return " / ".join(f"{row[field] / divisor:.{decimals}f}" for row in (hnsw, diskann))


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text("\n".join(summarize(args.input)) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
