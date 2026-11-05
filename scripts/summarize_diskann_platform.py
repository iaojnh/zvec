"""Keep native publication rows separate from mobile simulator measurements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PLATFORMS = [
    ("linux-x86_64", "Linux", "x86_64"),
    ("linux-arm64", "Linux", "ARM64"),
    ("windows-x86_64", "Windows", "x86_64"),
    ("windows-arm64", "Windows", "ARM64"),
    ("macos-arm64", "macOS", "ARM64 (Apple Silicon)"),
    ("macos-x86_64", "macOS", "x86_64 (Intel Mac)"),
    ("android-device", "Android", "64-bit physical device"),
    ("ios-device", "iOS", "64-bit physical device"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    reports = [json.loads(path.read_text()) for path in sorted(args.input.rglob("result.json"))]
    complete = {r["platform"]: r for r in reports if "comparable_recall" in r}
    table = ["| Platform | Architecture | DiskANN query RSS (MB) | RSS saved vs HNSW (%) | Recall@10 | P95 (ms) | QPS (1 thread) |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for key, system, arch in PLATFORMS:
        report = complete.get(key)
        if (report and report["comparable_recall"]
                and report["environment"] == "native CI host"
                and report["dataset"]["dataset"] != "synthetic-smoke"):
            q = report["diskann"]["query"]
            table.append(f'| {system} | {arch} | {q["steady_rss_bytes"] / 1e6:.2f} | '
                         f'{report["rss_saving_percent"]:.2f} | {q["recall_at_10"]:.4f} | '
                         f'{q["p95_ms"]:.3f} | {q["qps"]:.1f} |')
        else:
            table.append(f"| {system} | {arch} | | | | | |")
    (args.output / "publication_table.md").write_text("\n".join(table) + "\n")
    text = ["# DiskANN platform benchmark", "", *table, "",
            "CI hosts are virtualized runner machines. These are C++ core-index measurements after query-workload warmup, not full Collection or Python timings. Workload warmup does not guarantee OS page-cache hits; I/O backends and storage/cache states differ between platforms. Do not use these rows to rank operating systems.",
            "Memory is total query-process RSS/Windows working set (decimal MB), including mapped resident pages. Build runs in a separate process. Index-only RSS deltas are included in JSON.",
            "Recall calibration uses 200 queries; the remaining 1,000 queries are held out and measured three times. Each algorithm chooses its smallest budget reaching 0.95 calibration Recall@10.",
            "Rows require held-out Recall@10 >= 0.93 and a pairwise recall gap <= 0.02. Ineligible measurements remain in the full comparison below.",
            "Android/iOS simulator results are excluded from physical-device publication rows.", "",
            "## All measured results (including simulators)", "",
            "| Environment | Index | Query RSS MB | Index RSS delta MB | Recall@10 | Budget | P95 ms | QPS | Comparable recall |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for report in reports:
        for algorithm in ("hnsw", "diskann"):
            q = report.get(algorithm, {}).get("query")
            if not q:
                continue
            text.append(f'| {report["platform"]} ({report["environment"]}) | {algorithm} | '
                        f'{q["steady_rss_bytes"] / 1e6:.2f} | {q["index_rss_delta_bytes"] / 1e6:.2f} | '
                        f'{q["recall_at_10"]:.4f} | {q["budget"]} | {q["p95_ms"]:.3f} | '
                        f'{q["qps"]:.1f} | {report.get("comparable_recall", False)} |')
    text.extend(["", "## Test conditions", ""])
    for report in reports:
        details = {key: report[key] for key in ("host", "dataset", "measurement")}
        text.extend([f'### {report["platform"]}', "", "```json", json.dumps(details, indent=2), "```", ""])
    (args.output / "comparison.md").write_text("\n".join(text))
    (args.output / "results.json").write_text(json.dumps(reports, indent=2) + "\n")
    cache_reports = [json.loads(path.read_text()) for path in sorted(args.input.rglob("cache-comparison.json"))]
    if cache_reports:
        import compare_diskann_query_cache
        compare_diskann_query_cache.summarize(cache_reports, args.output / "cache-comparison.md")
        (args.output / "cache-comparison.json").write_text(json.dumps(cache_reports, indent=2) + "\n")
    print("\n".join(table))


if __name__ == "__main__":
    main()
