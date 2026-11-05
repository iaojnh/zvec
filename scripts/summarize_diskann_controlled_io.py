"""Summarize fixed-storage-cost replays without presenting them as SSD QPS."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

EXPECTED = {
    "linux-x86_64", "linux-arm64", "windows-x86_64", "windows-arm64",
    "macos-arm64", "macos-x86_64", "android-x86_64-emulator", "ios-arm64-simulator",
}


def summarize(root):
    reports = [json.loads(p.read_text()) for p in sorted(root.rglob("controlled-results.json"))]
    reports = [r for r in reports if r["label"] in EXPECTED]
    labels = [r["label"] for r in reports]
    if len(set(labels)) != len(labels):
        raise ValueError("Duplicate platform reports")
    if not reports:
        raise ValueError("No controlled model reports")
    canonical = reports[0]
    for report in reports:
        if report["kind"] != "controlled-storage-model":
            raise ValueError("Native SSD and model reports must not be mixed")
        if report["fixture"] != canonical["fixture"] or report["model"] != canonical["model"]:
            raise ValueError("Different canonical index/query fixture or storage costs")
    lines = [
        "# DiskANN under identical modeled storage costs", "",
        "Same trained index bytes, query budgets, and discrete-event I/O settings. "
        "CPU/RAM remain host-specific. Modeled I/O-only QPS is not measured SSD QPS; "
        "the full index image resides in RAM, so this mode does not measure product RSS.", "",
        "| Environment | Recall@10 | Reads/query | kB/query | Model I/O-only P95 ms | Model I/O-only QPS | Host memory-backed P95 ms | Host memory-backed QPS |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for report in sorted(reports, key=lambda r: r["label"]):
        query = report["measurements"]["diskann"]
        model = query["controlled_io"]
        count = query["queries"] * query["repetitions"]
        label = report["label"]
        if "emulator" in label or "simulator" in label:
            label += " **SIMULATOR**"
        lines.append(f"| {label} | {query['recall_at_10']:.4%} | {model['reads']/count:.3f} | "
                     f"{model['bytes']/count/1000:.3f} | {model['io_only_p95_ms']:.6f} | "
                     f"{model['io_only_qps']:.3f} | {query['p95_ms']:.6f} | {query['qps']:.3f} |")
    missing = sorted(EXPECTED - set(labels))
    if missing:
        lines += ["", "Missing/failed environments: " + ", ".join(missing)]
    return "\n".join(lines) + "\n", missing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    markdown, missing = summarize(args.results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown)
    print(markdown)
    if missing:
        raise SystemExit("Incomplete eight-environment run; partial summary retained")


if __name__ == "__main__":
    main()
