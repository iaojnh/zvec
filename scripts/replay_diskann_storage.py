"""Compare storage volumes with byte-identical trained indexes and fresh workers."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import benchmark_diskann_platform as benchmark


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    if args.source.resolve() == args.target.resolve():
        raise ValueError("Storage comparison needs two distinct data directories")
    args.output.mkdir(parents=True, exist_ok=True)
    shutil.copytree(args.source, args.target, dirs_exist_ok=True)
    manifest = {}
    for source in args.source.rglob("*"):
        if source.is_file():
            relative = source.relative_to(args.source)
            original = digest(source)
            if original != digest(args.target / relative):
                raise ValueError(f"Copied index/data mismatch: {relative}")
            manifest[str(relative)] = original
    baseline = json.loads(args.baseline.read_text())
    args.data = args.target
    args.transport = "native"
    args.repetitions = 3
    args.target_recall = 0.95
    report = {"platform": args.label,
              "source_path": str(args.source.resolve()),
              "target_path": str(args.target.resolve()),
              "same_trained_index_sha256": manifest,
              "rebuild": False, "baseline": baseline}
    for algorithm in ("hnsw", "diskann"):
        report[algorithm] = benchmark.measure(args, str(args.target.resolve()), algorithm, "query")
        original = baseline[algorithm]["query"]
        replay = report[algorithm]
        if abs(original["recall_at_10"] - replay["recall_at_10"]) > 0.0001:
            raise ValueError("Storage replay changed query recall")
        report[algorithm]["qps_speedup"] = replay["qps"] / original["qps"]
        report[algorithm]["p95_speedup"] = original["p95_ms"] / replay["p95_ms"]
        (args.output / "storage-comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    text = ["# Same-index storage comparison", "",
            "Indexes and queries were copied and SHA-256 verified; no rebuilding or query-parameter changes.", "",
            "| Index | Original volume P95 ms | Local NVMe P95 ms | Original QPS | Local NVMe QPS | Recall@10 |",
            "| --- | --- | --- | --- | --- | --- |"]
    for algorithm in ("hnsw", "diskann"):
        original, replay = baseline[algorithm]["query"], report[algorithm]
        text.append(f'| {algorithm} | {original["p95_ms"]:.3f} | {replay["p95_ms"]:.3f} | '
                    f'{original["qps"]:.1f} | {replay["qps"]:.1f} | {replay["recall_at_10"]:.4f} |')
    (args.output / "storage-comparison.md").write_text("\n".join(text) + "\n")
    print("\n".join(text))


if __name__ == "__main__":
    main()
