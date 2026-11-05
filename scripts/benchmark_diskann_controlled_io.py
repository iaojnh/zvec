"""Compare DiskANN under deterministic storage costs, separately from host CPU.

The model does not normalize CPU/RAM and never reports its I/O-only QPS as
native disk QPS. A canonical fixture carries byte-identical trained indexes
and fixed query budgets to every runner, including mobile simulators.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path

import benchmark_diskann_platform as native


def hashes(root):
    return {
        str(path.relative_to(root)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "controlled-fixture.json"
    }


def verify_fixture(root):
    fixture = json.loads((root / "controlled-fixture.json").read_text())
    if fixture["sha256"] != hashes(root):
        raise ValueError("Canonical dataset/index hash mismatch")
    for name in ("hnsw", "diskann"):
        if not any(path.startswith(name + ".index") for path in fixture["sha256"]):
            raise ValueError("Canonical trained index missing: " + name)
    return fixture


def copy_indexes(args, remote, fixture):
    # The native harness already transferred the three vector matrices.
    for name in fixture["sha256"]:
        if not name.startswith(("hnsw.index", "diskann.index")):
            continue
        source = args.data / name
        if args.transport == "android":
            native.run(["adb", "shell", "mkdir", "-p", str(Path(remote, name).parent)])
            native.run(["adb", "push", str(source), remote + "/" + name])
        elif args.transport == "ios":
            destination = Path(remote) / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
    if args.transport == "android":
        for name, expected in fixture["sha256"].items():
            if name not in ("base.fbin", "queries.fbin", "neighbors.ibin") and not name.startswith(
                ("hnsw.index", "diskann.index")
            ):
                continue
            actual = native.run(["adb", "shell", "sha256sum", remote + "/" + name],
                                capture_output=True).stdout.split()[0]
            if actual != expected:
                raise ValueError("Android transferred fixture hash mismatch: " + name)
    elif args.transport == "ios":
        for name, expected in fixture["sha256"].items():
            if name not in ("base.fbin", "queries.fbin", "neighbors.ibin") and not name.startswith(
                ("hnsw.index", "diskann.index")
            ):
                continue
            if hashlib.sha256((Path(remote) / name).read_bytes()).hexdigest() != expected:
                raise ValueError("iOS transferred fixture hash mismatch: " + name)


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Must be positive")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--prepare-fixture", action="store_true")
    parser.add_argument("--transport", choices=["native", "android", "ios"], default="native")
    parser.add_argument("--device", default="booted")
    parser.add_argument("--android-data-root")
    parser.add_argument("--latency-ns", type=positive, default=200000)
    parser.add_argument("--queue-depth", type=positive, default=8)
    parser.add_argument("--bytes-per-second", type=positive, default=1048576000)
    parser.add_argument("--repetitions", type=positive, default=3)
    parser.add_argument("--target-recall", type=float, default=0.95)
    args = parser.parse_args()
    if args.queue_depth > 128 or not 0 < args.target_recall <= 1:
        parser.error("Queue depth must be <=128 and recall in (0,1]")
    args.data = args.data.resolve()
    args.output = args.output.resolve()
    if args.data == args.output or args.data in args.output.parents or args.output in args.data.parents:
        parser.error("Data and output must be separate, non-nested directories")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.prepare_fixture:
        if args.transport != "native":
            parser.error("Prepare the canonical fixture on a native host")
        if (args.data / "controlled-fixture.json").exists():
            raise ValueError("Refusing to rebuild an existing canonical fixture")
        calibration = {}
        for algorithm in ("hnsw", "diskann"):
            native.measure(args, str(args.data), algorithm, "build")
            # Calibrate using the same memory-backed reader ordering as replay.
            args.query_env = {"ZVEC_BENCH_CONTROLLED_IO": "1"}
            calibration[algorithm] = native.measure(args, str(args.data), algorithm, "query")
        fixture = {
            "schema_version": 1, "sha256": hashes(args.data),
            "budgets": {key: value["budget"] for key, value in calibration.items()},
            "recall_at_10": {key: value["recall_at_10"] for key, value in calibration.items()},
            "dataset": json.loads((args.data / "dataset.json").read_text()),
        }
        (args.data / "controlled-fixture.json").write_text(json.dumps(fixture, indent=2) + "\n")
    fixture = verify_fixture(args.data)
    remote = native.setup_transport(args)
    copy_indexes(args, remote, fixture)
    report = {
        "schema_version": 1, "kind": "controlled-storage-model",
        "label": args.label, "host": native.metadata(),
        "fixture": fixture,
        "model": {"latency_ns": args.latency_ns, "queue_depth": args.queue_depth,
                  "bytes_per_second": args.bytes_per_second, "cache": "none"},
        "limits": ["CPU/RAM differ across hosts", "I/O-only costs exclude CPU and CPU/I/O overlap",
                   "Memory-backed query RSS includes the full index image; not product RSS",
                   "Host wall QPS is memory-backed execution, not native disk performance"],
        "measurements": {},
    }
    for algorithm in ("hnsw", "diskann"):
        args.query_env = {
            "ZVEC_BENCH_CONTROLLED_IO": "1",
            "ZVEC_BENCH_FIXED_BUDGET": str(fixture["budgets"][algorithm]),
            "ZVEC_BENCH_IO_LATENCY_NS": str(args.latency_ns),
            "ZVEC_BENCH_IO_QUEUE_DEPTH": str(args.queue_depth),
            "ZVEC_BENCH_IO_BYTES_PER_SECOND": str(args.bytes_per_second),
        }
        result = native.measure(args, remote, algorithm, "query")
        if result["budget"] != fixture["budgets"][algorithm] or not result["fixed_budget"]:
            raise ValueError("Canonical query budget was not applied: " + algorithm)
        if abs(result["recall_at_10"] - fixture["recall_at_10"][algorithm]) > 0.0001:
            raise ValueError("Recall changed from canonical fixture: " + algorithm)
        if algorithm == "diskann":
            model = result.get("controlled_io")
            if not model or model["kind"] != "discrete-event-no-cache" or not model["reads"]:
                raise ValueError("Controlled reader not active")
            for key in ("latency_ns", "queue_depth", "bytes_per_second"):
                if model[key] != report["model"][key]:
                    raise ValueError("I/O model setting mismatch: " + key)
            if not math.isfinite(model["io_only_qps"]) or model["io_only_qps"] <= 0:
                raise ValueError("Invalid model throughput")
        report["measurements"][algorithm] = result
    verify_fixture(args.data)
    query = report["measurements"]["diskann"]
    model = query["controlled_io"]
    count = query["queries"] * query["repetitions"]
    lines = [
        "# Controlled DiskANN storage model", "",
        "Synthetic I/O-only costs and actual memory-backed execution are separate. "
        "CPU/RAM are not normalized. These are not native SSD performance or product RSS results.", "",
        f"Fixture source: {fixture['dataset']['dataset']}. "
        "All runners reuse identical trained indexes and fixed query budgets.", "",
        "| Metric | Value |", "| --- | ---: |",
        f"| Model latency / depth / bandwidth | {args.latency_ns} ns / {args.queue_depth} / {args.bytes_per_second} B/s |",
        f"| DiskANN Recall@10 | {query['recall_at_10']:.4%} |",
        f"| Reads per query | {model['reads'] / count:.3f} |",
        f"| Read bytes per query | {model['bytes'] / count:.3f} |",
        f"| Batches per query | {model['batches'] / count:.3f} |",
        f"| Modeled I/O-only P95 | {model['io_only_p95_ms']:.6f} ms |",
        f"| Modeled I/O-only QPS | {model['io_only_qps']:.3f} |",
        f"| Host memory-backed wall P95 | {query['p95_ms']:.6f} ms |",
        f"| Host memory-backed wall QPS | {query['qps']:.3f} |", "",
    ]
    (args.output / "controlled-results.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "controlled-comparison.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
