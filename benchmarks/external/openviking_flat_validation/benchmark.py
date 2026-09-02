#!/usr/bin/env python3
"""OpenViking native-local Flat versus zvec Flat benchmark."""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib.metadata
import json
import math
import os
import platform
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import psutil


HERE = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = HERE / "data"
DEFAULT_WORK_DIR = HERE / ".workspace" / "runs"
DEFAULT_RESULTS_DIR = HERE / "results"
ZVEC_REPO = HERE.parents[2]


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    np.maximum(norms, np.finfo(np.float32).tiny, out=norms)
    return np.asarray(values / norms, dtype=np.float32, order="C")


def _dataset_key(args: argparse.Namespace) -> str:
    return f"clustered_n{args.n}_d{args.dim}_q{args.queries}_seed{args.seed}"


def prepare_dataset(args: argparse.Namespace) -> Path:
    data_dir = Path(args.data_dir).resolve() / _dataset_key(args)
    data_dir.mkdir(parents=True, exist_ok=True)
    base_path = data_dir / "base.npy"
    query_path = data_dir / "queries.npy"
    truth_path = data_dir / f"truth_top{args.topk}.npy"
    meta_path = data_dir / "meta.json"

    expected_meta = {
        "n": args.n,
        "dim": args.dim,
        "queries": args.queries,
        "topk": args.topk,
        "seed": args.seed,
        "query_noise": args.query_noise,
    }
    if all(path.exists() for path in (base_path, query_path, truth_path, meta_path)):
        actual_meta = json.loads(meta_path.read_text())
        if actual_meta == expected_meta:
            return data_dir

    rng = np.random.default_rng(args.seed)
    base = _normalize_rows(rng.standard_normal((args.n, args.dim), dtype=np.float32))
    anchors = rng.choice(args.n, size=args.queries, replace=args.queries > args.n)
    noise = rng.standard_normal((args.queries, args.dim), dtype=np.float32)
    queries = _normalize_rows(base[anchors] + np.float32(args.query_noise) * noise)

    truth = np.empty((args.queries, args.topk), dtype=np.int64)
    truth_batch = max(1, min(args.truth_batch, args.queries))
    base_t = base.T
    for start in range(0, args.queries, truth_batch):
        stop = min(start + truth_batch, args.queries)
        scores = queries[start:stop] @ base_t
        candidates = np.argpartition(scores, -args.topk, axis=1)[:, -args.topk :]
        candidate_scores = np.take_along_axis(scores, candidates, axis=1)
        order = np.argsort(candidate_scores, axis=1)[:, ::-1]
        truth[start:stop] = np.take_along_axis(candidates, order, axis=1)

    np.save(base_path, base, allow_pickle=False)
    np.save(query_path, queries, allow_pickle=False)
    np.save(truth_path, truth, allow_pickle=False)
    meta_path.write_text(json.dumps(expected_meta, indent=2, sort_keys=True) + "\n")
    return data_dir


def _directory_bytes(path: Path) -> int:
    return sum(entry.stat().st_size for entry in path.rglob("*") if entry.is_file())


def _percentiles(latencies_ms: list[float]) -> dict[str, float]:
    values = np.asarray(latencies_ms, dtype=np.float64)
    return {
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "mean_ms": float(values.mean()),
    }


def _recall_at_k(actual: list[list[int]], truth: np.ndarray) -> float:
    recalls = []
    for got, expected in zip(actual, truth, strict=True):
        recalls.append(len(set(got) & set(map(int, expected))) / len(expected))
    return float(np.mean(recalls))


def _max_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ZVEC_REPO,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _query_loop(
    query_one: Callable[[np.ndarray], list[int]],
    queries: np.ndarray,
    warmup: int,
    repeats: int,
    concurrency: int,
) -> tuple[list[list[int]], list[float], float]:
    first_pass: list[list[int]] = []
    latencies_ms: list[float] = []

    def timed_query(query: np.ndarray) -> tuple[list[int], float]:
        start = time.perf_counter_ns()
        ids = query_one(query)
        return ids, (time.perf_counter_ns() - start) / 1_000_000.0

    workers = max(1, concurrency)
    if workers == 1:
        for query in queries[: min(warmup, len(queries))]:
            query_one(query)
        wall_start = time.perf_counter()
        for repeat in range(repeats):
            for query in queries:
                ids, latency_ms = timed_query(query)
                latencies_ms.append(latency_ms)
                if repeat == 0:
                    first_pass.append(ids)
        return first_pass, latencies_ms, time.perf_counter() - wall_start

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(query_one, queries[: min(warmup, len(queries))]))

        wall_start = time.perf_counter()
        for repeat in range(repeats):
            results = executor.map(timed_query, queries)
            for ids, latency_ms in results:
                latencies_ms.append(latency_ms)
                if repeat == 0:
                    first_pass.append(ids)
        wall_s = time.perf_counter() - wall_start

    return first_pass, latencies_ms, wall_s


def run_openviking(
    args: argparse.Namespace,
    base: np.ndarray,
    queries: np.ndarray,
    db_path: Path,
) -> dict[str, Any]:
    from openviking.storage.vectordb.collection.local_collection import (
        get_or_create_local_collection,
    )

    quant = "int8" if args.mode == "int8" else "float"
    schema = {
        "CollectionName": f"openviking_flat_{args.mode}",
        "Fields": [
            {"FieldName": "id", "FieldType": "int64", "IsPrimaryKey": True},
            {"FieldName": "vector", "FieldType": "vector", "Dim": args.dim},
        ],
    }
    config = {"ttl_cleanup_seconds": 3600, "index_maintenance_seconds": 3600}

    start = time.perf_counter()
    collection = get_or_create_local_collection(
        meta_data=schema, path=str(db_path), config=config
    )
    create_s = time.perf_counter() - start

    start = time.perf_counter()
    for offset in range(0, len(base), args.batch_size):
        batch = [
            {"id": int(i), "vector": base[i].tolist()}
            for i in range(offset, min(offset + args.batch_size, len(base)))
        ]
        result = collection.upsert_data(batch)
        if len(result.ids) != len(batch):
            raise RuntimeError(
                f"OpenViking inserted {len(result.ids)} of {len(batch)} records"
            )
    load_s = time.perf_counter() - start

    start = time.perf_counter()
    collection.create_index(
        "default",
        {
            "IndexName": "default",
            "VectorIndex": {
                "IndexType": "flat",
                "Distance": "ip",
                "Quant": quant,
            },
            "ScalarIndex": [],
        },
    )
    collection.close()
    finalize_s = time.perf_counter() - start

    start = time.perf_counter()
    collection = get_or_create_local_collection(path=str(db_path), config=config)
    reopen_s = time.perf_counter() - start
    rss_after_open = psutil.Process().memory_info().rss

    def query_one(query: np.ndarray) -> list[int]:
        result = collection.search_by_vector(
            "default", dense_vector=query.tolist(), limit=args.topk, output_fields=[]
        )
        return [int(item.id) for item in result.data]

    first_pass, latencies_ms, wall_s = _query_loop(
        query_one, queries, args.warmup, args.repeats, args.concurrency
    )
    rss_after_warmup = psutil.Process().memory_info().rss
    collection.close()
    return {
        "backend": "openviking",
        "layout": "contiguous",
        "package_version": _package_version("openviking"),
        "create_s": create_s,
        "load_s": load_s,
        "finalize_s": finalize_s,
        "reopen_s": reopen_s,
        "query_ids": first_pass,
        "latencies_ms": latencies_ms,
        "query_wall_s": wall_s,
        "rss_after_open_bytes": rss_after_open,
        "rss_after_warmup_bytes": rss_after_warmup,
    }


def run_zvec(
    args: argparse.Namespace,
    base: np.ndarray,
    queries: np.ndarray,
    db_path: Path,
) -> dict[str, Any]:
    import zvec

    quantize_type = (
        zvec.QuantizeType.INT8
        if args.mode == "int8"
        else zvec.QuantizeType.UNDEFINED
    )
    use_contiguous = args.layout == "contiguous"
    schema = zvec.CollectionSchema(
        name=f"zvec_flat_{args.mode}_{args.layout}",
        vectors=zvec.VectorSchema(
            "vector",
            zvec.DataType.VECTOR_FP32,
            args.dim,
            index_param=zvec.FlatIndexParam(
                metric_type=zvec.MetricType.IP,
                quantize_type=quantize_type,
                use_contiguous_memory=use_contiguous,
            ),
        ),
    )

    start = time.perf_counter()
    collection = zvec.create_and_open(path=str(db_path), schema=schema)
    create_s = time.perf_counter() - start

    start = time.perf_counter()
    for offset in range(0, len(base), args.batch_size):
        batch = [
            zvec.Doc(id=str(i), vectors={"vector": base[i].tolist()})
            for i in range(offset, min(offset + args.batch_size, len(base)))
        ]
        statuses = collection.insert(batch)
        failed = [status for status in statuses if not status.ok()]
        if failed:
            raise RuntimeError(f"zvec failed to insert {len(failed)} records")
    load_s = time.perf_counter() - start

    start = time.perf_counter()
    collection.optimize()
    collection.close()
    finalize_s = time.perf_counter() - start

    start = time.perf_counter()
    collection = zvec.open(str(db_path))
    reopen_s = time.perf_counter() - start
    rss_after_open = psutil.Process().memory_info().rss

    def query_one(query: np.ndarray) -> list[int]:
        result = collection.query(
            zvec.Query(field_name="vector", vector=query.tolist()),
            topk=args.topk,
            include_vector=False,
            output_fields=[],
        )
        return [int(item.id) for item in result]

    first_pass, latencies_ms, wall_s = _query_loop(
        query_one, queries, args.warmup, args.repeats, args.concurrency
    )
    rss_after_warmup = psutil.Process().memory_info().rss
    collection.close()
    return {
        "backend": "zvec",
        "layout": args.layout,
        "package_version": _package_version("zvec"),
        "create_s": create_s,
        "load_s": load_s,
        "finalize_s": finalize_s,
        "reopen_s": reopen_s,
        "query_ids": first_pass,
        "latencies_ms": latencies_ms,
        "query_wall_s": wall_s,
        "rss_after_open_bytes": rss_after_open,
        "rss_after_warmup_bytes": rss_after_warmup,
    }


def run_case(args: argparse.Namespace) -> dict[str, Any]:
    data_dir = prepare_dataset(args)
    base = np.load(data_dir / "base.npy", mmap_mode="r")
    queries = np.load(data_dir / "queries.npy", mmap_mode="r")
    truth = np.load(data_dir / f"truth_top{args.topk}.npy", mmap_mode="r")

    db_name = f"{args.backend}_{args.mode}_{args.layout}_{_dataset_key(args)}"
    db_path = Path(args.work_dir).resolve() / db_name
    if db_path.exists():
        shutil.rmtree(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    if args.backend == "openviking":
        raw = run_openviking(args, base, queries, db_path)
    else:
        raw = run_zvec(args, base, queries, db_path)

    query_ids = raw.pop("query_ids")
    latencies_ms = raw.pop("latencies_ms")
    total_queries = args.queries * args.repeats
    result = {
        **raw,
        "mode": args.mode,
        "n": args.n,
        "dim": args.dim,
        "queries": args.queries,
        "topk": args.topk,
        "repeats": args.repeats,
        "batch_size": args.batch_size,
        "concurrency": args.concurrency,
        "recall_at_k": _recall_at_k(query_ids, truth),
        "qps": total_queries / raw["query_wall_s"],
        **_percentiles(latencies_ms),
        "disk_bytes": _directory_bytes(db_path),
        "peak_rss_bytes": _max_rss_bytes(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "zvec_git_head": _git_head(),
    }

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    return result


def _mib(value: int | float) -> float:
    return float(value) / (1024.0 * 1024.0)


def write_summary(results: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# OpenViking local Flat vs zvec Flat results",
        "",
        "All input vectors and queries were L2-normalized FP32 values and both",
        "backends searched with inner product. Each backend ran in a separate process.",
        "",
        "| mode | backend | layout | concurrency | version | Recall@K | QPS | p50 ms | p95 ms | p99 ms | load s | finalize s | RSS MiB | disk MiB |",
        "|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            "| {mode} | {backend} | {layout} | {concurrency} | {package_version} | "
            "{recall_at_k:.4f} | {qps:.1f} | {p50_ms:.3f} | {p95_ms:.3f} | "
            "{p99_ms:.3f} | {load_s:.2f} | {finalize_s:.2f} | {rss:.1f} | {disk:.1f} |".format(
                **result,
                rss=_mib(result["rss_after_warmup_bytes"]),
                disk=_mib(result["disk_bytes"]),
            )
        )
    lines.extend(
        [
            "",
            f"zvec checkout: `{results[0]['zvec_git_head']}`",
            f"Python: `{results[0]['python']}`",
            f"Platform: `{results[0]['platform']}`",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def run_matrix(args: argparse.Namespace) -> None:
    data_dir = prepare_dataset(args)
    print(f"dataset={data_dir}")
    results_dir = Path(args.results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    layouts = [part.strip() for part in args.zvec_layouts.split(",") if part.strip()]
    invalid = set(layouts) - {"contiguous", "default"}
    if invalid:
        raise ValueError(f"invalid zvec layouts: {sorted(invalid)}")

    cases = []
    for mode in ("int8", "fp32"):
        cases.append(("openviking", mode, "contiguous"))
        cases.extend(("zvec", mode, layout) for layout in layouts)

    results = []
    for backend, mode, layout in cases:
        output = results_dir / f"{backend}_{mode}_{layout}.json"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "run",
            "--backend",
            backend,
            "--mode",
            mode,
            "--layout",
            layout,
            "--output",
            str(output),
        ]
        for name in (
            "n",
            "dim",
            "queries",
            "topk",
            "seed",
            "query_noise",
            "truth_batch",
            "batch_size",
            "warmup",
            "repeats",
            "concurrency",
            "data_dir",
            "work_dir",
        ):
            command.extend([f"--{name.replace('_', '-')}", str(getattr(args, name))])
        print(f"running backend={backend} mode={mode} layout={layout}", flush=True)
        completed = subprocess.run(command, text=True)
        if completed.returncode != 0:
            raise RuntimeError(
                f"case failed: backend={backend} mode={mode} layout={layout}"
            )
        results.append(json.loads(output.read_text()))

    combined_path = results_dir / "results.json"
    combined_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    summary_path = results_dir / "summary.md"
    write_summary(results, summary_path)
    print(summary_path.read_text())


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--n", type=int, default=20_000)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--queries", type=int, default=500)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--query-noise", type=float, default=0.03)
    parser.add_argument("--truth-batch", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--work-dir", default=str(DEFAULT_WORK_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    add_common_arguments(prepare)

    run = subparsers.add_parser("run")
    add_common_arguments(run)
    run.add_argument("--backend", choices=("openviking", "zvec"), required=True)
    run.add_argument("--mode", choices=("int8", "fp32"), required=True)
    run.add_argument("--layout", choices=("contiguous", "default"), default="contiguous")
    run.add_argument("--output", required=True)

    matrix = subparsers.add_parser("matrix")
    add_common_arguments(matrix)
    matrix.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    matrix.add_argument("--zvec-layouts", default="contiguous,default")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.n < args.topk:
        raise ValueError("n must be at least topk")
    if args.command == "prepare":
        print(prepare_dataset(args))
    elif args.command == "run":
        run_case(args)
    else:
        run_matrix(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
