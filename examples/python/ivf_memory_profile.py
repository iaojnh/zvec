# Copyright 2025-present the zvec project
# SPDX-License-Identifier: Apache-2.0
"""Run an IVF profile using public APIs only; each stage uses a fresh process.

python ivf_memory_profile.py --profile ivf_memory_profile.json --root ./run \
    --storage buffer --stage build --result build.json
Then run mixed, verify and query with the same profile/root/storage arguments.
The CI wrapper adds independent RSS observation and an external memory limit.
"""

# This executable example prints its stage completion record.
# ruff: noqa: T201

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import zvec


def vectors(config, start, count):
    """Regenerate a bounded batch without retaining the corpus in Python."""
    rng = np.random.default_rng(config["seed"] + start // config["batch_size"])
    return rng.standard_normal((count, config["dimension"]), dtype=np.float32)


def add_truth(queries, batch, start, best_ids, best_distances, topk):
    distances = (
        np.sum(queries * queries, axis=1)[:, None]
        + np.sum(batch * batch, axis=1)[None, :]
        - 2 * queries @ batch.T
    )
    ids = np.broadcast_to(np.arange(start, start + len(batch)), distances.shape)
    distances = np.concatenate((best_distances, distances), axis=1)
    ids = np.concatenate((best_ids, ids), axis=1)
    selected = np.argsort(distances, axis=1, kind="stable")[:, :topk]
    return np.take_along_axis(ids, selected, axis=1), np.take_along_axis(
        distances, selected, axis=1
    )


def option(config, storage, read_only=False):
    return zvec.CollectionOption(
        read_only=read_only,
        enable_mmap=storage == "mmap",
        max_buffer_size=config["max_buffer_size"],
    )


def query(config, collection, vector):
    return collection.query(
        zvec.Query(
            "embedding",
            vector=vector,
            param=zvec.IVFQueryParam(nprobe=config["index"]["nprobe"]),
        ),
        topk=config["query"]["topk"],
        output_fields=[],
        include_vector=False,
    )


def check_hits(hits, expected, topk):
    ids = [hit.id for hit in hits]
    if len(ids) != topk or len(set(ids)) != topk:
        raise AssertionError("query returned incomplete/duplicate results")
    if not all(np.isfinite(hit.score) for hit in hits):
        raise AssertionError("query returned non-finite scores")
    return len(set(ids) & set(expected)) / topk


def check_statuses(statuses, count):
    if len(statuses) != count or not all(status.ok() for status in statuses):
        raise AssertionError(f"write failed: {statuses!r}")


def documents(config, start, batch, cycle=0):
    return [
        zvec.Doc(
            id=str(start + offset),
            vectors={"embedding": vector.tolist()},
            fields={"payload": f"{cycle}:" + "p" * config["payload_bytes"]},
        )
        for offset, vector in enumerate(batch)
    ]


def build(config, root, storage):
    settings = config["index"]
    schema = zvec.CollectionSchema(
        name="ivf_memory_profile",
        fields=[zvec.FieldSchema("payload", zvec.DataType.STRING)],
        vectors=[
            zvec.VectorSchema(
                "embedding",
                zvec.DataType.VECTOR_FP32,
                dimension=config["dimension"],
                index_param=zvec.IVFIndexParam(
                    metric_type=zvec.MetricType.L2,
                    n_list=settings["n_list"],
                    n_iters=settings["n_iters"],
                ),
            )
        ],
    )
    queries = vectors(config, 0, config["batch_size"])[: config["query"]["query_count"]]
    best_ids = np.empty((len(queries), 0), dtype=np.int64)
    best_distances = np.empty((len(queries), 0), dtype=np.float32)
    collection = zvec.create_and_open(
        str(root / "collection"), schema, option(config, storage)
    )
    inserted_at = time.monotonic()
    try:
        for start in range(0, config["rows"], config["batch_size"]):
            batch = vectors(
                config, start, min(config["batch_size"], config["rows"] - start)
            )
            best_ids, best_distances = add_truth(
                queries, batch, start, best_ids, best_distances, config["query"]["topk"]
            )
            docs = documents(config, start, batch)
            check_statuses(collection.insert(docs), len(docs))
        del docs, batch
        collection.flush()
        build_started = time.monotonic()
        collection.optimize()
        build_finished = time.monotonic()
        assert collection.stats.doc_count == config["rows"]
        assert collection.stats.index_completeness["embedding"] == 1
    finally:
        collection.close()
    fixture = {
        "queries": queries.tolist(),
        "truth": best_ids.tolist(),
        "distances": best_distances.tolist(),
        "rows": config["rows"],
    }
    (root / "fixture.json").write_text(json.dumps(fixture))
    return {
        "rows": config["rows"],
        "input_vector_bytes": config["rows"] * config["dimension"] * 4,
        "insert_and_flush_seconds": build_started - inserted_at,
        "build_seconds": build_finished - build_started,
    }


def extended_truth(config, fixture):
    queries = np.asarray(fixture["queries"], dtype=np.float32)
    best_ids = np.asarray(fixture["truth"], dtype=np.int64)
    best_distances = np.asarray(fixture["distances"], dtype=np.float32)
    # The first pass inserts new IDs; later passes update those same IDs. This
    # keeps data size bounded during sustained writes and identical across modes.
    for i in range(config["mixed"]["batches"]):
        start = config["rows"] + i * config["batch_size"]
        batch = vectors(config, start, config["batch_size"])
        best_ids, best_distances = add_truth(
            queries, batch, start, best_ids, best_distances, config["query"]["topk"]
        )
    return best_ids, best_distances


def mixed(config, root, storage):
    fixture = json.loads((root / "fixture.json").read_text())
    best_ids, best_distances = extended_truth(config, fixture)
    collection = zvec.open(str(root / "collection"), option(config, storage))
    stop = threading.Event()
    read_counts = [0] * config["mixed"]["readers"]

    def read_loop(reader):
        while not stop.is_set():
            hits = query(
                config,
                collection,
                fixture["queries"][read_counts[reader] % len(fixture["queries"])],
            )
            assert len(hits) == config["query"]["topk"]
            read_counts[reader] += 1

    started = time.monotonic()
    writes = cycles = 0
    try:
        with ThreadPoolExecutor(max_workers=len(read_counts)) as executor:
            readers = [executor.submit(read_loop, i) for i in range(len(read_counts))]
            reads_before_writes = sum(read_counts)
            try:
                while (
                    cycles == 0
                    or time.monotonic() - started < config["mixed"]["minimum_seconds"]
                ):
                    for i in range(config["mixed"]["batches"]):
                        for reader in readers:
                            if reader.done():
                                reader.result()
                                raise AssertionError("concurrent reader stopped early")
                        start = config["rows"] + i * config["batch_size"]
                        batch = vectors(config, start, config["batch_size"])
                        docs = documents(config, start, batch, cycles)
                        check_statuses(collection.upsert(docs), len(docs))
                        writes += len(docs)
                        collection.flush()
                    cycles += 1
                reads_during_writes = sum(read_counts) - reads_before_writes
                assert reads_during_writes > 0, (
                    "no queries completed during sustained writes"
                )
                collection.optimize()
            finally:
                stop.set()
            for reader in readers:
                reader.result()
        expected_rows = (
            config["rows"] + config["mixed"]["batches"] * config["batch_size"]
        )
        assert collection.stats.doc_count == expected_rows
        assert all(count > 0 for count in read_counts)
        new_id = str(config["rows"])
        assert (
            collection.fetch([new_id])[new_id]
            .field("payload")
            .startswith(f"{cycles - 1}:")
        )
    finally:
        stop.set()
        collection.close()
    fixture.update(
        truth=best_ids.tolist(), distances=best_distances.tolist(), rows=expected_rows
    )
    (root / "fixture.json").write_text(json.dumps(fixture))
    elapsed = time.monotonic() - started
    return {
        "rows": expected_rows,
        "write_operations": writes,
        "reader_queries": sum(read_counts),
        "reader_queries_during_writes": reads_during_writes,
        "cycles": cycles,
        "seconds": elapsed,
        "writes_per_second": writes / elapsed,
    }


def verify(config, root, storage):
    fixture = json.loads((root / "fixture.json").read_text())
    collection = zvec.open(str(root / "collection"), option(config, storage, True))
    try:
        assert collection.stats.doc_count == fixture["rows"]
        recalls = [
            check_hits(
                query(config, collection, vector),
                [str(i) for i in truth],
                config["query"]["topk"],
            )
            for vector, truth in zip(fixture["queries"], fixture["truth"], strict=True)
        ]
        assert min(recalls) >= config["acceptance"]["minimum_recall"]
        ids = ["0", str(config["rows"])]
        docs = collection.fetch(ids)
        assert set(docs) == set(ids)
        assert docs["0"].vector("embedding") == fixture["queries"][0]
    finally:
        collection.close()
    return {
        "rows": fixture["rows"],
        "recall": float(np.mean(recalls)),
        "fields_and_vectors_verified": True,
    }


def query_trials(config, root, storage, trial):
    fixture = json.loads((root / "fixture.json").read_text())
    collection = zvec.open(str(root / "collection"), option(config, storage, True))
    cases = []
    rng = np.random.default_rng(config["seed"] + trial)
    count = len(fixture["queries"])
    distributions = {
        "uniform": np.full(count, 1 / count),
        "hot_80_20": np.asarray([0.8, *([0.2 / (count - 1)] * (count - 1))]),
        "zipf": 1 / np.arange(1, count + 1, dtype=float),
        "repeat_90_10": np.asarray([0.9, *([0.1 / (count - 1)] * (count - 1))]),
    }
    try:
        for distribution in config["query"]["distributions"]:
            weights = distributions[distribution]
            sequence = rng.choice(count, size=1024, p=weights / weights.sum()).tolist()
            # Each case gets the same fixed warmup, outside its measured window.
            for i in range(config["query"]["warmup_queries"]):
                query(config, collection, fixture["queries"][sequence[i]])
            for i, vector in enumerate(fixture["queries"]):
                hits = query(config, collection, vector)
                assert (
                    check_hits(
                        hits,
                        [str(x) for x in fixture["truth"][i]],
                        config["query"]["topk"],
                    )
                    >= config["acceptance"]["minimum_recall"]
                )
            latencies = []
            recall_sum = 0.0
            started = time.monotonic()
            while (
                not latencies
                or time.monotonic() - started < config["query"]["minimum_seconds"]
            ):
                i = sequence[len(latencies) % len(sequence)]
                before = time.monotonic()
                hits = query(config, collection, fixture["queries"][i])
                latencies.append(time.monotonic() - before)
                recall_sum += check_hits(
                    hits, [str(x) for x in fixture["truth"][i]], config["query"]["topk"]
                )
            ended = time.monotonic()
            cases.append(
                {
                    "distribution": distribution,
                    "cache": "warm",
                    "trial": trial,
                    "start": started,
                    "end": ended,
                    "queries": len(latencies),
                    "qps": len(latencies) / (ended - started),
                    "p99_ms": float(np.quantile(latencies, 0.99)) * 1000,
                    "recall": recall_sum / len(latencies),
                }
            )
            time.sleep(0.15)  # Let the independent observer bracket the window.
    finally:
        collection.close()
    return {"cases": cases, "rows": fixture["rows"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--storage", choices=("buffer", "mmap"), default="buffer")
    parser.add_argument(
        "--stage", choices=("build", "mixed", "verify", "query"), required=True
    )
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--trial", type=int, default=1)
    args = parser.parse_args()
    config = json.loads(args.profile.read_text())
    assert config["schema_version"] == 1
    assert config["rows"] % config["batch_size"] == 0
    assert 2 <= config["query"]["query_count"] <= config["batch_size"]
    assert config["query"]["topk"] < config["rows"]
    args.root.mkdir(parents=True, exist_ok=True)
    result = {
        "stage": args.stage,
        "storage": args.storage,
        "profile_sha256": hashlib.sha256(args.profile.read_bytes()).hexdigest(),
        "profile": config,
        "trial": args.trial,
        "environment": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "zvec": zvec.__version__,
        },
        "passed": False,
    }
    try:
        zvec.init(
            memory_limit_mb=config["memory_limit_mb"]
            if args.storage == "buffer"
            else config["mmap_reference_memory_limit_mb"],
            query_threads=config["query_threads"],
            optimize_threads=config["optimize_threads"],
            log_level=zvec.LogLevel.INFO,
        )
        if args.stage == "query":
            details = query_trials(config, args.root, args.storage, args.trial)
        else:
            details = globals()[args.stage](config, args.root, args.storage)
        result.update(details, passed=True)
    except BaseException as error:
        result["error"] = f"{type(error).__name__}: {error}"
        result["traceback"] = traceback.format_exc()
        raise
    finally:
        with args.result.open("x") as output:
            json.dump(result, output, indent=2)
        print(
            json.dumps(
                {
                    k: v
                    for k, v in result.items()
                    if k in {"stage", "storage", "passed", "error", "rows"}
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
