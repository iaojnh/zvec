# Copyright 2025-present the zvec project
# SPDX-License-Identifier: Apache-2.0

"""Measure the installed candidate against the public Cohere 1M ground truth."""

# This is a standalone single-process benchmark with console evidence.
# ruff: noqa: T201, PT017, PT018, PLR0915
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from functools import partial
from pathlib import Path
from time import perf_counter

import numpy as np
import pyarrow.parquet as pq
from buffer_pool_acceptance import sample
from buffer_pool_metrics import POLICY, calibration

import zvec

ROWS = 1_000_000
DIMENSION = 768
TOPK = 10
QUERY_COUNT = 200
MIN_RECALL = POLICY["minimum_recall_at_10"]
TRIALS = POLICY["performance_trials"]
BUDGET_MIB = {"ivf": 2048, "diskann": 4096}


def io_bytes():
    counts = {}
    for line in Path("/proc/self/io").read_text().splitlines():
        key, value = line.split(":", 1)
        if key in ("read_bytes", "write_bytes"):
            counts[key] = int(value)
    return counts


def workload(data):
    test = pq.ParquetFile(data / "test.parquet")
    batch = next(test.iter_batches(batch_size=QUERY_COUNT, columns=["id", "emb"]))
    queries = dict(
        zip(batch.column(0).to_pylist(), batch.column(1).to_pylist(), strict=True)
    )
    truth = pq.read_table(data / "neighbors.parquet", columns=["id", "neighbors_id"])
    neighbors = {
        query_id: {str(value) for value in ids[:TOPK]}
        for query_id, ids in zip(
            truth["id"].to_pylist(), truth["neighbors_id"].to_pylist(), strict=True
        )
        if query_id in queries
    }
    assert len(queries) == len(neighbors) == QUERY_COUNT
    assert all(len(query) == DIMENSION for query in queries.values())
    assert all(len(ids) == TOPK for ids in neighbors.values())
    return queries, neighbors


def build(path, train, options, index_param, observe):
    schema = zvec.CollectionSchema(
        name="cohere_acceptance",
        vectors=[
            zvec.VectorSchema(
                "embedding",
                zvec.DataType.VECTOR_FP32,
                dimension=DIMENSION,
                index_param=index_param,
            )
        ],
    )
    parquet = pq.ParquetFile(train)
    assert parquet.metadata.num_rows == ROWS
    collection = zvec.create_and_open(path, schema, options)
    probes = {}
    try:
        start = perf_counter()
        io_start = io_bytes()
        inserted = 0
        for batch in parquet.iter_batches(
            batch_size=256, columns=["id", "emb"], use_threads=False
        ):
            docs = [
                zvec.Doc(id=str(row), vectors={"embedding": vector})
                for row, vector in zip(
                    batch.column(0).to_pylist(),
                    batch.column(1).to_pylist(),
                    strict=True,
                )
            ]
            if not probes:
                probes = {doc.id: doc.vector("embedding") for doc in docs[:3]}
            statuses = collection.insert(docs)
            assert len(statuses) == len(docs) and all(
                status.ok() for status in statuses
            )
            inserted += len(statuses)
            if inserted % 65_536 == 0:
                observe("inserting", rows=inserted)
        del docs, statuses, batch
        collection.flush()
        assert inserted == collection.stats.doc_count == ROWS
        observe("inserted", rows=inserted, seconds=perf_counter() - start)
        start = perf_counter()
        collection.optimize()
        assert collection.stats.index_completeness["embedding"] == 1
        observe(
            "optimized",
            seconds=perf_counter() - start,
            **{key: val - io_start[key] for key, val in io_bytes().items()},
        )
        probes = {
            key: doc.vector("embedding")
            for key, doc in collection.fetch(list(probes)).items()
        }
    finally:
        collection.close()
    observe("closed_after_build")
    return probes


def query_sequences(queries):
    ids = list(queries)
    rng = np.random.default_rng(2026)
    hot = list(rng.choice(ids[:40], 160)) + list(rng.choice(ids[40:], 40))
    rng.shuffle(hot)
    zipf_weights = 1 / np.arange(1, len(ids) + 1)
    return {
        "uniform": ids,
        "hot_80_20": hot,
        "zipf": list(rng.choice(ids, QUERY_COUNT, p=zipf_weights / zipf_weights.sum())),
        "repeat_90_10": [ids[0]] * 180 + ids[1:21],
    }


def cold_cache():
    # This is a dedicated ephemeral runner. Drop OS caches before opening the
    # collection, then require physical reads as evidence of a cold trial.
    subprocess.run(["sync"], check=True)
    subprocess.run(
        ["sudo", "tee", "/proc/sys/vm/drop_caches"],
        input="3\n",
        text=True,
        stdout=subprocess.DEVNULL,
        check=True,
    )


def mutations_and_recovery(
    path, mmap, probes, budget, limit, evidence, observe, query_param=None
):
    fixture = Path(str(path) + "-wal-fixture.json")
    fixture.write_text(json.dumps(probes))
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "_crash",
            str(path),
            str(int(mmap)),
            str(fixture),
            str(budget),
            str(limit),
            evidence.name,
        ],
        check=True,
        timeout=120,
    )
    try:
        reader = zvec.open(
            path, zvec.CollectionOption(read_only=True, enable_mmap=mmap)
        )
    except (ValueError, RuntimeError) as error:
        assert "requires WAL recovery" in str(error), str(error)
    else:
        reader.close()
        raise AssertionError("unclean WAL was accepted without recovery")
    writer = zvec.open(path, zvec.CollectionOption(enable_mmap=mmap))
    try:
        first = next(iter(probes))
        # COSINE may renormalize FP32 inputs when replaying the WAL.
        np.testing.assert_allclose(
            writer.fetch([first])[first].vector("embedding"),
            probes[first][::-1],
            rtol=1e-6,
            atol=1e-7,
        )
        assert all(status.ok() for status in writer.delete(list(probes)))
        assert not writer.fetch(list(probes))
        assert writer.stats.doc_count == ROWS - len(probes)
        assert all(
            status.ok()
            for status in writer.upsert(
                [
                    zvec.Doc(id=key, vectors={"embedding": vector})
                    for key, vector in probes.items()
                ]
            )
        )
        writer.flush()
        writer.optimize()
    finally:
        writer.close()
    reader = zvec.open(path, zvec.CollectionOption(read_only=True, enable_mmap=mmap))
    try:
        assert reader.stats.doc_count == ROWS
        docs = reader.fetch(list(probes))
        assert set(docs) == set(probes)
        for key, vector in probes.items():
            np.testing.assert_allclose(
                docs[key].vector("embedding"), vector, rtol=1e-6, atol=1e-7
            )
            hits = reader.query(
                zvec.Query("embedding", vector=vector, param=query_param),
                topk=1,
                output_fields=[],
            )
            assert hits[0].id == key
    finally:
        reader.close()
    fixture.unlink()
    observe(
        "million_mutations_recovered",
        operations=["update", "delete", "upsert", "flush", "optimize", "WAL", "reopen"],
    )


def run(kind, storage, data, path, evidence, role="candidate"):
    pq_cpu = __import__("pyarrow")
    pq_cpu.set_cpu_count(1)
    pq_cpu.set_io_thread_count(1)
    metric = zvec.MetricType.COSINE
    if kind == "ivf":
        index_param = zvec.IVFIndexParam(metric_type=metric, n_list=1024, n_iters=5)
        query_param = zvec.IVFQueryParam(nprobe=128)
    else:
        index_param = zvec.DiskAnnIndexParam(
            metric_type=metric,
            max_degree=64,
            list_size=100,
            pq_chunk_num=96,
        )
        query_param = zvec.DiskAnnQueryParam(list_size=400)

    budget_mib = BUDGET_MIB[kind]
    metadata = pq.ParquetFile(data / "shuffle_train.parquet").metadata
    decoder = 2 * max(
        metadata.row_group(i).total_byte_size for i in range(metadata.num_row_groups)
    )
    control = calibration(
        "buffer_pool_cohere",
        budget_mib,
        4,
        DIMENSION,
        256,
        decoder_bytes=decoder,
        fixture_bytes=QUERY_COUNT * DIMENSION * 32
        + 2 * (data / "neighbors.parquet").stat().st_size,
    )
    # mmap is a performance reference; its mapped-file RSS is not governed by
    # the Buffer Pool contract. Both roles use the same external cgroup cap.
    limit_kib = (
        control["rss_limit_kib"]
        if role == "candidate" and storage == "buffer"
        else None
    )
    observe = partial(sample, evidence, limit_kib=limit_kib)
    observe("calibration", **control, role=role)
    observe(
        "start",
        index=kind,
        storage=storage,
        rows=ROWS,
        dimension=DIMENSION,
        budget_mib=budget_mib,
        rss_limit_kib=limit_kib,
        min_recall=MIN_RECALL,
    )
    zvec.init(memory_limit_mb=budget_mib, query_threads=4, optimize_threads=1)
    queries, neighbors = workload(data)
    mmap = storage == "mmap"
    probes = build(
        path,
        data / "shuffle_train.parquet",
        zvec.CollectionOption(enable_mmap=mmap),
        index_param,
        observe,
    )

    options = zvec.CollectionOption(read_only=True, enable_mmap=mmap)
    for distribution, sequence in query_sequences(queries).items():
        for cache in ("cold", "hot"):
            for trial in range(1, TRIALS + 1):
                if cache == "cold":
                    cold_cache()
                open_io_start = io_bytes()
                open_start = perf_counter()
                collection = zvec.open(path, options)
                try:
                    assert collection.stats.doc_count == ROWS
                    observe("opened", trial=trial)
                    if cache == "hot":
                        for query_id in sequence:
                            collection.query(
                                zvec.Query(
                                    "embedding",
                                    vector=queries[query_id],
                                    param=query_param,
                                ),
                                topk=TOPK,
                                output_fields=[],
                            )
                    latencies = []
                    matched = 0
                    query_io_start = io_bytes()
                    io_start = open_io_start if cache == "cold" else query_io_start
                    trial_start = open_start if cache == "cold" else perf_counter()
                    for query_id in sequence:
                        vector = queries[query_id]
                        query = zvec.Query(
                            "embedding", vector=vector, param=query_param
                        )
                        start = perf_counter()
                        hits = collection.query(query, topk=TOPK, output_fields=[])
                        latencies.append(perf_counter() - start)
                        ids = {hit.id for hit in hits}
                        assert len(ids) == TOPK and all(
                            math.isfinite(hit.score) for hit in hits
                        )
                        matched += len(ids & neighbors[query_id])
                    elapsed = perf_counter() - trial_start
                    recall = matched / (QUERY_COUNT * TOPK)
                    metrics = {
                        "trial": trial,
                        "queries": QUERY_COUNT,
                        "recall_at_10": recall,
                        "distribution": distribution,
                        "cache": cache,
                        "role": role,
                        "qps": QUERY_COUNT / elapsed,
                        "p99_ms": float(np.percentile(latencies, 99)) * 1000,
                        "query_read_bytes": io_bytes()["read_bytes"]
                        - query_io_start["read_bytes"],
                        **{key: val - io_start[key] for key, val in io_bytes().items()},
                    }
                    observe("query_trial", **metrics)
                    print(metrics, flush=True)
                    if cache == "cold":
                        assert metrics["read_bytes"] > 0, (
                            "cold cache was not demonstrated"
                        )
                    assert recall >= MIN_RECALL, (
                        f"trial {trial}: recall@10 {recall:.3f} < {MIN_RECALL}"
                    )
                finally:
                    collection.close()
                observe("closed", trial=trial)
    if role == "candidate":
        mutations_and_recovery(
            path, mmap, probes, budget_mib, limit_kib, evidence, observe, query_param
        )
    observe("complete", role=role, index=kind, storage=storage)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "_crash":
        path, mmap, fixture, budget, limit, output = sys.argv[2:]
        zvec.init(memory_limit_mb=int(budget), query_threads=4, optimize_threads=1)
        probes = json.loads(Path(fixture).read_text())
        first = next(iter(probes))
        writer = zvec.open(path, zvec.CollectionOption(enable_mmap=bool(int(mmap))))
        assert writer.update(
            zvec.Doc(id=first, vectors={"embedding": probes[first][::-1]})
        ).ok()
        with Path(output).open("a") as log:
            sample(
                log, "unclean_writer", limit_kib=None if limit == "None" else int(limit)
            )
        os._exit(0)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", choices=tuple(BUDGET_MIB))
    parser.add_argument("storage", choices=("buffer", "mmap"))
    parser.add_argument("data", type=Path)
    parser.add_argument("collection")
    parser.add_argument("evidence", type=Path)
    parser.add_argument(
        "--role", choices=("candidate", "baseline"), default="candidate"
    )
    args = parser.parse_args()
    with args.evidence.open("x") as evidence:
        run(args.index, args.storage, args.data, args.collection, evidence, args.role)
