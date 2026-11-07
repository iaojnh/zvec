# Copyright 2025-present the zvec project
# SPDX-License-Identifier: Apache-2.0

"""Measure the installed candidate against the public Cohere 1M ground truth."""

import argparse
import math
from functools import partial
from pathlib import Path
from time import perf_counter

import numpy as np
import pyarrow.parquet as pq
import zvec

from buffer_pool_acceptance import sample

ROWS = 1_000_000
DIMENSION = 768
TOPK = 10
QUERY_COUNT = 200
MIN_RECALL = 0.90
TRIALS = 3
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
    queries = dict(zip(batch.column(0).to_pylist(), batch.column(1).to_pylist()))
    truth = pq.read_table(data / "neighbors.parquet", columns=["id", "neighbors_id"])
    neighbors = {
        query_id: {str(value) for value in ids[:TOPK]}
        for query_id, ids in zip(truth["id"].to_pylist(), truth["neighbors_id"].to_pylist())
        if query_id in queries
    }
    assert len(queries) == len(neighbors) == QUERY_COUNT
    assert all(len(query) == DIMENSION for query in queries.values())
    assert all(len(ids) == TOPK for ids in neighbors.values())
    return queries, neighbors


def build(path, train, options, index_param, observe):
    schema = zvec.CollectionSchema(
        name="cohere_acceptance",
        vectors=[zvec.VectorSchema(
            "embedding", zvec.DataType.VECTOR_FP32, dimension=DIMENSION,
            index_param=index_param,
        )],
    )
    parquet = pq.ParquetFile(train)
    assert parquet.metadata.num_rows == ROWS
    collection = zvec.create_and_open(path, schema, options)
    try:
        start = perf_counter()
        io_start = io_bytes()
        inserted = 0
        for batch in parquet.iter_batches(batch_size=256, columns=["id", "emb"], use_threads=False):
            docs = [zvec.Doc(id=str(row), vectors={"embedding": vector}) for row, vector in
                    zip(batch.column(0).to_pylist(), batch.column(1).to_pylist())]
            statuses = collection.insert(docs)
            assert len(statuses) == len(docs) and all(status.ok() for status in statuses)
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
        observe("optimized", seconds=perf_counter() - start,
                **{key: val - io_start[key] for key, val in io_bytes().items()})
    finally:
        collection.close()
    observe("closed_after_build")


def run(kind, storage, data, path, evidence):
    queries, neighbors = workload(data)
    metric = zvec.MetricType.COSINE
    if kind == "ivf":
        index_param = zvec.IVFIndexParam(metric_type=metric, n_list=1024, n_iters=5)
        query_param = zvec.IVFQueryParam(nprobe=128)
    else:
        index_param = zvec.DiskAnnIndexParam(
            metric_type=metric, max_degree=64, list_size=100, pq_chunk_num=96,
        )
        query_param = zvec.DiskAnnQueryParam(list_size=400)

    budget_mib = BUDGET_MIB[kind]
    limit_kib = (budget_mib + 128) * 1024
    observe = partial(sample, evidence, limit_kib=limit_kib)
    observe("start", index=kind, storage=storage, rows=ROWS, dimension=DIMENSION,
            budget_mib=budget_mib, rss_limit_kib=limit_kib, min_recall=MIN_RECALL)
    zvec.init(memory_limit_mb=budget_mib, query_threads=4, optimize_threads=2)
    mmap = storage == "mmap"
    build(path, data / "shuffle_train.parquet", zvec.CollectionOption(enable_mmap=mmap),
          index_param, observe)

    options = zvec.CollectionOption(read_only=True, enable_mmap=mmap)
    for trial in range(1, TRIALS + 1):
        collection = zvec.open(path, options)
        try:
            assert collection.stats.doc_count == ROWS
            observe("opened", trial=trial)
            latencies = []
            matched = 0
            io_start = io_bytes()
            trial_start = perf_counter()
            for query_id, vector in queries.items():
                query = zvec.Query("embedding", vector=vector, param=query_param)
                start = perf_counter()
                hits = collection.query(query, topk=TOPK, output_fields=[])
                latencies.append(perf_counter() - start)
                ids = {hit.id for hit in hits}
                assert len(ids) == TOPK and all(math.isfinite(hit.score) for hit in hits)
                matched += len(ids & neighbors[query_id])
            elapsed = perf_counter() - trial_start
            recall = matched / (QUERY_COUNT * TOPK)
            metrics = {
                "trial": trial, "queries": QUERY_COUNT, "recall_at_10": recall,
                "qps": QUERY_COUNT / elapsed,
                "p99_ms": float(np.percentile(latencies, 99)) * 1000,
                **{key: val - io_start[key] for key, val in io_bytes().items()},
            }
            observe("query_trial", **metrics)
            print(metrics, flush=True)
            assert recall >= MIN_RECALL, f"trial {trial}: recall@10 {recall:.3f} < {MIN_RECALL}"
        finally:
            collection.close()
        observe("closed", trial=trial)
    observe("complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", choices=tuple(BUDGET_MIB))
    parser.add_argument("storage", choices=("buffer", "mmap"))
    parser.add_argument("data", type=Path)
    parser.add_argument("collection")
    parser.add_argument("evidence", type=Path)
    args = parser.parse_args()
    with args.evidence.open("x") as evidence:
        run(args.index, args.storage, args.data, args.collection, evidence)
