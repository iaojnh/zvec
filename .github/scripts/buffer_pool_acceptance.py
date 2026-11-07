# Copyright 2025-present the zvec project
# SPDX-License-Identifier: Apache-2.0

"""Bounded Buffer Pool acceptance for a single index and process."""

import argparse
import json
import math
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from functools import partial
from hashlib import shake_256
from pathlib import Path

import numpy as np
import zvec

ROWS = 20_000
DIMENSION = 128
BATCH_SIZE = 128
ROUNDS = 36
BUDGET_MIB = 512
# Allow 64 MiB beyond the configured soft target for the CI process. This is
# the acceptance ceiling for this fixed workload, including the Python runtime.
RSS_LIMIT_KIB = (BUDGET_MIB + 64) * 1024
PROBE_IDS = (17, 101, 143)
PAYLOAD = "buffer-pool " * 64


def rss_kib():
    if sys.platform != "linux":
        import psutil

        memory = psutil.Process().memory_info()
        if sys.platform == "win32":
            return memory.rss // 1024, memory.peak_wset // 1024
        import resource

        return memory.rss // 1024, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024

    values = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        key, _, value = line.partition(":")
        if key in ("VmRSS", "VmHWM"):
            values[key] = int(value.split()[0])
    return values["VmRSS"], values["VmHWM"]


def sample(evidence, phase, *, limit_kib=RSS_LIMIT_KIB, **details):
    rss, peak = rss_kib()
    print(
        json.dumps({"phase": phase, "rss_kib": rss, "peak_kib": peak, **details}),
        file=evidence,
        flush=True,
    )
    assert peak <= limit_kib, f"{phase}: peak {peak} KiB exceeds {limit_kib} KiB"
    return rss


def check_reads(collection, probes, query_param, topk, rows):
    assert collection.stats.doc_count == rows
    doc_ids = [str(i) for i in PROBE_IDS]
    docs = collection.fetch(doc_ids)
    assert set(docs) == set(doc_ids)
    for doc_id in doc_ids:
        assert docs[doc_id].field("payload") == (
            "updated" if doc_id == doc_ids[0] else PAYLOAD + doc_id
        )
        assert docs[doc_id].vector("embedding") == probes[doc_id]
    hits = collection.query(
        zvec.Query("embedding", vector=probes[doc_ids[0]], param=query_param),
        topk=topk,
        include_vector=True,
    )
    assert len(hits) == topk
    assert hits[0].id == doc_ids[0] and hits[0].field("payload") == "updated"
    assert all(math.isfinite(hit.score) for hit in hits)
    return docs


def check_competing_collections(path, primary, read_primary, observe):
    schema = zvec.CollectionSchema(
        name="buffer_pool_competing",
        fields=[zvec.FieldSchema("payload", zvec.DataType.STRING)],
    )
    options = zvec.CollectionOption(enable_mmap=False)

    def write_batch(collection, cycle):
        docs = [
            zvec.Doc(id=str(i), fields={"payload": f"{cycle}:{i}:" + "v" * (i * 128)})
            for i in range(32)
        ]
        statuses = collection.update(docs)
        assert len(statuses) == len(docs) and all(status.ok() for status in statuses)
        collection.flush()
        assert collection.fetch(["0"])["0"].field("payload") == f"{cycle}:0:"
        if cycle % 4 == 0:
            collection.optimize()

    with ExitStack() as cleanup, ThreadPoolExecutor(max_workers=4) as workers:
        competing = []
        for shard in range(3):
            collection = zvec.create_and_open(f"{path}-shared-{shard}", schema, options)
            cleanup.callback(collection.close)
            competing.append(collection)
            for start in range(0, 1024, BATCH_SIZE):
                docs = [
                    zvec.Doc(
                        id=str(i),
                        fields={"payload": shake_256(str(i).encode()).hexdigest(2048)},
                    )
                    for i in range(start, start + BATCH_SIZE)
                ]
                assert all(status.ok() for status in collection.insert(docs))
            collection.flush()

        for cycle in range(1, 13):
            tasks = [workers.submit(read_primary, primary, topk=32)]
            tasks += [workers.submit(write_batch, collection, cycle) for collection in competing]
            for task in tasks:
                task.result()
            observe("shared_collections", cycle=cycle, collections=4, workers=4)


def run(
    kind, path, evidence, *, rows=ROWS, dimension=DIMENSION, budget_mib=BUDGET_MIB, mixed=False
):
    limit_kib = (budget_mib + 64) * 1024
    observe = partial(sample, evidence, limit_kib=limit_kib)
    observe(
        "start", index=kind, rows=rows, dimension=dimension, budget_mib=budget_mib,
        rss_limit_kib=limit_kib, rounds=ROUNDS,
    )
    zvec.init(memory_limit_mb=budget_mib, query_threads=2, optimize_threads=1)
    if kind == "ivf":
        index_param = zvec.IVFIndexParam(
            metric_type=zvec.MetricType.L2, n_list=32, n_iters=5
        )
        query_param = zvec.IVFQueryParam(nprobe=32)
    else:
        index_param = zvec.DiskAnnIndexParam(
            metric_type=zvec.MetricType.L2, max_degree=32, list_size=128, pq_chunk_num=0
        )
        query_param = zvec.DiskAnnQueryParam(list_size=256)

    schema = zvec.CollectionSchema(
        name="buffer_pool_acceptance",
        fields=[zvec.FieldSchema("payload", zvec.DataType.STRING)],
        vectors=[
            zvec.VectorSchema(
                "embedding", zvec.DataType.VECTOR_FP32, dimension=dimension,
                index_param=index_param,
            )
        ],
    )
    probes = {}
    read = partial(check_reads, probes=probes, query_param=query_param, rows=rows)
    rng = np.random.default_rng(2026)
    collection = zvec.create_and_open(
        path, schema, zvec.CollectionOption(enable_mmap=False)
    )
    try:
        for start in range(0, rows, BATCH_SIZE):
            count = min(BATCH_SIZE, rows - start)
            vectors = rng.standard_normal((count, dimension), dtype=np.float32)
            docs = []
            for row, vector in enumerate(vectors, start):
                doc_id = str(row)
                values = vector.tolist()
                if row in PROBE_IDS:
                    probes[doc_id] = values
                docs.append(
                    zvec.Doc(
                        id=doc_id, fields={"payload": PAYLOAD + doc_id},
                        vectors={"embedding": values},
                    )
                )
            statuses = collection.insert(docs)
            assert len(statuses) == count and all(status.ok() for status in statuses)
        del docs, statuses, vectors
        collection.flush()
        observe("inserted")
        collection.optimize()
        assert collection.stats.index_completeness["embedding"] == 1
        observe("optimized")
        update = zvec.Doc(id=str(PROBE_IDS[0]), fields={"payload": "updated"})
        assert collection.update(update).ok()
        collection.flush()
        read(collection, topk=32)
    finally:
        collection.close()
    observe("closed_after_build")

    options = zvec.CollectionOption(read_only=True, enable_mmap=False)
    if mixed:
        primary = zvec.open(path, options)
        try:
            check_competing_collections(path, primary, read, observe)
        finally:
            primary.close()

    closed_rss = []
    for cycle in range(1, ROUNDS + 1):
        collection = zvec.open(path, options)
        try:
            retained = read(collection, topk=32 if cycle % 2 else 1)
            observe("results_held", cycle=cycle)
        finally:
            collection.close()
        assert retained[str(PROBE_IDS[0])].field("payload") == "updated"
        del retained, collection
        closed_rss.append(observe("closed", cycle=cycle))

    # Allow 12 warm-up rounds, then compare two equally sized windows.
    prior_median = statistics.median(closed_rss[12:24])
    late_median = statistics.median(closed_rss[24:])
    drift_kib = late_median - prior_median
    observe("complete", closed_rss_drift_kib=drift_kib)
    assert drift_kib <= 8 * 1024, f"close RSS median grew by {drift_kib:.0f} KiB"
    assert max(closed_rss[24:]) <= prior_median + 16 * 1024, (
        "late close RSS exceeds stable band"
    )
    print(
        f"{kind}: {ROUNDS} rounds, peak RSS {rss_kib()[1]} KiB, "
        f"close drift {drift_kib:.0f} KiB"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", choices=("ivf", "diskann"))
    parser.add_argument("collection")
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--rows", type=int, default=ROWS)
    parser.add_argument("--dimension", type=int, default=DIMENSION)
    parser.add_argument("--budget-mib", type=int, default=BUDGET_MIB)
    parser.add_argument("--mixed", action="store_true")
    args = parser.parse_args()
    if args.rows <= max(PROBE_IDS) or args.dimension <= 0 or args.budget_mib <= 0:
        parser.error("rows must cover all probes, and dimension and budget must be positive")
    with args.evidence.open("x") as evidence:
        run(
            args.index, args.collection, evidence, rows=args.rows,
            dimension=args.dimension, budget_mib=args.budget_mib, mixed=args.mixed,
        )
