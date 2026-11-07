# Copyright 2025-present the zvec project
# SPDX-License-Identifier: Apache-2.0

"""Bounded Buffer Pool acceptance for a single index and process."""

# Standalone scenario assertions and console evidence are intentional here.
# ruff: noqa: T201, PT017, PT018, PLC0415, PLR0915
from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from functools import partial
from hashlib import shake_256
from pathlib import Path
from time import perf_counter
from typing import ClassVar

import numpy as np
from buffer_pool_metrics import calibration, stability
from buffer_pool_rss_monitor import ProcessRssMonitor

import zvec

ROWS = 20_000
DIMENSION = 128
BATCH_SIZE = 128
ROUNDS = 36
BUDGET_MIB = 512
RSS_LIMIT_KIB = BUDGET_MIB * 1024
PROBE_IDS = (17, 101, 143)
PAYLOAD = "buffer-pool " * 64


class Mallinfo2(ctypes.Structure):
    _fields_: ClassVar[list[tuple[str, type]]] = [
        (name, ctypes.c_size_t)
        for name in (
            "arena",
            "ordblks",
            "smblks",
            "hblks",
            "hblkhd",
            "usmblks",
            "fsmblks",
            "uordblks",
            "fordblks",
            "keepcost",
        )
    ]


_mallinfo2 = (
    getattr(ctypes.CDLL(None), "mallinfo2", None) if sys.platform == "linux" else None
)
if _mallinfo2 is not None:
    _mallinfo2.restype = Mallinfo2


def memory_details():
    details = {}
    if sys.platform == "linux":
        for line in Path("/proc/self/status").read_text().splitlines():
            key, _, value = line.partition(":")
            if key in ("RssAnon", "RssFile", "VmSwap", "Threads"):
                details[key] = int(value.split()[0])
        if _mallinfo2 is not None:
            info = _mallinfo2()
            details["glibc_heap_bytes"] = {
                name: getattr(info, name)
                for name in ("arena", "hblkhd", "uordblks", "fordblks")
            }
    arrow = sys.modules.get("pyarrow")
    if arrow is not None:
        details["python_arrow_bytes"] = arrow.total_allocated_bytes()
        details["python_arrow_allocator"] = arrow.default_memory_pool().backend_name
    return details


def rss_kib():
    if sys.platform != "linux":
        import psutil

        memory = psutil.Process().memory_info()
        if sys.platform == "win32":
            return memory.rss // 1024, memory.peak_wset // 1024
        import resource

        return memory.rss // 1024, resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss // 1024

    values = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        key, _, value = line.partition(":")
        if key in ("VmRSS", "VmHWM"):
            values[key] = int(value.split()[0])
    return values["VmRSS"], values["VmHWM"]


def sample(evidence, phase, *, limit_kib=RSS_LIMIT_KIB, **details):
    rss, peak = rss_kib()
    row = {
        "phase": phase,
        "rss_kib": rss,
        "peak_kib": peak,
        **memory_details(),
        **details,
    }
    print(
        json.dumps(row),
        file=evidence,
        flush=True,
    )
    if phase == "closed" and details.get("cycle", 0) % 6 == 0:
        print("MEMORY_DIAGNOSTIC " + json.dumps(row), flush=True)
    if limit_kib is not None:
        assert peak <= limit_kib, f"{phase}: peak {peak} KiB exceeds {limit_kib} KiB"
    return rss


def mutate(collection, probes, rows):
    """Exercise all write operations, then restore the exact fixed corpus."""
    ids = [str(i) for i in PROBE_IDS]
    assert collection.update(zvec.Doc(id=ids[2], fields={"payload": "temporary"})).ok()
    assert collection.fetch([ids[2]])[ids[2]].field("payload") == "temporary"
    assert all(s.ok() for s in collection.delete(ids[:2]))
    assert not collection.fetch(ids[:2])
    assert collection.stats.doc_count == rows - 2
    docs = [
        zvec.Doc(
            id=key,
            fields={"payload": "updated" if key == ids[0] else PAYLOAD + key},
            vectors={"embedding": probes[key]},
        )
        for key in ids
    ]
    assert all(s.ok() for s in collection.upsert(docs))
    collection.flush()
    assert collection.stats.doc_count == rows


def recover(path, probes, rows, budget, limit_kib, evidence):
    # A real unclean process exit leaves pending WAL; no destructor or flush.
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "_crash",
            path,
            str(budget),
            str(limit_kib),
            evidence.name,
        ],
        check=True,
        timeout=120,
    )
    try:
        reader = zvec.open(
            path, zvec.CollectionOption(read_only=True, enable_mmap=False)
        )
    except (ValueError, RuntimeError) as error:
        assert "requires WAL recovery" in str(error), str(error)
    else:
        reader.close()
        raise AssertionError("pending WAL was silently accepted read-only")
    writer = zvec.open(path, zvec.CollectionOption(enable_mmap=False))
    try:
        key = str(PROBE_IDS[2])
        assert writer.fetch([key])[key].field("payload") == "wal-recovered"
        mutate(writer, probes, rows)
    finally:
        writer.close()


def compaction_witness(path, observe):
    schema = zvec.CollectionSchema(
        name="compaction_witness",
        fields=[
            zvec.FieldSchema(
                "text", zvec.DataType.STRING, index_param=zvec.FtsIndexParam()
            )
        ],
        vectors=[
            zvec.VectorSchema(
                "v",
                zvec.DataType.VECTOR_FP32,
                dimension=4,
                index_param=zvec.FlatIndexParam(metric_type=zvec.MetricType.L2),
            )
        ],
    )
    collection = zvec.create_and_open(
        path + "-compaction", schema, zvec.CollectionOption(enable_mmap=False)
    )
    try:
        assert all(
            s.ok()
            for s in collection.insert(
                [
                    zvec.Doc(
                        id=str(i),
                        fields={"text": "hello compaction"},
                        vectors={"v": [float(i), 1.0, 0.0, 0.0]},
                    )
                    for i in range(100)
                ]
            )
        )
        collection.flush()
        collection.optimize()
        assert all(s.ok() for s in collection.delete([str(i) for i in range(40)]))
        # 40% tombstones exceed the product's 30% physical rebuild threshold.
        collection.optimize()
        assert collection.stats.doc_count == 60
        assert {doc.id for doc in collection.iter_docs()} == {
            str(i) for i in range(40, 100)
        }
        hits = collection.query(
            zvec.Query("text", fts=zvec.Fts(match_string="hello")), topk=100
        )
        assert {hit.id for hit in hits} == {str(i) for i in range(40, 100)}
    finally:
        collection.close()
    observe("compaction_verified", before_rows=100, after_rows=60, deleted_ratio=0.4)


def check_reads(collection, probes, query_param, topk, rows, vector_only=False):
    assert collection.stats.doc_count == rows
    doc_ids = [str(i) for i in PROBE_IDS]
    if vector_only:
        hits = collection.query(
            zvec.Query("embedding", vector=probes[doc_ids[0]], param=query_param),
            topk=topk,
            output_fields=[],
            include_vector=False,
        )
        assert len(hits) == topk and hits[0].id == doc_ids[0], (
            doc_ids[0],
            [(hit.id, hit.score) for hit in hits[:3]],
        )
        assert all(math.isfinite(hit.score) for hit in hits)
        return {hits[0].id: hits[0]}
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
    assert hits[0].id == doc_ids[0] and hits[0].field("payload") == "updated", (
        doc_ids[0],
        [(hit.id, hit.score, hit.field("payload")) for hit in hits[:3]],
    )
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
            tasks += [
                workers.submit(write_batch, collection, cycle)
                for collection in competing
            ]
            for task in tasks:
                task.result()
            observe("shared_collections", cycle=cycle, collections=4, workers=4)


def run(
    kind,
    path,
    evidence,
    *,
    rows=ROWS,
    dimension=DIMENSION,
    budget_mib=BUDGET_MIB,
    mixed=False,
    workers=1,
    soak_seconds=0,
    read_only=False,
    vector_only=False,
):
    control = calibration(
        "buffer_pool_acceptance",
        budget_mib,
        workers,
        dimension,
        BATCH_SIZE,
        len(PAYLOAD) + 32,
    )
    limit_kib = control["rss_limit_kib"]
    observe = partial(sample, evidence, limit_kib=limit_kib)
    observe("calibration", **control)
    observe(
        "start",
        index=kind,
        rows=rows,
        dimension=dimension,
        budget_mib=budget_mib,
        rss_limit_kib=limit_kib,
        rounds=ROUNDS,
    )
    zvec.init(
        memory_limit_mb=budget_mib,
        query_threads=workers,
        optimize_threads=1,
        log_level=zvec.LogLevel.WARN if soak_seconds else zvec.LogLevel.INFO,
    )
    if not read_only:
        compaction_witness(path, observe)
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
                "embedding",
                zvec.DataType.VECTOR_FP32,
                dimension=dimension,
                index_param=index_param,
            )
        ],
    )
    probes = {}
    read = partial(
        check_reads,
        probes=probes,
        query_param=query_param,
        rows=rows,
        vector_only=vector_only,
    )
    rng = np.random.default_rng(2026)
    if not read_only:
        collection = zvec.create_and_open(
            path, schema, zvec.CollectionOption(enable_mmap=False)
        )
    else:
        collection = None
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
                if read_only:
                    continue
                docs.append(
                    zvec.Doc(
                        id=doc_id,
                        fields={"payload": PAYLOAD + doc_id},
                        vectors={"embedding": values},
                    )
                )
            if read_only:
                continue
            statuses = collection.insert(docs)
            assert len(statuses) == count and all(status.ok() for status in statuses)
        if not read_only:
            del docs, statuses, vectors
            collection.flush()
            observe("inserted")
            collection.optimize()
            assert collection.stats.index_completeness["embedding"] == 1
            observe("optimized")
            mutate(collection, probes, rows)
            read(collection, topk=32)
            observe(
                "mutations_verified", operations=["update", "delete", "upsert", "flush"]
            )
    finally:
        if collection is not None:
            collection.close()
    observe("closed_after_build")
    if not read_only:
        recover(path, probes, rows, budget_mib, limit_kib, evidence)
        observe("wal_recovered")

    options = zvec.CollectionOption(read_only=True, enable_mmap=False)
    if mixed:
        primary = zvec.open(path, options)
        try:
            check_competing_collections(path, primary, read, observe)
        finally:
            primary.close()

    closed_rss = []
    cycle = 0
    while cycle < ROUNDS or (
        soak_seconds
        and (not closed_rss or perf_counter() - closed_rss[0][0] < soak_seconds)
    ):
        cycle += 1
        if soak_seconds:
            writer = zvec.open(path, zvec.CollectionOption(enable_mmap=False))
            try:
                mutate(writer, probes, rows)
                if cycle % 12 == 0:
                    writer.optimize()
            finally:
                writer.close()
        collection = zvec.open(path, options)
        try:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                tasks = [
                    executor.submit(read, collection, topk=32 if cycle % 2 else 1)
                    for _ in range(workers)
                ]
                results = [task.result() for task in tasks]
            del tasks
            retained = results[0]
            del results
            observe("results_held", cycle=cycle)
        finally:
            collection.close()
        if not vector_only:
            assert retained[str(PROBE_IDS[0])].field("payload") == "updated"
        del retained, collection
        closed_rss.append((perf_counter(), observe("closed", cycle=cycle)))

    metrics = stability(closed_rss, soak_seconds)
    observe(
        "complete",
        **metrics,
        workers=workers,
        read_only=read_only,
        mixed=mixed,
        rows=rows,
        dimension=dimension,
        budget_mib=budget_mib,
        projection="ids_scores" if vector_only else "fields_vectors",
    )
    print(
        f"{kind}: {cycle} rounds, peak RSS {rss_kib()[1]} KiB, "
        f"close drift {metrics['steady_growth_kib']:.0f} KiB"
    )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "_crash":
        path, budget, limit, output = sys.argv[2:]
        zvec.init(memory_limit_mb=int(budget), query_threads=1, optimize_threads=1)
        writer = zvec.open(path, zvec.CollectionOption(enable_mmap=False))
        assert writer.update(
            zvec.Doc(id=str(PROBE_IDS[2]), fields={"payload": "wal-recovered"})
        ).ok()
        with Path(output).open("a") as log:
            sample(log, "unclean_writer", limit_kib=int(limit))
        os._exit(0)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", choices=("ivf", "diskann"))
    parser.add_argument("collection")
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--rows", type=int, default=ROWS)
    parser.add_argument("--dimension", type=int, default=DIMENSION)
    parser.add_argument("--budget-mib", type=int, default=BUDGET_MIB)
    parser.add_argument("--mixed", action="store_true")
    parser.add_argument("--workers", type=int, choices=(1, 4, 8), default=1)
    parser.add_argument("--soak-seconds", type=int, default=0)
    parser.add_argument("--read-only", action="store_true")
    parser.add_argument("--vector-only", action="store_true")
    args = parser.parse_args()
    if args.rows <= max(PROBE_IDS) or args.dimension <= 0 or args.budget_mib <= 0:
        parser.error(
            "rows must cover all probes, and dimension and budget must be positive"
        )
    if args.soak_seconds < 0 or (args.read_only and (args.mixed or args.soak_seconds)):
        parser.error("soak requires a writable workload and a nonnegative duration")
    if args.vector_only and not args.read_only:
        parser.error("vector-only is the separate read-only capacity-boundary profile")
    with (
        args.evidence.open("x") as evidence,
        ProcessRssMonitor(evidence, args.budget_mib * 1024),
    ):
        run(
            args.index,
            args.collection,
            evidence,
            rows=args.rows,
            dimension=args.dimension,
            budget_mib=args.budget_mib,
            mixed=args.mixed,
            workers=args.workers,
            soak_seconds=args.soak_seconds,
            read_only=args.read_only,
            vector_only=args.vector_only,
        )
