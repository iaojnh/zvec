# Copyright 2025-present the zvec project
# SPDX-License-Identifier: Apache-2.0
"""Measure insertion, index construction, persistence and reopen in one budget."""

# Standalone assertions and evidence are the acceptance contract.
# ruff: noqa: T201, PT018
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import monotonic

import numpy as np
from buffer_pool_acceptance import PAYLOAD, PROBE_IDS, check_reads, rss_kib
from buffer_pool_metrics import POLICY, calibration
from buffer_pool_rss_monitor import ProcessRssMonitor

import zvec


def run(index, path, evidence, rows, dimension, budget_mib, build_workers):
    batch_size = 128
    control = calibration(
        "buffer_pool_build_memory",
        budget_mib,
        build_workers,
        dimension,
        batch_size,
        len(PAYLOAD) + 32,
    )

    def record(phase, **details):
        rss, peak = rss_kib()
        print(
            json.dumps(
                {
                    "phase": phase,
                    "monotonic_seconds": monotonic(),
                    "rss_kib": rss,
                    "peak_kib": peak,
                    **details,
                }
            ),
            file=evidence,
            flush=True,
        )
        print(f"{index}: {phase}; RSS {rss} KiB; peak {peak} KiB", flush=True)

    record("calibration", **control)
    vector_bytes = rows * dimension * 4
    assert vector_bytes > budget_mib * 1024**2, "vectors must exceed the build budget"
    record(
        "start",
        index=index,
        rows=rows,
        dimension=dimension,
        budget_mib=budget_mib,
        build_workers=build_workers,
        input_vector_bytes=vector_bytes,
        max_buffer_mib=1,
    )
    zvec.init(
        memory_limit_mb=budget_mib,
        query_threads=1,
        optimize_threads=build_workers,
        log_level=zvec.LogLevel.INFO,
    )
    params = (
        zvec.IVFIndexParam(metric_type=zvec.MetricType.L2, n_list=32, n_iters=5)
        if index == "ivf"
        else zvec.DiskAnnIndexParam(
            metric_type=zvec.MetricType.L2,
            max_degree=32,
            list_size=128,
            pq_chunk_num=0,
        )
    )
    query_param = (
        zvec.IVFQueryParam(nprobe=32)
        if index == "ivf"
        else zvec.DiskAnnQueryParam(list_size=256)
    )
    schema = zvec.CollectionSchema(
        name="build_memory",
        fields=[zvec.FieldSchema("payload", zvec.DataType.STRING)],
        vectors=[
            zvec.VectorSchema(
                "embedding",
                zvec.DataType.VECTOR_FP32,
                dimension=dimension,
                index_param=params,
            )
        ],
    )
    options = zvec.CollectionOption(enable_mmap=False, max_buffer_size=1024**2)
    collection = zvec.create_and_open(path, schema, options)
    probes = {}
    rng = np.random.default_rng(2026)
    try:
        for start in range(0, rows, batch_size):
            vectors = rng.standard_normal(
                (min(batch_size, rows - start), dimension),
                dtype=np.float32,
            )
            docs = []
            for row, vector in enumerate(vectors, start):
                key = str(row)
                values = vector.tolist()
                if row in PROBE_IDS:
                    probes[key] = values
                docs.append(
                    zvec.Doc(
                        key,
                        fields={
                            "payload": "updated"
                            if row == PROBE_IDS[0]
                            else PAYLOAD + key
                        },
                        vectors={"embedding": values},
                    )
                )
            statuses = collection.insert(docs)
            assert len(statuses) == len(docs) and all(s.ok() for s in statuses)
        del docs, statuses, vectors, values
        record("insert_finished")
        collection.flush()
        record("flush_finished")
        record("build_started")
        collection.optimize()
        assert collection.stats.index_completeness["embedding"] == 1
        record("build_finished", indexed_rows=collection.stats.doc_count)
    finally:
        collection.close()
    record("closed_after_build")
    reader = zvec.open(path, zvec.CollectionOption(read_only=True, enable_mmap=False))
    try:
        assert reader.stats.index_completeness["embedding"] == 1
        check_reads(reader, probes, query_param, 32, rows)
    finally:
        reader.close()
    record("reopened_verified", indexed_rows=rows)
    record(
        "complete",
        index=index,
        rows=rows,
        dimension=dimension,
        budget_mib=budget_mib,
        build_workers=build_workers,
        input_vector_bytes=vector_bytes,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", choices=("ivf", "diskann"))
    parser.add_argument("collection")
    parser.add_argument("evidence", type=Path)
    settings = POLICY["index_build"]
    for name in ("rows", "dimension", "budget_mib", "build_workers"):
        parser.add_argument(
            "--" + name.replace("_", "-"), type=int, default=settings[name]
        )
    args = parser.parse_args()
    assert args.rows > max(PROBE_IDS) and args.dimension > 0
    assert args.budget_mib > 0 and args.build_workers > 0
    with (
        args.evidence.open("x") as output,
        ProcessRssMonitor(output, args.budget_mib * 1024),
    ):
        run(
            args.index,
            args.collection,
            output,
            args.rows,
            args.dimension,
            args.budget_mib,
            args.build_workers,
        )
