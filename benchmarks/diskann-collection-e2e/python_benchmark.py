"""Execute one Collection benchmark phase against the installed Python wheel."""

import argparse
from contextlib import ExitStack
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import struct
import sys
from time import perf_counter_ns

import numpy as np
import psutil
import zvec


FIELD = "embedding"
TOPK = 10
CALIBRATION = 200
PASSES = 3
BUDGETS = (32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048)


def matrix_shape(stream):
    rows, dimension = struct.unpack("<II", stream.read(8))
    if rows == 0 or not 0 < dimension <= 4096:
        raise ValueError("Invalid fixture shape")
    return rows, dimension


def read_matrix(path, dtype, row_limit):
    with path.open("rb") as stream:
        rows, dimension = matrix_shape(stream)
        if rows > row_limit:
            raise ValueError(f"Too many rows in {path}")
        values = np.fromfile(stream, dtype=dtype, count=rows * dimension)
        if len(values) != rows * dimension or stream.read(1):
            raise ValueError(f"Invalid fixture size in {path}")
    return values.reshape(rows, dimension)


def check_collection(collection, index, count):
    field = collection.schema.vector(FIELD)
    expected = zvec.IndexType.DISKANN if index == "diskann" else zvec.IndexType.HNSW
    stats = collection.stats
    if (field is None or field.index_param is None or field.index_param.type != expected
            or stats.doc_count != count or stats.index_completeness.get(FIELD) != 1.0):
        raise RuntimeError("Unexpected index type, document count, or index completeness")


def build(data, path, index):
    base = read_matrix(data / "base.fbin", "<f4", 1_000_000)
    if index == "diskann":
        params = zvec.DiskAnnIndexParam(
            metric_type=zvec.MetricType.L2, max_degree=64, list_size=100, pq_chunk_num=32
        )
    else:
        params = zvec.HnswIndexParam(metric_type=zvec.MetricType.L2, m=32, ef_construction=200)
    schema = zvec.CollectionSchema(
        "collection_benchmark",
        vectors=[zvec.VectorSchema(FIELD, zvec.DataType.VECTOR_FP32, base.shape[1], params)],
    )

    start = perf_counter_ns()
    collection = zvec.create_and_open(path, schema, option=zvec.CollectionOption(enable_mmap=True))
    try:
        batch = 256
        for first in range(0, len(base), batch):
            docs = [
                zvec.Doc(id=str(i), vectors={FIELD: vector.tolist()})
                for i, vector in enumerate(base[first:first + batch], start=first)
            ]
            statuses = collection.insert(docs)
            if len(statuses) != len(docs) or any(not status.ok() for status in statuses):
                raise RuntimeError(f"Insert failed for batch at row {first}")
        collection.flush()
        write_ms = (perf_counter_ns() - start) / 1e6
        start = perf_counter_ns()
        collection.optimize(zvec.OptimizeOption(concurrency=2))
        optimize_ms = (perf_counter_ns() - start) / 1e6
        check_collection(collection, index, len(base))
    finally:
        collection.close()
    return dict(vectors=len(base), index_complete=True, write_ms=write_ms, optimize_ms=optimize_ms)


def query(data, path, index, linux_block_io=False, observe_storage=False, cold_storage=False):
    with (data / "base.fbin").open("rb") as stream:
        count, dimension = matrix_shape(stream)
    matrix = read_matrix(data / "queries.fbin", "<f4", 20_000)
    truth = read_matrix(data / "neighbors.ibin", "<u4", 20_000)
    if len(matrix) <= CALIBRATION or matrix.shape[1] != dimension or truth.shape != (len(matrix), TOPK):
        raise ValueError("Query and truth matrices disagree")
    vectors = matrix.tolist()
    neighbors = [set(row) for row in truth.tolist()]
    query_param = zvec.DiskAnnQueryParam if index == "diskann" else zvec.HnswQueryParam

    process = psutil.Process()
    baseline = process.memory_info().rss
    start = perf_counter_ns()
    collection = zvec.open(path, option=zvec.CollectionOption(read_only=True, enable_mmap=True))
    open_ms = (perf_counter_ns() - start) / 1e6
    after_open = process.memory_info().rss

    def evaluate(row, param, latencies=None):
        q = zvec.Query(field_name=FIELD, vector=vectors[row], param=param)
        start = perf_counter_ns()
        hits = collection.query(q, topk=TOPK, output_fields=[])
        if latencies is not None:
            latencies.append(perf_counter_ns() - start)
        ids = {int(hit.id) for hit in hits if math.isfinite(hit.score)}
        if len(hits) != TOPK or len(ids) != TOPK or any(not 0 <= i < count for i in ids):
            raise RuntimeError("Invalid top-k result")
        return len(ids & neighbors[row])

    with ExitStack() as resources:
        resources.callback(collection.close)
        check_collection(collection, index, count)
        for budget in BUDGETS:
            param = query_param(budget)
            recall = sum(evaluate(i, param) for i in range(CALIBRATION)) / (CALIBRATION * TOPK)
            if recall >= 0.95:
                break
        else:
            raise RuntimeError("Could not reach 95% calibration recall")

        held_out = range(CALIBRATION, len(matrix))
        if cold_storage:
            from storage_observation import apple_index_cache, process_disk_bytes

            evict = resources.enter_context(apple_index_cache(Path(path)))
            cold_reads = []
            pid = os.getpid()
        else:
            for row in held_out:
                evaluate(row, param)
        after_warmup = process.memory_info().rss
        peak = after_warmup
        latencies = []
        correct = 0
        if observe_storage:
            from storage_observation import desktop_observation, desktop_snapshot

            io_start = desktop_snapshot(os.getpid())
        if linux_block_io:
            from linux_io_sweep import QueryBlockIO

            block_io = QueryBlockIO(Path(path))
            before_io = block_io.snapshot()
        if linux_block_io or observe_storage:
            window_start = perf_counter_ns()
        for _ in range(PASSES):
            for row in held_out:
                if cold_storage:
                    evict()
                    storage_before = process_disk_bytes(pid)
                correct += evaluate(row, param, latencies)
                if cold_storage:
                    storage_after = process_disk_bytes(pid)
                    if storage_before is None or storage_after is None or storage_after <= storage_before:
                        raise RuntimeError(f"Cold query {row} has no accounted process storage reads")
                    cold_reads.append(storage_after - storage_before)
                if row % 50 == 0:
                    peak = max(peak, process.memory_info().rss)
        if linux_block_io or observe_storage:
            window_ns = perf_counter_ns() - window_start
        if linux_block_io:
            io_result = block_io.result(before_io, window_ns, len(latencies))
        if observe_storage:
            observation = desktop_observation(io_start, desktop_snapshot(os.getpid()))
        after_query = process.memory_info().rss

    latencies.sort()
    result = dict(
        vectors=count,
        dimension=dimension,
        queries_per_pass=len(held_out),
        passes=PASSES,
        budget=budget,
        calibration_recall_at_10=recall,
        recall_at_10=correct / (len(latencies) * TOPK),
        qps=len(latencies) * 1e9 / sum(latencies),
        p95_ms=latencies[math.ceil(len(latencies) * 0.95) - 1] / 1e6,
        p99_ms=latencies[math.ceil(len(latencies) * 0.99) - 1] / 1e6,
        open_ms=open_ms,
        baseline_rss_bytes=baseline,
        after_open_rss_bytes=after_open,
        after_warmup_rss_bytes=after_warmup,
        after_query_rss_bytes=after_query,
        sampled_peak_query_rss_bytes=max(peak, after_query),
        query_rss_delta_bytes=after_query - baseline,
    )
    if linux_block_io:
        result["linux_block_io"] = io_result
    if observe_storage:
        result["query_window_ns"] = window_ns
        result["storage_observation"] = observation
    if cold_storage:
        result["cold_storage"] = dict(
            verified_queries=len(cold_reads),
            query_ns=sum(latencies),
            storage_read_bytes=sum(cold_reads),
            min_storage_read_bytes=min(cold_reads),
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["build", "query"])
    parser.add_argument("index", choices=["hnsw", "diskann"])
    parser.add_argument("data", type=Path)
    parser.add_argument("collection")
    parser.add_argument("--linux-block-io", action="store_true")
    parser.add_argument("--observe-storage", action="store_true")
    parser.add_argument("--cold-storage", action="store_true")
    args = parser.parse_args()
    if args.linux_block_io and (args.phase, args.index) != ("query", "diskann"):
        parser.error("--linux-block-io applies to the DiskANN query phase")
    if args.observe_storage and (sys.platform not in ("darwin", "win32")
                                 or (args.phase, args.index) != ("query", "diskann")
                                 or args.linux_block_io):
        parser.error("--observe-storage applies to macOS and Windows DiskANN queries")
    if args.cold_storage and (sys.platform != "darwin" or not args.observe_storage):
        parser.error("--cold-storage requires a macOS DiskANN query with --observe-storage")
    if version("zvec") != "0.8.1":
        raise RuntimeError("This benchmark requires the Zvec 0.8.1 wheel")
    zvec.init(query_threads=1, optimize_threads=2)
    if args.phase == "build":
        result = build(args.data, args.collection, args.index)
    else:
        result = query(
            args.data, args.collection, args.index,
            args.linux_block_io, args.observe_storage, args.cold_storage,
        )
    result.update(phase=args.phase, index=args.index)
    print("ZVEC_COLLECTION_BENCH_RESULT " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
