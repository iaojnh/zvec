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


def input_config(config, directory):
    if "dataset" not in config:
        return config
    if directory is None:
        raise ValueError(
            "Cohere requires --input-dir; prepare it with ivf_cohere_input.py"
        )
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text())
    expected = {
        "dataset": config["dataset"]["name"],
        "rows": config["rows"],
        "dimension": config["dimension"],
        "query_count": config["query"]["query_count"],
        "topk": config["query"]["topk"],
        "normalized": True,
    }
    if any(manifest[key] != value for key, value in expected.items()):
        raise ValueError("prepared input differs from the selected profile")
    for name, info in manifest["files"].items():
        if Path(name).name != name or (directory / name).stat().st_size != info["size"]:
            raise ValueError("prepared input file is missing or truncated")
    return {
        **config,
        "_input_directory": directory,
        "_input_manifest": manifest,
        "_dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def document_ids(config, start, count):
    if "dataset" in config:
        if start < config["rows"]:
            return np.fromfile(
                config["_input_directory"] / "ids.i64",
                dtype="<i8",
                count=count,
                offset=start * 8,
            )
        start = config["_input_manifest"]["new_id_start"] + start - config["rows"]
    return np.arange(start, start + count, dtype=np.int64)


def vectors(config, start, count):
    """Regenerate a bounded batch without retaining the corpus in Python."""
    if "dataset" in config and start < config["rows"]:
        values = np.fromfile(
            config["_input_directory"] / "vectors.f32",
            dtype="<f4",
            count=count * config["dimension"],
            offset=start * config["dimension"] * 4,
        )
        return values.reshape(count, config["dimension"])
    rng = np.random.default_rng(config["seed"] + start // config["batch_size"])
    values = rng.standard_normal((count, config["dimension"]), dtype=np.float32)
    if config.get("index", {}).get("metric") == "COSINE":
        values /= np.linalg.norm(values, axis=1, keepdims=True)
    return values


def add_truth(
    queries, batch, start, best_ids, best_distances, topk, candidate_ids=None
):
    distances = (
        np.sum(queries * queries, axis=1)[:, None]
        + np.sum(batch * batch, axis=1)[None, :]
        - 2 * queries @ batch.T
    )
    ids = np.broadcast_to(
        np.arange(start, start + len(batch))
        if candidate_ids is None
        else candidate_ids,
        distances.shape,
    )
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
            id=str(key),
            vectors={"embedding": vector.tolist()},
            fields={"payload": f"{cycle}:" + "p" * config["payload_bytes"]},
        )
        for key, vector in zip(
            document_ids(config, start, len(batch)), batch, strict=True
        )
    ]


def build_queries(config):
    if "dataset" in config:
        path = config["_input_directory"] / "queries.f32"
        if (
            hashlib.sha256(path.read_bytes()).hexdigest()
            != config["_input_manifest"]["files"][path.name]["sha256"]
        ):
            raise ValueError("query fixture checksum mismatch")
        return np.fromfile(path, dtype="<f4").reshape(
            config["query"]["query_count"], config["dimension"]
        )
    return vectors(config, 0, config["batch_size"])[: config["query"]["query_count"]]


def check_input_truth(config, hashes, best_ids):
    if "dataset" not in config:
        return None
    manifest = config["_input_manifest"]
    for name, value in hashes.items():
        if value.hexdigest() != manifest["files"][name]["sha256"]:
            raise ValueError(f"input checksum mismatch: {name}")
    path = config["_input_directory"] / "neighbors.i64"
    if (
        hashlib.sha256(path.read_bytes()).hexdigest()
        != manifest["files"][path.name]["sha256"]
    ):
        raise ValueError("ground truth checksum mismatch")
    truth = np.fromfile(path, dtype="<i8").reshape(best_ids.shape)
    recall = np.mean(
        [len(set(a) & set(b)) / len(a) for a, b in zip(best_ids, truth, strict=True)]
    )
    if recall < 0.99:
        raise AssertionError(
            f"independent oracle differs from public Cohere truth: {recall}"
        )
    return float(recall)


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
                    metric_type=getattr(zvec.MetricType, settings["metric"]),
                    n_list=settings["n_list"],
                    n_iters=settings["n_iters"],
                ),
            )
        ],
    )
    queries = build_queries(config)
    hashes = {name: hashlib.sha256() for name in ("vectors.f32", "ids.i64")}
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
                queries,
                batch,
                start,
                best_ids,
                best_distances,
                config["query"]["topk"],
                document_ids(config, start, len(batch)),
            )
            hashes["vectors.f32"].update(batch.astype("<f4", copy=False).tobytes())
            hashes["ids.i64"].update(
                document_ids(config, start, len(batch))
                .astype("<i8", copy=False)
                .tobytes()
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
        "probe_id": str(document_ids(config, 0, 1)[0]),
        "new_id": str(document_ids(config, config["rows"], 1)[0]),
    }
    (root / "fixture.json").write_text(json.dumps(fixture))
    # Check the build-to-query transition before this process can discard any
    # retained allocator backing. Later stages also test a fresh process.
    verify(config, root, storage, after_mixed=False)
    return {
        "rows": config["rows"],
        "input_vector_bytes": config["rows"] * config["dimension"] * 4,
        "insert_and_flush_seconds": build_started - inserted_at,
        "build_seconds": build_finished - build_started,
        "post_build_query_verified": True,
        "public_truth_recall": check_input_truth(config, hashes, best_ids),
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
            queries,
            batch,
            start,
            best_ids,
            best_distances,
            config["query"]["topk"],
            document_ids(config, start, len(batch)),
        )
    return best_ids, best_distances


def pace_write(settings, started, completed):
    total_batches = settings["cycles"] * settings["batches"]
    deadline = started + settings["minimum_seconds"] * completed / total_batches
    time.sleep(max(0.0, deadline - time.monotonic()))


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
                for _ in range(config["mixed"]["cycles"]):
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
                        # Pace identical operation counts over the declared
                        # write interval. Faster storage must not accumulate
                        # more obsolete vector versions before the comparison.
                        pace_write(
                            config["mixed"],
                            started,
                            cycles * config["mixed"]["batches"] + i + 1,
                        )
                    cycles += 1
                write_phase_seconds = time.monotonic() - started
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
        assert collection.stats.index_completeness["embedding"] == 1
        assert all(count > 0 for count in read_counts)
        new_id = fixture["new_id"]
        assert (
            collection.fetch([new_id])[new_id]
            .field("payload")
            .startswith(f"{cycles - 1}:")
        )
    finally:
        stop.set()
        collection.close()
    fixture.update(
        last_cycle=cycles - 1,
        truth=best_ids.tolist(),
        distances=best_distances.tolist(),
        rows=expected_rows,
    )
    (root / "fixture.json").write_text(json.dumps(fixture))
    elapsed = time.monotonic() - started
    return {
        "rows": expected_rows,
        "write_operations": writes,
        "reader_queries": sum(read_counts),
        "reader_queries_during_writes": reads_during_writes,
        "cycles": cycles,
        "write_phase_seconds": write_phase_seconds,
        "seconds": elapsed,
        "writes_per_second": writes / elapsed,
    }


def verify(config, root, storage, after_mixed=True):
    fixture = json.loads((root / "fixture.json").read_text())
    collection = zvec.open(str(root / "collection"), option(config, storage, True))
    try:
        assert collection.stats.doc_count == fixture["rows"]
        probe_count = len(fixture["queries"]) if after_mixed else 1
        recalls = [
            check_hits(
                query(config, collection, vector),
                [str(i) for i in truth],
                config["query"]["topk"],
            )
            for vector, truth in zip(
                fixture["queries"][:probe_count],
                fixture["truth"][:probe_count],
                strict=True,
            )
        ]
        assert min(recalls) >= config["acceptance"]["minimum_recall"]
        probe_id = fixture["probe_id"]
        ids = [probe_id, fixture["new_id"]] if after_mixed else [probe_id]
        docs = collection.fetch(ids)
        assert set(docs) == set(ids)
        np.testing.assert_allclose(
            docs[probe_id].vector("embedding"),
            vectors(config, 0, 1)[0],
            rtol=1e-6,
            atol=1e-7,
        )
        assert docs[probe_id].field("payload") == "0:" + "p" * config["payload_bytes"]
        if after_mixed:
            new_id = fixture["new_id"]
            assert (
                docs[new_id].field("payload")
                == f"{fixture['last_cycle']}:" + "p" * config["payload_bytes"]
            )
            np.testing.assert_allclose(
                docs[new_id].vector("embedding"),
                vectors(config, config["rows"], config["batch_size"])[0],
                rtol=1e-6,
                atol=1e-7,
            )
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
    hot_count = max(1, count // 5)
    distributions = {
        "uniform": np.full(count, 1 / count),
        "hot_80_20": np.concatenate(
            (
                np.full(hot_count, 0.8 / hot_count),
                np.full(count - hot_count, 0.2 / (count - hot_count)),
            )
        ),
        "zipf": 1 / np.arange(1, count + 1, dtype=float),
        "repeat_90_10": np.asarray([0.9, *([0.1 / (count - 1)] * (count - 1))]),
    }
    try:
        # The corpus, projection and search parameters are identical in every
        # case. Check the complete fixed query set once per fresh process;
        # measured queries also validate their own results below.
        for i, vector in enumerate(fixture["queries"]):
            assert (
                check_hits(
                    query(config, collection, vector),
                    [str(x) for x in fixture["truth"][i]],
                    config["query"]["topk"],
                )
                >= config["acceptance"]["minimum_recall"]
            )
        for distribution in config["query"]["distributions"]:
            weights = distributions[distribution]
            sequence = rng.choice(count, size=1024, p=weights / weights.sum()).tolist()
            # Each case gets the same fixed warmup, outside its measured window.
            for i in range(config["query"]["warmup_queries"]):
                query(config, collection, fixture["queries"][sequence[i]])
            latencies = []
            recall_sum = 0.0
            started = time.monotonic()
            while (
                len(latencies) < config["query"]["minimum_queries"]
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
            print(
                json.dumps({"stage": "query_case", "storage": storage, **cases[-1]}),
                flush=True,
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
    parser.add_argument("--input-dir", type=Path)
    args = parser.parse_args()
    config = json.loads(args.profile.read_text())
    assert config["schema_version"] == 1
    assert 2 <= config["query"]["query_count"] <= config["batch_size"]
    assert config["query"]["topk"] < config["rows"]
    assert config["mixed"]["cycles"] > 0
    assert config["query"]["minimum_queries"] > 0
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
        runtime_config = input_config(config, args.input_dir)
        result["dataset_sha256"] = runtime_config.get("_dataset_sha256")
        zvec.init(
            memory_limit_mb=config["memory_limit_mb"]
            if args.storage == "buffer"
            else config["mmap_reference_memory_limit_mb"],
            query_threads=config["query_threads"],
            optimize_threads=config["optimize_threads"],
            log_level=zvec.LogLevel.INFO,
        )
        if args.stage == "query":
            details = query_trials(runtime_config, args.root, args.storage, args.trial)
        else:
            details = globals()[args.stage](runtime_config, args.root, args.storage)
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
