# Copyright 2025-present the zvec project
# SPDX-License-Identifier: Apache-2.0
"""Prepare streamable Cohere input before measuring the Zvec process.

The public Parquet column chunks require substantial Arrow decoder memory.
This separate fixture-preparation process emits normalized binary vectors/IDs
so the actual build can consume small bounded batches, without retaining an
input mmap. Preparation is not included in Zvec memory qualification.
"""

# Preparation metadata is a command-line result.
# ruff: noqa: T201

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024**2), b""):
            value.update(chunk)
    return value.hexdigest()


def normalize(vectors):
    values = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if not np.all(np.isfinite(values)) or not np.all(norms > 0):
        raise ValueError("Cohere vectors must be finite and nonzero")
    return values / norms


def fetch(config, source):
    source.mkdir(parents=True, exist_ok=True)
    for name, version in config["dataset"]["files"].items():
        if Path(name).name != name:
            raise ValueError("dataset input must be a filename")
        subprocess.run(
            [
                "curl",
                "--fail",
                "--location",
                "--retry",
                "2",
                "--max-time",
                "1200",
                "--header",
                f'If-Match: "{version["etag"]}"',
                "--max-filesize",
                str(version["size"]),
                f"https://assets.zilliz.com/benchmark/cohere_medium_1m/{name}",
                "--output",
                str(source / name),
            ],
            check=True,
        )


def train(config, source, output):
    parquet = pq.ParquetFile(
        source / "shuffle_train.parquet", buffer_size=65536, pre_buffer=False
    )
    if parquet.metadata.num_rows != config["rows"]:
        raise ValueError("unexpected Cohere row count")
    maximum_id = -1
    with (
        (output / "vectors.f32").open("xb") as vectors_file,
        (output / "ids.i64").open("xb") as ids_file,
    ):
        for batch in parquet.iter_batches(
            batch_size=config["batch_size"], columns=["id", "emb"], use_threads=False
        ):
            ids = np.asarray(batch.column(0).to_pylist(), dtype=np.int64)
            values = normalize(batch.column(1).to_pylist())
            if values.shape != (len(ids), config["dimension"]) or np.any(ids < 0):
                raise ValueError("unexpected Cohere ID/vector format")
            values.astype("<f4", copy=False).tofile(vectors_file)
            ids.astype("<i8", copy=False).tofile(ids_file)
            maximum_id = max(maximum_id, int(ids.max()))
    return maximum_id + 1


def queries(config, source, output):
    count, topk = config["query"]["query_count"], config["query"]["topk"]
    test = pq.ParquetFile(source / "test.parquet", buffer_size=65536, pre_buffer=False)
    batch = next(
        test.iter_batches(batch_size=count, columns=["id", "emb"], use_threads=False)
    )
    ids = batch.column(0).to_pylist()
    values = normalize(batch.column(1).to_pylist())
    if values.shape != (count, config["dimension"]) or len(set(ids)) != count:
        raise ValueError("unexpected Cohere queries")
    positions = {str(key): i for i, key in enumerate(ids)}
    neighbors = np.full((count, topk), -1, dtype="<i8")
    truth = pq.ParquetFile(
        source / "neighbors.parquet", buffer_size=65536, pre_buffer=False
    )
    for batch in truth.iter_batches(
        batch_size=32, columns=["id", "neighbors_id"], use_threads=False
    ):
        for key, nearest in zip(
            batch.column(0).to_pylist(), batch.column(1).to_pylist(), strict=True
        ):
            if str(key) in positions:
                neighbors[positions[str(key)]] = np.asarray(
                    nearest[:topk], dtype=np.int64
                )
    if np.any(neighbors < 0):
        raise ValueError("missing public ground truth")
    values.astype("<f4", copy=False).tofile(output / "queries.f32")
    neighbors.tofile(output / "neighbors.i64")


def prepare(config, source, output):
    versions = {}
    for name, expected in config["dataset"]["files"].items():
        path = source / name
        if path.stat().st_size != expected["size"]:
            raise ValueError(f"unexpected size for {name}")
        versions[name] = {**expected, "sha256": digest(path)}
    output.mkdir(parents=True, exist_ok=False)
    maximum_id = train(config, source, output)
    queries(config, source, output)
    manifest = {
        "dataset": config["dataset"]["name"],
        "rows": config["rows"],
        "dimension": config["dimension"],
        "query_count": config["query"]["query_count"],
        "topk": config["query"]["topk"],
        "new_id_start": maximum_id,
        "normalized": True,
        "source_files": versions,
        "files": {
            path.name: {"size": path.stat().st_size, "sha256": digest(path)}
            for path in sorted(output.iterdir())
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.profile.read_text())
    if args.download:
        fetch(config, args.source)
    pq_cpu = __import__("pyarrow")
    pq_cpu.set_cpu_count(1)
    pq_cpu.set_io_thread_count(1)
    prepare(config, args.source, args.output)


if __name__ == "__main__":
    main()
