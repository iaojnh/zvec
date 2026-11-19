"""Prepare a shared SIFT subset and exhaustive ground truth for every runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import urllib.request
from pathlib import Path

import h5py
import numpy as np


def write_matrix(path, values):
    values = np.ascontiguousarray(values)
    with path.open("wb") as output:
        output.write(struct.pack("<II", *values.shape))
        output.write(values.tobytes())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=100_000)
    parser.add_argument("--queries", type=int, default=1200)
    parser.add_argument("--synthetic", action="store_true", help="Local smoke test only")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = "https://ann-benchmarks.com/sift-128-euclidean.hdf5"
    if args.synthetic:
        rng = np.random.default_rng(20261105)
        base = rng.normal(size=(args.count, 32)).astype("<f4")
        queries = rng.normal(size=(args.queries, 32)).astype("<f4")
        source = "synthetic-smoke-only"
    else:
        hdf5_path = args.output / "sift-128-euclidean.hdf5"
        request = urllib.request.Request(source, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=180) as response:
            with hdf5_path.open("wb") as output:
                while chunk := response.read(8 * 1024 * 1024):
                    output.write(chunk)
        with h5py.File(hdf5_path, "r") as dataset:
            if not 10 <= args.count <= len(dataset["train"]):
                raise ValueError("Requested vector count exceeds source dataset")
            if not 201 <= args.queries <= len(dataset["test"]):
                raise ValueError("Need 200 calibration queries and a held-out query set")
            base = np.asarray(dataset["train"][: args.count], dtype="<f4")
            queries = np.asarray(dataset["test"][: args.queries], dtype="<f4")
        hdf5_path.unlink()
    # Recompute exact ground truth for the selected subset; never filter the
    # full-million neighbor list, which would give an incorrect Recall@10.
    neighbors = np.empty((len(queries), 10), dtype="<u4")
    norms = np.einsum("ij,ij->i", base, base)
    for start in range(0, len(queries), 32):
        batch = queries[start : start + 32]
        distances = norms[None, :] - 2 * batch @ base.T
        distances += np.einsum("ij,ij->i", batch, batch)[:, None]
        ids = np.argpartition(distances, 9, axis=1)[:, :10]
        nearest = np.take_along_axis(distances, ids, axis=1)
        ids = np.take_along_axis(ids, np.argsort(nearest, axis=1), axis=1)
        neighbors[start : start + len(batch)] = ids
    write_matrix(args.output / "base.fbin", base)
    write_matrix(args.output / "queries.fbin", queries)
    write_matrix(args.output / "neighbors.ibin", neighbors)
    manifest = {
        "dataset": "synthetic-smoke" if args.synthetic else "SIFT-128 first-N subset",
        "source": source,
        "vectors": len(base),
        "dimension": base.shape[1],
        "dtype": "FP32",
        "metric": "squared Euclidean",
        "calibration_queries": 200,
        "evaluation_queries": len(queries) - 200,
        "ground_truth": "exhaustive squared Euclidean, recomputed for this subset",
        "sha256": {
            name: hashlib.file_digest((args.output / name).open("rb"), "sha256").hexdigest()
            for name in ("base.fbin", "queries.fbin", "neighbors.ibin")
        },
    }
    (args.output / "dataset.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
