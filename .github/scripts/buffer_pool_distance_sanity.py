# Copyright 2025-present the zvec project
# SPDX-License-Identifier: Apache-2.0
"""Reject inactive runtime-dispatched kernels before comparing performance."""

from __future__ import annotations

import math
from pathlib import Path
from tempfile import TemporaryDirectory

import zvec


def run():
    dimension = 128
    zvec.init(memory_limit_mb=512, query_threads=1, optimize_threads=1)
    schema = zvec.CollectionSchema(
        name="distance_sanity",
        vectors=[
            zvec.VectorSchema(
                "v",
                zvec.DataType.VECTOR_FP32,
                dimension=dimension,
                index_param=zvec.FlatIndexParam(metric_type=zvec.MetricType.L2),
            )
        ],
    )
    with TemporaryDirectory(prefix="zvec-distance-sanity-") as directory:
        collection = zvec.create_and_open(
            str(Path(directory) / "db"),
            schema,
            zvec.CollectionOption(enable_mmap=True),
        )
        try:
            assert all(
                status.ok()
                for status in collection.insert(
                    [
                        zvec.Doc(id="one", vectors={"v": [1.0] * dimension}),
                        zvec.Doc(id="two", vectors={"v": [2.0] * dimension}),
                    ]
                )
            )
            collection.flush()
            for value in (0.0, 0.5):
                hits = collection.query(
                    zvec.Query("v", vector=[value] * dimension),
                    topk=2,
                    output_fields=[],
                    include_vector=False,
                )
                assert [hit.id for hit in hits] == ["one", "two"]
                expected = [
                    dimension * (candidate - value) ** 2 for candidate in (1, 2)
                ]
                assert all(
                    math.isclose(hit.score, score, rel_tol=1e-6, abs_tol=1e-6)
                    for hit, score in zip(hits, expected, strict=True)
                ), ([(hit.id, hit.score) for hit in hits], expected)
        finally:
            collection.close()


if __name__ == "__main__":
    run()
