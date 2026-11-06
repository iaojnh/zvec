"""Managed scans and nested filters work with one admitted CPU/I/O worker."""

from pathlib import Path
import subprocess
import sys

import zvec


def verify(path):
    zvec.init(memory_limit_mb=128, query_threads=1, optimize_threads=1)
    options = zvec.CollectionOption.__new__(zvec.CollectionOption)
    options.__setstate__((False, False, 1024 * 1024))
    schema = zvec.CollectionSchema(
        "executor_budget",
        fields=[zvec.FieldSchema("value", zvec.DataType.INT64)],
        vectors=[
            zvec.VectorSchema(
                "vector",
                zvec.DataType.VECTOR_FP32,
                2,
                index_param=zvec.FlatIndexParam(),
            )
        ],
    )
    coll = zvec.create_and_open(str(path), schema, option=options)
    for segment in range(2):
        assert all(
            status.ok()
            for status in coll.insert(
                [
                    zvec.Doc(
                        str(segment * 10 + row),
                        fields={"value": row},
                        vectors={"vector": [float(row), 1.0]},
                    )
                    for row in range(10)
                ]
            )
        )
        coll.flush()
    # An unindexed scalar filter creates a nested forward-bitmap plan before
    # vector recall. It must not wait for the outer plan's sole CPU worker.
    for _ in range(3):
        hits = coll.query(
            zvec.Query("vector", vector=[7.0, 1.0]),
            filter="value >= 5",
            topk=20,
            output_fields=["value"],
        )
        assert len(hits) == 10
        assert all(hit.field("value") >= 5 for hit in hits)
        rows = list(coll.iter_docs(output_fields=["value"]))
        assert len(rows) == 20
    coll.close()


def test_single_worker_scan_and_nested_filter(tmp_path):
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), str(tmp_path / "db")],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    verify(sys.argv[1])
