"""Public query preparation admits copies before NumPy and STL conversion."""

from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import zvec


def verify(path, case):
    zvec.init(memory_limit_mb=128, query_threads=2, optimize_threads=1)
    options = zvec.CollectionOption.__new__(zvec.CollectionOption)
    options.__setstate__((False, False, 1024 * 1024))
    schema = zvec.CollectionSchema(
        "query_budget",
        fields=[
            zvec.FieldSchema(
                "group", zvec.DataType.INT64, index_param=zvec.InvertIndexParam()
            )
        ],
        vectors=[
            zvec.VectorSchema(
                "dense", zvec.DataType.VECTOR_FP32, 4, index_param=zvec.FlatIndexParam()
            ),
            zvec.VectorSchema(
                "sparse",
                zvec.DataType.SPARSE_VECTOR_FP32,
                index_param=zvec.FlatIndexParam(),
            ),
        ],
    )
    coll = zvec.create_and_open(str(path), schema, option=options)
    query = zvec.Query("dense", vector=[1.0, 2.0, 3.0, 4.0])
    if case == "routes":
        with pytest.raises(MemoryError):
            coll.query([query] * 32, filter=" " * (1024 * 1024))
    else:
        large = zvec.Query("dense", vector=np.zeros(1_000_000, dtype=np.float64))
        with pytest.raises(MemoryError):
            coll.query(large)

    assert coll.insert(
        zvec.Doc(
            "one",
            fields={"group": 1},
            vectors={
                "dense": [1.0, 2.0, 3.0, 4.0],
                "sparse": {1: 2.0, 3: 4.0},
            },
        )
    ).ok()
    for field in ("dense", "sparse"):
        result = coll.query(zvec.Query(field, id="one"), topk=1)
        assert len(result) == 1 and result[0].id == "one"
    result = coll.query(
        (q for q in [query]), topk=1, output_fields=(name for name in ["group"])
    )
    assert result[0].field("group") == 1

    coll.close()


@pytest.mark.parametrize(
    "case", ["routes", "normal"]
)
def test_query_preparation_admission_and_recovery(tmp_path, case):
    completed = subprocess.run(
        [sys.executable, __file__, str(tmp_path / "db"), case],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


if __name__ == "__main__":
    verify(Path(sys.argv[1]), sys.argv[2])
