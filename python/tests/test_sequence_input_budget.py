"""Budget the actual Python sequence consumed by write/query conversion."""

from array import array
import gc
from pathlib import Path
import subprocess
import sys
import tracemalloc

import numpy as np
import pytest
import zvec


def verify(path, case):
    zvec.init(memory_limit_mb=128, query_threads=2, optimize_threads=1)
    options = zvec.CollectionOption.__new__(zvec.CollectionOption)
    options.__setstate__((False, False, 1024 * 1024))
    schema = zvec.CollectionSchema(
        "sequence_budget",
        fields=[
            zvec.FieldSchema("values", zvec.DataType.ARRAY_STRING),
            zvec.FieldSchema("group", zvec.DataType.INT64),
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
    if case in ("field_name", "vector_name"):
        name = "文" * (4 * 1024 * 1024)
        before = sys.getsizeof(name)
        doc = zvec.Doc(
            "large",
            **(
                {"fields": {name: 1}}
                if case == "field_name"
                else {"vectors": {name: [1, 2, 3, 4]}}
            ),
        )
        with pytest.raises(MemoryError):
            coll.insert(doc)
        assert sys.getsizeof(name) == before  # No persistent UTF-8 cache.
        del doc, name
    elif case == "write":
        payload = "x" * (1024 * 1024)

        class Expanded(list):
            consumed = 0

            def __iter__(self):
                for _ in range(128):
                    self.consumed += 1
                    yield payload

        values = Expanded([""])
        with pytest.raises(MemoryError):
            coll.insert(zvec.Doc("large", fields={"values": values, "group": 1}))
        assert values.consumed < 128
    else:
        field = "dense"
        if case == "buffer":
            vector = memoryview(array("d", [1]) * 600_000)
        elif case == "sparse":

            class ShortDict(dict):
                def __len__(self):
                    return 1

            field = "sparse"
            vector = ShortDict.fromkeys(range(120_000), 1.0)
        else:
            vector = range(4_000_000)
        query = zvec.Query(field, vector=vector)
        measure_range = case in ("normal", "internal", "group")
        if measure_range:
            tracemalloc.start()
        with pytest.raises(MemoryError):
            if case == "internal":
                coll.query_internal_ids(query)
            elif case == "group":
                coll.group_by_query(query, group_by_field_name="group")
            else:
                coll.query(query)
        if measure_range:
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            assert peak < 1024 * 1024  # Reject before expanding the range.
        del query, vector
    gc.collect()
    assert coll.stats.doc_count == 0

    class OnceList(list):
        def __len__(self):
            raise AssertionError("Admission must not trust overridden len")

        def __iter__(self):
            assert not getattr(self, "consumed", False)
            self.consumed = True
            yield "actual"

    class NativeDict(dict):
        def __len__(self):
            raise AssertionError("Sparse casts traverse native dict entries")

    assert coll.insert(
        zvec.Doc(
            "small",
            fields={"values": OnceList(["ignored"]), "group": 1},
            vectors={"dense": [1, 2, 3, 4], "sparse": NativeDict({1: 1.0, 3: 3.0})},
        )
    ).ok()
    assert coll.fetch("small")["small"].field("values") == ["actual"]

    class OnceTuple(tuple):
        def __len__(self):
            return 1

        def __iter__(self):
            assert not getattr(self, "consumed", False)
            self.consumed = True
            yield from (1, 2, 3, 4)

    # Preserve small sequences, buffers and nested flattening. Subclasses are
    # normalized once, so estimator and NumPy cannot consume different data.
    for vector in (
        range(1, 5),
        OnceTuple((99,)),
        [range(1, 5)],
        memoryview(array("f", [1, 2, 3, 4])),
    ):
        hits = coll.query(zvec.Query("dense", vector=vector), topk=1)
        assert len(hits) == 1 and hits[0].id == "small"

    class OpaqueArray:
        def __len__(self):
            return 4

        def __array__(self, *args, **kwargs):
            raise AssertionError("Opaque conversion must not run before admission")

    with pytest.raises(TypeError, match="buffer or numeric sequence"):
        coll.query(zvec.Query("dense", vector=OpaqueArray()))

    class OpaqueInterface:
        __array_interface__ = {}

        def __len__(self):
            return 4

    with pytest.raises(TypeError, match="buffer or numeric sequence"):
        coll.query(zvec.Query("dense", vector=OpaqueInterface()))
    with pytest.raises(TypeError, match="buffer or numeric sequence"):
        coll.query(zvec.Query("dense", vector=[OpaqueInterface()]))
    hits = coll.query(zvec.Query("dense", vector=[np.float32(x) for x in range(1, 5)]))
    assert hits[0].id == "small"
    coll.close()


@pytest.mark.parametrize(
    "case",
    [
        "write",
        "field_name",
        "vector_name",
        "normal",
        "internal",
        "group",
        "buffer",
        "sparse",
    ],
)
def test_sequence_admission_and_recovery(tmp_path, case):
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), str(tmp_path / "db"), case],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    verify(sys.argv[1], sys.argv[2])
