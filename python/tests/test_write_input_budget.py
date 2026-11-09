"""Reject write conversion before mutation and release admission for retry."""

import gc
from pathlib import Path
import subprocess
import sys

import pytest
import zvec


def verify(path, operation, direct):
    from zvec.model.convert import convert_to_cpp_doc

    zvec.init(memory_limit_mb=128, query_threads=2, optimize_threads=1)
    options = zvec.CollectionOption.__new__(zvec.CollectionOption)
    options.__setstate__((False, False, 1024 * 1024))
    schema = zvec.CollectionSchema(
        "write_budget", fields=[zvec.FieldSchema("payload", zvec.DataType.STRING)]
    )
    coll = zvec.create_and_open(str(path), schema, option=options)
    # The caller shares ONE large string; all batch expansion belongs to zvec.
    doc = zvec.Doc("large", fields={"payload": "文" * (512 * 1024)})
    batch = [doc] * 64
    if direct:
        native = convert_to_cpp_doc(doc, schema)
        batch = [native] * 64
        call = getattr(coll._obj, operation.capitalize())
    else:
        call = getattr(coll, operation)
    try:
        call(batch)
    except MemoryError:
        pass
    else:
        raise AssertionError("Binding copies must be rejected before writing")
    gc.collect()
    assert coll.stats.doc_count == 0
    small = zvec.Doc("small", fields={"payload": "中文🙂"})
    assert coll.insert(small).ok()
    assert getattr(coll, operation)([small])[0].ok() == (operation != "insert")
    assert coll.fetch("small")["small"].field("payload") == "中文🙂"
    # Streaming Python document inputs must also retain incremental admission.
    assert coll.insert(
        zvec.Doc("stream" + str(i), fields={"payload": "ok"}) for i in range(3)
    )
    coll.close()



def verify_iterable(path):
    zvec.init(memory_limit_mb=128, query_threads=2, optimize_threads=1)
    options = zvec.CollectionOption.__new__(zvec.CollectionOption)
    options.__setstate__((False, False, 1024 * 1024))
    schema = zvec.CollectionSchema(
        "array_budget", fields=[zvec.FieldSchema("values", zvec.DataType.ARRAY_INT64)]
    )
    coll = zvec.create_and_open(str(path), schema, option=options)
    # No len() and no pre-existing list: the binding must admit incremental
    # staging instead of letting the STL caster consume the entire generator.
    source = (i for i in range(1_000_000))
    try:
        coll.insert(zvec.Doc("large", fields={"values": source}))
    except MemoryError:
        pass
    else:
        raise AssertionError("Unbounded iterable conversion must be budgeted")
    assert next(source) < 1_000_000
    gc.collect()
    assert coll.insert(zvec.Doc("small", fields={"values": iter([1, 2, 3])})).ok()
    assert coll.fetch("small")["small"].field("values") == [1, 2, 3]
    coll.close()



def run_worker(tmp_path, operation, direct=False):
    completed = subprocess.run(
        [sys.executable, __file__, str(tmp_path / "db"), operation, str(int(direct))],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr



@pytest.mark.parametrize("direct", [False, True])
def test_write_input_admission_and_recovery(tmp_path, direct):
    run_worker(tmp_path, "insert", direct)



def test_unknown_length_field_iterable_is_admitted_incrementally(tmp_path):
    run_worker(tmp_path, "iterable")



if __name__ == "__main__":
    if sys.argv[2] == "iterable":
        verify_iterable(Path(sys.argv[1]))
    else:
        verify(Path(sys.argv[1]), sys.argv[2], bool(int(sys.argv[3])))
