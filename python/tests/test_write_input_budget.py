"""Reject binding copies before materialization, in an isolated process."""

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


def verify_statuses(path):
    zvec.init(memory_limit_mb=128, query_threads=2, optimize_threads=1)
    options = zvec.CollectionOption.__new__(zvec.CollectionOption)
    options.__setstate__((False, False, 1024 * 1024))
    schema = zvec.CollectionSchema(
        "status_budget", fields=[zvec.FieldSchema("value", zvec.DataType.INT64)]
    )
    coll = zvec.create_and_open(str(path), schema, option=options)
    held = []
    for _ in range(64):
        try:
            held.append(coll.delete(["missing"] * 1024))
        except MemoryError:
            break
    else:
        raise AssertionError("Retained write statuses must participate in admission")
    assert held
    held.clear()
    gc.collect()
    assert len(coll.delete(["missing"] * 1024)) == 1024
    assert coll.insert(zvec.Doc("small", fields={"value": 1})).ok()
    coll.close()


def verify_long_status_messages(path):
    zvec.init(memory_limit_mb=256, query_threads=2, optimize_threads=1)
    options = zvec.CollectionOption.__new__(zvec.CollectionOption)
    options.__setstate__((False, False, 1024 * 1024))
    schema = zvec.CollectionSchema(
        "status_message_budget", fields=[zvec.FieldSchema("value", zvec.DataType.INT64)]
    )
    coll = zvec.create_and_open(str(path), schema, option=options)
    # Reuse one caller string. Each failed delete produces an independently
    # owned native status message containing the complete key.
    key = "x" * (1024 * 1024)
    held = []
    for _ in range(48):
        try:
            held.append(coll.delete(key))
        except MemoryError:
            break
    else:
        raise AssertionError("Retained native status messages escaped admission")
    assert held and key in held[0].message()
    held.clear()
    gc.collect()
    assert key in coll.delete(key).message()
    coll.close()


def run_worker(tmp_path, operation, direct=False):
    completed = subprocess.run(
        [sys.executable, __file__, str(tmp_path / "db"), operation, str(int(direct))],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize("operation", ["insert", "update", "upsert"])
@pytest.mark.parametrize("direct", [False, True])
def test_write_input_admission_and_recovery(tmp_path, operation, direct):
    run_worker(tmp_path, operation, direct)


def test_unknown_length_field_iterable_is_admitted_incrementally(tmp_path):
    run_worker(tmp_path, "iterable")


def test_retained_write_statuses_reject_growth_and_release_for_retry(tmp_path):
    run_worker(tmp_path, "statuses")


def test_retained_long_write_status_messages_use_budget(tmp_path):
    run_worker(tmp_path, "long_status_messages")


if __name__ == "__main__":
    if sys.argv[2] == "iterable":
        verify_iterable(Path(sys.argv[1]))
    elif sys.argv[2] == "statuses":
        verify_statuses(Path(sys.argv[1]))
    elif sys.argv[2] == "long_status_messages":
        verify_long_status_messages(Path(sys.argv[1]))
    else:
        verify(Path(sys.argv[1]), sys.argv[2], bool(int(sys.argv[3])))
