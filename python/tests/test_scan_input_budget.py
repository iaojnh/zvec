"""Iterator and filter-delete inputs must be admitted before binding copies."""

from pathlib import Path
import subprocess
import sys

import pytest
import zvec


def verify(path, operation):
    zvec.init(memory_limit_mb=128, query_threads=2, optimize_threads=1)
    options = zvec.CollectionOption.__new__(zvec.CollectionOption)
    options.__setstate__((False, False, 1024 * 1024))
    schema = zvec.CollectionSchema(
        "scan_budget", fields=[zvec.FieldSchema("payload", zvec.DataType.STRING)]
    )
    coll = zvec.create_and_open(str(path), schema, option=options)
    if operation == "iterate":
        with pytest.raises(MemoryError):
            coll.iter_docs(output_fields=["payload"] * 4_000_000)
    else:
        value = "payload = '" + "文" * (8 * 1024 * 1024) + "'"
        original_size = sys.getsizeof(value)
        with pytest.raises(MemoryError):
            coll.delete_by_filter(value)
        assert sys.getsizeof(value) == original_size
        del value
    assert coll.stats.doc_count == 0
    assert coll.insert(zvec.Doc("one", fields={"payload": "中文🙂"})).ok()
    with coll.iter_docs(output_fields=(name for name in ["payload"])) as docs:
        rows = list(docs)
        assert len(rows) == 1 and rows[0].field("payload") == "中文🙂"
    coll.delete_by_filter("payload = '中文🙂'")
    assert coll.stats.doc_count == 0
    coll.close()


@pytest.mark.parametrize("operation", ["iterate", "delete"])
def test_scan_input_admission_and_recovery(tmp_path, operation):
    completed = subprocess.run(
        [sys.executable, __file__, str(tmp_path / "db"), operation],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


if __name__ == "__main__":
    verify(Path(sys.argv[1]), sys.argv[2])
