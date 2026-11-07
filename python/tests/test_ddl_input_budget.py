"""Column DDL rejects large input copies and remains usable afterwards."""

from pathlib import Path
import subprocess
import sys

import pytest
import zvec


def verify(path):
    zvec.init(memory_limit_mb=128, query_threads=2, optimize_threads=1)
    options = zvec.CollectionOption.__new__(zvec.CollectionOption)
    options.__setstate__((False, False, 1024 * 1024))
    schema = zvec.CollectionSchema(
        "ddl_budget", fields=[zvec.FieldSchema("value", zvec.DataType.INT64)]
    )
    coll = zvec.create_and_open(str(path), schema, option=options)
    with pytest.raises(MemoryError):
        coll.add_column(
            zvec.FieldSchema("denied", zvec.DataType.INT64),
            expression="1" + " " * (16 * 1024 * 1024),
        )
    assert coll.schema.field("denied") is None
    # Persist multiple segments to exercise concurrent preparation, decoding,
    # scalar expansion, casts and nullable arrays through the managed pool.
    for segment in range(3):
        assert all(
            status.ok()
            for status in coll.insert(
                [
                    zvec.Doc(str(segment * 10 + i), fields={"value": i})
                    for i in range(10)
                ]
            )
        )
        coll.flush()
    coll.add_column(
        zvec.FieldSchema("empty", zvec.DataType.DOUBLE, nullable=True),
        option=zvec.AddColumnOption(concurrency=2),
    )
    coll.add_column(
        zvec.FieldSchema("computed", zvec.DataType.FLOAT),
        expression="value + 2",
        option=zvec.AddColumnOption(concurrency=2),
    )
    coll.alter_column(
        "value",
        field_schema=zvec.FieldSchema("value", zvec.DataType.DOUBLE),
        option=zvec.AlterColumnOption(concurrency=2),
    )
    coll.close()
    coll = zvec.open(str(path), option=options)
    doc = coll.fetch("12")["12"]
    assert doc.field("value") == 2.0
    assert doc.field("computed") == 4.0
    assert doc.field("empty") is None
    coll.close()


def test_ddl_admission_and_managed_column_operations(tmp_path):
    completed = subprocess.run(
        [sys.executable, __file__, str(tmp_path / "db")],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


if __name__ == "__main__":
    verify(Path(sys.argv[1]))
