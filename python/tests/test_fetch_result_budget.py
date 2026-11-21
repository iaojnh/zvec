"""Key public result admission and ownership regressions."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys

import pytest
import zvec


def vector_for(kind, dimension, row):
    values = [float((i + row) % 5 - 2) for i in range(dimension)]
    if kind == "VECTOR_INT8":
        return [int(v) for v in values]
    if kind.startswith("SPARSE"):
        return {i: v for i, v in enumerate(values) if v}
    return values



def build(path, kind, mmap, dimension, count, *, payload_size=0):
    zvec.init(memory_limit_mb=512, query_threads=2, optimize_threads=1)
    schema = zvec.CollectionSchema(
        name="result_budget",
        fields=[zvec.FieldSchema("payload", zvec.DataType.STRING)],
        vectors=[
            zvec.VectorSchema(
                "vector",
                getattr(zvec.DataType, kind),
                dimension=0 if kind.startswith("SPARSE") else dimension,
                index_param=zvec.FlatIndexParam(metric_type=zvec.MetricType.IP),
            )
        ],
    )
    if kind == "VECTOR_FP64":
        # This enum is exposed, but public Collection schemas reject it.
        with pytest.raises(ValueError, match="Invalid schema.*VECTOR_FP64"):
            zvec.create_and_open(
                str(path), schema, option=zvec.CollectionOption(enable_mmap=mmap)
            )
        return
    coll = zvec.create_and_open(
        str(path), schema, option=zvec.CollectionOption(enable_mmap=mmap)
    )
    for first in range(0, count, 8):
        docs = [
            zvec.Doc(
                id=str(i),
                fields={"payload": "x" * payload_size + "中文🙂" + str(i)},
                vectors={"vector": vector_for(kind, dimension, i)},
            )
            for i in range(first, min(first + 8, count))
        ]
        assert all(status.ok() for status in coll.insert(docs))
    coll.flush()
    coll.close()



def verify(path, kind, mmap, dimension, count):
    zvec.init(memory_limit_mb=128, query_threads=2, optimize_threads=1)
    coll = zvec.open(
        str(path), option=zvec.CollectionOption(read_only=True, enable_mmap=mmap)
    )

    def small_fetch(first):
        ids = [str(i) for i in range(first, first + 8)]
        docs = coll.fetch(ids + [ids[0], "missing"])
        assert set(docs) == set(ids)
        for key, doc in docs.items():
            assert doc.fields == {"payload": "中文🙂" + key}
            assert doc.vector("vector") == vector_for(kind, dimension, int(key))
        return len(docs)

    ids = [str(i) for i in range(count)]
    if dimension == 1024 and not mmap:
        # Native FP32 data is only 2MiB. Python boxes/slots exceed the
        # result allowance and must fail before accumulating all chunks.
        with pytest.raises(MemoryError, match="Python fetch results"):
            coll.fetch(ids)
    else:
        assert len(coll.fetch(ids)) == count
    assert len(coll.fetch(ids, include_vector=False)) == count
    assert coll.fetch([]) == {}
    # A rejected call must release the materialization slot. Exercise several
    # chunks, eight callers, deduplication and unchanged values after failure.
    with ThreadPoolExecutor(max_workers=8) as workers:
        assert sum(workers.map(small_fetch, [0, 8, 16, 24] * 2)) == 64
    held = coll.fetch(["0"])
    coll.close()
    assert held["0"].vector("vector") == vector_for(kind, dimension, 0)



def run_worker(mode, path, kind, mmap, dimension, count):
    completed = subprocess.run(
        [
            sys.executable,
            __file__,
            mode,
            str(path),
            kind,
            str(int(mmap)),
            str(dimension),
            str(count),
        ],
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr



def test_large_vector_result_admission_and_recovery(tmp_path):
    mmap = False
    path = tmp_path / "large"
    run_worker("build", path, "VECTOR_FP32", mmap, 1024, 512)
    run_worker("verify", path, "VECTOR_FP32", mmap, 1024, 512)



def verify_query_result_copy(path, kind, mmap, dimension, count):
    # Give native decoding enough headroom so this test reaches Python-copy
    # admission on each allocator/platform, rather than a prior native denial.
    zvec.init(memory_limit_mb=512, query_threads=2, optimize_threads=1)
    coll = zvec.open(str(path), zvec.CollectionOption(read_only=True, enable_mmap=mmap))
    try:
        with pytest.raises(MemoryError, match="Python query results"):
            coll.query(
                zvec.Query("vector", vector=vector_for(kind, dimension, 0)),
                topk=count,
                include_vector=True,
            )
        hits = coll.query(
            zvec.Query("vector", vector=vector_for(kind, dimension, 0)),
            topk=1,
            include_vector=True,
        )
        assert len(hits) == 1
        assert hits[0].vector("vector") == vector_for(kind, dimension, int(hits[0].id))
    finally:
        coll.close()



def test_large_query_result_copy_admission_and_recovery(tmp_path):
    path = tmp_path / "large_query"
    run_worker("build", path, "VECTOR_FP32", False, 4096, 2048)
    run_worker("query_copy", path, "VECTOR_FP32", False, 4096, 2048)



def verify_retained(path, kind, mmap, dimension, count):
    zvec.init(memory_limit_mb=128, query_threads=2, optimize_threads=1)
    coll = zvec.open(str(path), option=zvec.CollectionOption(read_only=True))
    held = []
    rejected = False
    for _ in range(32):
        try:
            docs = coll.fetch([str(i) for i in range(32)])
        except MemoryError:
            rejected = True
            break
        # Destroy the outer dict and all Docs; the extracted vectors must
        # still retain admission, including across Collection close/reopen.
        held.extend(docs[str(i)].vector("vector") for i in range(32))
        del docs
    assert rejected, "Caller-held payloads must participate in admission"
    coll.close()
    assert held[0] == vector_for(kind, dimension, 0)
    coll = zvec.open(str(path), option=zvec.CollectionOption(read_only=True))
    with pytest.raises(MemoryError):
        coll.fetch([str(i) for i in range(32)])
    held.clear()
    assert len(coll.fetch([str(i) for i in range(32)])) == 32
    coll.close()



def test_retained_vectors_reject_growth_and_release_for_retry(tmp_path):
    path = tmp_path / "retained"
    run_worker("build", path, "VECTOR_FP32", False, 1024, 64)
    run_worker("retained", path, "VECTOR_FP32", False, 1024, 64)



def verify_exception(path, kind, mmap, dimension, count):
    from zvec.model import _budgeted_result

    zvec.init(memory_limit_mb=128, query_threads=2, optimize_threads=1)
    coll = zvec.open(str(path), option=zvec.CollectionOption(read_only=True))
    original = _budgeted_result._retain_result_memory

    def fail_after_wrapping(results, owner):
        original(results, owner)
        raise RuntimeError("injected post-wrap failure")

    _budgeted_result._retain_result_memory = fail_after_wrapping
    captured = None
    try:
        coll.fetch(["0"])
    except RuntimeError as error:
        captured = error
    finally:
        _budgeted_result._retain_result_memory = original
    assert captured is not None
    # The traceback still owns the capsule. It must not own the producer's
    # mutex, or a second thread will hang waiting for materialization.
    with ThreadPoolExecutor(max_workers=1) as workers:
        assert len(workers.submit(coll.fetch, ["0"]).result(timeout=10)) == 1
    captured = None
    coll.close()



def test_failed_wrapper_does_not_keep_materialization_lock(tmp_path):
    path = tmp_path / "exception"
    run_worker("build", path, "VECTOR_FP32", False, 64, 8)
    run_worker("exception", path, "VECTOR_FP32", False, 64, 8)



if __name__ == "__main__":
    action, target, vector_kind, use_mmap, dim, rows = sys.argv[1:]
    worker = {"build": build, "verify": verify,
              "query_copy": verify_query_result_copy,
              "retained": verify_retained, "exception": verify_exception}[action]
    worker(Path(target), vector_kind, bool(int(use_mmap)), int(dim), int(rows))
