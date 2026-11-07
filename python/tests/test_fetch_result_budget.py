"""Public Python result admission, isolated from process-global initialization."""

from concurrent.futures import ThreadPoolExecutor
import copy
import json
from pathlib import Path
import pickle
import subprocess
import sys
import weakref

import pytest
import zvec
from zvec.model._budgeted_result import _retain_result_memory


KINDS = (
    "VECTOR_FP16",
    "VECTOR_FP32",
    "VECTOR_FP64",
    "VECTOR_INT8",
    "SPARSE_VECTOR_FP16",
    "SPARSE_VECTOR_FP32",
)


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
        with pytest.raises(MemoryError, match="Python query results"):
            coll.query(
                zvec.Query("vector", vector=vector_for(kind, dimension, 0)),
                topk=count,
                include_vector=True,
            )
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


@pytest.mark.parametrize("kind", KINDS)
def test_vector_fetch_materialization(tmp_path, kind):
    path = tmp_path / "vectors"
    run_worker("build", path, kind, False, 64, 64)
    if kind != "VECTOR_FP64":
        run_worker("verify", path, kind, False, 64, 64)


@pytest.mark.parametrize("mmap", [False, True])
def test_large_vector_result_admission_and_recovery(tmp_path, mmap):
    path = tmp_path / "large"
    run_worker("build", path, "VECTOR_FP32", mmap, 1024, 512)
    run_worker("verify", path, "VECTOR_FP32", mmap, 1024, 512)


class Owner:
    pass


@pytest.mark.parametrize("payload", ["vector", "sparse", "text", "array_text"])
def test_extracted_payload_keeps_result_owner(payload):
    owner = Owner()
    reference = weakref.ref(owner)
    rows = {
        "a": (
            "a",
            1.0,
            {"text": "中文🙂", "array_text": ["long field"]},
            {"vector": [1.25, 2.5], "sparse": {2: 1.25}},
        )
    }
    _retain_result_memory(rows, owner)
    doc = zvec.Doc._from_tuple(rows["a"])
    value = doc.vectors[payload] if payload in doc.vectors else doc.fields[payload]
    if payload == "array_text":
        value = value[0]
    del doc, rows, owner
    assert reference() is not None
    assert json.loads(json.dumps(value)) is not None
    assert pickle.loads(pickle.dumps(value)) == value
    assert copy.deepcopy(value) == value
    del value
    assert reference() is None


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


def verify_native_fields(path, kind, mmap, dimension, count):
    zvec.init(memory_limit_mb=128, query_threads=2, optimize_threads=1)
    coll = zvec.open(str(path), option=zvec.CollectionOption(read_only=True))
    native = coll._obj.Fetch(["0"])["0"]
    data_type = getattr(zvec.DataType, kind)
    held = []
    for _ in range(64):
        try:
            held.append(native.get_any("vector", data_type))
        except MemoryError:
            break
    else:
        raise AssertionError("Native field copies must retain Python admission")
    assert held and held[0] == vector_for(kind, dimension, 0)
    del native
    coll.close()
    coll = zvec.open(str(path), option=zvec.CollectionOption(read_only=True))
    native = coll._obj.Fetch(["0"])["0"]
    with pytest.raises(MemoryError):
        native.get_any("vector", data_type)
    held.clear()
    assert native.get_any("vector", data_type) == vector_for(kind, dimension, 0)
    coll.close()


def test_native_field_copies_hold_and_release_budget(tmp_path):
    path = tmp_path / "native_fields"
    run_worker("build", path, "VECTOR_FP32", False, 8192, 2)
    run_worker("native_fields", path, "VECTOR_FP32", False, 8192, 2)


def verify_native_doc_state(path, kind, mmap, dimension, count):
    from zvec._zvec import _Doc

    zvec.init(memory_limit_mb=128, query_threads=2, optimize_threads=1)
    coll = zvec.open(str(path), option=zvec.CollectionOption(read_only=True))
    native = coll._obj.Fetch(["0"])["0"]
    key = native.pk()
    names = native.field_names()
    assert isinstance(key, str) and key == "0"
    assert isinstance(names, list) and "vector" in names
    field_name = next(name for name in names if name == "vector")
    del names

    def hold_until_rejected(copy):
        held = []
        for _ in range(96):
            try:
                held.append(copy())
            except MemoryError:
                break
        else:
            pytest.fail("Native document copies must retain admission")
        assert held
        with pytest.raises(MemoryError):
            copy()
        held.clear()
        return copy()

    state = hold_until_rejected(native.__getstate__)
    assert isinstance(state, bytes) and len(state) >= 256 * 1024
    assert pickle.loads(pickle.dumps(state)) == state
    plain_state = bytes(state)
    del state

    def restore():
        doc = _Doc.__new__(_Doc)
        doc.__setstate__(plain_state)
        return doc

    assert hold_until_rejected(restore).pk() == key
    assert pickle.loads(pickle.dumps(native)).get_any(
        "payload", zvec.DataType.STRING
    ).endswith("中文🙂0")
    assert key._memory_owner is not None
    assert field_name._memory_owner is not None
    coll.close()


def build_native_doc_state(path, kind, mmap, dimension, count):
    build(path, kind, mmap, dimension, count, payload_size=256 * 1024)


def test_native_doc_state_holds_and_releases_budget(tmp_path):
    path = tmp_path / "native_doc_state"
    run_worker("build_native_doc_state", path, "VECTOR_FP32", False, 8, 1)
    run_worker("native_doc_state", path, "VECTOR_FP32", False, 8, 1)


def verify_rerank_budget(path, kind, mmap, dimension, count):
    from zvec._zvec import _Doc, _reranker_rerank, _RrfParams
    from zvec.extension import RrfReRanker, WeightedReRanker

    zvec.init(memory_limit_mb=128, query_threads=2, optimize_threads=1)
    field = zvec.VectorSchema(
        "vector",
        zvec.DataType.VECTOR_FP32,
        dimension=8,
        index_param=zvec.FlatIndexParam(metric_type=zvec.MetricType.IP),
    )
    doc = zvec.Doc(id="k" * 1024, fields={"source": "first"})
    duplicate = zvec.Doc(id=doc.id, fields={"source": "second"})
    for reranker, fields in (
        (RrfReRanker(), None),
        (WeightedReRanker([1.0]), [field]),
    ):
        with pytest.raises(MemoryError):
            reranker.rerank([[doc] * 24_000], topn=1, fields=fields)
        result = reranker.rerank([[doc, duplicate]], topn=1, fields=fields)[0]
        assert result.id == doc.id
        assert result.fields == doc.fields
        assert doc.score is None

    native = _Doc()
    native.set_pk("native")
    native.set_score(0.5)
    with pytest.raises(MemoryError):
        _reranker_rerank(_RrfParams(), [[native] * 400_000], (), 1)
    assert native.score() == 0.5
    assert _reranker_rerank(_RrfParams(), [[native]], (), 1)[0].pk() == "native"

    reranker = RrfReRanker()
    wide = zvec.Doc(id="wide", fields={str(i): i for i in range(1024)})
    held = []
    for _ in range(96):
        try:
            held.append(reranker.rerank([[wide]], topn=1)[0].fields)
        except MemoryError:
            break
    else:
        pytest.fail("Extracted reranked fields must retain result admission")
    assert held and held[0] == wide.fields
    held.clear()
    assert reranker.rerank([[wide]], topn=1)[0].fields == wide.fields


def test_standalone_rerank_admission_and_recovery(tmp_path):
    run_worker("rerank_budget", tmp_path, "VECTOR_FP32", False, 8, 1)


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


def verify_fetch_inputs(path, kind, mmap, dimension, count):
    zvec.init(memory_limit_mb=128, query_threads=2, optimize_threads=1)
    coll = zvec.open(str(path), option=zvec.CollectionOption(read_only=True))

    # Input copies must be admitted even when every lookup misses. Repeating
    # one Python string also keeps the caller's own heap small in this test.
    oversized = ["missing"] * 200_000
    for fetch in (coll.fetch, coll._obj.Fetch):
        with pytest.raises(MemoryError):
            fetch(oversized)
        with pytest.raises(MemoryError):
            fetch(["0"], output_fields=["payload"] * 200_000)
        with pytest.raises(MemoryError):
            fetch(["x" * (8 * 1024 * 1024)])
        assert fetch(["0", "missing", "0"])["0"] is not None

    for include_vector in (False, True):
        # These input copies fit. Unique IDs additionally need hash nodes and
        # bucket growth, whereas duplicates require only one hash entry.
        assert (
            coll.fetch(["missing000000"] * 70_000, include_vector=include_vector) == {}
        )
        # With the shared allowance nearly full, either the next hash growth
        # or native batch admission can be the first to reject the request.
        with pytest.raises(MemoryError):
            coll.fetch(
                [f"missing{i:06d}" for i in range(70_000)],
                include_vector=include_vector,
            )
        assert (
            coll.fetch(
                [f"missing{i}" for i in range(2_000)], include_vector=include_vector
            )
            == {}
        )

    # Retaining tiny results must not retain each call's large input lease.
    held = [coll.fetch(["0"] * 10_000) for _ in range(16)]
    assert all(set(docs) == {"0"} for docs in held)
    held.clear()

    def recover(_):
        with pytest.raises(MemoryError):
            coll.fetch(oversized)
        return set(coll.fetch(["0", "1", "0"]))

    with ThreadPoolExecutor(max_workers=8) as workers:
        assert all(keys == {"0", "1"} for keys in workers.map(recover, range(16)))
    coll.close()


def verify_fetch_iterables(path, kind, mmap, dimension, count):
    zvec.init(memory_limit_mb=128, query_threads=2, optimize_threads=1)
    coll = zvec.open(str(path), option=zvec.CollectionOption(read_only=True))
    for fetch in (coll.fetch, coll._obj.Fetch):
        docs = fetch(("0", b"1", bytearray(b"0"), "中文🙂\0missing"))
        assert {key for key, doc in docs.items() if doc is not None} == {"0", "1"}
        assert set(fetch((str(i % 8) for i in range(1_024)))) == set(map(str, range(8)))
        assert set(fetch(["0"], output_fields=(name for name in ["payload"]))) == {"0"}

        def reentrant_ids():
            # Input iteration must not hold the materialization mutex while
            # executing caller code, which may itself fetch from the database.
            assert set(fetch(["1"])) == {"1"}
            yield "0"

        assert set(fetch(reentrant_ids())) == {"0"}

        consumed = 0

        def oversized_ids():
            nonlocal consumed
            for _ in range(200_000):
                consumed += 1
                yield "missing"

        with pytest.raises(MemoryError):
            fetch(oversized_ids())
        assert consumed < 200_000

        def failing_ids():
            yield "0"
            raise RuntimeError("injected input iteration failure")

        with pytest.raises(RuntimeError, match="input iteration failure"):
            fetch(failing_ids())
        with pytest.raises(TypeError):
            fetch([object()])
        assert set(fetch(["0"])) == {"0"}
    coll.close()


@pytest.mark.parametrize("mode", ["inputs", "iterables"])
def test_fetch_input_and_workspace_admission(tmp_path, mode):
    path = tmp_path / mode
    run_worker("build", path, "VECTOR_FP32", False, 64, 8)
    run_worker(mode, path, "VECTOR_FP32", False, 64, 8)


def verify_fetch_dict_results(path, kind, mmap, dimension, count):
    zvec.init(memory_limit_mb=128, query_threads=2, optimize_threads=1)
    coll = zvec.open(str(path), option=zvec.CollectionOption(read_only=True))
    fetch = coll._obj.Fetch
    missing = [f"missing{i:06d}" for i in range(2_048)]
    held = []
    for _ in range(64):
        try:
            result = fetch(missing)
        except MemoryError:
            break
        assert len(result) == len(missing)
        assert all(doc is None for doc in result.values())
        held.append(tuple(result))
        del result
    else:
        pytest.fail("Keys extracted from missing-ID results must retain admission")
    assert held
    held.clear()
    result = fetch(["0", "missing"])
    assert result["0"] is not None and result["missing"] is None
    for include_vector in (False, True):
        docs = coll.fetch(["0"], include_vector=include_vector)
        key = next(iter(docs))
        assert type(docs) is dict
        assert key is docs["0"].id
        assert key._memory_owner is not None
    coll.close()


def test_fetch_dict_keys_hold_and_release_budget(tmp_path):
    path = tmp_path / "fetch_dict"
    run_worker("build", path, "VECTOR_FP32", False, 64, 8)
    run_worker("fetch_dict", path, "VECTOR_FP32", False, 64, 8)


def verify_id_results(path, kind, mmap, dimension, count):
    zvec.init(memory_limit_mb=128, query_threads=2, optimize_threads=1)
    coll = zvec.open(str(path), option=zvec.CollectionOption(read_only=True))
    with pytest.raises(MemoryError):
        coll.resolve_internal_ids([-1] * 1_000_000)
    assert coll.resolve_internal_ids([]) == []
    assert coll.resolve_internal_ids([-1, 0, 9999]) == [None, "0", None]
    held = []
    rejected = False
    query = zvec.Query("vector", vector=vector_for(kind, dimension, 0))
    for _ in range(32):
        try:
            ids, scores = coll.query_internal_ids(
                query, topk=100_000, return_scores=True
            )
        except MemoryError:
            rejected = True
            break
        # A view of only the scores must retain the shared native owner.
        held.append(scores[::2])
        del ids, scores
    assert rejected, "Caller-held arrays must participate in admission"
    coll.close()
    coll = zvec.open(str(path), option=zvec.CollectionOption(read_only=True))
    with pytest.raises(MemoryError):
        coll.query_internal_ids(query, topk=100_000, return_scores=True)
    held.clear()
    assert coll.query_internal_ids(query, topk=100_000).shape == (100_000,)

    # The outer sequence needs an owner even when every element is None.
    held = []
    rejected = False
    for _ in range(32):
        try:
            held.append(coll.resolve_internal_ids([-1] * 20_000))
        except MemoryError:
            rejected = True
            break
    assert rejected, "Retained all-missing ID results must consume admission"
    held.clear()
    assert coll.resolve_internal_ids([-1] * 20_000) == [None] * 20_000
    for _ in range(32):
        try:
            # An extracted key keeps the admission after its outer list dies.
            held.append(coll.resolve_internal_ids([0] * 20_000)[0])
        except MemoryError:
            break
    else:
        pytest.fail("Extracted resolved keys must retain admission")
    assert all(value == "0" for value in held)
    held.clear()

    def reentrant_ids():
        yield coll.query_internal_ids(query, topk=1)[0]
        yield -1

    assert coll.resolve_internal_ids(reentrant_ids())[1] is None
    with pytest.raises(TypeError):
        coll.resolve_internal_ids([object()])
    assert coll.resolve_internal_ids([0]) == ["0"]
    with ThreadPoolExecutor(max_workers=4) as workers:
        assert all(
            result == ["0", None]
            for result in workers.map(
                lambda _: coll.resolve_internal_ids([0, -1]), range(8)
            )
        )
    coll.close()


def verify_document_results(path, kind, mmap, dimension, count, *, grouped):
    zvec.init(memory_limit_mb=128, query_threads=2, optimize_threads=1)
    coll = zvec.open(str(path), option=zvec.CollectionOption(read_only=True))
    held = []
    rejected = False

    def take_results():
        if grouped:
            groups = coll.group_by_query(
                zvec.Query("vector", vector=vector_for(kind, dimension, 0)),
                "payload",
                group_count=count,
                topk_per_group=1,
                include_vector=True,
            )
            return [doc.vector("vector") for group in groups for doc in group.docs]
        with coll.iter_docs() as docs:
            return [doc.vector("vector") for doc in docs]

    for _ in range(32):
        try:
            held.extend(take_results())
        except MemoryError:
            rejected = True
            break
    assert rejected, "Escaping Python vectors must retain result admission"
    assert held
    coll.close()
    assert len(held[0]) == dimension
    coll = zvec.open(str(path), option=zvec.CollectionOption(read_only=True))
    with pytest.raises(MemoryError):
        take_results()
    held.clear()
    assert len(take_results()) == count
    coll.close()


def verify_iterator_results(*args):
    verify_document_results(*args, grouped=False)


def verify_group_results(*args):
    verify_document_results(*args, grouped=True)


@pytest.mark.parametrize("mode", ["id_results", "iterator_results", "group_results"])
def test_remaining_result_paths_hold_and_release_budget(tmp_path, mode):
    path = tmp_path / mode
    run_worker("build", path, "VECTOR_FP32", False, 1024, 64)
    run_worker(mode, path, "VECTOR_FP32", False, 1024, 64)


def build_group_keys(*args):
    build(*args, payload_size=128 * 1024)


def verify_group_keys(path, kind, mmap, dimension, count):
    zvec.init(memory_limit_mb=128, query_threads=2, optimize_threads=1)
    coll = zvec.open(str(path), option=zvec.CollectionOption(read_only=True))

    def keys():
        return [
            group.group_by_value
            for group in coll.group_by_query(
                zvec.Query("vector", vector=vector_for(kind, dimension, 0)),
                "payload",
                group_count=count,
                topk_per_group=1,
                output_fields=[],
            )
        ]

    held = []
    for _ in range(64):
        try:
            held.extend(keys())
        except MemoryError:
            break
    else:
        pytest.fail("Extracted group keys must retain admission")
    assert held and all(len(key) >= 128 * 1024 for key in held)
    held.clear()
    assert len(keys()) == count
    coll.close()


def test_group_keys_hold_and_release_budget(tmp_path):
    path = tmp_path / "group_keys"
    run_worker("build_group_keys", path, "VECTOR_FP32", False, 8, 4)
    run_worker("group_keys", path, "VECTOR_FP32", False, 8, 4)


if __name__ == "__main__":
    action, target, vector_kind, use_mmap, dim, rows = sys.argv[1:]
    worker = {
        "build": build,
        "build_group_keys": build_group_keys,
        "group_keys": verify_group_keys,
        "verify": verify,
        "retained": verify_retained,
        "native_fields": verify_native_fields,
        "build_native_doc_state": build_native_doc_state,
        "native_doc_state": verify_native_doc_state,
        "rerank_budget": verify_rerank_budget,
        "exception": verify_exception,
        "inputs": verify_fetch_inputs,
        "iterables": verify_fetch_iterables,
        "fetch_dict": verify_fetch_dict_results,
        "id_results": verify_id_results,
        "iterator_results": verify_iterator_results,
        "group_results": verify_group_results,
    }[action]
    worker(Path(target), vector_kind, bool(int(use_mmap)), int(dim), int(rows))
