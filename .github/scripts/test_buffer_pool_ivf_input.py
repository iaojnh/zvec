# Copyright 2025-present the zvec project
# SPDX-License-Identifier: Apache-2.0
"""The bounded fixture must preserve IDs, cosine truth and source integrity."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    examples = Path(__file__).resolve().parents[2] / "examples/python"
    prep = module("ivf_input_test_prep", examples / "ivf_cohere_input.py")
    monkeypatch.setitem(sys.modules, "zvec", SimpleNamespace())
    public = module("ivf_input_test_public", examples / "ivf_memory_profile.py")
    raw = np.random.default_rng(19).normal(size=(12, 7))
    ids = np.random.default_rng(20).permutation(np.arange(12) * 7 + 1000)
    queries = raw[[2, 6, 9]]
    unit = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    q_unit = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    order = np.argsort(-q_unit @ unit.T, axis=1)[:, :3]
    truth = ids[order]
    source = tmp_path / "source"
    source.mkdir()
    pq.write_table(
        pa.table({"id": ids, "emb": raw.tolist()}), source / "shuffle_train.parquet"
    )
    pq.write_table(
        pa.table({"id": [10, 20, 30], "emb": queries.tolist()}), source / "test.parquet"
    )
    pq.write_table(
        pa.table({"id": [10, 20, 30], "neighbors_id": truth.tolist()}),
        source / "neighbors.parquet",
    )
    config = {
        "rows": 12,
        "dimension": 7,
        "batch_size": 5,
        "seed": 2026,
        "index": {"metric": "COSINE"},
        "query": {"query_count": 3, "topk": 3},
        "dataset": {
            "name": "cohere-medium-1m",
            "files": {
                file.name: {"size": file.stat().st_size, "etag": "fixture"}
                for file in source.iterdir()
            },
        },
    }
    output = tmp_path / "input"
    manifest = prep.prepare(config, source, output)
    return public, config, output, manifest, unit, truth


def test_streamed_fixture_preserves_sparse_ids_and_public_cosine_truth(prepared):
    public, config, output, manifest, expected, truth = prepared
    runtime = public.input_config(config, output)
    queries = public.build_queries(runtime)
    best_ids = np.empty((3, 0), dtype=np.int64)
    distances = np.empty((3, 0), dtype=np.float32)
    hashes = {name: hashlib.sha256() for name in ("vectors.f32", "ids.i64")}
    for start in range(0, 12, 5):
        count = min(5, 12 - start)
        batch = public.vectors(runtime, start, count)
        ids = public.document_ids(runtime, start, count)
        assert not isinstance(batch, np.memmap)
        np.testing.assert_allclose(batch, expected[start : start + count], atol=1e-6)
        best_ids, distances = public.add_truth(
            queries, batch, start, best_ids, distances, 3, ids
        )
        hashes["vectors.f32"].update(batch.tobytes())
        hashes["ids.i64"].update(ids.tobytes())
    np.testing.assert_array_equal(best_ids, truth)
    assert public.check_input_truth(runtime, hashes, best_ids) == 1.0
    assert manifest["files"]["vectors.f32"]["size"] == 12 * 7 * 4
    assert public.document_ids(runtime, 12, 2).tolist() == [1078, 1079]
    np.testing.assert_allclose(
        np.linalg.norm(public.vectors(runtime, 12, 5), axis=1), 1, atol=1e-6
    )


@pytest.mark.parametrize(
    "damage", ["missing_input", "wrong_rows", "truncated", "query_corrupt"]
)
def test_invalid_fixture_is_rejected_before_build_or_query(prepared, damage):
    public, config, output, _, _, _ = prepared
    if damage == "wrong_rows":
        path = output / "manifest.json"
        value = json.loads(path.read_text())
        value["rows"] += 1
        path.write_text(json.dumps(value))
    elif damage == "truncated":
        (output / "vectors.f32").write_bytes(b"broken")
    elif damage == "query_corrupt":
        path = output / "queries.f32"
        values = bytearray(path.read_bytes())
        values[0] ^= 1
        path.write_bytes(values)
    with pytest.raises(ValueError, match=r"requires|differs|truncated|checksum"):
        public.build_queries(
            public.input_config(config, None if damage == "missing_input" else output)
        )


def test_appended_ids_update_exact_truth_without_reusing_dataset_row_numbers(prepared):
    public, config, output, _, _, _ = prepared
    runtime = public.input_config(config, output)
    query = public.vectors(runtime, 12, 1)
    old_ids = np.asarray([[1000, 1007, 1014]])
    old_distances = np.asarray([[1, 2, 3]], dtype=np.float32)
    ids, _ = public.add_truth(
        query, query, 12, old_ids, old_distances, 3, public.document_ids(runtime, 12, 1)
    )
    assert ids[0, 0] == 1078


@pytest.mark.parametrize(("second_matches", "passes"), [(10, True), (8, False)])
def test_ann_recall_uses_the_fixed_query_set_mean(
    prepared, monkeypatch, second_matches, passes
):
    public, _, _, _, _, _ = prepared
    hits = []
    for matches in (8, second_matches):
        keys = [*range(matches), *range(100, 110 - matches)]
        hits.append([SimpleNamespace(id=str(key), score=1.0) for key in keys])
    returned = iter(hits)
    monkeypatch.setattr(public, "query", lambda *_: next(returned))
    config = {"query": {"topk": 10}, "acceptance": {"minimum_recall": 0.9}}
    if passes:
        assert public.verify_queries(
            config, None, [[0], [1]], [list(range(10))] * 2
        ) == pytest.approx(0.9)
    else:
        with pytest.raises(AssertionError, match="mean recall@10"):
            public.verify_queries(config, None, [[0], [1]], [list(range(10))] * 2)
