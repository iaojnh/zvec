# Copyright 2025-present the zvec project
# SPDX-License-Identifier: Apache-2.0
"""ANN misses are allowed; corrupt, stale or malformed results are not."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from buffer_pool_query_contract import verify_ann_hits, verify_l2_results


@dataclass
class Document:
    id: str
    score: float
    values: list[float]
    payload: str = "current"

    def vector(self, name):
        assert name == "embedding"
        return self.values

    def field(self, name):
        assert name == "payload"
        return self.payload


def test_valid_nonself_ann_result_is_not_corruption():
    # The query is also stored as ID 17; a truncated ANN search can miss it.
    hits = [Document("10", 2.0, [1.0, 1.0])]
    documents = {"10": Document("10", 2.0, [1.0, 1.0])}
    verify_ann_hits(hits, 1, 20)
    verify_l2_results(hits, [0.0, 0.0], documents)


@pytest.mark.parametrize(
    "hits",
    [
        [],
        [Document("10", float("nan"), [1.0, 1.0])],
        [Document("10", float("inf"), [1.0, 1.0])],
        [Document("-1", 2.0, [1.0, 1.0])],
        [Document("20", 2.0, [1.0, 1.0])],
        [Document("unknown", 2.0, [1.0, 1.0])],
    ],
)
def test_bad_ann_results_still_fail(hits):
    with pytest.raises(AssertionError):
        verify_ann_hits(hits, 1, 20)


def test_duplicate_ids_fail():
    with pytest.raises(AssertionError, match="duplicate"):
        verify_ann_hits([Document("10", 2.0, [1.0])] * 2, 2, 20)


@pytest.mark.parametrize("damage", ["score", "vector", "field", "missing"])
def test_stale_or_incorrect_query_data_still_fails(damage):
    hit = Document("10", 2.0, [1.0, 1.0])
    documents = {"10": Document("10", 2.0, [1.0, 1.0])}
    if damage == "score":
        hit.score = 0.0
    elif damage == "vector":
        hit.values = [2.0, 2.0]
    elif damage == "field":
        hit.payload = "old"
    else:
        documents.clear()
    with pytest.raises(AssertionError):
        verify_l2_results([hit], [0.0, 0.0], documents)
