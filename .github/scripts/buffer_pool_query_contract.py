# Copyright 2025-present the zvec project
# SPDX-License-Identifier: Apache-2.0
"""Check ANN result integrity independently of statistical recall."""

from __future__ import annotations

import math


def verify_ann_hits(hits, topk, rows):
    assert len(hits) == topk, "incomplete query results"
    ids = [hit.id for hit in hits]
    assert len(set(ids)) == topk, "duplicate query IDs"
    assert all(key.isdecimal() and 0 <= int(key) < rows for key in ids), (
        "query ID outside the fixed corpus"
    )
    assert all(math.isfinite(hit.score) for hit in hits), "nonfinite query score"


def verify_l2_results(hits, query, documents):
    assert set(documents) == {hit.id for hit in hits}, "query IDs cannot be fetched"
    for hit in hits:
        stored = documents[hit.id]
        vector = hit.vector("embedding")
        assert vector == stored.vector("embedding"), "query returned a stale vector"
        assert hit.field("payload") == stored.field("payload"), (
            "query returned a stale field"
        )
        assert len(vector) == len(query), "query returned the wrong dimension"
        exact = math.fsum((a - b) ** 2 for a, b in zip(vector, query, strict=True))
        assert math.isclose(hit.score, exact, rel_tol=1e-4, abs_tol=1e-4), (
            hit.id,
            hit.score,
            exact,
        )
