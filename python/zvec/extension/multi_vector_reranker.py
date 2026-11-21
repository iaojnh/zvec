# Copyright 2025-present the zvec project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from zvec._zvec import (
    _CallbackParams,
    _rerank_python_docs,
    _RrfParams,
    _WeightedParams,
)

from ..model.doc import Doc, DocList
from .rerank_function import RerankFunction

if TYPE_CHECKING:
    from ..model.schema import FieldSchema, VectorSchema


class RrfReRanker(RerankFunction):
    """Re-ranker using Reciprocal Rank Fusion (RRF) for multi-vector search.

    RRF combines results from multiple vector queries without requiring
    relevance scores. The RRF score for a document at rank r is:
        score = 1 / (k + r + 1)
    where k is the rank constant.

    Args:
        rank_constant: RRF smoothing constant (default: 60).
            Higher values reduce the influence of rank position.

    Example:
        >>> reranker = RrfReRanker(rank_constant=60)
        >>> merged = reranker.rerank([results_a, results_b], topn=10)
    """

    def __init__(self, rank_constant: int = 60):
        self._rank_constant = rank_constant

    @property
    def rank_constant(self) -> int:
        """int: RRF rank constant."""
        return self._rank_constant

    def _to_cpp_params(self):
        return _RrfParams(self._rank_constant)

    def rerank(
        self,
        query_results: list[list[Doc]],
        topn: int = 10,
        *,
        fields: list[FieldSchema | VectorSchema] | None = None,  # noqa: ARG002
    ) -> DocList:
        """Apply RRF to combine multiple query results via C++ reranker."""
        return _rerank_python_docs(self._to_cpp_params(), query_results, (), topn)


class WeightedReRanker(RerankFunction):
    """Re-ranker that combines scores using per-sub-query weights.

    Each sub-query's score is normalized by metric type (automatic when used
    via collection.multi_query), then multiplied by the corresponding weight.

    Args:
        weights: Per-sub-query weights. Length must match the number of
            sub-queries.

    Example:
        >>> reranker = WeightedReRanker([0.7, 0.3])
        >>> merged = reranker.rerank([results_a, results_b], topn=10,
        ...                          fields=field_schemas)
    """

    def __init__(self, weights: list[float]):
        self._weights = list(weights)

    @property
    def weights(self) -> list[float]:
        """list[float]: Per-sub-query weights."""
        return self._weights

    def _to_cpp_params(self):
        return _WeightedParams(self._weights)

    def rerank(
        self,
        query_results: list[list[Doc]],
        topn: int = 10,
        *,
        fields: list[FieldSchema | VectorSchema] | None = None,
    ) -> DocList:
        """Combine scores from multiple sub-queries using weighted sum via C++ reranker.

        Args:
            query_results: Per-sub-query document lists.
            topn: Maximum results to return.
            fields: Per-sub-query Python FieldSchema/VectorSchema objects
                (required for score normalization by metric type).

        Raises:
            ValueError: If fields is None (required for normalization).
        """
        if not fields:
            raise ValueError(
                "WeightedReRanker.rerank() requires 'fields' for score normalization. "
                "Pass field schemas via fields= parameter."
            )
        return _rerank_python_docs(
            self._weights, query_results, (f._get_object() for f in fields), topn
        )


class CallbackReRanker(RerankFunction):
    """Re-ranker that delegates to a user-provided callback.

    The callback receives sub-query results, field schemas, and topn.

    Args:
        callback: A callable with signature
            (results: list[list[Doc]], fields: list, topn: int) -> list[Doc]

    Example:
        >>> def my_rerank(results, fields, topn):
        ...     # custom logic
        ...     return merged[:topn]
        >>> reranker = CallbackReRanker(my_rerank)
        >>> merged = reranker.rerank([results_a, results_b], topn=10)
    """

    def __init__(self, callback: Callable):
        self._callback = callback

    def _to_cpp_params(self):
        return _CallbackParams(self._callback)

    def rerank(
        self,
        query_results: list[list[Doc]],
        topn: int = 10,
        *,
        fields: list[FieldSchema | VectorSchema] | None = None,
    ) -> DocList:
        """Invoke the callback to re-rank documents."""
        return self._callback(query_results, fields, topn)
