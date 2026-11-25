# Copyright 2025-present the zvec project
# SPDX-License-Identifier: Apache-2.0
"""Keep native admission with returned payloads, not just the outer Doc.

These private built-in subclasses preserve list/dict/string behavior. A batch
shares one conservative lease until its last original payload is released.
Copies, slices, serialization and user-added values are caller allocations;
they are not a promise to budget the application's entire Python heap.
"""

from __future__ import annotations


class _ResultList(list):
    __slots__ = ("_memory_owner",)

    def __init__(self, values, owner):
        super().__init__(values)
        self._memory_owner = owner

    def __reduce_ex__(self, protocol):
        return list, (list(self),)


class _ResultDict(dict):
    __slots__ = ("_memory_owner",)

    def __init__(self, values, owner):
        super().__init__(values)
        self._memory_owner = owner

    def __reduce_ex__(self, protocol):
        return dict, (dict(self),)


class _ResultString(str):
    __slots__ = ("_memory_owner",)

    def __new__(cls, value, owner):
        result = super().__new__(cls, value)
        result._memory_owner = owner
        return result

    def __reduce_ex__(self, protocol):
        return str, (str(self),)


class _ResultBytes(bytes):
    def __new__(cls, value, owner):
        result = super().__new__(cls, value)
        result._memory_owner = owner
        return result

    def __reduce_ex__(self, protocol):
        return bytes, (bytes(self),)


def _retain_field(value, owner):
    if isinstance(value, str):
        return _ResultString(value, owner)
    if isinstance(value, bytes):
        return _ResultBytes(value, owner)
    if isinstance(value, list):
        # Schema arrays are homogeneous. Avoid a Python call per numeric
        # element; only strings require their own owner when extracted.
        first = next((item for item in value if item is not None), None)
        if isinstance(first, str):
            return _ResultList(
                (
                    _ResultString(item, owner) if item is not None else None
                    for item in value
                ),
                owner,
            )
        return _ResultList(value, owner)
    if isinstance(value, dict):
        # Sparse vectors have numeric keys and values.
        return _ResultDict(value, owner)
    return value


def _retain_fields(fields, owner):
    if fields is None:
        return None
    return _ResultDict(
        (
            (_ResultString(name, owner), _retain_field(value, owner))
            for name, value in fields.items()
        ),
        owner,
    )


def _retain_document(document, owner, key=None):
    if document is None:
        return None
    pk, score, fields, vectors = document
    if (
        not isinstance(key, _ResultString)
        or key._memory_owner is not owner
        or key != pk
    ):
        key = _ResultString(pk, owner)
    return (
        key,
        score,
        _retain_fields(fields, owner),
        _retain_fields(vectors, owner),
    )


def _retain_result_memory(results, owner):
    # Keep the public outer dict/list and four-item tuple layout unchanged.
    # Replace each batch entry progressively to bound overlapping list slots.
    if isinstance(results, dict):
        for key, document in results.items():
            results[key] = _retain_document(document, owner, key)
    else:
        for index, document in enumerate(results):
            results[index] = _retain_document(document, owner)


def _retain_id_result_memory(values, owner):
    return _ResultList(
        (
            _ResultString(value, owner) if value is not None else None
            for value in values
        ),
        owner,
    )
