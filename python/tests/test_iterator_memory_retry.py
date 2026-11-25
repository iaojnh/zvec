# Copyright 2025-present the zvec project
# SPDX-License-Identifier: Apache-2.0
"""Python conversion failures must not consume or retain iterator rows."""

import gc
import weakref
from unittest.mock import MagicMock

import pytest
from zvec.model import collection as collection_module
from zvec.model.collection import DocIterator


@pytest.mark.parametrize("rejections", [1, 3])
def test_repeated_conversion_refusal_preserves_row(monkeypatch, rejections):
    native = MagicMock()
    native.__next__.side_effect = ["first", "second", StopIteration]
    remaining = rejections

    def convert(row, schema):
        nonlocal remaining
        if remaining:
            remaining -= 1
            raise MemoryError("conversion refused")
        return row

    monkeypatch.setattr(collection_module, "convert_to_py_doc", convert)
    iterator = DocIterator(native, object())
    for _ in range(rejections):
        with pytest.raises(MemoryError, match="conversion refused"):
            next(iterator)
        assert native.__next__.call_count == 1
        native.close.assert_not_called()
    assert list(iterator) == ["first", "second"]
    native.close.assert_called_once()


def test_explicit_close_releases_refused_row(monkeypatch):
    class Row:
        pass

    refs = []

    def new_row():
        row = Row()
        refs.append(weakref.ref(row))
        return row

    def reject(row, schema):
        raise MemoryError("conversion refused")

    native = MagicMock()
    native.__next__.side_effect = new_row
    monkeypatch.setattr(collection_module, "convert_to_py_doc", reject)
    iterator = DocIterator(native, object())
    with pytest.raises(MemoryError):
        next(iterator)
    assert refs[0]() is not None
    iterator.close()
    gc.collect()
    assert refs[0]() is None
    native.close.assert_called_once()
    with pytest.raises(StopIteration):
        next(iterator)
    assert native.__next__.call_count == 1


@pytest.mark.parametrize("error", [ValueError, KeyboardInterrupt])
def test_terminal_conversion_error_still_closes(monkeypatch, error):
    def reject(row, schema):
        raise error("terminal conversion error")

    native = MagicMock()
    native.__next__.return_value = object()
    monkeypatch.setattr(collection_module, "convert_to_py_doc", reject)
    iterator = DocIterator(native, object())
    with pytest.raises(error):
        next(iterator)
    native.close.assert_called_once()
    with pytest.raises(StopIteration):
        next(iterator)


def test_native_memory_refusal_can_be_retried(monkeypatch):
    native = MagicMock()
    native.__next__.side_effect = [MemoryError("native refused"), "row", StopIteration]
    monkeypatch.setattr(collection_module, "convert_to_py_doc", lambda row, schema: row)
    iterator = DocIterator(native, object())
    with pytest.raises(MemoryError, match="native refused"):
        next(iterator)
    native.close.assert_not_called()
    assert list(iterator) == ["row"]
    native.close.assert_called_once()
