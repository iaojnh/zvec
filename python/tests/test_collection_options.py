# Copyright 2025-present the zvec project
# SPDX-License-Identifier: Apache-2.0
"""Public write-buffer configuration, including saved option compatibility."""

import pickle

import pytest
import zvec


def test_public_write_buffer_size_round_trips():
    option = zvec.CollectionOption(enable_mmap=False, max_buffer_size=1024**2)
    assert option.max_buffer_size == 1024**2
    restored = pickle.loads(pickle.dumps(option))
    assert restored.max_buffer_size == option.max_buffer_size
    assert restored.enable_mmap is False
    assert restored.read_only is False
    assert "max_buffer_size" in repr(option)


def test_existing_collection_option_arguments_remain_supported():
    option = zvec.CollectionOption(True, False)
    assert option.read_only is True
    assert option.enable_mmap is False
    assert option.max_buffer_size > 0


@pytest.mark.parametrize("size", [0, -1, 2**32, 2**64, 1.5, "1048576"])
def test_invalid_public_write_buffer_size(size):
    with pytest.raises((ValueError, TypeError, OverflowError)):
        zvec.CollectionOption(max_buffer_size=size)
