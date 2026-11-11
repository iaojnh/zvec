// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <limits>
#include <stdexcept>
#include "core/utility/appendable_build_buffer.h"
#include "bitpacked_posting_list.h"

namespace zvec::fts {

// Sparse local IDs read as length one until explicitly written. Newly grown
// temporary storage is zero-filled, so holes need no count-sized
// initialization.
class PostingDocLengths {
 public:
  void set(uint32_t id, uint32_t length) {
    const uint64_t bytes = (static_cast<uint64_t>(id) + 1) * sizeof(uint32_t);
    if (bytes > std::numeric_limits<size_t>::max())
      throw std::length_error("Doc lengths overflow");
    int ret = 0;
    if (bytes > data_.size()) ret = data_.resize(static_cast<size_t>(bytes));
    if (ret == 0)
      ret = data_.write(static_cast<size_t>(id) * sizeof(uint32_t), &length,
                        sizeof(length));
    if (ret == core::IndexError_NoMemory) throw std::bad_alloc();
    if (ret != 0) throw std::runtime_error("Cannot store posting doc length");
  }
  uint32_t get(uint32_t id) const {
    const uint64_t offset = static_cast<uint64_t>(id) * sizeof(uint32_t);
    if (offset >= data_.size()) return 1;
    uint32_t length = 0;
    if (data_.read(static_cast<size_t>(offset), &length, sizeof(length)) != 0)
      throw std::runtime_error("Cannot read posting doc length");
    return length ? length : 1;
  }

 private:
  core::AppendableBuildBuffer data_;
};

// One pageable representation of a term. Bounded staging avoids tiny writes;
// the encoder replays a block at a time without materializing the whole term.
class PostingBuildBuffer {
 public:
  void append(uint32_t id, uint32_t tf, uint32_t length) {
    if (count_ == UINT32_MAX) throw std::length_error("Too many postings");
    if (count_ && id <= last_id_)
      throw std::invalid_argument("Unsorted posting IDs");
    if (pending_ == rows_.size()) flush_block();
    rows_[pending_++] = {id, tf, length};
    last_id_ = id;
    ++count_;
  }
  void clear() {
    count_ = pending_ = stored_ = 0;
    check(data_.resize(0));
  }
  size_t size() const {
    return count_;
  }
  bool empty() const {
    return !count_;
  }
  bool paged() const {
    return data_.paged();
  }

  BitPackedPostingList::EncodedPosting encode(uint64_t df,
                                              const BM25Scorer &scorer) const {
    return BitPackedPostingList::encode_blocks(
        count_, df, scorer,
        [&](size_t start, size_t n, uint32_t *ids, uint32_t *tfs,
            uint32_t *lengths) {
          std::array<Entry, BitPackedPostingList::DOCS_PER_BLOCK> scratch;
          const Entry *entries = rows_.data();
          if (start < stored_) {
            check(data_.read(start * sizeof(Entry), scratch.data(),
                             n * sizeof(Entry)));
            entries = scratch.data();
          }
          for (size_t i = 0; i < n; ++i) {
            ids[i] = entries[i].id;
            tfs[i] = entries[i].tf;
            lengths[i] = entries[i].length;
          }
        });
  }

 private:
  struct Entry {
    uint32_t id, tf, length;
  };
  static void check(int ret) {
    if (ret == core::IndexError_NoMemory) throw std::bad_alloc();
    if (ret != 0)
      throw std::runtime_error("Posting scratch I/O failed: " +
                               std::to_string(ret));
  }
  void flush_block() {
    if (stored_ > std::numeric_limits<size_t>::max() / sizeof(Entry) - pending_)
      throw std::length_error("Posting scratch size overflow");
    check(data_.resize((stored_ + pending_) * sizeof(Entry)));
    check(data_.write(stored_ * sizeof(Entry), rows_.data(),
                      pending_ * sizeof(Entry)));
    stored_ += pending_;
    pending_ = 0;
  }
  core::AppendableBuildBuffer data_;
  std::array<Entry, BitPackedPostingList::DOCS_PER_BLOCK> rows_{};
  size_t count_{0}, stored_{0}, pending_{0};
  uint32_t last_id_{0};
};
}  // namespace zvec::fts
