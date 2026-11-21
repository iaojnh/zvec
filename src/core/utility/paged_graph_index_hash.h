// Copyright 2025-present the zvec project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>
#include <zvec/core/framework/index_error.h>
#include <zvec/core/framework/index_storage.h>

namespace zvec::core {

// Persistent open-addressing layout shared by dense, sparse and RaBitQ HNSW.
// Callers
// serialize insertion against readers (the entity's upper-neighbor lock).
// A lookup returns a value, never a pointer into an evictable page or TLS.
template <class Key, class Val, class Broker, Val EmptyVal = 0>
class PagedGraphIndexHash {
  static_assert(std::is_integral<Key>::value && std::is_unsigned<Key>::value,
                "Hash keys must be unsigned integers");
  static_assert(std::is_trivially_copyable<Val>::value,
                "Hash values must be persistable records");

 public:
  struct Item {
    Key first;
    Val second;
  };
  using const_iterator = std::optional<Item>;

  int init(const typename Broker::Pointer &broker, uint32_t chunk_size,
           uint32_t factor, size_t max, float expansion_ratio) {
    if (!broker || !chunk_size || !factor || !std::isfinite(expansion_ratio) ||
        expansion_ratio <= 1.0f)
      return IndexError_InvalidArgument;
    cleanup();
    broker_ = broker;
    const size_t items =
        (static_cast<size_t>(chunk_size) + sizeof(Item) - 1) / sizeof(Item);
    const size_t bits = static_cast<size_t>(std::ceil(std::log2(items)));
    if (bits >= 32) return IndexError_Overflow;
    slot_items_ = uint32_t{1} << bits;
    if (slot_items_ > std::numeric_limits<size_t>::max() / sizeof(Item))
      return IndexError_Overflow;
    const double range =
        static_cast<double>(slot_items_) * factor / expansion_ratio;
    if (range < 1.0 || range >= std::ldexp(1.0, sizeof(Key) * 8))
      return IndexError_InvalidArgument;
    mask_bits_ = static_cast<uint32_t>(std::floor(std::log2(range)));
    if (mask_bits_ >= std::numeric_limits<size_t>::digits)
      return IndexError_Overflow;
    const size_t key_range = size_t{1} << mask_bits_;
    const size_t max_slots = max / key_range + (max % key_range != 0);
    const size_t count =
        broker_->get_chunk_cnt(Broker::CHUNK_TYPE_NEIGHBOR_INDEX);
    // Opening an existing index must also work with a smaller current size
    // estimate. As before, all persisted slots remain readable.
    max_slots_ = std::max(max_slots, count);
    slots_.reserve(max_slots_);
    for (size_t i = 0; i < count; ++i) {
      auto chunk = broker_->get_chunk(Broker::CHUNK_TYPE_NEIGHBOR_INDEX, i);
      if (!chunk ||
          chunk->data_size() < static_cast<size_t>(slot_items_) * sizeof(Item))
        return IndexError_InvalidFormat;
      slots_.emplace_back(std::move(chunk));
    }
    return 0;
  }

  int cleanup() {
    slots_.clear();
    slots_.shrink_to_fit();
    broker_.reset();
    slot_items_ = mask_bits_ = 0;
    max_slots_ = 0;
    return 0;
  }

  std::nullopt_t end() const {
    return std::nullopt;
  }

  const_iterator find(Key key, int *status = nullptr) const {
    if (status) *status = 0;
    const size_t idx = static_cast<size_t>(key >> mask_bits_);
    if (idx >= slots_.size()) return end();
    Item item{};
    size_t offset = 0;
    int ret = probe(slots_[idx], key, &item, &offset);
    if (ret != 0) {
      if (status && ret != IndexError_IndexFull) *status = ret;
      return end();
    }
    return item.second != EmptyVal ? const_iterator(item) : end();
  }

  bool insert(Key key, Val val) {
    if (!broker_ || val == EmptyVal) return false;
    const size_t idx = static_cast<size_t>(key >> mask_bits_);
    if (idx >= max_slots_) return false;
    while (idx >= slots_.size()) {
      if (!alloc_slot(slots_.size())) return false;
    }
    Item item{};
    size_t offset = 0;
    if (probe(slots_[idx], key, &item, &offset) != 0) return false;
    item.first = key;
    item.second = val;
    return slots_[idx].chunk->write(offset, &item, sizeof(item)) ==
           sizeof(item);
  }

 private:
  struct Slot {
    explicit Slot(IndexStorage::Segment::Pointer source)
        : chunk(std::move(source)), base(chunk->base_data()) {}
    IndexStorage::Segment::Pointer chunk;
    const uint8_t *base;
  };

  int probe(const Slot &slot, Key key, Item *item, size_t *offset) const {
    uint32_t position = static_cast<uint32_t>(key) & (slot_items_ - 1);
    for (uint32_t tried = 0; tried < slot_items_; ++tried) {
      *offset = static_cast<size_t>(position) * sizeof(Item);
      if (slot.base) {
        // Stable mmap storage keeps the direct-address fast path.
        std::memcpy(item, slot.base + *offset, sizeof(Item));
      } else {
        IndexStorage::MemoryBlock block;
        if (slot.chunk->read_borrowed(*offset, block, sizeof(Item)) !=
                sizeof(Item) ||
            !block.data())
          return IndexError_ReadData;
        std::memcpy(item, block.data(), sizeof(Item));
      }
      if (item->first == key || item->second == EmptyVal) return 0;
      position = (position + 1) & (slot_items_ - 1);
    }
    return IndexError_IndexFull;
  }

  bool alloc_slot(size_t idx) {
    const size_t bytes = static_cast<size_t>(slot_items_) * sizeof(Item);
    auto allocated =
        broker_->alloc_chunk(Broker::CHUNK_TYPE_NEIGHBOR_INDEX, idx, bytes);
    if (allocated.first != 0 || !allocated.second) return false;
    auto chunk = std::move(allocated.second);
    if (chunk->resize(bytes) != bytes) return false;
    // Initialize persisted empty entries, not a private shadow copy. This also
    // supports storage backends that do not promise zero-filled new segments.
    const std::array<uint8_t, 4096> zeros{};
    for (size_t offset = 0; offset < bytes; offset += zeros.size()) {
      const size_t n = std::min(zeros.size(), bytes - offset);
      if (chunk->write(offset, zeros.data(), n) != n) return false;
    }
    slots_.emplace_back(std::move(chunk));
    return true;
  }

  typename Broker::Pointer broker_;
  std::vector<Slot> slots_;
  uint32_t slot_items_{0};
  uint32_t mask_bits_{0};
  size_t max_slots_{0};
};

}  // namespace zvec::core
