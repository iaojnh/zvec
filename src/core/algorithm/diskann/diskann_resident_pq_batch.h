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

#include <array>
#include <cstring>
#include <limits>
#include <zvec/ailego/buffer/budgeted_buffer.h>
#include <zvec/ailego/buffer/vector_page_table.h>
#include <zvec/core/framework/index_storage.h>

namespace zvec {
namespace core {

// An optional full-hit path for one PQ distance batch. All scratch growth
// precedes page acquisition. A miss releases every pin before the caller
// enters the normal admission/bypass path, and no pin survives distance scoring.
class ResidentPqBatch {
 public:
  ~ResidentPqBatch() { release(); }
  ResidentPqBatch() = default;
  ResidentPqBatch(const ResidentPqBatch &) = delete;
  ResidentPqBatch &operator=(const ResidentPqBatch &) = delete;

  bool read(ailego::VecBufferPool &pool, const IndexStorage::Segment &segment,
            const uint32_t *ids, size_t count, size_t code_size,
            ailego::BudgetedVector<const void *> &views) {
    release();
    const size_t page_size = ailego::kVectorPageSize;
    const size_t capacity = segment.capacity();
    const size_t base = segment.data_offset();
    if (!count || count > kMaxCodes || !ids || !code_size ||
        code_size > page_size || code_size > capacity || pool.writable() ||
        base > (std::numeric_limits<size_t>::max)() - capacity)
      return false;

    // Scratch remains charged to the shared budget; growth must precede pins.
    try {
      if (page_ids_.size() < count * 2) page_ids_.resize(count * 2);
      if (pages_.size() < count * 2) pages_.resize(count * 2);
      views.resize(count);
    } catch (const std::bad_alloc &) {
      return false;
    }
    struct Code {
      size_t within;
      uint16_t first, last;
    };
    std::array<Code, kMaxCodes> codes;
    std::array<uint16_t, kHashSlots> slots{};
    size_t pages = 0, crossing = 0;
    auto page_index = [&](size_t page) -> uint16_t {
      size_t slot = (page * size_t{0x9e3779b1}) & (kHashSlots - 1);
      while (slots[slot]) {
        const size_t index = slots[slot] - 1;
        if (page_ids_[index] == page) return static_cast<uint16_t>(index);
        slot = (slot + 1) & (kHashSlots - 1);
      }
      const auto index = static_cast<uint16_t>(pages++);
      page_ids_[index] = static_cast<ailego::block_id_t>(page);
      slots[slot] = index + 1;
      return index;
    };
    const size_t max_id = (capacity - code_size) / code_size;
    for (size_t i = 0; i < count; ++i) {
      if (ids[i] > max_id) return false;
      const size_t absolute = base + size_t{ids[i]} * code_size;
      const size_t page = absolute / page_size;
      if (page > (std::numeric_limits<ailego::block_id_t>::max)())
        return false;
      auto &code = codes[i];
      code.within = absolute % page_size;
      code.first = code.last = page_index(page);
      if (code_size > page_size - code.within) {
        if (page == (std::numeric_limits<ailego::block_id_t>::max)())
          return false;
        code.last = page_index(page + 1);
        ++crossing;
      }
    }
    // Both allocations can reclaim pages. They must happen before any pin.
    try {
      if (crossing_codes_.size() < crossing * code_size)
        crossing_codes_.resize(crossing * code_size);
    } catch (const std::bad_alloc &) {
      return false;
    }
    if (!pool.try_acquire_resident_pages(page_ids_.data(), pages,
                                          pages_.data()))
      return false;
    pool_ = &pool;
    page_count_ = pages;
    size_t copied = 0;
    for (size_t i = 0; i < count; ++i) {
      const auto &code = codes[i];
      const char *first = pages_[code.first] + code.within;
      if (code.first == code.last) {
        views[i] = first;
      } else {
        auto *destination = crossing_codes_.data() + copied;
        const size_t first_size = page_size - code.within;
        std::memcpy(destination, first, first_size);
        std::memcpy(destination + first_size, pages_[code.last],
                    code_size - first_size);
        views[i] = destination;
        copied += code_size;
      }
    }
    return true;
  }

  void release() noexcept {
    if (pool_) pool_->release_pages(page_ids_.data(), page_count_);
    pool_ = nullptr;
    page_count_ = 0;
  }

 private:
  static constexpr size_t kMaxCodes = 256;
  static constexpr size_t kHashSlots = 1024;
  ailego::BudgetedVector<ailego::block_id_t> page_ids_;
  ailego::BudgetedVector<char *> pages_;
  ailego::BudgetedVector<uint8_t> crossing_codes_;
  ailego::VecBufferPool *pool_{nullptr};
  size_t page_count_{0};
};

}  // namespace core
}  // namespace zvec
