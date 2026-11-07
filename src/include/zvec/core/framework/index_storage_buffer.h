// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <cstring>
#include <limits>
#include <utility>
#include <zvec/ailego/hash/crc32c.h>
#include <zvec/core/framework/index_dumper.h>
#include <zvec/core/framework/index_storage_temporary.h>

namespace zvec::core {

// Owned, growable scratch. With a configured budget, payloads are pageable;
// only explicitly bounded inline storage and row copies occupy workspace.
// Share a Temporary owner between buffers to reuse one page table. Resize
// requires exclusive access; reads and disjoint writes may run concurrently.
class IndexStorage::Buffer {
 public:
  Buffer() = default;
  explicit Buffer(Temporary::Pointer owner)
      : shared_owner_(owner != nullptr), owner_(std::move(owner)) {}
  Buffer(const Buffer &) = delete;
  Buffer &operator=(const Buffer &) = delete;
  Buffer(Buffer &&other) noexcept {
    swap(other);
  }
  Buffer &operator=(Buffer &&other) noexcept {
    Buffer replacement(std::move(other));
    swap(replacement);
    return *this;
  }

  int init(size_t bytes, size_t inline_limit = 0) {
    if (inline_limit > 64 * 1024) return IndexError_InvalidArgument;
    Buffer candidate(shared_owner_ ? owner_ : nullptr);
    candidate.inline_limit_ = inline_limit;
    candidate.chunk_size_ = std::max(bytes, kChunk);
    int ret = candidate.resize(bytes);
    if (ret == 0) swap(candidate);
    return ret;
  }

  int resize(size_t bytes) try {
    if (bytes > static_cast<size_t>(std::numeric_limits<int64_t>::max()))
      return IndexError_InvalidArgument;
    const bool require_pages =
        owner_ || (ailego::MemoryLimitPool::get_instance().capacity() &&
                   bytes > inline_limit_);
    if (require_pages && !paged_) {
      Buffer candidate(owner_);
      candidate.paged_ = true;
      candidate.chunk_size_ = chunk_size_;
      int ret = candidate.grow(bytes);
      const size_t copy = std::min(size_, bytes);
      if (ret == 0 && copy) ret = candidate.write(0, memory_.data(), copy);
      if (ret == 0) swap(candidate);
      return ret;
    }
    if (paged_) return grow(bytes);
    memory_.resize(bytes);
    size_ = bytes;
    return 0;
  } catch (const std::bad_alloc &) {
    return IndexError_NoMemory;
  } catch (const std::exception &) {
    return IndexError_Runtime;
  }

  int append(const void *data, size_t bytes) {
    if ((!data && bytes) || bytes > SIZE_MAX - size_)
      return IndexError_InvalidArgument;
    const size_t previous = size_;
    int ret = resize(previous + bytes);
    if (ret == 0) ret = write(previous, data, bytes);
    if (ret != 0) size_ = previous;
    return ret;
  }

  int read(size_t offset, void *out, size_t bytes) const try {
    if (!in_range(offset, bytes) || (!out && bytes))
      return IndexError_InvalidArgument;
    if (!bytes) return 0;
    if (!paged_) {
      std::memcpy(out, memory_.data() + offset, bytes);
      return 0;
    }
    auto *dest = static_cast<uint8_t *>(out);
    while (bytes) {
      const size_t n = std::min(bytes, chunk_size_ - offset % chunk_size_);
      const auto &segment = segments_[offset / chunk_size_];
      if (segment->fetch(offset % chunk_size_, dest, n) != n)
        return segment->read_error();
      offset += n;
      dest += n;
      bytes -= n;
    }
    return 0;
  } catch (const std::bad_alloc &) {
    return IndexError_NoMemory;
  } catch (const std::exception &) {
    return IndexError_ReadData;
  }

  // Return an independently owned, budgeted row. It remains valid across
  // subsequent reads and buffer destruction, including cross-page records.
  int read(size_t offset, size_t bytes, MemoryBlock &block) const try {
    block = {};
    if (!in_range(offset, bytes)) return IndexError_InvalidArgument;
    using Row = ailego::BudgetedVector<uint64_t>;
    auto row = std::allocate_shared<Row>(
        ailego::BudgetedAllocator<Row>{},
        bytes / sizeof(uint64_t) + (bytes % sizeof(uint64_t) != 0));
    int ret = read(offset, row->data(), bytes);
    if (ret == 0) block = MemoryBlock::MakeSharedView(row->data(), row);
    return ret;
  } catch (const std::bad_alloc &) {
    return IndexError_NoMemory;
  }

  int write(size_t offset, const void *data, size_t bytes) try {
    if (!in_range(offset, bytes) || (!data && bytes))
      return IndexError_InvalidArgument;
    if (!bytes) return 0;
    if (!paged_) {
      std::memcpy(memory_.data() + offset, data, bytes);
      return 0;
    }
    const auto *src = static_cast<const uint8_t *>(data);
    while (bytes) {
      const size_t n = std::min(bytes, chunk_size_ - offset % chunk_size_);
      const auto &segment = segments_[offset / chunk_size_];
      if (segment->write(offset % chunk_size_, src, n) != n)
        return segment->write_error();
      offset += n;
      src += n;
      bytes -= n;
    }
    return 0;
  } catch (const std::bad_alloc &) {
    return IndexError_NoMemory;
  } catch (const std::exception &) {
    return IndexError_WriteData;
  }

  int dump(const IndexDumper::Pointer &dumper, const std::string &id,
           size_t alignment = 32) const try {
    if (!dumper || !alignment) return IndexError_InvalidArgument;
    if (!size_) return 0;
    ailego::BudgetedVector<uint8_t> scratch;
    if (paged_) scratch.resize(std::min(size_, size_t{64 * 1024}));
    uint32_t crc = 0;
    for (size_t offset = 0; offset < size_;) {
      const size_t bytes =
          paged_ ? std::min(scratch.size(), size_ - offset) : size_;
      const void *data = memory_.data();
      if (paged_) {
        int ret = read(offset, scratch.data(), bytes);
        if (ret != 0) return ret;
        data = scratch.data();
      }
      crc = ailego::Crc32c::Hash(data, bytes, crc);
      if (dumper->write(data, bytes) != bytes) return IndexError_WriteData;
      offset += bytes;
    }
    const size_t padding_size = (alignment - size_ % alignment) % alignment;
    const uint8_t padding[4096]{};
    for (size_t left = padding_size; left;) {
      const size_t bytes = std::min(left, sizeof(padding));
      if (dumper->write(padding, bytes) != bytes) return IndexError_WriteData;
      left -= bytes;
    }
    return dumper->append(id, size_, padding_size, crc);
  } catch (const std::bad_alloc &) {
    return IndexError_NoMemory;
  } catch (const std::exception &) {
    return IndexError_WriteData;
  }

  size_t size() const {
    return size_;
  }
  bool paged() const {
    return paged_;
  }
  int truncate(size_t bytes) {
    if (bytes > size_) return IndexError_InvalidArgument;
    size_ = bytes;
    return 0;
  }

 private:
  bool in_range(size_t offset, size_t bytes) const {
    return offset <= size_ && bytes <= size_ - offset;
  }
  int grow(size_t bytes) {
    if (bytes && !owner_) {
      int ret = Temporary::CreateEmpty("", &owner_);
      if (ret != 0) return ret;
    }
    const size_t count = bytes / chunk_size_ + (bytes % chunk_size_ != 0);
    while (segments_.size() < count) {
      Segment::Pointer segment;
      int ret = owner_->append(chunk_size_, &segment);
      if (ret != 0) return ret;
      segments_.push_back(std::move(segment));
    }
    size_ = bytes;
    return 0;
  }
  void swap(Buffer &other) noexcept {
    using std::swap;
    swap(size_, other.size_);
    swap(inline_limit_, other.inline_limit_);
    swap(chunk_size_, other.chunk_size_);
    swap(paged_, other.paged_);
    swap(shared_owner_, other.shared_owner_);
    swap(owner_, other.owner_);
    swap(segments_, other.segments_);
    swap(memory_, other.memory_);
  }
  static constexpr size_t kChunk = 1024 * 1024;
  size_t size_{0};
  size_t inline_limit_{0};
  size_t chunk_size_{kChunk};
  bool paged_{false};
  bool shared_owner_{false};
  Temporary::Pointer owner_;
  ailego::BudgetedVector<Segment::Pointer> segments_;
  ailego::BudgetedVector<uint8_t> memory_;
};

}  // namespace zvec::core
