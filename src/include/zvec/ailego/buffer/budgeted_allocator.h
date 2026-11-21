// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <limits>
#include <memory>
#include <zvec/ailego/buffer/process_rss_allocation.h>

namespace zvec::ailego {
// For live, non-evictable workspaces. Charge before allocating and reject
// pressure explicitly; never silently allocate beyond the configured pool.
template <typename T, bool ProcessRss = false>
class BudgetedAllocator {
  struct alignas(alignof(T) > alignof(ProcessRssBudget::MaterializationStamp)
                     ? alignof(T)
                     : alignof(ProcessRssBudget::MaterializationStamp)) Header {
    size_t charged;
    ProcessRssBudget::MaterializationStamp creation;
  };
  // No allocation may exceed PTRDIFF_MAX. Keep the ownership bit in the
  // otherwise unused high bit, preserving the ordinary pool allocation size.
  static constexpr size_t kProcessRss =
      size_t{1} << (std::numeric_limits<size_t>::digits - 1);

 public:
  using value_type = T;
  // sparsehash still uses the pre-allocator_traits interface.
  using pointer = T *;
  using const_pointer = const T *;
  using reference = T &;
  using const_reference = const T &;
  using size_type = size_t;
  using difference_type = ptrdiff_t;
  template <typename U>
  // The legacy allocator protocol requires this exact spelling.
  // NOLINTNEXTLINE(readability-identifier-naming)
  struct rebind {
    using other = BudgetedAllocator<U, ProcessRss>;
  };
  size_t max_size() const noexcept {
    // T can be a pointer slot (e.g. a hash bucket), not the pointed-to object.
    // NOLINTNEXTLINE(bugprone-sizeof-expression)
    return (kProcessRss - 1 - sizeof(Header)) / sizeof(T);
  }
  using is_always_equal = std::true_type;
  BudgetedAllocator() = default;
  template <typename U>
  BudgetedAllocator(const BudgetedAllocator<U, ProcessRss> &) {}
  static bool charged(const T *ptr) {
    return ptr && (reinterpret_cast<const Header *>(ptr) - 1)->charged != 0;
  }
  T *allocate(size_t n) {
    using Kind = MemoryLimitPool::PendingAllocation::Kind;
    return allocate_with_kind(n,
                              ProcessRss || ProcessRssAllocationScope::active()
                                  ? Kind::kProcessRss
                                  : Kind::kWorkspace);
  }

 protected:
  T *allocate_with_kind(size_t n, MemoryLimitPool::PendingAllocation::Kind kind,
                        bool track = true) {
    if (n > max_size()) throw std::bad_alloc();
    // Count T slots even when T itself is a pointer.
    // NOLINTNEXTLINE(bugprone-sizeof-expression)
    const size_t bytes = sizeof(Header) + n * sizeof(T);
    auto &pool = MemoryLimitPool::get_instance();
    using Pending = MemoryLimitPool::PendingAllocation;
    const bool process_rss = kind == Pending::Kind::kProcessRss;
    const size_t charge = track && (process_rss || pool.capacity()) ? bytes : 0;
    Pending pending(pool, kind);
    if (!pending.reserve(charge)) throw std::bad_alloc();
    void *raw = ::operator new(bytes, std::align_val_t{alignof(Header)});
    const auto creation = pending.commit();
    auto *header = new (raw)
        Header{charge | (charge && process_rss ? kProcessRss : 0), creation};
    return reinterpret_cast<T *>(header + 1);
  }

 public:
  void deallocate(T *ptr, size_t) noexcept {
    if (!ptr) return;
    auto *header = reinterpret_cast<Header *>(ptr) - 1;
    const size_t charged = header->charged & ~kProcessRss;
    const auto creation = header->creation;
    const bool process_rss = (header->charged & kProcessRss) != 0;
    ::operator delete(header, std::align_val_t{alignof(Header)});
    if (charged) {
      auto &pool = MemoryLimitPool::get_instance();
      if (process_rss) {
        pool.release_heap_process_backing(charged, creation);
      } else {
        pool.release_heap_external(
            charged, ProcessRssBudget::HeapProjection::Full(charged, creation));
      }
    }
  }
  template <typename U>
  bool operator==(const BudgetedAllocator<U, ProcessRss> &) const {
    return true;
  }
  template <typename U>
  bool operator!=(const BudgetedAllocator<U, ProcessRss> &) const {
    return false;
  }
};

// Keep existing, unbudgeted allocation behavior outside the explicit
// process RSS scope. This lets shared models opt into DiskANN accounting
// without changing the capacity contract of their other callers.
template <typename T>
class ScopedRssAllocator : public BudgetedAllocator<T> {
 public:
  ScopedRssAllocator() = default;
  template <typename U>
  ScopedRssAllocator(const ScopedRssAllocator<U> &) {}
  template <typename U>
  // The allocator protocol requires this spelling.
  // NOLINTNEXTLINE(readability-identifier-naming)
  struct rebind {
    using other = ScopedRssAllocator<U>;
  };
  T *allocate(size_t n) {
    using Kind = MemoryLimitPool::PendingAllocation::Kind;
    return this->allocate_with_kind(n, Kind::kProcessRss,
                                    ProcessRssAllocationScope::active());
  }
};

// Optional acceleration metadata must neither wait nor evict consumers that
// the scalar operation does not need. Reuse the same header and release path.
template <typename T>
class OptionalBudgetedAllocator : public BudgetedAllocator<T> {
 public:
  OptionalBudgetedAllocator() = default;
  template <typename U>
  OptionalBudgetedAllocator(const OptionalBudgetedAllocator<U> &) {}
  template <typename U>
  // The allocator protocol requires this name to preserve optional admission.
  // NOLINTNEXTLINE(readability-identifier-naming)
  struct rebind {
    using other = OptionalBudgetedAllocator<U>;
  };
  T *allocate(size_t n) {
    return this->allocate_with_kind(
        n, MemoryLimitPool::PendingAllocation::Kind::kLockedExternal);
  }
};
}  // namespace zvec::ailego
