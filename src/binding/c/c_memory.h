// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>
#include "db/common/binding_memory.h"
#include "db/common/result_memory_allowance.h"

namespace zvec::detail {

// C payload pointers must remain ordinary, aligned malloc pointers. Track
// owners separately instead of inspecting memory before pointers also accepted
// by zvec_free (legacy malloc allocations and caller-created C structures).
class CResultAllocations {
 public:
  ~CResultAllocations() {
    // A caller's static cleanup may free payloads after this registry has
    // retired its leases. Such pointers still need ordinary free, not a
    // lookup in destroyed registry storage.
    current_.store(nullptr, std::memory_order_release);
  }
  static CResultAllocations &instance() {
    // Destruction order: registry, allowance, then process pool.
    (void)SharedResultMemoryAllowance();
    static CResultAllocations allocations;
    current_.store(&allocations, std::memory_order_release);
    return allocations;
  }

  static void release(void *ptr) noexcept {
    if (!ptr) return;
    if (auto *allocations = current_.load(std::memory_order_acquire))
      allocations->free(ptr);
    else
      std::free(ptr);
  }

  void *allocate(size_t bytes) {
    // Charge payload, registry node/bucket growth and shared ownership before
    // allocating any of them. Retained heap backing uses the common high water.
    ResultMemoryAllowance::Lease admission(&SharedResultMemoryAllowance());
    admission.add(BindingMemoryBytes(1, bytes, 256));
    auto owner = std::make_shared<ResultMemoryAllowance::Lease>(
        &SharedResultMemoryAllowance());
    owner->absorb(admission);
    return allocate_owned(bytes, owner);
  }

  // The caller has already admitted the array and its backing owner.
  void *allocate_owned(size_t bytes, const std::shared_ptr<void> &owner) {
    std::unique_ptr<void, decltype(&std::free)> data(
        std::malloc(bytes ? bytes : 1), &std::free);
    if (!data) throw std::bad_alloc();
    {
      std::lock_guard<std::mutex> lock(mutex_);
      // Keep the local owner until after data is freed if insertion/rehash
      // throws. A moved owner could otherwise refund admission too early.
      owners_.emplace(data.get(), owner);
    }
    return data.release();
  }

  void free(void *ptr) noexcept {
    if (!ptr) return;
    std::shared_ptr<void> owner;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto found = owners_.find(ptr);
      if (found != owners_.end()) {
        owner = std::move(found->second);
        owners_.erase(found);
      }
    }
    std::free(ptr);
    // Release active capacity only after the payload has been destroyed.
  }

 private:
  inline static std::atomic<CResultAllocations *> current_{nullptr};
  std::mutex mutex_;
  std::unordered_map<void *, std::shared_ptr<void>> owners_;
};

inline void *AllocateCResult(size_t bytes) {
  return CResultAllocations::instance().allocate(bytes);
}

inline void FreeCResult(void *ptr) noexcept {
  CResultAllocations::release(ptr);
}

inline void *AllocateCResultArray(size_t count, size_t width) {
  const size_t bytes = BindingMemoryBytes(count, width);
  auto *data = AllocateCResult(bytes);
  std::memset(data, 0, bytes);
  return data;
}

inline char *CopyCResultString(const std::string &value) {
  auto *data = static_cast<char *>(
      AllocateCResult(BindingMemoryBytes(1, value.size(), 1)));
  std::memcpy(data, value.c_str(), value.size() + 1);
  return data;
}

// Used before C arrays become vectors, including all-missing Fetch requests.
// No payload allocation is allowed before the complete estimate is admitted.
inline void ReserveCStrings(ResultMemoryAllowance::Lease &memory,
                            const char *const *strings, size_t count,
                            size_t extra_per_item = 0) {
  memory.add(
      BindingMemoryBytes(count, 3 * sizeof(std::string) + extra_per_item));
  size_t bytes = 0;
  for (size_t i = 0; i < count; ++i) {
    if (strings[i])
      bytes = BindingMemoryBytes(1, std::strlen(strings[i]),
                                 BindingMemoryBytes(1, bytes, 32));
  }
  memory.add(bytes);
}

}  // namespace zvec::detail
