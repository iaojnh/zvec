// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string_view>
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

  template <typename Populate>
  void *allocate(size_t bytes, Populate populate) {
    // Charge payload, registry node/bucket growth and shared ownership before
    // allocating any of them. Retained heap backing uses the common high water.
    ResultMemoryAllowance::Lease admission(&SharedResultMemoryAllowance());
    ResultMemoryAllowance::PendingCopy pending;
    admission.add_pending(BindingMemoryBytes(1, bytes, 256), pending);
    pending.start();
    auto owner = std::make_shared<ResultMemoryAllowance::Lease>(
        &SharedResultMemoryAllowance());
    owner->absorb(admission);
    std::unique_ptr<void, decltype(&release)> data(
        allocate_owned(bytes, owner), &release);
    populate(data.get());
    return data.release();
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
  return CResultAllocations::instance().allocate(bytes, [](void *) {});
}

template <typename Populate>
void *AllocateCResult(size_t bytes, Populate populate) {
  return CResultAllocations::instance().allocate(bytes, populate);
}

inline void FreeCResult(void *ptr) noexcept {
  CResultAllocations::release(ptr);
}

inline void *AllocateCResultArray(size_t count, size_t width) {
  const size_t bytes = BindingMemoryBytes(count, width);
  return AllocateCResult(bytes,
                         [bytes](void *data) { std::memset(data, 0, bytes); });
}

inline void *CopyCResultBuffer(const void *source, size_t bytes) {
  return AllocateCResult(bytes, [source, bytes](void *data) {
    if (bytes) std::memcpy(data, source, bytes);
  });
}

inline char *CopyCResultString(std::string_view value) {
  return static_cast<char *>(
      AllocateCResult(BindingMemoryBytes(1, value.size(), 1), [value](void *data) {
        auto *text = static_cast<char *>(data);
        if (!value.empty()) std::memcpy(text, value.data(), value.size());
        text[value.size()] = '\0';
      }));
}

// Used before C arrays become vectors, including all-missing Fetch requests.
// No payload allocation is allowed before the complete estimate is admitted.
inline void ReserveCStrings(ResultMemoryAllowance::Lease &memory,
                            ResultMemoryAllowance::PendingCopy &pending,
                            const char *const *strings, size_t count,
                            size_t extra_per_item = 0) {
  memory.add_pending(
      BindingMemoryBytes(count, 3 * sizeof(std::string) + extra_per_item),
      pending);
  size_t bytes = 0;
  for (size_t i = 0; i < count; ++i) {
    if (strings[i])
      bytes = BindingMemoryBytes(1, std::strlen(strings[i]),
                                 BindingMemoryBytes(1, bytes, 32));
  }
  memory.add_pending(bytes, pending);
}

}  // namespace zvec::detail
