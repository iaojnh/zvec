// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <atomic>
#include <mutex>
#include <thread>
#include <zvec/core/framework/index_storage.h>

namespace zvec::core {

// Small scalar snapshots keep one charged allocation per worker. Retention is
// bounded per storage, and close releases idle buffers even on live workers.
// Live results own their slot independently of the storage and reading thread.
class ReusableReadScratch {
 public:
  static constexpr size_t kMaxBytes = 64UL << 10;
  static constexpr size_t kMaxWorkers = 16;
  using Block = IndexStorage::MemoryBlock;

  ReusableReadScratch() : generation_(next_generation()) {}
  ReusableReadScratch(const ReusableReadScratch &) = delete;
  ReusableReadScratch &operator=(const ReusableReadScratch &) = delete;

  template <typename Allocate>
  Block acquire(size_t bytes, size_t alignment, bool deep_copy,
                Allocate allocate) try {
    if (bytes > kMaxBytes) return allocate();
    struct Recent {
      uint64_t generation{0};
      Entry *entry{nullptr};
    };
    // Non-owning TLS is validated by a never-reused generation before any
    // dereference. It retains neither storage nor budget after close/exit.
    static thread_local std::array<Recent, 8> recent;
    static thread_local size_t next = 0;
    for (const auto &hit : recent) {
      if (hit.generation == generation_) {
        return hit.entry->slot->acquire(hit.entry->slot, bytes, alignment,
                                        deep_copy, allocate);
      }
    }

    std::unique_lock<std::mutex> lock(mutex_);
    const auto thread = std::this_thread::get_id();
    Entry *empty = nullptr;
    for (auto &entry : entries_) {
      if (entry.slot && entry.thread == thread) {
        recent[next++ % recent.size()] = {generation_, &entry};
        lock.unlock();
        return entry.slot->acquire(entry.slot, bytes, alignment, deep_copy,
                                   allocate);
      }
      if (!entry.slot && !empty) empty = &entry;
    }
    if (!empty) {
      lock.unlock();
      return allocate();
    }

    // Allocate the required payload first. Optional cache bookkeeping must
    // never make a read fail that fits through the original allocation path.
    Block backing = allocate();
    const bool first = metadata_.bytes() == 0;
    try {
      if (first) metadata_.add(sizeof(entries_));
      auto slot = std::allocate_shared<Slot>(ailego::BudgetedAllocator<Slot>{},
                                             std::move(backing), bytes);
      empty->thread = thread;
      empty->slot = std::move(slot);
      recent[next++ % recent.size()] = {generation_, empty};
    } catch (const std::bad_alloc &) {
      if (first) metadata_.retain_at_most(0);
      return backing;
    }
    lock.unlock();
    return empty->slot->acquire(empty->slot, bytes, alignment, deep_copy,
                                allocate);
  } catch (const std::bad_alloc &) {
    // Idle cache backing is optional. Return it before retrying a required
    // allocation; never discard a buffer still referenced by a result.
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto &entry : entries_) {
      if (entry.slot) entry.slot->discard_idle();
    }
    return allocate();
  }

  // Called after storage readers stop. Retained result blocks remain valid.
  void clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    generation_ = next_generation();
    for (auto &entry : entries_) entry = {};
    metadata_.retain_at_most(0);
  }

 private:
  class Slot : public Block::ScratchLease {
   public:
    Slot(Block backing, size_t capacity)
        : backing_(std::move(backing)), capacity_(capacity) {}

    template <typename Allocate>
    Block acquire(const std::shared_ptr<Slot> &self, size_t bytes,
                  size_t alignment, bool deep_copy, Allocate allocate) {
      std::unique_lock<std::mutex> lock(mutex_);
      if (readers_) {
        lock.unlock();
        return allocate();
      }
      if (!fits(bytes, alignment)) {
        // Replacement remains charged alongside old backing until success.
        auto replacement = allocate();
        backing_ = std::move(replacement);
        capacity_ = bytes;
      }
      readers_ = 1;
      return Block::MakeReusedScratch(const_cast<void *>(backing_.data()), self,
                                      deep_copy);
    }

    void retain() noexcept override {
      std::lock_guard<std::mutex> lock(mutex_);
      ++readers_;
    }
    void release() noexcept override {
      std::lock_guard<std::mutex> lock(mutex_);
      assert(readers_ != 0);
      --readers_;
    }
    bool reusable(size_t bytes, size_t alignment) const noexcept override {
      std::lock_guard<std::mutex> lock(mutex_);
      return readers_ == 1 && fits(bytes, alignment);
    }

    void discard_idle() {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!readers_) {
        backing_.reset();
        capacity_ = 0;
      }
    }

   private:
    bool fits(size_t bytes, size_t alignment) const {
      return capacity_ >= bytes &&
             (!alignment ||
              reinterpret_cast<uintptr_t>(backing_.data()) % alignment == 0);
    }
    mutable std::mutex mutex_;
    size_t readers_{0};
    Block backing_;
    size_t capacity_{0};
  };

  struct Entry {
    std::thread::id thread;
    std::shared_ptr<Slot> slot;
  };
  static uint64_t next_generation() {
    static std::atomic<uint64_t> next{1};
    return next.fetch_add(1, std::memory_order_relaxed);
  }

  std::mutex mutex_;
  uint64_t generation_;
  ailego::MemoryReservation metadata_;
  std::array<Entry, kMaxWorkers> entries_{};
};
}  // namespace zvec::core
