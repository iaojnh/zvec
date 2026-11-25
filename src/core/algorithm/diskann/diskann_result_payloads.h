// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <limits>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>
#include <zvec/ailego/buffer/memory_reservation.h>

namespace zvec::core {

// allocate_shared determines the actual control-block size through rebind.
// Reserve all blocks together, then transfer each block's share to the
// reservation for that document. The allocator releases backing after the
// vector payload has been destroyed, independently of the other results.
class DiskAnnResultPayloads {
  using Pool = ailego::MemoryLimitPool;
  struct Charge {
    Charge *next{nullptr};
    ailego::MemoryReservation reservation;
  };

 public:
  explicit DiskAnnResultPayloads(size_t count)
      : count_(count),
        pending_(Pool::get_instance(),
                 Pool::PendingAllocation::Kind::kWorkspace),
        previous_(std::exchange(active_, this)) {}
  ~DiskAnnResultPayloads() { active_ = previous_; }

  template <typename T>
  struct Allocator {
    using value_type = T;
    using is_always_equal = std::true_type;

    Allocator() = default;
    template <typename U>
    Allocator(const Allocator<U> &) noexcept {}

    T *allocate(size_t count) {
      auto *batch = active_;
      if (!batch || count != 1) throw std::bad_alloc();
      constexpr size_t bytes = sizeof(Block) + sizeof(T);
      batch->reserve(bytes);
      auto *raw = ::operator new(bytes, std::align_val_t{alignof(Block)});
      auto *block = new (raw) Block{};
      block->charge.next = batch->first_;
      batch->first_ = &block->charge;
      ++batch->allocated_;
      return reinterpret_cast<T *>(block + 1);
    }

    void deallocate(T *value, size_t) noexcept {
      auto *block = reinterpret_cast<Block *>(value) - 1;
      auto reservation = std::move(block->charge.reservation);
      block->~Block();
      ::operator delete(block, std::align_val_t{alignof(Block)});
    }

    template <typename U>
    bool operator==(const Allocator<U> &) const noexcept {
      return true;
    }
    template <typename U>
    bool operator!=(const Allocator<U> &other) const noexcept {
      return !(*this == other);
    }

   private:
    struct alignas(alignof(T) > alignof(Charge) ? alignof(T) : alignof(Charge))
        Block {
      Charge charge;
    };
  };

  void attach_vector(ailego::MemoryReservation reservation) noexcept {
    first_->reservation.swap(reservation);
  }

  void commit() {
    if (!charged_) return;
    auto controls = ailego::MemoryReservation::commit_heap(
        pending_, allocated_ * slot_bytes_);
    for (auto *entry = first_; entry; entry = entry->next) {
      entry->reservation.absorb(controls.take(slot_bytes_));
    }
  }

 private:
  void reserve(size_t bytes) {
    if (allocated_ == count_) throw std::bad_alloc();
    if (slot_bytes_) {
      if (slot_bytes_ != bytes) throw std::bad_alloc();
      return;
    }
    slot_bytes_ = bytes;
    if (!Pool::get_instance().capacity()) return;
    if (count_ > (std::numeric_limits<size_t>::max)() / bytes ||
        !pending_.reserve(count_ * bytes))
      throw std::bad_alloc();
    charged_ = true;
  }

  size_t count_;
  Pool::PendingAllocation pending_;
  size_t slot_bytes_{0};
  size_t allocated_{0};
  Charge *first_{nullptr};
  bool charged_{false};
  DiskAnnResultPayloads *previous_;
  inline static thread_local DiskAnnResultPayloads *active_{nullptr};
};

}  // namespace zvec::core
