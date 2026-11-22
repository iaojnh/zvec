// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <condition_variable>
#include <cstddef>
#include <mutex>

namespace zvec::core {

// A nested index read can join active readers ahead of a waiting writer.
// Ordinary reads leave the next turn to writers that are already waiting.
class HnswGraphMutex {
 public:
  void lock() {
    std::unique_lock lock(state_mutex_);
    ++waiting_writers_;
    available_.wait(lock, [this] { return !writing_ && readers_ == 0; });
    --waiting_writers_;
    writing_ = true;
  }

  void unlock() {
    std::lock_guard lock(state_mutex_);
    writing_ = false;
    available_.notify_all();
  }

  void lock_shared(bool nested) {
    std::unique_lock lock(state_mutex_);
    available_.wait(lock, [this, nested] {
      return !writing_ && (waiting_writers_ == 0 || (nested && readers_ != 0));
    });
    ++readers_;
  }

  void unlock_shared() {
    std::lock_guard lock(state_mutex_);
    if (--readers_ == 0) available_.notify_all();
  }

 private:
  std::mutex state_mutex_;
  std::condition_variable available_;
  std::size_t readers_{0};
  std::size_t waiting_writers_{0};
  bool writing_{false};
};

}  // namespace zvec::core
