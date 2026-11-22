// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <condition_variable>
#include <cstddef>
#include <mutex>

namespace zvec::core {

// Ordinary reads leave the next turn to queued writers. A nested read may
// join readers ahead of them, but cannot wait for an active writer.
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

  void lock_shared() {
    std::unique_lock lock(state_mutex_);
    available_.wait(lock,
                    [this] { return !writing_ && waiting_writers_ == 0; });
    ++readers_;
  }

  bool try_lock_shared_for_nested_read() {
    std::lock_guard lock(state_mutex_);
    if (writing_) return false;
    ++readers_;
    return true;
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
