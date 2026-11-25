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

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <zvec/ailego/parallel/thread_pool.h>
#include <zvec/ailego/pattern/singleton.h>

namespace zvec {

class GlobalConfig;

class GlobalResource : public ailego::Singleton<GlobalResource> {
 public:
  int initialize();

  bool initialized() const {
    return initialized_.load(std::memory_order_acquire);
  }

  ailego::ThreadPool *query_thread_pool() {
    if (initialize() != 0) {
      return nullptr;
    }
    return query_thread_pool_.get();
  }

  ailego::ThreadPool *optimize_thread_pool() {
    if (initialize() != 0) {
      return nullptr;
    }
    return optimize_thread_pool_.get();
  }

 private:
  friend class GlobalConfig;

  // Explicit init publishes the prepared resources with its config snapshot.
  int prepare(uint64_t memory_limit_bytes, uint32_t query_thread_count,
              bool query_thread_binding, uint32_t optimize_thread_count,
              bool optimize_thread_binding);
  int initialize_locked(uint64_t memory_limit_bytes,
                        uint32_t query_thread_count, bool query_thread_binding,
                        uint32_t optimize_thread_count,
                        bool optimize_thread_binding,
                        bool preserve_existing_pool);
  void log_initialized() const noexcept;
  void reserve_native_memory(uint32_t query_workers, uint32_t optimize_workers);
  int configure_memory_policy(uint64_t memory_limit_bytes);

  std::atomic<bool> initialized_{false};
  std::mutex initialization_mutex_;
  // Protected by initialization_mutex_. Once DB-owned pool setup starts,
  // failure is terminal: its partial state must not become a standalone pool.
  bool initialization_failed_{false};
  // A standalone pool can supply capacity before lazy DB workers are created.
  // Explicit startup must still install the native/RSS policy before success.
  bool memory_policy_ready_{false};
  uint64_t memory_limit_bytes_{0};
  uint32_t query_thread_count_{0};
  uint32_t optimize_thread_count_{0};
  bool query_thread_binding_{false};
  bool optimize_thread_binding_{false};
  std::unique_ptr<ailego::ThreadPool> query_thread_pool_;
  std::unique_ptr<ailego::ThreadPool> optimize_thread_pool_;
};

}  // namespace zvec
