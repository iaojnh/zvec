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

#include <algorithm>
#include <exception>
#include <thread>
#include <utility>
#include <zvec/ailego/buffer/budgeted_allocator.h>
#include <zvec/ailego/parallel/thread_pool.h>
#include <zvec/ailego/pattern/closure.h>
#include <zvec/core/framework/index_storage.h>

namespace zvec {
namespace core {

/*! Index Threads
 *  Index ThreadPool maintains multiple threads resources to execute the tasks
 *  concurrently
 */
class IndexThreads {
 public:
  using Pointer = std::shared_ptr<IndexThreads>;

  /*! Threads Task Group
   *  Manage of a group of sub-tasks which can be seen as a big task,
   *  so we can wait all sub-tasks finished, or get the status of them
   */
  class TaskGroup {
   public:
    using Pointer = std::shared_ptr<TaskGroup>;

    //! Destructor
    virtual ~TaskGroup() = default;

    //! Submit a task to be executed asynchronous
    virtual void submit(ailego::ClosureHandler &&task) = 0;

    //! Check if the group is finished
    virtual bool is_finished() const = 0;

    // A failed group stops accepting useful work. Running operations must
    // cooperate with cancellation; wait_finish still joins every worker.
    virtual bool cancelled() const {
      return false;
    }

    //! Wait until all tasks in group finished
    virtual void wait_finish() = 0;

    // Keep closure construction inside the same failure boundary as submission.
    template <typename... Arguments>
    void execute(Arguments &&...arguments) {
      try {
        submit(ailego::Closure::New(std::forward<Arguments>(arguments)...));
      } catch (...) {
        cancel_pending();
        wait_finish_no_throw();
        throw;
      }
    }

    // Cleanup joins cannot replace the original failure or throw from a
    // destructor. Successful paths use wait_finish() to observe worker errors.
    void wait_finish_no_throw() noexcept {
      try {
        wait_finish();
      } catch (...) {
      }
    }

   protected:
    virtual void cancel_pending() {}
  };

  //! Destructor
  virtual ~IndexThreads() = default;

  //! Retrieve thread count in pool
  virtual size_t count() const = 0;

  //! Stop all threads
  virtual void stop() = 0;

  //! Submit a task to be executed asynchronous
  virtual void submit(ailego::ClosureHandler &&task) = 0;

  //! Make a task group
  virtual TaskGroup::Pointer make_group() = 0;

  //! Get the current work thread index
  virtual int indexof_this() const = 0;
  // Run bounded workers with private workspace slots. The callback returns
  // an index status for (ordinal, worker). Errors cancel remaining iterations
  // and all workers are joined before returning, including failed submission.
  template <typename Operation>
  int parallel_for(size_t records, Operation &&operation,
                   size_t max_workers = 0) try {
    if (!records) return 0;
    const size_t workers = std::min(
        records, max_workers ? std::min(max_workers, count()) : count());
    if (!workers) return IndexError_InvalidArgument;
    if (records > SIZE_MAX - workers) return IndexError_Overflow;
    std::atomic<size_t> next{0};
    std::atomic<int> result{0};
    auto group = make_group();
    if (!group) return IndexError_NoMemory;
    try {
      for (size_t worker = 0; worker < workers; ++worker) {
        group->execute([&, worker] {
          while (!result.load(std::memory_order_acquire) &&
                 !group->cancelled()) {
            const size_t ordinal = next.fetch_add(1, std::memory_order_relaxed);
            if (ordinal >= records) break;
            const int ret = operation(ordinal, worker);
            if (ret != 0) {
              int expected = 0;
              result.compare_exchange_strong(expected, ret);
            }
          }
        });
      }
      group->wait_finish();
    } catch (...) {
      result.store(IndexError_Runtime, std::memory_order_release);
      group->wait_finish_no_throw();
      throw;
    }
    return result.load();
  } catch (const std::bad_alloc &) {
    return IndexError_NoMemory;
  } catch (const IndexStorage::Exception &error) {
    return error.code;
  }
};

/*! Single Queue Index Threads
 */
class SingleQueueIndexThreads : public IndexThreads {
 public:
  /*! Single Queue Index Threads Task Group
   */
  class SingleQueueTaskGroup : public TaskGroup {
   public:
    using Pointer = std::shared_ptr<SingleQueueTaskGroup>;

    //! Constructor
    explicit SingleQueueTaskGroup(
        ailego::ThreadPool::TaskGroup::Pointer task_group)
        : task_group_(std::move(task_group)),
          state_(
              std::allocate_shared<State>(ailego::BudgetedAllocator<State>{})) {
    }

    ~SingleQueueTaskGroup() override {
      if (std::uncaught_exceptions())
        state_->cancelled.store(true, std::memory_order_release);
      // Never leave workers using a caller's stack after the group is released.
      task_group_->wait_finish();
    }

    //! Submit a task to be executed asynchronous
    void submit(ailego::ClosureHandler &&task) override {
      while (task_group_->pending_count() >= kMaxQueueSize) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
      try {
        auto state = state_;
        task_group_->enqueue_and_wake(
            ailego::Closure::New([state, task = std::move(task)]() {
              if (state->cancelled.load(std::memory_order_acquire)) return;
              try {
                task->run();
              } catch (...) {
                std::lock_guard<std::mutex> lock(state->mutex);
                if (!state->error) state->error = std::current_exception();
                state->cancelled.store(true, std::memory_order_release);
              }
            }));
      } catch (...) {
        state_->cancelled.store(true, std::memory_order_release);
        task_group_->wait_finish();
        throw;
      }
    }

    bool cancelled() const override {
      return state_->cancelled.load(std::memory_order_acquire);
    }

    //! Check if the group is finished
    bool is_finished() const override {
      return task_group_->is_finished();
    }

    //! Wait until all tasks in group finished
    void wait_finish() override {
      task_group_->wait_finish();
      std::exception_ptr error;
      {
        std::lock_guard<std::mutex> lock(state_->mutex);
        error = state_->error;
      }
      if (error) std::rethrow_exception(error);
    }

   private:
    void cancel_pending() override {
      state_->cancelled.store(true, std::memory_order_release);
    }

    //! Members
    struct State {
      std::atomic<bool> cancelled{false};
      std::mutex mutex;
      std::exception_ptr error;
    };
    ailego::ThreadPool::TaskGroup::Pointer task_group_{};
    std::shared_ptr<State> state_;
  };

  //! Constructor
  SingleQueueIndexThreads(uint32_t size, bool binding)
      : pool_(
            size > 0 ? size : std::max(std::thread::hardware_concurrency(), 1u),
            binding) {}

  //! Constructor
  explicit SingleQueueIndexThreads(bool binding)
      : SingleQueueIndexThreads(0, binding) {}

  //! Constructor
  SingleQueueIndexThreads() : SingleQueueIndexThreads{false} {}

  //! Destructor
  ~SingleQueueIndexThreads() override = default;

  //! Retrieve thread count in pool
  size_t count() const override {
    return pool_.count();
  }

  //! Stop all threads
  void stop() override {
    pool_.stop();
  }

  //! Submit a task to be executed asynchronous
  void submit(ailego::ClosureHandler &&task) override {
    while (pool_.pending_count() >= kMaxQueueSize) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    pool_.enqueue_and_wake(std::move(task));
  }

  //! Make a task group
  TaskGroup::Pointer make_group() override {
    return std::make_shared<SingleQueueTaskGroup>(pool_.make_group());
  }

  //! Get the current work thread index
  int indexof_this() const override {
    return pool_.indexof_this();
  }

 public:
  //! Disable them
  SingleQueueIndexThreads(const SingleQueueIndexThreads &) = delete;
  SingleQueueIndexThreads(SingleQueueIndexThreads &&) = delete;
  SingleQueueIndexThreads &operator=(const SingleQueueIndexThreads &) = delete;

 private:
  static constexpr size_t kMaxQueueSize = 4096u;

  //! Members
  ailego::ThreadPool pool_{};
};

/*! Borrowed Single Queue Index Threads
 *
 *  Adapts an existing thread pool to IndexThreads. The caller must keep the
 *  pool alive for the lifetime of this object.
 */
class BorrowedSingleQueueIndexThreads : public IndexThreads {
 public:
  //! Constructor
  explicit BorrowedSingleQueueIndexThreads(ailego::ThreadPool &pool)
      : pool_(pool) {}

  //! Destructor
  ~BorrowedSingleQueueIndexThreads() override = default;

  //! Retrieve thread count in pool
  size_t count() const override {
    return pool_.count();
  }

  //! Stop all threads
  void stop() override {
    pool_.stop();
  }

  //! Submit a task to be executed asynchronous
  void submit(ailego::ClosureHandler &&task) override {
    while (pool_.pending_count() >= kMaxQueueSize) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    pool_.enqueue_and_wake(std::move(task));
  }

  //! Make a task group
  TaskGroup::Pointer make_group() override {
    return std::make_shared<SingleQueueIndexThreads::SingleQueueTaskGroup>(
        pool_.make_group());
  }

  //! Get the current work thread index
  int indexof_this() const override {
    return pool_.indexof_this();
  }

 public:
  //! Disable them
  BorrowedSingleQueueIndexThreads(const BorrowedSingleQueueIndexThreads &) =
      delete;
  BorrowedSingleQueueIndexThreads(BorrowedSingleQueueIndexThreads &&) = delete;
  BorrowedSingleQueueIndexThreads &operator=(
      const BorrowedSingleQueueIndexThreads &) = delete;

 private:
  static constexpr size_t kMaxQueueSize = 4096u;

  //! Members
  ailego::ThreadPool &pool_;
};

}  // namespace core
}  // namespace zvec
