// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0.
#pragma once
#if defined(ZVEC_DISKANN_CONTROLLED_IO_BENCHMARK)
#include <algorithm>
#include <chrono>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#if defined(__APPLE__)
#include <cerrno>
#include <sys/event.h>
#include <unistd.h>
#endif
#include "algorithm/diskann/diskann_benchmark_hook.h"

namespace zvec::core::benchmark {
using PaceClock = std::chrono::steady_clock;
struct PaceConfig {
  uint64_t iops{2000};
  uint64_t bytes_per_second{16777216};
  uint32_t depth{8};
};
struct PaceStats {
  uint64_t reads{0}, bytes{0}, batches{0}, wait_ns{0};
  uint64_t depth_wait_ns{0};
  uint32_t max_reserved_reads{0};
};

// Two real-time token buckets, bounded to one depth-sized burst. Credits carry
// across queries, including CPU and native-I/O time; no per-query clock reset.
class NativeRateLimiter {
 public:
  explicit NativeRateLimiter(PaceConfig config) : config_(config) {
    if (!config.depth || config.depth > MAX_IO_DEPTH || config.iops > 1000000000ULL ||
        config.bytes_per_second > 1000000000000ULL) {
      throw std::invalid_argument("Invalid native pacing configuration");
    }
    const auto now = Now();
    io_finish_ = now - Budget(config.depth, config.iops);
    byte_finish_ = now - Budget(uint64_t(config.depth) * 4096, config.bytes_per_second);
  }
  NativeRateLimiter(const NativeRateLimiter &) = delete;
  NativeRateLimiter &operator=(const NativeRateLimiter &) = delete;
  const PaceConfig &config() const { return config_; }
  PaceStats stats() const { std::lock_guard<std::mutex> lock(mutex_); return stats_; }
  void reset_peak() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (reserved_reads_) throw std::logic_error("Cannot reset active pacing observations");
    stats_.max_reserved_reads = 0;
  }
  void acquire_slots(uint32_t reads) {
    if (!reads || reads > config_.depth) throw std::invalid_argument("Invalid I/O reservation");
    const auto before = Now();
    std::unique_lock<std::mutex> lock(mutex_);
    const auto ticket = next_ticket_++;
    slots_.wait(lock, [&] { return ticket == serving_ticket_ && reserved_reads_ + reads <= config_.depth; });
    ++serving_ticket_;
    reserved_reads_ += reads;
    stats_.max_reserved_reads = std::max(stats_.max_reserved_reads, reserved_reads_);
    stats_.depth_wait_ns += static_cast<uint64_t>(Now() - before);
    lock.unlock();
    slots_.notify_all();
  }
  void release_slots(uint32_t reads) {
    { std::lock_guard<std::mutex> lock(mutex_);
      if (reads > reserved_reads_) std::terminate();
      reserved_reads_ -= reads;
    }
    slots_.notify_all();
  }
  const char *timer_kind() const {
#if defined(_WIN32)
    return fallback_timer_.load() ? "Windows per-worker waitable timers (fallback used)" :
                                   "Windows per-worker high-resolution waitable timers";
#elif defined(__APPLE__)
    return "Apple kqueue NOTE_CRITICAL monotonic timer";
#else
    return "steady_clock sleep_until";
#endif
  }

  void admit(uint64_t reads, uint64_t bytes) {
    if (!reads || reads > config_.depth || !bytes) throw std::invalid_argument("Invalid paced batch");
    int64_t deadline;
    {
    std::lock_guard<std::mutex> lock(mutex_);
    const int64_t now = Now();
    deadline = now;
    if (config_.iops) {
      io_finish_ = std::max(io_finish_, now - Budget(config_.depth, config_.iops)) +
                   Budget(reads, config_.iops);
      deadline = std::max(deadline, io_finish_);
    }
    if (config_.bytes_per_second) {
      byte_finish_ = std::max(byte_finish_, now - Budget(uint64_t(config_.depth) * 4096,
                                                        config_.bytes_per_second)) +
                     Budget(bytes, config_.bytes_per_second);
      deadline = std::max(deadline, byte_finish_);
    }
    }
    const auto waited = Wait(deadline);
    std::lock_guard<std::mutex> lock(mutex_);
    stats_.wait_ns += waited;
    ++stats_.batches;
    stats_.reads += reads;
    stats_.bytes += bytes;
  }

 private:
  static int64_t Now() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               PaceClock::now().time_since_epoch()).count();
  }
  static int64_t Budget(uint64_t units, uint64_t rate) {
    if (!rate) return 0;
    if (units > uint64_t((std::numeric_limits<int64_t>::max)()) / 1000000000ULL)
      throw std::overflow_error("Pacing budget overflow");
    const auto value = units * 1000000000ULL;
    return static_cast<int64_t>(value / rate + (value % rate != 0));
  }
  uint64_t Wait(int64_t deadline) {
    const auto before = Now();
    if (before >= deadline) return 0;
#if defined(_WIN32)
    struct Timer {
      HANDLE handle{CreateWaitableTimerExW(nullptr, nullptr, 0x2, TIMER_MODIFY_STATE | SYNCHRONIZE)};
      bool high_resolution{handle != nullptr};
      Timer() {
        if (!handle) handle = CreateWaitableTimerExW(nullptr, nullptr, 0, TIMER_MODIFY_STATE | SYNCHRONIZE);
        if (!handle) throw std::runtime_error("Cannot create pacing timer");
      }
      ~Timer() { CloseHandle(handle); }
    };
    static thread_local Timer timer;
    if (!timer.high_resolution) fallback_timer_.store(true);
#elif defined(__APPLE__)
    struct Timer {
      int queue{kqueue()};
      Timer() { if (queue < 0) throw std::runtime_error("Cannot create pacing timer queue"); }
      ~Timer() { ::close(queue); }
    };
    static thread_local Timer timer;
#endif
    while (Now() < deadline) {
#if defined(_WIN32)
      LARGE_INTEGER due{};
      due.QuadPart = -std::max<int64_t>(1, (deadline - Now() + 99) / 100);
      if (!SetWaitableTimer(timer.handle, &due, 0, nullptr, nullptr, FALSE) ||
          WaitForSingleObject(timer.handle, INFINITE) != WAIT_OBJECT_0)
        throw std::runtime_error("Native pacing timer failed");
#elif defined(__APPLE__)
      // Ordinary sleep_until can coalesce millisecond waits by an entire timer
      // quantum. Request minimal coalescing without spinning or changing QoS.
      const auto remaining = deadline - Now();
      if (remaining <= 0) break;
      struct kevent change{}, event{};
      EV_SET(&change, 1, EVFILT_TIMER, EV_ADD | EV_ONESHOT,
             NOTE_NSECONDS | NOTE_CRITICAL, remaining, nullptr);
      int result = kevent(timer.queue, &change, 1, &event, 1, nullptr);
      while (result < 0 && errno == EINTR)
        result = kevent(timer.queue, nullptr, 0, &event, 1, nullptr);
      if (result != 1 || (event.flags & EV_ERROR))
        throw std::runtime_error("Native pacing timer failed");
#else
      std::this_thread::sleep_until(PaceClock::time_point(
          std::chrono::duration_cast<PaceClock::duration>(std::chrono::nanoseconds(deadline))));
#endif
    }
    return static_cast<uint64_t>(std::max<int64_t>(0, Now() - before));
  }
  PaceConfig config_;
  PaceStats stats_;
  int64_t io_finish_{0}, byte_finish_{0};
  mutable std::mutex mutex_;
  std::condition_variable slots_;
  uint64_t next_ticket_{0}, serving_ticket_{0};
  uint32_t reserved_reads_{0};
  std::atomic<bool> fallback_timer_{false};
};

class RateLimitedReader final : public AlignedFileReader {
 public:
  RateLimitedReader(std::shared_ptr<AlignedFileReader> native, NativeRateLimiter &limiter)
      : native_(std::move(native)), limiter_(limiter) {
    if (!native_) throw std::invalid_argument("Native reader required");
  }
  void open(const std::string &path) override { native_->open(path); }
  void close() override { std::lock_guard<std::mutex> lock(pending_mutex_); pending_.clear(); native_->close(); }
  bool requires_io_context() const override { return native_->requires_io_context(); }
  void release_io_ctx(IOContext &ctx) override { native_->release_io_ctx(ctx); }
  std::string cache_policy() const {
#if defined(_WIN32)
    return "Windows FILE_FLAG_NO_BUFFERING; device/host cache not measured";
#elif defined(__ANDROID__)
    auto reader = std::dynamic_pointer_cast<LinuxAlignedFileReader>(native_);
    return reader && reader->direct_io_requested() ? "Android O_DIRECT requested" : "Android buffered fallback";
#elif defined(__APPLE__)
    auto reader = std::dynamic_pointer_cast<LinuxAlignedFileReader>(native_);
    return reader && reader->cache_invalidation_requested() ? "Apple per-range eviction" : "Apple advisory/buffered fallback";
#else
    return "Linux native reader; page-cache bypass checked by OS read bytes";
#endif
  }
  int read(std::vector<AlignedRead> &requests, IOContext &ctx, bool = false) override {
    PendingBatch batch;
    int ret = submit(batch, requests, ctx);
    if (ret) return ret;
    std::vector<uint32_t> completed;
    while (batch.n_reaped < batch.n_submitted) {
      ret = get_completed(batch, ctx, 1, completed);
      if (ret <= 0) return -1;
    }
    return 0;
  }
  int submit(PendingBatch &batch, std::vector<AlignedRead> &requests, IOContext &ctx) override {
    if (requests.size() > MAX_IO_DEPTH) return -1;
    { std::lock_guard<std::mutex> lock(pending_mutex_);
      if (pending_.count(&batch)) return -1;
    }
    batch.n_submitted = static_cast<uint32_t>(requests.size());
    batch.n_reaped = 0;
    if (requests.empty()) return 0;
    auto state = std::make_shared<State>(limiter_);
    state->requests = requests;
    {
      std::lock_guard<std::mutex> lock(pending_mutex_);
      if (!pending_.emplace(&batch, state).second) return -1;
    }
    int ret;
    try { ret = Start(*state, ctx); }
    catch (...) { std::lock_guard<std::mutex> lock(pending_mutex_); pending_.erase(&batch); throw; }
    if (ret) { std::lock_guard<std::mutex> lock(pending_mutex_); pending_.erase(&batch); return ret; }
    return 0;
  }
  int get_completed(PendingBatch &batch, IOContext &ctx, int minimum,
                    std::vector<uint32_t> &completed) override {
    completed.clear();
    if (minimum < 0 || batch.n_reaped > batch.n_submitted) return -1;
    std::shared_ptr<State> owner;
    {
      std::lock_guard<std::mutex> lock(pending_mutex_);
      auto found = pending_.find(&batch);
      if (found == pending_.end()) return batch.n_reaped == batch.n_submitted ? 0 : -1;
      owner = found->second;
    }
    auto &state = *owner;
    if (state.thread != std::this_thread::get_id()) return -1;
    auto failed = [&](int code) {
      std::lock_guard<std::mutex> lock(pending_mutex_);
      pending_.erase(&batch);
      return code;
    };
    std::vector<uint32_t> indices;
    const int need = std::min(minimum, static_cast<int>(state.current.n_submitted - state.current.n_reaped));
    int ret;
    try { ret = native_->get_completed(state.current, ctx, need, indices); }
    catch (...) { failed(-1); throw; }
    if (ret < 0) return failed(ret);
    for (auto index : indices) {
      if (index >= state.chunk.size()) return failed(-1);
      completed.push_back(static_cast<uint32_t>(state.first + index));
    }
    if (indices.size() > state.reserved) return failed(-1);
    state.reserved -= static_cast<uint32_t>(indices.size());
    limiter_.release_slots(static_cast<uint32_t>(indices.size()));
    batch.n_reaped += static_cast<uint32_t>(indices.size());
    if (state.current.n_reaped == state.current.n_submitted) {
      if (state.next < state.requests.size()) {
        try { ret = Start(state, ctx); }
        catch (...) { std::lock_guard<std::mutex> lock(pending_mutex_); pending_.erase(&batch); throw; }
        if (ret) return failed(ret);
      } else {
        std::lock_guard<std::mutex> lock(pending_mutex_);
        pending_.erase(&batch);
      }
    }
    return static_cast<int>(completed.size());
  }

 private:
  struct State {
    explicit State(NativeRateLimiter &value) : limiter(value) {}
    ~State() { if (reserved) limiter.release_slots(reserved); }
    NativeRateLimiter &limiter;
    uint32_t reserved{0};
    std::thread::id thread{std::this_thread::get_id()};
    PendingBatch current;
    std::vector<AlignedRead> requests, chunk;
    size_t first{0}, next{0};
  };
  int Start(State &state, IOContext &ctx) {
    state.first = state.next;
    state.next = std::min(state.requests.size(), state.first + limiter_.config().depth);
    state.chunk.assign(state.requests.begin() + state.first, state.requests.begin() + state.next);
    uint64_t bytes = 0;
    for (const auto &request : state.chunk) bytes += request.len;
    limiter_.acquire_slots(static_cast<uint32_t>(state.chunk.size()));
    state.reserved = static_cast<uint32_t>(state.chunk.size());
    limiter_.admit(state.chunk.size(), bytes);
    state.current = PendingBatch{};
    return native_->submit(state.current, state.chunk, ctx);
  }
  std::shared_ptr<AlignedFileReader> native_;
  NativeRateLimiter &limiter_;
  std::mutex pending_mutex_;
  std::unordered_map<PendingBatch *, std::shared_ptr<State>> pending_;
};

inline NativeRateLimiter *&ActiveLimiter() { static thread_local NativeRateLimiter *value = nullptr; return value; }
inline std::weak_ptr<RateLimitedReader> &ActivePacedReader() {
  static thread_local std::weak_ptr<RateLimitedReader> value;
  return value;
}
inline std::shared_ptr<AlignedFileReader> MakePacedReader(
    const std::string &, std::shared_ptr<AlignedFileReader> native) {
  if (!ActiveLimiter()) throw std::runtime_error("Native limiter not configured");
  auto reader = std::make_shared<RateLimitedReader>(std::move(native), *ActiveLimiter());
  ActivePacedReader() = reader;
  return reader;
}
class PacedScope {
 public:
  explicit PacedScope(NativeRateLimiter &limiter) {
    if (ControlledReaderFactory()) throw std::runtime_error("Conflicting benchmark reader modes");
    ActiveLimiter() = &limiter;
    ControlledReaderFactory() = MakePacedReader;
  }
  ~PacedScope() { ControlledReaderFactory() = nullptr; ActiveLimiter() = nullptr; }
};
}  // namespace zvec::core::benchmark
#endif
