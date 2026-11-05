// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0.
#pragma once
#if defined(ZVEC_DISKANN_CONTROLLED_IO_BENCHMARK)
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
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
#if defined(_WIN32)
    // High-resolution waitable timers avoid the ordinary Windows tick quantum.
    timer_ = CreateWaitableTimerExW(nullptr, nullptr, 0x2, TIMER_MODIFY_STATE | SYNCHRONIZE);
    high_resolution_timer_ = timer_ != nullptr;
    if (!timer_) timer_ = CreateWaitableTimerExW(nullptr, nullptr, 0, TIMER_MODIFY_STATE | SYNCHRONIZE);
    if (!timer_) throw std::runtime_error("Cannot create pacing timer");
#endif
  }
  ~NativeRateLimiter() {
#if defined(_WIN32)
    if (timer_) CloseHandle(timer_);
#endif
  }
  NativeRateLimiter(const NativeRateLimiter &) = delete;
  NativeRateLimiter &operator=(const NativeRateLimiter &) = delete;
  const PaceConfig &config() const { return config_; }
  const PaceStats &stats() const { return stats_; }
  const char *timer_kind() const {
#if defined(_WIN32)
    return high_resolution_timer_ ? "Windows high-resolution waitable timer" : "Windows waitable timer";
#else
    return "steady_clock sleep_until";
#endif
  }

  void admit(uint64_t reads, uint64_t bytes) {
    if (!reads || reads > config_.depth || !bytes) throw std::invalid_argument("Invalid paced batch");
    const int64_t now = Now();
    int64_t deadline = now;
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
    Wait(deadline);
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
  void Wait(int64_t deadline) {
    const auto before = Now();
    if (before >= deadline) return;
    while (Now() < deadline) {
#if defined(_WIN32)
      LARGE_INTEGER due{};
      due.QuadPart = -std::max<int64_t>(1, (deadline - Now() + 99) / 100);
      if (!SetWaitableTimer(timer_, &due, 0, nullptr, nullptr, FALSE) ||
          WaitForSingleObject(timer_, INFINITE) != WAIT_OBJECT_0)
        throw std::runtime_error("Native pacing timer failed");
#else
      std::this_thread::sleep_until(PaceClock::time_point(
          std::chrono::duration_cast<PaceClock::duration>(std::chrono::nanoseconds(deadline))));
#endif
    }
    stats_.wait_ns += static_cast<uint64_t>(std::max<int64_t>(0, Now() - before));
  }
  PaceConfig config_;
  PaceStats stats_;
  int64_t io_finish_{0}, byte_finish_{0};
#if defined(_WIN32)
  HANDLE timer_{nullptr};
  bool high_resolution_timer_{false};
#endif
};

class RateLimitedReader final : public AlignedFileReader {
 public:
  RateLimitedReader(std::shared_ptr<AlignedFileReader> native, NativeRateLimiter &limiter)
      : native_(std::move(native)), limiter_(limiter) {
    if (!native_) throw std::invalid_argument("Native reader required");
  }
  void open(const std::string &path) override { native_->open(path); }
  void close() override { pending_.clear(); native_->close(); }
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
    if (pending_.count(&batch) || requests.size() > MAX_IO_DEPTH) return -1;
    batch.n_submitted = static_cast<uint32_t>(requests.size());
    batch.n_reaped = 0;
    if (requests.empty()) return 0;
    auto state = std::make_unique<State>();
    state->requests = requests;
    int ret = Start(*state, ctx);
    if (ret) return ret;
    pending_[&batch] = std::move(state);
    return 0;
  }
  int get_completed(PendingBatch &batch, IOContext &ctx, int minimum,
                    std::vector<uint32_t> &completed) override {
    completed.clear();
    if (minimum < 0 || batch.n_reaped > batch.n_submitted) return -1;
    auto found = pending_.find(&batch);
    if (found == pending_.end()) return batch.n_reaped == batch.n_submitted ? 0 : -1;
    auto &state = *found->second;
    std::vector<uint32_t> indices;
    const int need = std::min(minimum, static_cast<int>(state.current.n_submitted - state.current.n_reaped));
    int ret = native_->get_completed(state.current, ctx, need, indices);
    if (ret < 0) return ret;
    for (auto index : indices) {
      if (index >= state.chunk.size()) return -1;
      completed.push_back(static_cast<uint32_t>(state.first + index));
    }
    batch.n_reaped += static_cast<uint32_t>(indices.size());
    if (state.current.n_reaped == state.current.n_submitted) {
      if (state.next < state.requests.size()) {
        ret = Start(state, ctx);
        if (ret) return ret;
      } else {
        pending_.erase(found);
      }
    }
    return static_cast<int>(completed.size());
  }

 private:
  struct State {
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
    limiter_.admit(state.chunk.size(), bytes);
    state.current = PendingBatch{};
    return native_->submit(state.current, state.chunk, ctx);
  }
  std::shared_ptr<AlignedFileReader> native_;
  NativeRateLimiter &limiter_;
  std::unordered_map<PendingBatch *, std::unique_ptr<State>> pending_;
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
