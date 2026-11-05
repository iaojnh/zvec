// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0.
#pragma once
#include <chrono>
#include <condition_variable>
#include <exception>
#include <cstdint>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace zvec::core::benchmark {
// Workers stay alive across warmup/timing/RSS sampling, retaining their search
// contexts and scratch buffers. A phase barrier separates all measurements.
class ConcurrentQueryWorkers {
 public:
  explicit ConcurrentQueryWorkers(uint32_t count) {
    if (!count || count > 8) throw std::invalid_argument("Invalid query worker count");
    try {
      for (uint32_t worker = 0; worker < count; ++worker)
        threads_.emplace_back([this, worker] { Work(worker); });
    } catch (...) { Stop(); throw; }
    std::unique_lock<std::mutex> lock(mutex_);
    done_.wait(lock, [&] { return ready_ == count; });
  }
  ~ConcurrentQueryWorkers() { Stop(); }
  void Run(std::function<void(uint32_t)> task, const std::function<void()> &sample = {}) {
    std::unique_lock<std::mutex> lock(mutex_);
    task_ = std::move(task);
    completed_ = 0;
    error_ = nullptr;
    ++epoch_;
    start_.notify_all();
    while (completed_ != threads_.size()) {
      done_.wait_for(lock, std::chrono::milliseconds(5));
      if (sample) {
        lock.unlock();
        std::exception_ptr error;
        try { sample(); } catch (...) { error = std::current_exception(); }
        lock.lock();
        if (error && !error_) error_ = error;
      }
    }
    task_ = {};
    if (error_) std::rethrow_exception(error_);
  }

 private:
  void Stop() {
    { std::lock_guard<std::mutex> lock(mutex_); stopping_ = true; }
    start_.notify_all();
    for (auto &thread : threads_) if (thread.joinable()) thread.join();
  }
  void Work(uint32_t worker) {
    std::unique_lock<std::mutex> lock(mutex_);
    ++ready_;
    done_.notify_all();
    uint64_t seen = 0;
    for (;;) {
      start_.wait(lock, [&] { return stopping_ || epoch_ != seen; });
      if (stopping_) return;
      seen = epoch_;
      auto task = task_;
      lock.unlock();
      std::exception_ptr error;
      try { task(worker); } catch (...) { error = std::current_exception(); }
      task = {};
      lock.lock();
      if (error && !error_) error_ = error;
      ++completed_;
      done_.notify_all();
    }
  }
  std::vector<std::thread> threads_;
  std::mutex mutex_;
  std::condition_variable start_, done_;
  std::function<void(uint32_t)> task_;
  std::exception_ptr error_;
  uint64_t epoch_{0};
  uint32_t ready_{0}, completed_{0};
  bool stopping_{false};
};
}  // namespace zvec::core::benchmark
