// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace zvec::core::benchmark {

// Discrete-event storage model. No OS timer, sleep, real disk or CPU clock
// affects these costs. Every request pays latency plus a serialized transfer;
// queue_depth limits concurrently serviced requests. CPU/I/O overlap and device
// cache are deliberately outside the model, not silently inferred from it.
class IOModel {
 public:
  IOModel(uint64_t latency_ns, uint32_t queue_depth, uint64_t bytes_per_second)
      : latency_ns_(latency_ns), bytes_per_second_(bytes_per_second),
        lanes_(CheckedDepth(queue_depth), 0) {
    if (!latency_ns || !queue_depth || queue_depth > 128 || !bytes_per_second) {
      throw std::invalid_argument("Positive latency/bandwidth and depth 1..128 required");
    }
  }

  void reset() {
    now_ns_ = transfer_ns_ = requests_ = bytes_ = batches_ = 0;
    std::fill(lanes_.begin(), lanes_.end(), 0);
  }

  uint64_t schedule(uint64_t bytes) {
    if (!bytes) throw std::invalid_argument("Empty I/O request");
    // Bound arithmetic so large/untrusted benchmark inputs cannot wrap clocks.
    if (bytes > std::numeric_limits<uint64_t>::max() / 1000000000ULL) {
      throw std::overflow_error("I/O request too large");
    }
    auto lane = std::min_element(lanes_.begin(), lanes_.end());
    const uint64_t start = std::max(now_ns_, *lane);
    const uint64_t transfer_start = std::max(start, transfer_ns_);
    const uint64_t numerator = bytes * 1000000000ULL;
    const uint64_t duration = numerator / bytes_per_second_ +
                             (numerator % bytes_per_second_ != 0);
    if (transfer_start > std::numeric_limits<uint64_t>::max() - duration ||
        transfer_start + duration > std::numeric_limits<uint64_t>::max() - latency_ns_ ||
        bytes_ > std::numeric_limits<uint64_t>::max() - bytes) {
      throw std::overflow_error("I/O clock or byte counter overflow");
    }
    transfer_ns_ = transfer_start + duration;
    *lane = transfer_ns_ + latency_ns_;
    ++requests_;
    bytes_ += bytes;
    return *lane;
  }

  void begin_batch() { ++batches_; }
  void advance(uint64_t completion_ns) { now_ns_ = std::max(now_ns_, completion_ns); }
  uint64_t now_ns() const { return now_ns_; }
  uint64_t requests() const { return requests_; }
  uint64_t bytes() const { return bytes_; }
  uint64_t batches() const { return batches_; }
  uint64_t latency_ns() const { return latency_ns_; }
  uint64_t bytes_per_second() const { return bytes_per_second_; }
  size_t queue_depth() const { return lanes_.size(); }

 private:
  static uint32_t CheckedDepth(uint32_t depth) {
    if (!depth || depth > 128) throw std::invalid_argument("Queue depth must be 1..128");
    return depth;
  }
  uint64_t latency_ns_, bytes_per_second_;
  std::vector<uint64_t> lanes_;
  uint64_t now_ns_{0}, transfer_ns_{0}, requests_{0}, bytes_{0}, batches_{0};
};
}  // namespace zvec::core::benchmark
