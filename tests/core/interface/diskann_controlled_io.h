// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0.
#pragma once
#if defined(ZVEC_DISKANN_CONTROLLED_IO_BENCHMARK)
#include <cstring>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include "algorithm/diskann/diskann_benchmark_hook.h"
#include "benchmarks/diskann-controlled-io/io_model.h"

namespace zvec::core::benchmark {
class ControlledReader final : public AlignedFileReader {
 public:
  explicit ControlledReader(IOModel &model) : model_(model) {}
  void open(const std::string &path) override {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot preload controlled index: " + path);
    image_.assign(std::istreambuf_iterator<char>(input), {});
    if (input.bad() || image_.empty()) throw std::runtime_error("Invalid controlled index");
  }
  void close() override { image_.clear(); pending_.clear(); }
  bool requires_io_context() const override { return false; }
  void release_io_ctx(IOContext &) override {}
  uint64_t image_bytes() const { return image_.size(); }

  int read(std::vector<AlignedRead> &requests, IOContext &ctx, bool = false) override {
    PendingBatch batch;
    int result = submit(batch, requests, ctx);
    if (result) return result;
    std::vector<uint32_t> completed;
    return get_completed(batch, ctx, batch.n_submitted, completed) < 0 ? -1 : 0;
  }

  int submit(PendingBatch &batch, std::vector<AlignedRead> &requests,
             IOContext &) override {
    if (pending_.count(&batch) || requests.size() > MAX_IO_DEPTH) return -1;
    for (const auto &request : requests) {
      if (!request.buf || !request.len || request.offset > image_.size() ||
          request.len > image_.size() - request.offset) return -1;
    }
    batch.n_submitted = static_cast<uint32_t>(requests.size());
    batch.n_reaped = 0;
    if (requests.empty()) return 0;
    auto &events = pending_[&batch];
    model_.begin_batch();
    for (uint32_t i = 0; i < requests.size(); ++i) {
      const auto &request = requests[i];
      std::memcpy(request.buf, image_.data() + request.offset, request.len);
      events.push_back({model_.schedule(request.len), i});
    }
    std::sort(events.begin(), events.end());
    return 0;
  }

  int get_completed(PendingBatch &batch, IOContext &, int min_completed,
                    std::vector<uint32_t> &completed) override {
    completed.clear();
    if (min_completed < 0 || batch.n_reaped > batch.n_submitted) return -1;
    auto found = pending_.find(&batch);
    if (found == pending_.end()) return batch.n_reaped == batch.n_submitted ? 0 : -1;
    auto &events = found->second;
    const uint32_t remaining = batch.n_submitted - batch.n_reaped;
    const uint32_t count = std::min(remaining, static_cast<uint32_t>(min_completed));
    if (!count) return 0;
    for (uint32_t n = 0; n < count; ++n) {
      const auto &event = events[batch.n_reaped++];
      model_.advance(event.first);
      completed.push_back(event.second);
    }
    if (batch.n_reaped == batch.n_submitted) pending_.erase(found);
    return static_cast<int>(completed.size());
  }

 private:
  IOModel &model_;
  std::vector<char> image_;
  std::unordered_map<PendingBatch *, std::vector<std::pair<uint64_t, uint32_t>>> pending_;
};

inline IOModel *&ActiveModel() { static thread_local IOModel *model = nullptr; return model; }
inline std::weak_ptr<ControlledReader> &ActiveReader() {
  static thread_local std::weak_ptr<ControlledReader> reader;
  return reader;
}
inline std::shared_ptr<AlignedFileReader> MakeControlledReader(const std::string &path) {
  if (!ActiveModel()) throw std::runtime_error("Controlled model not configured");
  auto reader = std::make_shared<ControlledReader>(*ActiveModel());
  reader->open(path);
  ActiveReader() = reader;
  return reader;
}
class ControlledScope {
 public:
  explicit ControlledScope(IOModel &model) {
    if (ActiveModel()) throw std::runtime_error("Nested controlled benchmark");
    ActiveModel() = &model;
    ControlledReaderFactory() = MakeControlledReader;
  }
  ~ControlledScope() { ControlledReaderFactory() = nullptr; ActiveModel() = nullptr; }
};
}  // namespace zvec::core::benchmark
#endif
