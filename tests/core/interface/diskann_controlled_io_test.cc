// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0.
#include <gtest/gtest.h>
#include <array>
#include "benchmarks/diskann-controlled-io/io_model.h"
#include "diskann_controlled_io.h"
#include "diskann_paced_io.h"
#include <chrono>
#include <cstdio>
#include <fstream>

namespace zvec::core::benchmark {
TEST(DiskAnnIOModel, SerialAndParallelRequests) {
  IOModel serial(200000, 1, 1024000000);
  EXPECT_EQ(204000U, serial.schedule(4096));
  EXPECT_EQ(408000U, serial.schedule(4096));
  IOModel parallel(200000, 2, 1024000000);
  EXPECT_EQ(204000U, parallel.schedule(4096));
  EXPECT_EQ(208000U, parallel.schedule(4096));
  EXPECT_EQ(408000U, parallel.schedule(4096));
}
TEST(DiskAnnIOModel, BandwidthAndDependentBatches) {
  IOModel model(200000, 8, 4096000);
  const auto first = model.schedule(4096);
  EXPECT_EQ(1200000U, first);
  EXPECT_EQ(2200000U, model.schedule(4096));
  model.advance(2200000);
  EXPECT_EQ(3400000U, model.schedule(4096));
  EXPECT_EQ(3U, model.requests());
  EXPECT_EQ(12288U, model.bytes());
  model.reset();
  EXPECT_EQ(0U, model.now_ns());
  EXPECT_EQ(0U, model.requests());
  EXPECT_EQ(first, model.schedule(4096));
}
TEST(DiskAnnIOModel, RejectInvalidAndOverflowInputs) {
  EXPECT_THROW(IOModel(0, 1, 1), std::invalid_argument);
  EXPECT_THROW(IOModel(1, 0, 1), std::invalid_argument);
  EXPECT_THROW(IOModel(1, 129, 1), std::invalid_argument);
  EXPECT_THROW(IOModel(1, UINT32_MAX, 1), std::invalid_argument);
  EXPECT_THROW(IOModel(1, 1, 0), std::invalid_argument);
  IOModel model(200000, 1, 1024000000);
  EXPECT_THROW(model.schedule(0), std::invalid_argument);
  EXPECT_THROW(model.schedule(UINT64_MAX), std::overflow_error);
  model.advance(UINT64_MAX);
  EXPECT_THROW(model.schedule(4096), std::overflow_error);
}

#if defined(ZVEC_DISKANN_CONTROLLED_IO_BENCHMARK)
TEST(DiskAnnNativePacing, WaitsInRealTimeAndKeepsCreditsAcrossCalls) {
  NativeRateLimiter limiter(PaceConfig{200, 819200, 1});
  const auto start = PaceClock::now();
  for (int i = 0; i < 10; ++i) limiter.admit(1, 4096);
  const auto ms = std::chrono::duration<double, std::milli>(PaceClock::now() - start).count();
  EXPECT_GE(ms, 40.0);
  EXPECT_GE(limiter.stats().wait_ns, 30000000U);
  EXPECT_EQ(10U, limiter.stats().reads);
  EXPECT_EQ(40960U, limiter.stats().bytes);
}
TEST(DiskAnnNativePacing, EnforcesBandwidthIndependently) {
  NativeRateLimiter limiter(PaceConfig{0, 409600, 1});
  const auto start = PaceClock::now();
  for (int i = 0; i < 5; ++i) limiter.admit(1, 4096);
  const auto elapsed_ms = std::chrono::duration<double, std::milli>(PaceClock::now() - start).count();
  EXPECT_GE(elapsed_ms, 35.0);
  EXPECT_THROW(NativeRateLimiter(PaceConfig{1, 1, 0}), std::invalid_argument);
  EXPECT_THROW(NativeRateLimiter(PaceConfig{1, 1, 129}), std::invalid_argument);
}
class ImmediateNativeReader : public AlignedFileReader {
 public:
  size_t largest_batch{0};
  bool closed{false};
  void open(const std::string &) override {}
  void close() override { closed = true; }
  bool requires_io_context() const override { return false; }
  void release_io_ctx(IOContext &) override {}
  int read(std::vector<AlignedRead> &, IOContext &, bool = false) override { return -1; }
  int submit(PendingBatch &batch, std::vector<AlignedRead> &requests, IOContext &) override {
    largest_batch = std::max(largest_batch, requests.size());
    batch.n_submitted = static_cast<uint32_t>(requests.size());
    batch.n_reaped = 0;
    for (const auto &r : requests) std::memset(r.buf, static_cast<int>(r.offset % 251), r.len);
    return 0;
  }
  int get_completed(PendingBatch &batch, IOContext &, int, std::vector<uint32_t> &completed) override {
    completed.clear();
    while (batch.n_reaped < batch.n_submitted) completed.push_back(batch.n_reaped++);
    return static_cast<int>(completed.size());
  }
};
TEST(DiskAnnNativePacing, SplitsNativeBatchesAndPreservesCompletionIndices) {
  auto native = std::make_shared<ImmediateNativeReader>();
  NativeRateLimiter limiter(PaceConfig{0, 0, 4});
  RateLimitedReader reader(native, limiter);
  std::vector<char> allocation(17 * 4096 + 4096);
  auto data = reinterpret_cast<char *>((reinterpret_cast<uintptr_t>(allocation.data()) + 4095) & ~uintptr_t(4095));
  std::vector<AlignedRead> requests;
  for (size_t i = 0; i < 17; ++i) requests.emplace_back(i * 4096, 4096, data + i * 4096);
  IOContext ctx = nullptr;
  PendingBatch batch;
  ASSERT_EQ(0, reader.submit(batch, requests, ctx));
  EXPECT_EQ(-1, reader.submit(batch, requests, ctx));
  std::vector<uint32_t> seen, completed;
  while (batch.n_reaped < batch.n_submitted) {
    ASSERT_GT(reader.get_completed(batch, ctx, 1, completed), 0);
    seen.insert(seen.end(), completed.begin(), completed.end());
  }
  ASSERT_EQ(17U, seen.size());
  for (size_t i = 0; i < seen.size(); ++i) {
    EXPECT_EQ(i, seen[i]);
    EXPECT_EQ(static_cast<char>((i * 4096) % 251), data[i * 4096]);
  }
  EXPECT_LE(native->largest_batch, 4U);
  EXPECT_EQ(17U, limiter.stats().reads);
  EXPECT_EQ(5U, limiter.stats().batches);
  reader.close();
  EXPECT_TRUE(native->closed);
}
TEST(DiskAnnIOModel, PayloadCompletionLifecycleAndBounds) {
  const auto path = testing::TempDir() + "zvec-controlled-" + std::to_string(
      std::chrono::steady_clock::now().time_since_epoch().count());
  struct Remove { std::string path; ~Remove() { std::remove(path.c_str()); } } cleanup{path};
  std::vector<char> payload(12288);
  for (size_t i = 0; i < payload.size(); ++i) payload[i] = static_cast<char>(i % 251);
  { std::ofstream output(path, std::ios::binary); output.write(payload.data(), payload.size()); }
  IOModel model(200000, 2, 1024000000);
  ControlledReader reader(model);
  reader.open(path);
  IOContext ctx = nullptr;
  alignas(4096) std::array<char, 8192> buffers{};
  std::vector<AlignedRead> requests{
      AlignedRead(0, 4096, buffers.data()),
      AlignedRead(8192, 4096, buffers.data() + 4096)};
  PendingBatch batch;
  ASSERT_EQ(0, reader.submit(batch, requests, ctx));
  EXPECT_EQ(-1, reader.submit(batch, requests, ctx));
  std::vector<uint32_t> completed;
  ASSERT_EQ(1, reader.get_completed(batch, ctx, 1, completed));
  EXPECT_EQ(0U, completed.at(0));
  EXPECT_EQ(204000U, model.now_ns());
  ASSERT_EQ(1, reader.get_completed(batch, ctx, 1, completed));
  EXPECT_EQ(1U, completed.at(0));
  EXPECT_EQ(208000U, model.now_ns());
  EXPECT_EQ(0, reader.get_completed(batch, ctx, 1, completed));
  EXPECT_TRUE(std::equal(buffers.begin(), buffers.begin() + 4096, payload.begin()));
  EXPECT_TRUE(std::equal(buffers.begin() + 4096, buffers.end(), payload.begin() + 8192));
  model.reset();
  EXPECT_EQ(0, reader.read(requests, ctx));
  EXPECT_EQ(208000U, model.now_ns());
  requests[0].offset = payload.size();
  EXPECT_EQ(-1, reader.read(requests, ctx));
}
#endif
}  // namespace zvec::core::benchmark
