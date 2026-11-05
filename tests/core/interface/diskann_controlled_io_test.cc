// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0.
#include <gtest/gtest.h>
#include <array>
#include "benchmarks/diskann-controlled-io/io_model.h"
#include "diskann_controlled_io.h"
#include "diskann_paced_io.h"
#include "diskann_concurrent_queries.h"
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
TEST(DiskAnnNativePacing, SustainsCommonRateWithoutTimerQuantumOversleep) {
  NativeRateLimiter limiter(PaceConfig{2000, 16777216, 8});
  const auto start = PaceClock::now();
  for (int i = 0; i < 128; ++i) limiter.admit(8, 32768);
  const double seconds = std::chrono::duration<double>(PaceClock::now() - start).count();
  EXPECT_GE(1024 / seconds, 1800.0);
  EXPECT_LE(1024 / seconds, 2100.0);
  EXPECT_GT(limiter.stats().wait_ns, 0U);
}
class ImmediateNativeReader : public AlignedFileReader {
 public:
  std::atomic<size_t> largest_batch{0};
  std::atomic<uint32_t> active_reads{0}, peak_active_reads{0};
  uint32_t pause_ms{0};
  bool closed{false};
  void open(const std::string &) override {}
  void close() override { closed = true; }
  bool requires_io_context() const override { return false; }
  void release_io_ctx(IOContext &) override {}
  int read(std::vector<AlignedRead> &, IOContext &, bool = false) override { return -1; }
  int submit(PendingBatch &batch, std::vector<AlignedRead> &requests, IOContext &) override {
    auto previous = largest_batch.load();
    while (previous < requests.size() && !largest_batch.compare_exchange_weak(previous, requests.size())) {}
    batch.n_submitted = static_cast<uint32_t>(requests.size());
    batch.n_reaped = 0;
    const auto active = active_reads.fetch_add(batch.n_submitted) + batch.n_submitted;
    auto peak = peak_active_reads.load();
    while (peak < active && !peak_active_reads.compare_exchange_weak(peak, active)) {}
    if (pause_ms) std::this_thread::sleep_for(std::chrono::milliseconds(pause_ms));
    for (const auto &r : requests) std::memset(r.buf, static_cast<int>(r.offset % 251), r.len);
    return 0;
  }
  int get_completed(PendingBatch &batch, IOContext &, int, std::vector<uint32_t> &completed) override {
    completed.clear();
    while (batch.n_reaped < batch.n_submitted) completed.push_back(batch.n_reaped++);
    active_reads.fetch_sub(static_cast<uint32_t>(completed.size()));
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
  EXPECT_LE(native->largest_batch.load(), 4U);
  EXPECT_EQ(17U, limiter.stats().reads);
  EXPECT_EQ(5U, limiter.stats().batches);
  reader.close();
  EXPECT_TRUE(native->closed);
}
TEST(DiskAnnNativePacing, ConcurrentWorkersShareOneRateAndDepthBudget) {
  for (const auto config : {PaceConfig{1000, 0, 8}, PaceConfig{0, 4096000, 8}, PaceConfig{0, 0, 8}}) {
  auto native = std::make_shared<ImmediateNativeReader>();
  // Slow native I/O is only needed for depth contention. Ordinary OS sleep
  // can coalesce beyond the rate interval, hiding a correctly idle limiter.
  native->pause_ms = (config.iops || config.bytes_per_second) ? 0 : 2;
  NativeRateLimiter limiter(config);
  RateLimitedReader reader(native, limiter);
  ConcurrentQueryWorkers workers(8);
  std::atomic<uint32_t> successful{0};
  const auto start = PaceClock::now();
  workers.Run([&](uint32_t worker) {
    alignas(4096) std::array<char, 8192> data{};
    IOContext ctx = nullptr;
    for (uint32_t i = 0; i < 16; ++i) {
      std::vector<AlignedRead> requests{
          AlignedRead(uint64_t(worker * 32 + i * 2) * 4096, 4096, data.data()),
          AlignedRead(uint64_t(worker * 32 + i * 2 + 1) * 4096, 4096, data.data() + 4096)};
      if (reader.read(requests, ctx) || data[0] != static_cast<char>(requests[0].offset % 251) ||
          data[4096] != static_cast<char>(requests[1].offset % 251))
        throw std::runtime_error("Concurrent native payload/completion mismatch");
      successful.fetch_add(2);
    }
  });
  const auto seconds = std::chrono::duration<double>(PaceClock::now() - start).count();
  EXPECT_EQ(256U, successful.load());
  EXPECT_EQ(256U, limiter.stats().reads);
  EXPECT_EQ(256U * 4096, limiter.stats().bytes);
  if (config.iops || config.bytes_per_second)
    EXPECT_GE(seconds, 0.23);  // Neither rate budget may multiply by eight.
  EXPECT_LE(limiter.stats().max_reserved_reads, 8U);
  EXPECT_LE(native->peak_active_reads.load(), 8U);
  EXPECT_EQ(0U, native->active_reads.load());
  if (config.iops || config.bytes_per_second) EXPECT_GT(limiter.stats().wait_ns, 0U);
  reader.close();
  }
}
TEST(DiskAnnConcurrentWorkers, ExecuteTogetherAndRetainThreadsAcrossPhases) {
  ConcurrentQueryWorkers workers(4);
  std::mutex mutex;
  std::condition_variable all_entered;
  uint32_t entered = 0;
  std::array<std::thread::id, 4> first{};
  workers.Run([&](uint32_t worker) {
    first[worker] = std::this_thread::get_id();
    std::unique_lock<std::mutex> lock(mutex);
    ++entered;
    all_entered.notify_all();
    if (!all_entered.wait_for(lock, std::chrono::seconds(3), [&] { return entered == 4; }))
      throw std::runtime_error("Query workers serialized unexpectedly");
  });
  std::array<std::thread::id, 4> second{};
  workers.Run([&](uint32_t worker) { second[worker] = std::this_thread::get_id(); });
  EXPECT_EQ(first, second);
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
