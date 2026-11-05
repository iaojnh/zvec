// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0.
#include <gtest/gtest.h>
#include <array>
#include "benchmarks/diskann-controlled-io/io_model.h"
#include "diskann_controlled_io.h"
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
