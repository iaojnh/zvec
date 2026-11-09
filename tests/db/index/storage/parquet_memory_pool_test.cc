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

#include "db/index/storage/parquet_memory_pool.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <system_error>
#include <thread>
#include <ailego/pattern/scope_guard.h>
#include <arrow/api.h>
#include <arrow/buffer.h>
#include <gtest/gtest.h>
#include <parquet/arrow/reader.h>
#include <parquet/file_reader.h>
#include <parquet/statistics.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include "db/index/storage/chunked_file_writer.h"
#include "db/index/storage/fetch_memory.h"
#include "db/index/storage/parquet_writer.h"
#include "db/index/storage/store_helper.h"
#include "../../../ailego/buffer/pool_payload_budget.h"
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace zvec::ailego {

struct MemoryLimitPoolTestPeer {
  static bool configure_rss(MemoryLimitPool &pool, size_t limit) {
    pool.stop_background_evictor();
    return pool.process_rss_.configure(limit, +[] { return rss; });
  }
  static void refresh_rss(MemoryLimitPool &pool) {
    pool.process_rss_.refresh(true);
  }
  static size_t rss_available(MemoryLimitPool &pool) {
    return pool.process_rss_.available();
  }
  static constexpr size_t rss = 32 * 1024 * 1024;
};

}  // namespace zvec::ailego

namespace zvec {
namespace detail {
namespace {

constexpr int64_t kThreshold = ParquetMemoryPool::kMappedAllocationThreshold;

void RemoveTemporaryParquetFile(const std::filesystem::path &path) {
  // ScopeGuard's destructor must not throw when Windows still has a file open.
  // Report cleanup failures through gtest instead of terminating the process.
  std::error_code error;
  std::filesystem::remove(path, error);
  EXPECT_FALSE(error) << "Cannot remove " << path << ": " << error.message();
}

class RejectingMemoryPool : public arrow::ProxyMemoryPool {
 public:
  RejectingMemoryPool()
      : arrow::ProxyMemoryPool(arrow::default_memory_pool()) {}

  arrow::Status Allocate(int64_t, int64_t, uint8_t **) override {
    return arrow::Status::OutOfMemory("Injected small allocation failure");
  }
};

TEST(ParquetMemoryPoolTest, ReallocationPreservesDataAcrossHeapAndMapping) {
  arrow::ProxyMemoryPool fallback(arrow::default_memory_pool());
  ParquetMemoryPool pool(&fallback);
  uint8_t *buffer = nullptr;
  ASSERT_TRUE(pool.Allocate(1024, &buffer).ok());
  std::memset(buffer, 29, 1024);
  ASSERT_TRUE(pool.Reallocate(1024, 4096, &buffer).ok());
  EXPECT_EQ(4096, pool.bytes_allocated());
  EXPECT_EQ(4096, fallback.bytes_allocated());
  EXPECT_EQ(0, pool.num_mapped_allocations());
  EXPECT_EQ(29, buffer[1023]);
  std::memset(buffer, 29, 4096);
  ASSERT_TRUE(pool.Reallocate(4096, kThreshold, &buffer).ok());
  EXPECT_EQ(0, fallback.bytes_allocated());
  EXPECT_EQ(kThreshold, pool.bytes_allocated());
  EXPECT_EQ(kThreshold + 4096, pool.max_memory());
  EXPECT_TRUE(std::all_of(buffer, buffer + 4096,
                          [](uint8_t byte) { return byte == 29; }));
  ASSERT_TRUE(pool.Reallocate(kThreshold, 8192, &buffer).ok());
  EXPECT_EQ(0, pool.mapped_bytes());
  EXPECT_EQ(8192, fallback.bytes_allocated());
  EXPECT_EQ(8192, pool.bytes_allocated());
  EXPECT_EQ(kThreshold + 8192, pool.max_memory());
  EXPECT_TRUE(std::all_of(buffer, buffer + 4096,
                          [](uint8_t byte) { return byte == 29; }));
  pool.Free(buffer, 8192);
  EXPECT_EQ(0, pool.bytes_allocated());
  EXPECT_EQ(0, fallback.bytes_allocated());
  EXPECT_EQ(4, pool.num_allocations());
}

TEST(ParquetMemoryPoolTest,
     InvalidAndOverflowingRequestsDoNotChangeStatistics) {
  ParquetMemoryPool pool;
  uint8_t *buffer = nullptr;
  EXPECT_TRUE(pool.Allocate(-1, &buffer).IsInvalid());
  EXPECT_TRUE(pool.Allocate(4096, 0, &buffer).IsInvalid());
  EXPECT_TRUE(pool.Allocate(4096, 65, &buffer).IsInvalid());
  EXPECT_TRUE(pool.Allocate(4096, nullptr).IsInvalid());
  EXPECT_TRUE(pool.Allocate(std::numeric_limits<int64_t>::max(), &buffer)
                  .IsOutOfMemory());
  EXPECT_TRUE(pool.Reallocate(0, 4096, nullptr).IsInvalid());
  EXPECT_TRUE(pool.Reallocate(1024, 4096, &buffer).IsInvalid());
  EXPECT_EQ(nullptr, buffer);
  EXPECT_EQ(0, pool.bytes_allocated());
  EXPECT_EQ(0, pool.mapped_bytes());
  EXPECT_EQ(0, pool.num_allocations());
  EXPECT_EQ(0, pool.total_bytes_allocated());
}

TEST(ParquetMemoryPoolTest,
     FailedReallocationPreservesOriginalDataAndCounters) {
  RejectingMemoryPool fallback;
  ParquetMemoryPool pool(&fallback);
  uint8_t *buffer = nullptr;
  ASSERT_TRUE(pool.Allocate(kThreshold, &buffer).ok());
  std::memset(buffer, 11, kThreshold);
  uint8_t *original = buffer;
  const int64_t mapped = pool.mapped_bytes();
  ASSERT_TRUE(pool.Reallocate(kThreshold, 0, &buffer).IsOutOfMemory());
  EXPECT_TRUE(pool.Reallocate(kThreshold, 1024, &buffer).IsOutOfMemory());
  EXPECT_TRUE(
      pool.Reallocate(kThreshold, std::numeric_limits<int64_t>::max(), &buffer)
          .IsOutOfMemory());
  EXPECT_EQ(original, buffer);
  EXPECT_EQ(11, buffer[kThreshold - 1]);
  EXPECT_EQ(kThreshold, pool.bytes_allocated());
  EXPECT_EQ(mapped, pool.mapped_bytes());
  EXPECT_EQ(1, pool.num_allocations());
  EXPECT_EQ(1, pool.num_mapped_allocations());
  pool.Free(buffer, kThreshold);
  EXPECT_EQ(0, pool.bytes_allocated());
  EXPECT_EQ(0, pool.mapped_bytes());
}

class ParquetBudgetTest : public testing::Test {
 protected:
  void SetUp() override {
    previous_capacity_ = budget_.capacity();
    ASSERT_EQ(0u, budget_.used());
    ASSERT_EQ(0, budget_.init(1024 * 1024));
  }
  void TearDown() override {
    EXPECT_EQ(0u, budget_.used());
    EXPECT_EQ(0, budget_.init(previous_capacity_));
  }
  ailego::PoolPayloadBudget &budget_ =
      ailego::PoolPayloadBudget::get_instance();
  size_t previous_capacity_{0};
};

TEST_F(ParquetBudgetTest,
       RepeatedWriterBatchesDoNotAccumulateOneBufferedRowGroup) {
  ASSERT_EQ(0, budget_.init(8 * 1024 * 1024));
  const auto path =
      std::filesystem::temp_directory_path() /
      ("zvec_parquet_budget_" +
       std::to_string(
           std::chrono::steady_clock::now().time_since_epoch().count()));
  auto cleanup =
      ailego::ScopeGuard::Make([&] { RemoveTemporaryParquetFile(path); });
  auto schema = arrow::schema({arrow::field("payload", arrow::binary())});
  arrow::BinaryBuilder builder;
  for (int i = 0; i < 1024; ++i)
    ASSERT_TRUE(
        builder.Append(std::string(1024, 'x') + std::to_string(i)).ok());
  auto batch =
      arrow::RecordBatch::Make(schema, 1024, {builder.Finish().ValueOrDie()});
  auto writer =
      ChunkedFileWriter::Open(path.string(), schema, FileFormat::PARQUET);
  ASSERT_NE(nullptr, writer);
  for (int i = 0; i < 32; ++i) {
    ASSERT_TRUE(writer->write(*batch).ok());
    EXPECT_LE(budget_.used(), budget_.capacity());
  }
  ASSERT_GT(GetParquetMemoryPool()->total_bytes_allocated(), 0);
  ASSERT_TRUE(writer->close().ok());
  writer.reset();
  EXPECT_EQ(0u, budget_.used());
  auto reader = parquet::ParquetFileReader::OpenFile(path.string(), false);
  EXPECT_EQ(32 * 1024, reader->metadata()->num_rows());
  EXPECT_GE(reader->metadata()->num_row_groups(), 32);
  for (int i = 0; i < reader->metadata()->num_row_groups(); ++i)
    EXPECT_LE(reader->metadata()->RowGroup(i)->num_rows(), 1024);
}

TEST_F(ParquetBudgetTest, FailedBufferedWriteDoesNotResumeOrReportSuccess) {
  ASSERT_EQ(0, budget_.init(8 * 1024 * 1024));
  const auto path =
      std::filesystem::temp_directory_path() /
      ("zvec_parquet_failed_batch_" +
       std::to_string(
           std::chrono::steady_clock::now().time_since_epoch().count()));
  auto cleanup =
      ailego::ScopeGuard::Make([&] { RemoveTemporaryParquetFile(path); });
  auto schema = arrow::schema({arrow::field("payload", arrow::binary())});
  arrow::BinaryBuilder builder;
  ASSERT_TRUE(builder.Append(std::string(4096, 'x')).ok());
  auto batch =
      arrow::RecordBatch::Make(schema, 1, {builder.Finish().ValueOrDie()});
  auto writer =
      ChunkedFileWriter::Open(path.string(), schema, FileFormat::PARQUET);
  ASSERT_NE(nullptr, writer);
  const size_t remaining = budget_.capacity() - budget_.used();
  ASSERT_TRUE(budget_.try_charge_external(remaining));
  auto status = writer->write(*batch);
  budget_.release_external(remaining);
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status, writer->write(*batch));
  EXPECT_EQ(status, writer->close());
  writer.reset();
  EXPECT_EQ(0u, budget_.used());
}

TEST_F(ParquetBudgetTest, SmallBuffersCanBeFreedOnAnotherThread) {
  ParquetMemoryPool pool(arrow::default_memory_pool(), true);
  std::array<uint8_t *, 64> buffers{};
  for (auto &buffer : buffers) ASSERT_TRUE(pool.Allocate(1024, &buffer).ok());
  std::array<std::thread, 4> workers;
  for (size_t i = 0; i < workers.size(); ++i) {
    workers[i] = std::thread([&, i] {
      for (size_t j = i; j < buffers.size(); j += workers.size()) {
        std::memset(buffers[j], 41, 1024);
        uint8_t *temporary = nullptr;
        auto status = pool.Allocate(1024, &temporary);
        EXPECT_TRUE(status.ok());
        if (temporary) pool.Free(temporary, 1024);
        pool.Free(buffers[j], 1024);
      }
    });
  }
  for (auto &worker : workers) worker.join();
  EXPECT_EQ(0, pool.bytes_allocated());
  EXPECT_EQ(0u, pool.charged_bytes());
  EXPECT_EQ(0u, budget_.used());
  EXPECT_LE(budget_.committed(), budget_.capacity());
}

TEST_F(ParquetBudgetTest,
       ReallocationReservesOverlapAndFailurePreservesOldData) {
  ParquetMemoryPool pool(arrow::default_memory_pool(), true);
  uint8_t *buffer = nullptr;
  ASSERT_TRUE(pool.Allocate(kThreshold, &buffer).ok());
  std::memset(buffer, 17, kThreshold);
  auto *original = buffer;
  const auto charged = budget_.external_used();
  // New capacity alone fits, but old + new backing cannot coexist.
  EXPECT_TRUE(
      pool.Reallocate(kThreshold, 3 * kThreshold, &buffer).IsOutOfMemory());
  EXPECT_EQ(original, buffer);
  EXPECT_EQ(charged, budget_.external_used());
  EXPECT_EQ(17, buffer[kThreshold - 1]);
  ASSERT_TRUE(pool.Reallocate(kThreshold, 2 * kThreshold, &buffer).ok());
  EXPECT_EQ(17, buffer[kThreshold - 1]);
  EXPECT_EQ(pool.mapped_bytes(), budget_.external_used());
  ASSERT_TRUE(pool.Reallocate(2 * kThreshold, 1024, &buffer).ok());
  EXPECT_EQ(17, buffer[1023]);
  ASSERT_TRUE(pool.Reallocate(1024, 0, &buffer).ok());
  EXPECT_NE(nullptr, buffer);
  EXPECT_EQ(0u, budget_.external_used());
  pool.Free(buffer, 0);
}

TEST_F(ParquetBudgetTest, FetchBatchCanShrinkMappedBufferAtItsLimit) {
  ParquetMemoryPool parent(arrow::default_memory_pool(), true);
  FetchMemoryPool pool(&parent, 2 * kThreshold);
  uint8_t *buffer = nullptr;
  constexpr int64_t kAlignment = 64;
  ASSERT_TRUE(pool.Allocate(2 * kThreshold, kAlignment, &buffer).ok());
  auto *original = buffer;
  const auto mapped = parent.mapped_bytes();
  constexpr int64_t kNewSize = 8192;
  std::memset(buffer, 41, kNewSize);
  ASSERT_TRUE(
      pool.Reallocate(2 * kThreshold, kNewSize, kAlignment, &buffer).ok());
  EXPECT_FALSE(pool.exceeded());
  EXPECT_EQ(original, buffer);
  EXPECT_EQ(41, buffer[kNewSize - 1]);
  EXPECT_EQ(kNewSize, pool.bytes_allocated());
  EXPECT_EQ(2 * kThreshold, pool.max_memory());
  EXPECT_LT(parent.mapped_bytes(), mapped);
  EXPECT_TRUE(
      pool.Reallocate(kNewSize, 2 * kThreshold, kAlignment, &buffer)
          .IsOutOfMemory());
  EXPECT_TRUE(pool.exceeded());
  EXPECT_EQ(original, buffer);
  pool.Free(buffer, kNewSize, kAlignment);
  EXPECT_EQ(0, pool.bytes_allocated());
  EXPECT_EQ(0u, budget_.external_used());
}

}  // namespace
}  // namespace detail
}  // namespace zvec
