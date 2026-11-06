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
#include "db/index/storage/parquet_writer.h"
#include "db/index/storage/store_helper.h"
#include "../../../ailego/buffer/pool_payload_budget.h"
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace zvec {
namespace detail {
namespace {

constexpr int64_t kThreshold = ParquetMemoryPool::kMappedAllocationThreshold;

class RejectingMemoryPool : public arrow::ProxyMemoryPool {
 public:
  RejectingMemoryPool()
      : arrow::ProxyMemoryPool(arrow::default_memory_pool()) {}

  arrow::Status Allocate(int64_t, int64_t, uint8_t **) override {
    return arrow::Status::OutOfMemory("Injected small allocation failure");
  }
};

TEST(ParquetMemoryPoolTest, SmallAllocationsUseFallback) {
  arrow::ProxyMemoryPool fallback(arrow::default_memory_pool());
  ParquetMemoryPool pool(&fallback);
  uint8_t *buffer = nullptr;
  ASSERT_TRUE(pool.Allocate(kThreshold - 1, &buffer).ok());
  ASSERT_NE(nullptr, buffer);
  EXPECT_EQ(0u, reinterpret_cast<uintptr_t>(buffer) % 64);
  EXPECT_EQ(kThreshold - 1, fallback.bytes_allocated());
  EXPECT_EQ(kThreshold - 1, pool.bytes_allocated());
  EXPECT_EQ(0, pool.mapped_bytes());
  EXPECT_EQ(0, pool.num_mapped_allocations());
  pool.Free(buffer, kThreshold - 1);
  EXPECT_EQ(0, pool.bytes_allocated());
  EXPECT_EQ(0, fallback.bytes_allocated());
  EXPECT_EQ(kThreshold - 1, pool.max_memory());
  EXPECT_EQ(kThreshold - 1, pool.total_bytes_allocated());
  EXPECT_EQ(1, pool.num_allocations());
}

TEST(ParquetMemoryPoolTest, ThresholdAllocationsUseIndependentMappings) {
  RejectingMemoryPool fallback;
  ParquetMemoryPool pool(&fallback);
  uint8_t *first = nullptr;
  uint8_t *second = nullptr;
  ASSERT_TRUE(pool.Allocate(kThreshold, &first).ok());
  ASSERT_TRUE(pool.Allocate(kThreshold + 1, &second).ok());
  ASSERT_NE(first, second);
  std::memset(first, 17, kThreshold);
  std::memset(second, 31, kThreshold + 1);
  EXPECT_EQ(2 * kThreshold + 1, pool.bytes_allocated());
  EXPECT_GT(pool.mapped_bytes(), pool.bytes_allocated());
  EXPECT_EQ(2, pool.num_mapped_allocations());
  const int64_t before = pool.mapped_bytes();
  pool.Free(first, kThreshold);
  EXPECT_GT(pool.mapped_bytes(), 0);
  EXPECT_LT(pool.mapped_bytes(), before);
  EXPECT_EQ(31, second[kThreshold]);
  pool.Free(second, kThreshold + 1);
  EXPECT_EQ(0, pool.bytes_allocated());
  EXPECT_EQ(0, pool.mapped_bytes());
}

TEST(ParquetMemoryPoolTest, PreservesSmallAndLargeAlignment) {
  ParquetMemoryPool pool;
  const std::array<int64_t, 4> alignments{16, 64, 4096, 65536};
  const std::array<int64_t, 2> sizes{1024, kThreshold + 7};
  for (const int64_t alignment : alignments) {
    for (const int64_t size : sizes) {
      uint8_t *buffer = nullptr;
      ASSERT_TRUE(pool.Allocate(size, alignment, &buffer).ok());
      EXPECT_EQ(0u,
                reinterpret_cast<uintptr_t>(buffer) %
                    static_cast<uintptr_t>(std::max<int64_t>(64, alignment)));
      buffer[0] = 1;
      buffer[size - 1] = 2;
      pool.Free(buffer, size, alignment);
    }
  }
  EXPECT_EQ(0, pool.bytes_allocated());
  EXPECT_EQ(0, pool.mapped_bytes());
}

TEST(ParquetMemoryPoolTest, ReallocationCrossesThresholdInBothDirections) {
  arrow::ProxyMemoryPool fallback(arrow::default_memory_pool());
  ParquetMemoryPool pool(&fallback);
  uint8_t *buffer = nullptr;
  ASSERT_TRUE(pool.Allocate(4096, &buffer).ok());
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
  EXPECT_EQ(3, pool.num_allocations());
}

TEST(ParquetMemoryPoolTest,
     LargeReallocationPreservesDataAndReleasesOldMapping) {
  ParquetMemoryPool pool;
  uint8_t *buffer = nullptr;
  constexpr int64_t alignment = 65536;
  ASSERT_TRUE(pool.Allocate(kThreshold, alignment, &buffer).ok());
  std::memset(buffer, 41, kThreshold);
  ASSERT_TRUE(
      pool.Reallocate(kThreshold, kThreshold * 2, alignment, &buffer).ok());
  EXPECT_EQ(0u, reinterpret_cast<uintptr_t>(buffer) % alignment);
  EXPECT_TRUE(std::all_of(buffer, buffer + kThreshold,
                          [](uint8_t byte) { return byte == 41; }));
  EXPECT_EQ(3 * kThreshold, pool.max_memory());
  EXPECT_EQ(2 * kThreshold, pool.bytes_allocated());
  EXPECT_LT(pool.mapped_bytes(), 3 * kThreshold);
  ASSERT_TRUE(
      pool.Reallocate(kThreshold * 2, kThreshold, alignment, &buffer).ok());
  EXPECT_EQ(41, buffer[kThreshold - 1]);
  EXPECT_EQ(2, pool.num_mapped_allocations());
  pool.Free(buffer, kThreshold, alignment);
  EXPECT_EQ(0, pool.bytes_allocated());
  EXPECT_EQ(0, pool.mapped_bytes());
}

TEST(ParquetMemoryPoolTest, MappedShrinkKeepsAddressAlignmentAndPrefix) {
  ParquetMemoryPool pool;
  for (const int64_t alignment : {64, 4096, 65536}) {
    SCOPED_TRACE(alignment);
    uint8_t *buffer = nullptr;
    ASSERT_TRUE(pool.Allocate(4 * kThreshold, alignment, &buffer).ok());
    std::memset(buffer, 37, 4 * kThreshold);
    auto *original = buffer;
    const auto mapped = pool.mapped_bytes();
    const auto allocations = pool.num_mapped_allocations();
    const auto total = pool.total_mapped_bytes_allocated();
    ASSERT_TRUE(
        pool.Reallocate(4 * kThreshold, kThreshold + 100, alignment, &buffer)
            .ok());
    EXPECT_EQ(original, buffer);
    EXPECT_EQ(0u, reinterpret_cast<uintptr_t>(buffer) % alignment);
    EXPECT_TRUE(std::all_of(buffer, buffer + kThreshold + 100,
                            [](uint8_t byte) { return byte == 37; }));
    EXPECT_LT(pool.mapped_bytes(), mapped);
#if defined(_WIN32)
    MEMORY_BASIC_INFORMATION tail{};
    ASSERT_NE(0u, ::VirtualQuery(buffer + 3 * kThreshold, &tail, sizeof(tail)));
    EXPECT_EQ(MEM_RESERVE, tail.State);
#endif
    EXPECT_EQ(allocations, pool.num_mapped_allocations());
    EXPECT_EQ(total, pool.total_mapped_bytes_allocated());
    // Logical accounting changes even when no complete OS page can be freed.
    const auto retained = pool.mapped_bytes();
    ASSERT_TRUE(
        pool.Reallocate(kThreshold + 100, kThreshold + 99, alignment, &buffer)
            .ok());
    EXPECT_EQ(original, buffer);
    EXPECT_EQ(retained, pool.mapped_bytes());
    EXPECT_EQ(kThreshold + 99, pool.bytes_allocated());
    pool.Free(buffer, kThreshold + 99, alignment);
    EXPECT_EQ(0, pool.mapped_bytes());
    EXPECT_EQ(0, pool.bytes_allocated());
  }
}

TEST(ParquetMemoryPoolTest, SmallReallocationUsesFallback) {
  arrow::ProxyMemoryPool fallback(arrow::default_memory_pool());
  ParquetMemoryPool pool(&fallback);
  uint8_t *buffer = nullptr;
  ASSERT_TRUE(pool.Allocate(1024, &buffer).ok());
  std::memset(buffer, 19, 1024);
  ASSERT_TRUE(pool.Reallocate(1024, 8192, &buffer).ok());
  EXPECT_EQ(8192, pool.bytes_allocated());
  EXPECT_EQ(8192, fallback.bytes_allocated());
  EXPECT_EQ(0, pool.num_mapped_allocations());
  EXPECT_EQ(19, buffer[1023]);
  pool.Free(buffer, 8192);
  EXPECT_EQ(0, pool.bytes_allocated());
}

TEST(ParquetMemoryPoolTest, ZeroAllocationAndReallocationAreSafe) {
  for (int64_t size : {int64_t{1024}, kThreshold}) {
    SCOPED_TRACE(size);
    arrow::ProxyMemoryPool fallback(arrow::default_memory_pool());
    ParquetMemoryPool pool(&fallback);
    uint8_t *buffer = nullptr;
    ASSERT_TRUE(pool.Allocate(0, &buffer).ok());
    EXPECT_NE(nullptr, buffer);
    ASSERT_TRUE(pool.Reallocate(0, 0, &buffer).ok());
    EXPECT_NE(nullptr, buffer);
    EXPECT_EQ(0, pool.bytes_allocated());
    EXPECT_EQ(0, pool.mapped_bytes());
    pool.Free(buffer, 0);

    buffer = nullptr;
    ASSERT_TRUE(pool.Allocate(size, &buffer).ok());
    ASSERT_NE(nullptr, buffer);
    ASSERT_TRUE(pool.Reallocate(size, 0, &buffer).ok());
    EXPECT_NE(nullptr, buffer);
    EXPECT_EQ(0, pool.bytes_allocated());
    EXPECT_EQ(0, pool.mapped_bytes());
    EXPECT_EQ(0, fallback.bytes_allocated());
    ASSERT_TRUE(pool.Reallocate(0, size, &buffer).ok());
    ASSERT_NE(nullptr, buffer);
    pool.Free(buffer, size);
    EXPECT_EQ(0, pool.bytes_allocated());
    EXPECT_EQ(0, pool.mapped_bytes());
    EXPECT_EQ(0, fallback.bytes_allocated());
  }
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

TEST(ParquetMemoryPoolTest, ArrowResizableBufferUsesPoolUntilReleased) {
  ParquetMemoryPool pool;
  auto result = arrow::AllocateResizableBuffer(kThreshold, &pool);
  ASSERT_TRUE(result.ok()) << result.status().ToString();
  auto buffer = std::move(result).ValueOrDie();
  EXPECT_EQ(kThreshold, pool.bytes_allocated());
  ASSERT_TRUE(buffer->Resize(kThreshold * 2).ok());
  EXPECT_EQ(kThreshold * 2, pool.bytes_allocated());
  ASSERT_TRUE(buffer->Resize(0).ok());
  EXPECT_NE(nullptr, buffer->data());
  EXPECT_EQ(0, pool.mapped_bytes());
  buffer.reset();
  EXPECT_EQ(0, pool.bytes_allocated());
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

TEST_F(ParquetBudgetTest, ScalarConversionAdmitsBeforeAllocationAndCanRetry) {
  arrow::Int32Builder values_builder;
  ASSERT_TRUE(values_builder.Append(7).ok());
  auto values_result = values_builder.Finish();
  ASSERT_TRUE(values_result.ok());
  auto values = values_result.ValueOrDie();
  const std::vector<std::shared_ptr<arrow::Scalar>> samples{
      std::make_shared<arrow::Int32Scalar>(7),
      std::make_shared<arrow::StringScalar>("payload"),
      std::make_shared<arrow::BinaryScalar>("binary"),
      std::make_shared<arrow::ListScalar>(values)};
  ParquetMemoryPool pool(arrow::default_memory_pool(), true);
  for (const auto &sample : samples) {
    SCOPED_TRACE(sample->type->ToString());
    const std::vector<std::shared_ptr<arrow::Scalar>> input{sample};
    std::shared_ptr<arrow::Array> output = values;
    ASSERT_TRUE(budget_.try_charge_external(budget_.capacity()));
    {
      auto release = ailego::ScopeGuard::Make(
          [&] { budget_.release_external(budget_.capacity()); });
      EXPECT_TRUE(ConvertScalarVectorToArrayByType(input, &output, &pool)
                      .IsOutOfMemory());
      EXPECT_EQ(values, output);
      EXPECT_EQ(0, pool.bytes_allocated());
    }
    ASSERT_TRUE(ConvertScalarVectorToArrayByType(input, &output, &pool).ok());
    auto actual = output->GetScalar(0);
    ASSERT_TRUE(actual.ok());
    EXPECT_TRUE(actual.ValueOrDie()->Equals(*sample));
    EXPECT_GT(pool.charged_bytes(), 0u);
    actual = arrow::Status::Invalid("Release result before checking budget");
    output.reset();
    EXPECT_EQ(0u, budget_.used());
  }
}

TEST_F(ParquetBudgetTest, ReservesBackingCapacityBeforeAllocation) {
  arrow::ProxyMemoryPool fallback(arrow::default_memory_pool());
  ParquetMemoryPool pool(&fallback, true);
  uint8_t *small = nullptr, *large = nullptr, *denied = nullptr;
  ASSERT_TRUE(pool.Allocate(1024, &small).ok());
  EXPECT_EQ(0, fallback.bytes_allocated());
  const auto small_backing = budget_.external_used();
  EXPECT_GT(budget_.external_used(), 1024u);
  ASSERT_TRUE(pool.Allocate(kThreshold, 65536, &large).ok());
  EXPECT_EQ(small_backing + pool.mapped_bytes(), budget_.external_used());
  EXPECT_EQ(pool.charged_bytes(), budget_.external_used());
  const auto allocations = pool.num_allocations();
  EXPECT_TRUE(pool.Allocate(1024 * 1024, &denied).IsOutOfMemory());
  EXPECT_EQ(nullptr, denied);
  EXPECT_EQ(allocations, pool.num_allocations());
  pool.Free(large, kThreshold, 65536);
  pool.Free(small, 1024);
  EXPECT_EQ(0u, pool.charged_bytes());
}

TEST_F(ParquetBudgetTest,
       RepeatedWriterBatchesDoNotAccumulateOneBufferedRowGroup) {
  ASSERT_EQ(0, budget_.init(8 * 1024 * 1024));
  const auto path =
      std::filesystem::temp_directory_path() /
      ("zvec_parquet_budget_" +
       std::to_string(
           std::chrono::steady_clock::now().time_since_epoch().count()));
  auto cleanup =
      ailego::ScopeGuard::Make([&] { std::filesystem::remove(path); });
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

TEST_F(ParquetBudgetTest, SmallBackingAdmissionFailureCanRetry) {
  RejectingMemoryPool fallback;
  ParquetMemoryPool pool(&fallback, true);
  uint8_t *buffer = nullptr;
  ASSERT_TRUE(budget_.try_charge_external(budget_.capacity()));
  EXPECT_TRUE(pool.Allocate(1024, &buffer).IsOutOfMemory());
  EXPECT_EQ(nullptr, buffer);
  EXPECT_EQ(budget_.capacity(), budget_.external_used());
  EXPECT_EQ(0u, pool.charged_bytes());
  EXPECT_EQ(0, pool.num_allocations());
  budget_.release_external(budget_.capacity());
  ASSERT_TRUE(pool.Allocate(1024, &buffer).ok());
  pool.Free(buffer, 1024);
  EXPECT_EQ(0u, budget_.used());
}

TEST_F(ParquetBudgetTest, SmallBatchesShareBoundedRowGroups) {
  ASSERT_EQ(0, budget_.init(64 * 1024 * 1024));
  const auto path =
      std::filesystem::temp_directory_path() /
      ("zvec_parquet_small_batches_" +
       std::to_string(
           std::chrono::steady_clock::now().time_since_epoch().count()));
  auto cleanup =
      ailego::ScopeGuard::Make([&] { std::filesystem::remove(path); });
  auto schema = arrow::schema({arrow::field("payload", arrow::binary())});
  arrow::BinaryBuilder builder;
  for (int i = 0; i < 64; ++i)
    ASSERT_TRUE(builder.Append(std::string(64, 'x') + std::to_string(i)).ok());
  auto array = builder.Finish().ValueOrDie();
  auto batch = arrow::RecordBatch::Make(schema, 64, {array});
  auto table = arrow::Table::FromRecordBatches({batch}).ValueOrDie();
  auto writer =
      ChunkedFileWriter::Open(path.string(), schema, FileFormat::PARQUET);
  ASSERT_NE(nullptr, writer);
  for (int i = 0; i < 256; ++i) {
    auto status = i % 2 ? writer->write(*batch) : writer->write(*table);
    ASSERT_TRUE(status.ok()) << status.ToString();
    EXPECT_LE(budget_.used(), budget_.capacity());
  }
  ASSERT_TRUE(writer->close().ok());
  writer.reset();
  EXPECT_EQ(0u, budget_.used());
  auto file = parquet::ParquetFileReader::OpenFile(path.string(), false);
  EXPECT_EQ(16384, file->metadata()->num_rows());
  ASSERT_EQ(2, file->metadata()->num_row_groups());
  EXPECT_EQ(8192, file->metadata()->RowGroup(0)->num_rows());
  EXPECT_EQ(8192, file->metadata()->RowGroup(1)->num_rows());
  auto input = arrow::io::ReadableFile::Open(path.string()).ValueOrDie();
  auto reader = parquet::arrow::OpenFile(input, arrow::default_memory_pool())
                    .ValueOrDie();
  std::shared_ptr<arrow::Table> actual;
  ASSERT_TRUE(reader->ReadTable(&actual).ok());
  auto expected =
      arrow::Table::FromRecordBatches(
          std::vector<std::shared_ptr<arrow::RecordBatch>>(256, batch))
          .ValueOrDie();
  EXPECT_TRUE(actual->Equals(*expected));
}

TEST_F(ParquetBudgetTest, SlicedBatchesUseReferencedBytesAndExplicitRowLimit) {
  ASSERT_EQ(0, budget_.init(64 * 1024 * 1024));
  const auto path =
      std::filesystem::temp_directory_path() /
      ("zvec_parquet_sliced_batches_" +
       std::to_string(
           std::chrono::steady_clock::now().time_since_epoch().count()));
  auto cleanup =
      ailego::ScopeGuard::Make([&] { std::filesystem::remove(path); });
  auto schema = arrow::schema({arrow::field("payload", arrow::binary())});
  arrow::BinaryBuilder builder;
  for (int i = 0; i < 8192; ++i)
    ASSERT_TRUE(builder.Append(std::string(512, 'a') + std::to_string(i)).ok());
  auto batch =
      arrow::RecordBatch::Make(schema, 8192, {builder.Finish().ValueOrDie()});
  std::vector<std::shared_ptr<arrow::RecordBatch>> expected_batches;
  {
    ParquetWriter writer(path.string(), 1000);
    for (int i = 0; i < 32; ++i) {
      auto slice = batch->Slice(i * 64, 64);
      expected_batches.push_back(slice);
      ASSERT_TRUE(writer.insert_batch(slice).ok());
    }
    ASSERT_TRUE(writer.finalize().ok());
  }
  EXPECT_EQ(0u, budget_.used());
  auto file = parquet::ParquetFileReader::OpenFile(path.string(), false);
  EXPECT_EQ(2048, file->metadata()->num_rows());
  ASSERT_EQ(3, file->metadata()->num_row_groups());
  for (int i = 0; i < 3; ++i)
    EXPECT_LE(file->metadata()->RowGroup(i)->num_rows(), 1000);
  auto input = arrow::io::ReadableFile::Open(path.string()).ValueOrDie();
  auto reader = parquet::arrow::OpenFile(input, arrow::default_memory_pool())
                    .ValueOrDie();
  std::shared_ptr<arrow::Table> actual;
  ASSERT_TRUE(reader->ReadTable(&actual).ok());
  auto expected =
      arrow::Table::FromRecordBatches(expected_batches).ValueOrDie();
  EXPECT_TRUE(actual->Equals(*expected));
}

TEST_F(ParquetBudgetTest, FailedBufferedWriteDoesNotResumeOrReportSuccess) {
  ASSERT_EQ(0, budget_.init(8 * 1024 * 1024));
  const auto path =
      std::filesystem::temp_directory_path() /
      ("zvec_parquet_failed_batch_" +
       std::to_string(
           std::chrono::steady_clock::now().time_since_epoch().count()));
  auto cleanup =
      ailego::ScopeGuard::Make([&] { std::filesystem::remove(path); });
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

TEST_F(ParquetBudgetTest,
       WritersOmitUnusedPageIndexesAndPreserveRowGroupStatistics) {
  ASSERT_EQ(0, budget_.init(64 * 1024 * 1024));
  auto schema = arrow::schema({arrow::field("row", arrow::int64()),
                               arrow::field("payload", arrow::utf8())});
  arrow::Int64Builder rows;
  arrow::StringBuilder strings;
  for (int i = 0; i < 64; ++i) {
    ASSERT_TRUE(rows.Append(i).ok());
    ASSERT_TRUE((i % 11 == 0 ? strings.AppendNull()
                             : strings.Append("value_" + std::to_string(i)))
                    .ok());
  }
  auto batch = arrow::RecordBatch::Make(
      schema, 64, {rows.Finish().ValueOrDie(), strings.Finish().ValueOrDie()});
  auto table = arrow::Table::FromRecordBatches({batch}).ValueOrDie();
  auto expected = arrow::Table::FromRecordBatches({batch, batch}).ValueOrDie();
  for (int entry = 0; entry < 3; ++entry) {
    SCOPED_TRACE(entry);
    const auto path =
        std::filesystem::temp_directory_path() /
        ("zvec_parquet_no_page_index_" + std::to_string(entry) + "_" +
         std::to_string(
             std::chrono::steady_clock::now().time_since_epoch().count()));
    auto cleanup =
        ailego::ScopeGuard::Make([&] { std::filesystem::remove(path); });
    if (entry == 0) {
      auto writer =
          ChunkedFileWriter::Open(path.string(), schema, FileFormat::PARQUET);
      ASSERT_NE(nullptr, writer);
      ASSERT_TRUE(writer->write(*batch).ok());
      ASSERT_TRUE(writer->write(*table).ok());
      ASSERT_TRUE(writer->close().ok());
    } else {
      ParquetWriter writer(path.string(), 32);
      if (entry == 1) {
        ASSERT_TRUE(writer.insert_batch(batch).ok());
        ASSERT_TRUE(writer.insert_batch(batch).ok());
      } else {
        auto input =
            arrow::RecordBatchReader::Make({batch, batch}, schema).ValueOrDie();
        ASSERT_TRUE(writer.insert(std::move(input)).ok());
      }
      ASSERT_TRUE(writer.finalize().ok());
    }
    EXPECT_EQ(0u, budget_.used());
    auto file = parquet::ParquetFileReader::OpenFile(path.string(), false);
    ASSERT_EQ(128, file->metadata()->num_rows());
    ASSERT_GT(file->metadata()->num_row_groups(), 0);
    for (int group = 0; group < file->metadata()->num_row_groups(); ++group) {
      auto metadata = file->metadata()->RowGroup(group);
      ASSERT_EQ(2, metadata->num_columns());
      for (int column = 0; column < metadata->num_columns(); ++column) {
        auto chunk = metadata->ColumnChunk(column);
        EXPECT_FALSE(chunk->GetColumnIndexLocation().has_value());
        EXPECT_FALSE(chunk->GetOffsetIndexLocation().has_value());
        ASSERT_TRUE(chunk->is_stats_set());
        ASSERT_NE(nullptr, chunk->statistics());
        EXPECT_TRUE(chunk->statistics()->HasMinMax());
      }
    }
    auto input = arrow::io::ReadableFile::Open(path.string()).ValueOrDie();
    auto reader = parquet::arrow::OpenFile(input, arrow::default_memory_pool())
                      .ValueOrDie();
    std::shared_ptr<arrow::Table> actual;
    ASSERT_TRUE(reader->ReadTable(&actual).ok());
    EXPECT_TRUE(actual->Equals(*expected));
  }
}

TEST_F(ParquetBudgetTest, SmallSlotsReuseBackingAndKeepItCharged) {
  RejectingMemoryPool fallback;
  ParquetMemoryPool pool(&fallback, true);
  uint8_t *first = nullptr, *second = nullptr, *replacement = nullptr;
  ASSERT_TRUE(pool.Allocate(64, &first).ok());
  ASSERT_TRUE(pool.Allocate(64, &second).ok());
  const auto backing = budget_.external_used();
  EXPECT_EQ(64u * 1024, backing);
  pool.Free(first, 64);
  EXPECT_EQ(backing, budget_.external_used());
  EXPECT_EQ(backing, pool.charged_bytes());
  auto &native = ailego::MemoryLimitPool::get_instance();
  using Pressure = ailego::ProcessRssBudget::PressureLevel;
  native.set_memory_pressure(Pressure::kCritical);
  const auto blocked = pool.Allocate(64, &replacement);
  EXPECT_TRUE(blocked.IsOutOfMemory());
  if (blocked.ok()) pool.Free(replacement, 64);
  native.set_memory_pressure(Pressure::kNormal);
  ASSERT_TRUE(pool.Allocate(64, &replacement).ok());
  EXPECT_EQ(first, replacement);
  EXPECT_EQ(backing, budget_.external_used());
  pool.Free(second, 64);
  pool.Free(replacement, 64);
  EXPECT_EQ(0u, budget_.used());
  EXPECT_EQ(0u, pool.charged_bytes());
  // Empty chunks stay committed, not silently refunded to another allocator.
  EXPECT_EQ(backing, budget_.committed());
  EXPECT_EQ(backing, budget_.stats().retained_idle_bytes);
  const auto reuses = budget_.stats().retained_block_reuses;
  ASSERT_TRUE(pool.Allocate(64, &replacement).ok());
  EXPECT_GT(budget_.stats().retained_block_reuses, reuses);
  EXPECT_EQ(backing, budget_.committed());
  pool.Free(replacement, 64);
  // Pressure may reclaim idle backing before admitting a different use.
  ASSERT_TRUE(budget_.try_charge_workspace(budget_.capacity()));
  EXPECT_EQ(0u, budget_.stats().retained_idle_bytes);
  budget_.release_external(budget_.capacity());
}

TEST_F(ParquetBudgetTest, ManagedSmallBuffersHonorAlignmentAndSizeClasses) {
  RejectingMemoryPool fallback;
  ParquetMemoryPool pool(&fallback, true);
  for (const int64_t alignment : {16, 64, 4096, 65536}) {
    for (const int64_t size : {1, 64, 65, 1024, 4096, 8128, 8192, 262143}) {
      SCOPED_TRACE(size);
      SCOPED_TRACE(alignment);
      uint8_t *buffer = nullptr;
      ASSERT_TRUE(pool.Allocate(size, alignment, &buffer).ok());
      EXPECT_EQ(0u,
                reinterpret_cast<uintptr_t>(buffer) %
                    static_cast<uintptr_t>(std::max<int64_t>(64, alignment)));
      std::memset(buffer, 23, size);
      EXPECT_EQ(pool.charged_bytes(), budget_.external_used());
      EXPECT_LE(budget_.committed(), budget_.capacity());
      pool.Free(buffer, size, alignment);
      EXPECT_EQ(0u, pool.charged_bytes());
      EXPECT_EQ(0u, budget_.used());
    }
  }
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

TEST_F(ParquetBudgetTest, MappedShrinkNeedsNoOverlapBudget) {
  ParquetMemoryPool pool(arrow::default_memory_pool(), true);
  uint8_t *buffer = nullptr;
  ASSERT_TRUE(pool.Allocate(2 * kThreshold, &buffer).ok());
  std::memset(buffer, 43, 2 * kThreshold);
  auto *original = buffer;
  const auto mapped = pool.mapped_bytes();
  const auto remaining = budget_.capacity() - budget_.used();
  ASSERT_TRUE(budget_.try_charge_external(remaining));
  auto refund =
      ailego::ScopeGuard::Make([&] { budget_.release_external(remaining); });
  ASSERT_TRUE(pool.Reallocate(2 * kThreshold, kThreshold, &buffer).ok());
  EXPECT_EQ(original, buffer);
  EXPECT_EQ(43, buffer[kThreshold - 1]);
  EXPECT_EQ(1, pool.num_mapped_allocations());
  EXPECT_EQ(2 * kThreshold, pool.max_memory());
  EXPECT_LT(pool.mapped_bytes(), mapped);
  EXPECT_EQ(pool.mapped_bytes(), pool.charged_bytes());
  EXPECT_EQ(remaining + pool.charged_bytes(), budget_.external_used());
  EXPECT_EQ(pool.charged_bytes() + remaining, budget_.committed());
  pool.Free(buffer, kThreshold);
  EXPECT_EQ(0u, pool.charged_bytes());
  EXPECT_EQ(remaining, budget_.external_used());
}

TEST_F(ParquetBudgetTest, MappedShrinkWithinPageRetainsBackingCharge) {
  ParquetMemoryPool pool(arrow::default_memory_pool(), true);
  uint8_t *buffer = nullptr;
  ASSERT_TRUE(pool.Allocate(kThreshold + 100, &buffer).ok());
  auto *original = buffer;
  const auto mapped = pool.mapped_bytes();
  ASSERT_TRUE(pool.Reallocate(kThreshold + 100, kThreshold + 99, &buffer).ok());
  EXPECT_EQ(original, buffer);
  EXPECT_EQ(mapped, pool.mapped_bytes());
  EXPECT_EQ(mapped, pool.charged_bytes());
  EXPECT_EQ(mapped, budget_.external_used());
  EXPECT_EQ(kThreshold + 99, pool.bytes_allocated());
  pool.Free(buffer, kThreshold + 99);
  EXPECT_EQ(0u, budget_.external_used());
  EXPECT_EQ(0u, budget_.committed());
}

TEST_F(ParquetBudgetTest, MappedShrinkAdmitsPreBudgetAllocation) {
  ASSERT_EQ(0, budget_.init(0));
  ParquetMemoryPool pool(arrow::default_memory_pool(), true);
  uint8_t *buffer = nullptr;
  ASSERT_TRUE(pool.Allocate(2 * kThreshold, &buffer).ok());
  std::memset(buffer, 47, 2 * kThreshold);
  auto *original = buffer;
  ASSERT_EQ(0u, pool.charged_bytes());
  ASSERT_EQ(0, budget_.init(1024 * 1024));
  ASSERT_TRUE(budget_.try_charge_external(budget_.capacity()));
  EXPECT_TRUE(
      pool.Reallocate(2 * kThreshold, kThreshold, &buffer).IsOutOfMemory());
  EXPECT_EQ(original, buffer);
  EXPECT_EQ(47, buffer[kThreshold - 1]);
  budget_.release_external(budget_.capacity());
  ASSERT_TRUE(pool.Reallocate(2 * kThreshold, kThreshold, &buffer).ok());
  EXPECT_NE(original, buffer);
  EXPECT_EQ(47, buffer[kThreshold - 1]);
  EXPECT_EQ(pool.mapped_bytes(), pool.charged_bytes());
  EXPECT_EQ(pool.charged_bytes(), budget_.external_used());
  pool.Free(buffer, kThreshold);
  EXPECT_EQ(0u, budget_.used());
}

TEST_F(ParquetBudgetTest, FreeUsesRecordedChargeAcrossBudgetInitialization) {
  ASSERT_EQ(0, budget_.init(0));
  ParquetMemoryPool pool(arrow::default_memory_pool(), true);
  uint8_t *small = nullptr, *large = nullptr;
  ASSERT_TRUE(pool.Allocate(1024, &small).ok());
  ASSERT_TRUE(pool.Allocate(kThreshold, &large).ok());
  EXPECT_EQ(0u, pool.charged_bytes());
  ASSERT_EQ(0, budget_.init(1024 * 1024));
  pool.Free(small, 1024);
  pool.Free(large, kThreshold);
  EXPECT_EQ(0u, budget_.external_used());
}

TEST_F(ParquetBudgetTest, ConcurrentDecodersShareOneByteBudget) {
  ParquetMemoryPool pool(arrow::default_memory_pool(), true);
  std::atomic<size_t> ready{0}, admitted{0};
  std::array<std::thread, 8> workers;
  for (auto &worker : workers) {
    worker = std::thread([&] {
      uint8_t *buffer = nullptr;
      auto status = pool.Allocate(kThreshold, &buffer);
      if (status.ok())
        admitted.fetch_add(1);
      else
        EXPECT_TRUE(status.IsOutOfMemory());
      EXPECT_LE(budget_.used(), budget_.capacity());
      ready.fetch_add(1);
      while (ready.load() != workers.size()) std::this_thread::yield();
      if (buffer) pool.Free(buffer, kThreshold);
    });
  }
  for (auto &worker : workers) worker.join();
  // Page rounding makes four 256-KiB mappings exceed the 1-MiB budget.
  EXPECT_EQ(3u, admitted.load());
  EXPECT_EQ(0u, budget_.external_used());
  EXPECT_EQ(0u, pool.charged_bytes());
}

}  // namespace
}  // namespace detail
}  // namespace zvec
