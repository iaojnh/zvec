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
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <future>
#include <limits>
#include <system_error>
#include <thread>
#include <vector>
#include <ailego/pattern/scope_guard.h>
#include <ailego/utility/memory_helper.h>
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
#include "db/index/storage/request_scratch.h"
#include "db/index/storage/store_helper.h"
#include "../../../ailego/buffer/pool_payload_budget.h"
#if defined(__APPLE__)
#include <TargetConditionals.h>
#endif
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace zvec::ailego {
struct MemoryLimitPoolTestPeer {
  static void stop_background(MemoryLimitPool &pool) {
    pool.stop_background_evictor();
  }
  static void start_background(MemoryLimitPool &pool) {
    pool.start_background_evictor();
  }
  static bool prove(MemoryLimitPool &pool) {
    const auto token = pool.process_rss_.begin_resident_sample();
    return pool.process_rss_.publish_resident_sample(
        token, MemoryLimitPool::read_background_resident_sample());
  }
  static size_t room(MemoryLimitPool &pool) {
    return pool.process_rss_.available();
  }
  static bool hold_registry_until_waiter(MemoryLimitPool &pool,
                                         std::promise<void> &ready,
                                         std::atomic<bool> &stop) {
    std::lock_guard<std::mutex> lock(pool.cache_mapping_mutex_);
    ready.set_value();
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!stop.load() && !pool.mandatory_waiters_.load() &&
           std::chrono::steady_clock::now() < deadline)
      std::this_thread::yield();
    return pool.mandatory_waiters_.load() != 0;
  }
  static bool configure_rss(MemoryLimitPool &pool) {
    return pool.process_rss_.configure(
        size_t{128} * 1024 * 1024,
        +[]() -> size_t { return size_t{16} * 1024 * 1024; });
  }
  static void refresh_rss(MemoryLimitPool &pool) {
    pool.process_rss_.refresh(true);
  }
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

#if (defined(__linux__) && !defined(__ANDROID__)) || \
    (defined(__APPLE__) && TARGET_OS_OSX)
TEST_F(ParquetBudgetTest, RequiredSmallSlotWaitsForOwnershipConfirmation) {
  ASSERT_EQ(0, budget_.init(256UL << 20));
  auto &native = ailego::MemoryLimitPool::get_instance();
  ailego::MemoryLimitPoolTestPeer::stop_background(native);
  auto restore = ailego::ScopeGuard::Make([&] {
    EXPECT_TRUE(native.configure_process_rss_limit(0));
    ailego::MemoryLimitPoolTestPeer::start_background(native);
  });
  ParquetMemoryPool allocator(arrow::default_memory_pool(), true, true);
  uint8_t *first = nullptr, *second = nullptr;
  ASSERT_TRUE(allocator.Allocate(1024, 64, &first).ok());
  auto release = ailego::ScopeGuard::Make([&] {
    if (second) allocator.Free(second, 1024, 64);
    allocator.Free(first, 1024, 64);
  });
  std::memset(first, 0x55, 1024);
  auto *workspace = native.try_acquire_workspace_buffer(64UL << 20);
  ASSERT_NE(nullptr, workspace);
  auto unmap = ailego::ScopeGuard::Make(
      [&] { EXPECT_TRUE(native.release_workspace_buffer(workspace)); });
  std::memset(workspace->data, 0x66, workspace->bytes);
  ASSERT_TRUE(native.configure_process_rss_limit(
      ailego::MemoryHelper::SelfRSS() + (32UL << 20)));
  ASSERT_TRUE(ailego::MemoryLimitPoolTestPeer::prove(native));
  ailego::MemoryLimitPoolTestPeer::refresh_rss(native);
  ASSERT_EQ(0u, ailego::MemoryLimitPoolTestPeer::room(native));
  std::promise<void> ready;
  std::atomic<bool> stop{false}, waited{false};
  std::thread holder([&] {
    waited = ailego::MemoryLimitPoolTestPeer::hold_registry_until_waiter(
        native, ready, stop);
  });
  auto join = ailego::ScopeGuard::Make([&] {
    stop = true;
    if (holder.joinable()) holder.join();
  });
  ready.get_future().wait();
  ailego::MemoryLimitPoolTestPeer::start_background(native);
  const auto status = allocator.Allocate(1024, 64, &second);
  stop = true;
  holder.join();
  ASSERT_TRUE(status.ok()) << status;
  EXPECT_TRUE(waited);
  EXPECT_EQ(0x55, first[0]);
  EXPECT_EQ(0x66, workspace->data[0]);
}
#endif

TEST_F(ParquetBudgetTest, FetchScratchRetainsChargeAndBoundsReallocation) {
  ASSERT_EQ(0, budget_.init(8 * 1024 * 1024));
  ParquetMemoryPool backing(arrow::default_memory_pool(), true, true);
  FetchMemoryPool bounded(&backing, 128 * 1024);
  {
    FetchScratchMemoryPool scratch(&bounded, true);
    ASSERT_TRUE(scratch.enabled());
    const auto allocations = backing.num_allocations();
    const auto charged = budget_.external_used();
    ASSERT_GE(charged, 64u * 1024);
    {
      std::vector<std::unique_ptr<arrow::Buffer>> buffers;
      for (int i = 0; i < 32; ++i) {
        auto buffer = arrow::AllocateBuffer(64, &scratch);
        ASSERT_TRUE(buffer.ok());
        EXPECT_EQ(0u, reinterpret_cast<uintptr_t>((*buffer)->data()) % 64);
        std::memset((*buffer)->mutable_data(), i, 64);
        buffers.push_back(std::move(*buffer));
      }
      auto growing = arrow::AllocateResizableBuffer(64, &scratch).ValueOrDie();
      std::memset(growing->mutable_data(), 37, 64);
      ASSERT_TRUE(growing->Resize(4096).ok());
      EXPECT_EQ(37, growing->data()[63]);
      ASSERT_TRUE(growing->Resize(128, true).ok());
      EXPECT_EQ(37, growing->data()[63]);
      EXPECT_EQ(allocations, backing.num_allocations());
      EXPECT_EQ(charged, budget_.external_used());
    }
    EXPECT_EQ(0, scratch.bytes_allocated());
    EXPECT_EQ(64 * 1024, bounded.bytes_allocated());
    // Freed suballocations must not refund the still-owned backing or allow
    // a direct allocation to exceed the batch's total retained working set.
    auto refused = arrow::AllocateBuffer(96 * 1024, &scratch);
    EXPECT_TRUE(refused.status().IsOutOfMemory());
    EXPECT_TRUE(bounded.exceeded());
    EXPECT_EQ(charged, budget_.external_used());
  }
  EXPECT_EQ(0, bounded.bytes_allocated());
  EXPECT_EQ(0, backing.bytes_allocated());
  EXPECT_EQ(0u, budget_.external_used());
}

TEST_F(ParquetBudgetTest, FetchScratchKeepsSmallAllocationUnderPressure) {
  ASSERT_EQ(0, budget_.init(8 * 1024 * 1024));
  const size_t occupied = budget_.capacity() - 128 * 1024;
  ASSERT_TRUE(budget_.try_charge_external(occupied));
  auto release =
      ailego::ScopeGuard::Make([&] { budget_.release_external(occupied); });
  ParquetMemoryPool backing(arrow::default_memory_pool(), true, true);
  FetchMemoryPool bounded(&backing, 128 * 1024);
  FetchScratchMemoryPool scratch(&bounded, true);
  EXPECT_FALSE(scratch.enabled());
  auto small_buffer = arrow::AllocateBuffer(64, &bounded);
  ASSERT_TRUE(small_buffer.ok()) << small_buffer.status().ToString();
  EXPECT_LE(budget_.used(), budget_.capacity());
  small_buffer = std::unique_ptr<arrow::Buffer>{};
  EXPECT_EQ(occupied, budget_.external_used());
}

TEST_F(ParquetBudgetTest, FetchScratchConcurrentBatchesReleaseTheirBacking) {
  ASSERT_EQ(0, budget_.init(8 * 1024 * 1024));
  ParquetMemoryPool backing(arrow::default_memory_pool(), true, true);
  std::atomic<size_t> failures{0};
  std::vector<std::thread> threads;
  for (size_t worker = 0; worker < 4; ++worker) {
    threads.emplace_back([&, worker] {
      for (int round = 0; round < 64; ++round) {
        FetchMemoryPool bounded(&backing, 128 * 1024);
        FetchScratchMemoryPool scratch(&bounded, true);
        auto buffer = arrow::AllocateBuffer(4096, &scratch);
        if (!buffer.ok()) {
          ++failures;
          continue;
        }
        std::memset((*buffer)->mutable_data(), worker + round, 4096);
        std::this_thread::yield();
        if ((*buffer)->data()[4095] != static_cast<uint8_t>(worker + round))
          ++failures;
        if (budget_.used() > budget_.capacity()) ++failures;
      }
    });
  }
  for (auto &thread : threads) thread.join();
  EXPECT_EQ(0u, failures.load());
  EXPECT_EQ(0, backing.bytes_allocated());
  EXPECT_EQ(0u, budget_.external_used());
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
  ASSERT_GE(pool.mapped_bytes(), 0);
  EXPECT_EQ(static_cast<size_t>(pool.mapped_bytes()), budget_.external_used());
  ASSERT_TRUE(pool.Reallocate(2 * kThreshold, 1024, &buffer).ok());
  EXPECT_EQ(17, buffer[1023]);
  ASSERT_TRUE(pool.Reallocate(1024, 0, &buffer).ok());
  EXPECT_NE(nullptr, buffer);
  EXPECT_EQ(0u, budget_.external_used());
  pool.Free(buffer, 0);
}

TEST_F(ParquetBudgetTest, SmallBuffersCanBeFreedByDifferentThreads) {
  ASSERT_EQ(0, budget_.init(8 * 1024 * 1024));
  ParquetMemoryPool pool(arrow::default_memory_pool(), true);
  constexpr size_t threads = 8, count = 112;
  constexpr std::array<int64_t, 7> sizes{24, 160, 480, 992, 1990, 3990, 8000};
  std::array<std::array<uint8_t *, count>, threads> buffers{};
  std::atomic<size_t> failures{0};
  std::vector<std::thread> workers;
  for (size_t worker = 0; worker < threads; ++worker) {
    workers.emplace_back([&, worker] {
      for (size_t i = 0; i < count; ++i) {
        const auto size = sizes[i % sizes.size()];
        if (!pool.Allocate(size, &buffers[worker][i]).ok()) {
          ++failures;
          continue;
        }
        std::memset(buffers[worker][i],
                    static_cast<uint8_t>(worker * count + i), size);
      }
    });
  }
  for (auto &worker : workers) worker.join();
  workers.clear();
  for (size_t worker = 0; worker < threads; ++worker) {
    workers.emplace_back([&, worker] {
      const size_t owner = (worker + 1) % threads;
      for (size_t i = 0; i < count; ++i) {
        auto *buffer = buffers[owner][i];
        if (!buffer) continue;
        const auto size = sizes[i % sizes.size()];
        const auto pattern = static_cast<uint8_t>(owner * count + i);
        for (int64_t byte = 0; byte < size; ++byte)
          if (buffer[byte] != pattern) ++failures;
        pool.Free(buffer, size);
      }
    });
  }
  for (auto &worker : workers) worker.join();
  EXPECT_EQ(0u, failures.load());
  EXPECT_EQ(0, pool.bytes_allocated());
  EXPECT_EQ(0u, pool.charged_bytes());
  EXPECT_EQ(0u, budget_.external_used());
}

TEST_F(ParquetBudgetTest, SmallAllocationsCanRaceLastSlotRelease) {
  ASSERT_EQ(0, budget_.init(8 * 1024 * 1024));
  ParquetMemoryPool pool(arrow::default_memory_pool(), true);
  constexpr std::array<int64_t, 7> sizes{24, 160, 480, 992, 1990, 3990, 8000};
  std::atomic<size_t> failures{0}, ready{0};
  std::vector<std::thread> workers;
  for (size_t worker = 0; worker < 8; ++worker) {
    workers.emplace_back([&, worker] {
      ++ready;
      while (ready.load() != 8) std::this_thread::yield();
      for (size_t i = 0; i < 2000; ++i) {
        const auto size = sizes[i % sizes.size()];
        uint8_t *buffer = nullptr;
        if (!pool.Allocate(size, &buffer).ok()) {
          ++failures;
          continue;
        }
        const auto pattern = static_cast<uint8_t>(worker * 31 + i);
        std::memset(buffer, pattern, size);
        std::this_thread::yield();
        for (int64_t byte = 0; byte < size; ++byte)
          if (buffer[byte] != pattern) ++failures;
        pool.Free(buffer, size);
      }
    });
  }
  for (auto &worker : workers) worker.join();
  EXPECT_EQ(0u, failures.load());
  EXPECT_EQ(0, pool.bytes_allocated());
  EXPECT_EQ(0u, pool.charged_bytes());
  EXPECT_EQ(0u, budget_.external_used());
}

TEST_F(ParquetBudgetTest, RefusedSmallSlotDoesNotRetainBackingOrLoseCapacity) {
  auto &native = ailego::MemoryLimitPool::get_instance();
  using Pressure = ailego::ProcessRssBudget::PressureLevel;
  ASSERT_TRUE(ailego::MemoryLimitPoolTestPeer::configure_rss(native));
  auto restore = ailego::ScopeGuard::Make([&] {
    native.set_memory_pressure(Pressure::kNormal);
    EXPECT_TRUE(native.configure_process_rss_limit(0));
  });
  ParquetMemoryPool pool(arrow::default_memory_pool(), true);
  auto first = arrow::AllocateBuffer(64, &pool).ValueOrDie();
  const auto charged = pool.charged_bytes();
  native.set_memory_pressure(Pressure::kCritical);
  for (int i = 0; i < 1024; ++i) {
    uint8_t *refused = nullptr;
    auto status = pool.Allocate(64, &refused);
    EXPECT_TRUE(status.IsOutOfMemory()) << status.ToString();
    if (status.ok()) pool.Free(refused, 64);
  }
  EXPECT_EQ(charged, pool.charged_bytes());
  native.set_memory_pressure(Pressure::kNormal);
  ailego::MemoryLimitPoolTestPeer::refresh_rss(native);
  auto second = arrow::AllocateBuffer(64, &pool);
  ASSERT_TRUE(second.ok()) << second.status().ToString();
  EXPECT_EQ(charged, pool.charged_bytes());
  second = std::unique_ptr<arrow::Buffer>{};
  first.reset();
  EXPECT_EQ(0u, pool.charged_bytes());
  EXPECT_EQ(0u, budget_.external_used());
}

TEST_F(ParquetBudgetTest, ScratchOverflowStillRequiresSharedQuota) {
  auto &native = ailego::MemoryLimitPool::get_instance();
  const size_t remaining = native.capacity() - native.used();
  ASSERT_TRUE(native.try_charge_external(remaining));
  {
    auto restore =
        ailego::ScopeGuard::Make([&] { native.release_external(remaining); });
    RequestScratchArray<int, 16> inline_values(16);
    inline_values[15] = 42;
    EXPECT_EQ(42, inline_values[15]);
    EXPECT_THROW((RequestScratchArray<int, 16>(17)), std::bad_alloc);
    EXPECT_EQ(native.capacity(), native.used());
  }
  {
    RequestScratchArray<int, 16> large(128);
    EXPECT_GT(native.external_used(), 0u);
    large[127] = 37;
    EXPECT_EQ(37, large[127]);
  }
  EXPECT_EQ(0u, native.external_used());
}

TEST_F(ParquetBudgetTest, ScratchReleasesOwnedPinsOnEarlyExit) {
  for (size_t count : {2u, 17u}) {
    auto owner = std::make_shared<int>(42);
    std::weak_ptr<int> weak = owner;
    auto read = [&] {
      RequestScratchArray<std::shared_ptr<int>, 2> pins(count);
      pins[0] = std::move(owner);
      EXPECT_FALSE(weak.expired());
      return false;  // Same lifetime boundary as a failed cache lookup.
    };
    EXPECT_FALSE(read());
    EXPECT_TRUE(weak.expired());
    EXPECT_EQ(0u, budget_.external_used());
  }
}

TEST_F(ParquetBudgetTest, QueryAndCachePoolsShareTheirLifetimeOwner) {
  auto cache = GetParquetMemoryPool();
  auto query = GetParquetQueryMemoryPool();
  EXPECT_NE(cache.get(), query.get());
  EXPECT_FALSE(cache.owner_before(query));
  EXPECT_FALSE(query.owner_before(cache));
}

TEST_F(ParquetBudgetTest, QueryMediumBuffersReuseAdmittedBacking) {
  ASSERT_EQ(0, budget_.init(128 * 1024 * 1024));
  ParquetMemoryPool pool(arrow::default_memory_pool(), true, true);
  auto &native = ailego::MemoryLimitPool::get_instance();
  const auto before = native.stats().retained_block_reuses;
  for (int i = 0; i < 3; ++i) {
    auto buffer = arrow::AllocateBuffer(1024 * 1024, &pool).ValueOrDie();
    std::memset(buffer->mutable_data(), i + 1, buffer->size());
    EXPECT_GE(pool.charged_bytes(), static_cast<size_t>(buffer->size()));
    EXPECT_LE(native.used(), native.capacity());
  }
  EXPECT_GT(native.stats().retained_block_reuses, before);
  EXPECT_EQ(0, pool.num_mapped_allocations());
  EXPECT_EQ(0u, pool.charged_bytes());
}

TEST_F(ParquetBudgetTest, TightQueryBudgetsKeepReclaimableMappings) {
  ASSERT_EQ(0, budget_.init(8 * 1024 * 1024));
  ParquetMemoryPool pool(arrow::default_memory_pool(), true, true);
  auto buffer =
      arrow::AllocateResizableBuffer(1024 * 1024, &pool).ValueOrDie();
  EXPECT_EQ(1, pool.num_mapped_allocations());
  const auto before = pool.charged_bytes();
  ASSERT_TRUE(buffer->Resize(1024, true).ok());
  EXPECT_LT(pool.charged_bytes(), before);
  buffer.reset();
  EXPECT_EQ(0u, pool.charged_bytes());
}

TEST_F(ParquetBudgetTest, ReusedQueryBufferShrinksWithoutAdditionalQuota) {
  ASSERT_EQ(0, budget_.init(128 * 1024 * 1024));
  ParquetMemoryPool pool(arrow::default_memory_pool(), true, true);
  auto buffer =
      arrow::AllocateResizableBuffer(1024 * 1024, &pool).ValueOrDie();
  std::memset(buffer->mutable_data(), 37, buffer->size());
  auto *original = buffer->data();
  const auto charged = pool.charged_bytes();
  auto &native = ailego::MemoryLimitPool::get_instance();
  const size_t remaining = native.capacity() - native.used();
  ASSERT_TRUE(native.try_charge_external(remaining));
  auto restore =
      ailego::ScopeGuard::Make([&] { native.release_external(remaining); });
  ASSERT_TRUE(buffer->Resize(1024, true).ok());
  EXPECT_EQ(original, buffer->data());
  EXPECT_EQ(37, buffer->data()[1023]);
  EXPECT_EQ(charged, pool.charged_bytes());
  EXPECT_EQ(1024, pool.bytes_allocated());
}

TEST_F(ParquetBudgetTest, LimitedHeadroomFallsBackToShrinkableMapping) {
  ASSERT_EQ(0, budget_.init(128 * 1024 * 1024));
  auto &native = ailego::MemoryLimitPool::get_instance();
  const size_t charge = native.capacity() - native.used() - 2 * 1024 * 1024;
  ASSERT_TRUE(native.try_charge_external(charge));
  auto restore =
      ailego::ScopeGuard::Make([&] { native.release_external(charge); });
  ParquetMemoryPool pool(arrow::default_memory_pool(), true, true);
  auto buffer =
      arrow::AllocateResizableBuffer(1024 * 1024, &pool).ValueOrDie();
  EXPECT_EQ(1, pool.num_mapped_allocations());
  const auto before = pool.charged_bytes();
  ASSERT_TRUE(buffer->Resize(1024, true).ok());
  EXPECT_LT(pool.charged_bytes(), before);
}

TEST_F(ParquetBudgetTest, QueryReuseObeysRssPressureAndRecovers) {
  ASSERT_EQ(0, budget_.init(128 * 1024 * 1024));
  auto &native = ailego::MemoryLimitPool::get_instance();
  using Pressure = ailego::ProcessRssBudget::PressureLevel;
  ASSERT_TRUE(ailego::MemoryLimitPoolTestPeer::configure_rss(native));
  auto restore = ailego::ScopeGuard::Make([&] {
    native.set_memory_pressure(Pressure::kNormal);
    EXPECT_TRUE(native.configure_process_rss_limit(0));
  });
  ParquetMemoryPool pool(arrow::default_memory_pool(), true, true);
  auto first =
      arrow::AllocateResizableBuffer(1024 * 1024, &pool).ValueOrDie();
  std::memset(first->mutable_data(), 19, first->size());
  native.set_memory_pressure(Pressure::kCritical);
  auto refused = arrow::AllocateBuffer(1024 * 1024, &pool);
  EXPECT_TRUE(refused.status().IsOutOfMemory());
  ASSERT_TRUE(first->Resize(1024, true).ok());
  EXPECT_EQ(19, first->data()[1023]);
  first.reset();
  native.set_memory_pressure(Pressure::kNormal);
  ailego::MemoryLimitPoolTestPeer::refresh_rss(native);
  auto recovered = arrow::AllocateBuffer(1024 * 1024, &pool);
  EXPECT_TRUE(recovered.ok()) << recovered.status().ToString();
}

}  // namespace
}  // namespace detail
}  // namespace zvec
