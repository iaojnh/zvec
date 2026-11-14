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
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <mutex>
#include <set>
#include <thread>
#include <type_traits>
#include <arrow/api.h>
#include <arrow/io/file.h>
#include <arrow/result.h>
#include <arrow/table.h>
#include <gtest/gtest.h>
#include <parquet/arrow/writer.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/ailego/buffer/external_cache.h>
#include <zvec/ailego/buffer/memory_reservation.h>
#include "db/common/query_result_memory.h"
#include "db/index/storage/bufferpool_forward_store.h"
#include "db/index/storage/fetch_memory.h"
#include "db/index/storage/lazy_record_batch_reader.h"
#include "db/index/storage/parquet_buffer_pool.h"
#include "db/index/storage/parquet_memory_pool.h"
#include "db/index/storage/parquet_metadata_budget.h"
#include "db/index/storage/selected_rows_plan.h"
#include "db/index/storage/store_helper.h"
#include "utils/utils.h"
#include "parquet_buffer_pool_test_peer.h"
#include "../../../ailego/buffer/pool_payload_budget.h"
#include "../../common/memory_budget_test_helper.h"

using namespace zvec;

namespace {
size_t ParquetCacheEntryCount() {
  return ParquetBufferPoolTestPeer::entry_count();
}
}  // namespace

TEST(ParquetMetadataBudgetTest, ReusesConcurrentHighWaterAndRejectsOverflow) {
  auto &pool = ailego::MemoryLimitPool::get_instance();
  ASSERT_EQ(0, pool.init(64 * 1024 * 1024));
  const auto before = pool.metadata_used();
  {
    detail::ParquetMetadataBudget budget;
    {
      detail::ParquetMetadataBudget::Lease first(budget);
      first.acquire(1024);
      {
        detail::ParquetMetadataBudget::Lease second(budget);
        second.acquire(2048);
        EXPECT_EQ(3072u, budget.reserved());
      }
      EXPECT_EQ(before + 3072, pool.metadata_used());
      EXPECT_THROW(first.acquire(std::numeric_limits<size_t>::max()),
                   std::bad_alloc);
      EXPECT_EQ(3072u, budget.reserved());
    }
    detail::ParquetMetadataBudget::Lease reopened(budget);
    reopened.acquire(3072);
    EXPECT_EQ(before + 3072, pool.metadata_used());
  }
  EXPECT_EQ(before, pool.metadata_used());
  EXPECT_THROW(detail::ParquetMetadataBudget::estimate(
                   std::numeric_limits<size_t>::max()),
               std::bad_alloc);
}

TEST(ParquetMetadataBudgetTest, FailedAdmissionDoesNotConsumeReservation) {
  auto &pool = ailego::MemoryLimitPool::get_instance();
  ASSERT_EQ(0, pool.init(64 * 1024 * 1024));
  detail::ParquetMetadataBudget budget;
  detail::ParquetMetadataBudget::Lease lease(budget);
  EXPECT_THROW(lease.acquire(pool.capacity() + 1), std::bad_alloc);
  EXPECT_EQ(0u, budget.reserved());
  EXPECT_NO_THROW(lease.acquire(1024));
}

TEST(ParquetMetadataBudgetTest, LeavesDecodeHeadroomAfterRejectedOpen) {
  auto &pool = ailego::MemoryLimitPool::get_instance();
  ASSERT_EQ(0, pool.init(64 * 1024 * 1024));
  detail::ParquetMetadataBudget budget;
  const size_t limit = pool.capacity() / 2;
  {
    detail::ParquetMetadataBudget::Lease first(budget);
    first.acquire(limit);
    detail::ParquetMetadataBudget::Lease rejected(budget);
    EXPECT_THROW(rejected.acquire(1), std::bad_alloc);
    EXPECT_EQ(limit, budget.reserved());
  }
  // Reopening reuses the retained allowance; admission failure has not
  // increased it or poisoned subsequent small operations.
  detail::ParquetMetadataBudget::Lease reopened(budget);
  EXPECT_NO_THROW(reopened.acquire(4096));
  EXPECT_EQ(limit, budget.reserved());
}

TEST(FetchMemoryTest, ReservesPeakBeforeGrowingAndReleasesOnFailure) {
  detail::FetchMemoryPool pool(arrow::default_memory_pool(), 10 * 1024);
  uint8_t *buffer = nullptr;
  ASSERT_TRUE(pool.Allocate(1024, 64, &buffer).ok());
  ASSERT_TRUE(pool.Reallocate(1024, 8192, 64, &buffer).ok());
  EXPECT_EQ(8192, pool.bytes_allocated());
  auto *previous = buffer;
  EXPECT_TRUE(pool.Reallocate(8192, 16384, 64, &buffer).IsOutOfMemory());
  EXPECT_EQ(previous, buffer);
  EXPECT_EQ(8192, pool.bytes_allocated());
  EXPECT_TRUE(pool.exceeded());
  pool.Free(buffer, 8192, 64);
  EXPECT_EQ(0, pool.bytes_allocated());
}

TEST(FetchMemoryTest, BinaryScalarViewSharesOwnedBytesAndHandlesSlices) {
  for (const auto &type : {arrow::utf8(), arrow::binary()}) {
    std::unique_ptr<arrow::BinaryBuilder> owner;
    if (type->id() == arrow::Type::STRING) {
      owner = std::make_unique<arrow::StringBuilder>();
    } else {
      owner = std::make_unique<arrow::BinaryBuilder>();
    }
    auto &builder = *owner;
    ASSERT_TRUE(builder.Append("unused").ok());
    ASSERT_TRUE(builder.Append(std::string("a\0b", 3)).ok());
    ASSERT_TRUE(builder.AppendNull().ok());
    std::shared_ptr<arrow::Array> array;
    ASSERT_TRUE(builder.Finish(&array).ok());
    auto slice = array->Slice(1);
    auto column = std::make_shared<arrow::ChunkedArray>(slice);
    auto scalar = detail::FetchScalarView(*column, 0);
    ASSERT_TRUE(scalar.ok());
    auto binary = std::static_pointer_cast<arrow::BaseBinaryScalar>(*scalar);
    EXPECT_EQ(type->id(), binary->type->id());
    const auto &source = static_cast<const arrow::BinaryArray &>(*slice);
    EXPECT_EQ(source.value_data()->data() + source.value_offset(0),
              binary->value->data());
    auto null_scalar = detail::FetchScalarView(*column, 1);
    ASSERT_TRUE(null_scalar.ok());
    EXPECT_FALSE((*null_scalar)->is_valid);
    EXPECT_FALSE(detail::FetchScalarView(*column, 2).ok());
    array.reset();
    slice.reset();
    column.reset();
    EXPECT_EQ(std::string_view("a\0b", 3), binary->view());
  }
}

class BufferPoolStoreTest : public testing::Test {
 protected:
  void SetUp() override {
    ASSERT_TRUE(detail::MemoryBudgetTestPeer::reset_idle_allowances());
    // The Parquet cache outlives individual tests. Reusing a file name is not
    // a new cache identity on every platform, even after deleting the file.
    // Include an invocation counter so --gtest_repeat is isolated as well.
    static std::atomic<uint64_t> sequence{0};
    parquet_path = "bufferpool_store_test_" +
                   std::to_string(sequence.fetch_add(1)) + ".parquet";
    auto s = test::TestHelper::WriteTestFile(parquet_path, FileFormat::PARQUET);
    if (!s.ok()) {
      std::cout << "err: " << s.message() << std::endl;
      exit(1);
    }
    ASSERT_EQ(0, zvec::ailego::PoolPayloadBudget::get_instance().init(
                     10 * 1024 * 1024));
  }

  void TearDown() override {
    EXPECT_TRUE(detail::MemoryBudgetTestPeer::reset_idle_allowances());
    // Releasing the last Arrow view only makes a column evictable; deleting
    // the backing file does not remove its process-wide cached payload.
    ailego::BlockEvictionQueue::get_instance().batch_recycle(1024);
    EXPECT_EQ(0u, ailego::PoolPayloadBudget::get_instance().external_used());
    if (std::filesystem::exists(parquet_path)) {
      std::filesystem::remove(parquet_path);
    }
  }
  std::string parquet_path;
};

TEST_F(BufferPoolStoreTest, PointFetchStreamsWhenWholeColumnCannotFit) {
  constexpr int rows = 8192;
  const auto value = [](int row) {
    return std::to_string(row) + std::string(2048, 'x');
  };
  {
    arrow::StringBuilder strings;
    arrow::Int64Builder numbers;
    arrow::UInt64Builder doc_ids;
    arrow::StringBuilder user_ids;
    for (int row = 0; row < rows; ++row) {
      ASSERT_TRUE(
          (row % 11 ? strings.Append(value(row)) : strings.AppendNull()).ok());
      ASSERT_TRUE(numbers.Append(row * 3).ok());
      ASSERT_TRUE(doc_ids.Append(row).ok());
      ASSERT_TRUE(user_ids.Append(std::to_string(row)).ok());
    }
    auto text = strings.Finish();
    auto numeric = numbers.Finish();
    auto doc_id = doc_ids.Finish();
    auto user_id = user_ids.Finish();
    ASSERT_TRUE(text.ok());
    ASSERT_TRUE(numeric.ok());
    ASSERT_TRUE(doc_id.ok());
    ASSERT_TRUE(user_id.ok());
    auto table = arrow::Table::Make(
        arrow::schema({arrow::field("wide", arrow::utf8()),
                       arrow::field("number", arrow::int64()),
                       arrow::field(GLOBAL_DOC_ID, arrow::uint64()),
                       arrow::field(USER_ID, arrow::utf8())}),
        {*text, *numeric, *doc_id, *user_id});
    auto output = arrow::io::FileOutputStream::Open(parquet_path);
    ASSERT_TRUE(output.ok());
    parquet::WriterProperties::Builder properties;
    properties.disable_dictionary();
    properties.data_pagesize(64 * 1024);
    ASSERT_TRUE(parquet::arrow::WriteTable(*table, arrow::default_memory_pool(),
                                           *output, rows, properties.build())
                    .ok());
    ASSERT_TRUE((*output)->Close().ok());
  }
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  ASSERT_TRUE(store->open().ok());
  const std::vector<std::string> doc_columns{GLOBAL_DOC_ID, USER_ID, "wide"};
  auto doc_schema = arrow::schema({arrow::field(GLOBAL_DOC_ID, arrow::uint64()),
                                   arrow::field(USER_ID, arrow::utf8()),
                                   arrow::field("wide", arrow::utf8())});
  // An optional all-column pin must yield instead of aborting a scalar read.
  EXPECT_EQ(nullptr, store->fetch_doc(doc_columns, doc_schema, rows - 1));
  auto single = store->fetch(doc_columns, rows - 1);
  ASSERT_NE(nullptr, single);
  ASSERT_EQ(3u, single->values.size());
  EXPECT_EQ(value(rows - 1), single->values[2].scalar()->ToString());
  // The decoded string column is larger than the fixture's 10 MiB pool.
  // Preserve unsorted requests, duplicates, nulls and cross-column ordering.
  const std::vector<int> selected{8191, 0, 4095, 4095, 256, 7000};
  auto result = store->fetch_checked({"wide", "number"}, selected,
                                     detail::GetParquetMemoryPool().get());
  ASSERT_TRUE(result.ok()) << result.status().ToString();
  ASSERT_EQ(selected.size(), static_cast<size_t>((*result)->num_rows()));
  for (size_t i = 0; i < selected.size(); ++i) {
    auto text = (*result)->column(0)->GetScalar(i);
    auto number = (*result)->column(1)->GetScalar(i);
    ASSERT_TRUE(text.ok());
    ASSERT_TRUE(number.ok());
    EXPECT_EQ(selected[i] % 11 != 0, (*text)->is_valid);
    if ((*text)->is_valid) EXPECT_EQ(value(selected[i]), (*text)->ToString());
    EXPECT_EQ(selected[i] * 3,
              std::static_pointer_cast<arrow::Int64Scalar>(*number)->value);
  }
  std::vector<int> all_rows(rows);
  for (int row = 0; row < rows; ++row) all_rows[row] = row;
  auto too_large = store->fetch_checked({"wide"}, all_rows,
                                        detail::GetParquetMemoryPool().get());
  ASSERT_FALSE(too_large.ok());
  EXPECT_TRUE(too_large.status().IsOutOfMemory());
  store.reset();
  EXPECT_EQ(value(selected[0]),
            (*result)->column(0)->GetScalar(0).ValueOrDie()->ToString());
}

TEST_F(BufferPoolStoreTest, PathAndOpenedFileUseTheSameCacheIdentity) {
  auto file = arrow::io::ReadableFile::Open(parquet_path);
  ASSERT_TRUE(file.ok());
  ParquetBufferID path_key(parquet_path, 0, 0);
  ParquetBufferID opened_key(parquet_path, *file, 0, 0);
  EXPECT_TRUE(ParquetBufferIDEqual{}(path_key, opened_key));
  EXPECT_EQ(ParquetBufferIDHash{}(path_key), ParquetBufferIDHash{}(opened_key));

  const auto original = std::filesystem::last_write_time(parquet_path);
  std::filesystem::last_write_time(parquet_path,
                                   original + std::chrono::seconds(2));
  auto refreshed = opened_key.for_column(1, 2);
  ParquetBufferID changed(parquet_path, 1, 2);
  EXPECT_TRUE(ParquetBufferIDEqual{}(changed, refreshed));
  EXPECT_NE(opened_key.mtime, refreshed.mtime);
}

TEST_F(BufferPoolStoreTest, ClosingReaderRetiresPathAcquiredColumns) {
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  ASSERT_TRUE(store->open().ok());
  std::shared_ptr<arrow::ChunkedArray> escaped;
  {
    auto handle = ParquetBufferPool::get_instance().acquire_buffer(
        ParquetBufferID(parquet_path, 0, 0));
    escaped = handle.data();
    ASSERT_NE(nullptr, escaped);
  }
  const auto expected = escaped->GetScalar(0).ValueOrDie()->ToString();
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  store.reset();
  EXPECT_GT(pool.external_used(), 0u);
  EXPECT_EQ(expected, escaped->GetScalar(0).ValueOrDie()->ToString());
  escaped.reset();
  EXPECT_EQ(0u, pool.external_used());
}

TEST_F(BufferPoolStoreTest, MetadataHintRequiresMatchingFileIdentity) {
  auto reader = parquet::ParquetFileReader::OpenFile(parquet_path);
  ParquetBufferID source(parquet_path, 0, 0);
  source.metadata = reader->metadata();
  ParquetBufferID hinted(parquet_path, 1, 0, source);
  EXPECT_EQ(reader->metadata(), hinted.metadata.lock());
  ParquetBufferID plain(parquet_path, 1, 0);
  EXPECT_TRUE(ParquetBufferIDEqual{}(hinted, plain));
  EXPECT_EQ(ParquetBufferIDHash{}(hinted), ParquetBufferIDHash{}(plain));

  auto wrong_source = source;
  ++wrong_source.file_id;
  EXPECT_TRUE(
      ParquetBufferID(parquet_path, 1, 0, wrong_source).metadata.expired());
  wrong_source = source;
  ++wrong_source.mtime;
  EXPECT_TRUE(
      ParquetBufferID(parquet_path, 1, 0, wrong_source).metadata.expired());
  wrong_source = ParquetBufferID(parquet_path + ".other", 0, 0);
  wrong_source.file_id = source.file_id;
  wrong_source.mtime = source.mtime;
  wrong_source.metadata = source.metadata;
  EXPECT_TRUE(
      ParquetBufferID(parquet_path, 1, 0, wrong_source).metadata.expired());

  reader.reset();
  EXPECT_TRUE(hinted.metadata.expired());
}

TEST_F(BufferPoolStoreTest, MultiColumnFetchRefreshesIdentityBetweenCalls) {
  const auto write_rows = [&](int64_t start) {
    arrow::Int64Builder first, second;
    ASSERT_TRUE(first.AppendValues({start, start + 1}).ok());
    ASSERT_TRUE(second.AppendValues({start + 100, start + 101}).ok());
    std::shared_ptr<arrow::Array> a, b;
    ASSERT_TRUE(first.Finish(&a).ok());
    ASSERT_TRUE(second.Finish(&b).ok());
    auto schema = arrow::schema(
        {arrow::field("a", arrow::int64()), arrow::field("b", arrow::int64())});
    auto output = arrow::io::FileOutputStream::Open(parquet_path);
    ASSERT_TRUE(output.ok());
    ASSERT_TRUE(parquet::arrow::WriteTable(*arrow::Table::Make(schema, {a, b}),
                                           arrow::default_memory_pool(),
                                           *output, 1)
                    .ok());
    ASSERT_TRUE((*output)->Close().ok());
  };
  ASSERT_NO_FATAL_FAILURE(write_rows(10));
  const auto original_mtime = std::filesystem::last_write_time(parquet_path);
  BufferPoolForwardStore store(parquet_path);
  ASSERT_TRUE(store.open().ok());
  const auto check_rows = [&](int64_t start) {
    auto batch = store.fetch({"b", "a", "b"}, std::vector<int>{1, 0, 1});
    ASSERT_NE(nullptr, batch);
    for (int column = 0; column < 3; ++column) {
      const int64_t base = start + (column == 1 ? 0 : 100);
      EXPECT_EQ(std::to_string(base + 1),
                batch->column(column)->GetScalar(0).ValueOrDie()->ToString());
      EXPECT_EQ(std::to_string(base),
                batch->column(column)->GetScalar(1).ValueOrDie()->ToString());
    }
    auto scalar = store.fetch({"a", "b"}, 1);
    ASSERT_NE(nullptr, scalar);
    EXPECT_EQ(start + 1, std::static_pointer_cast<arrow::Int64Scalar>(
                             scalar->values[0].scalar())
                             ->value);
    EXPECT_EQ(start + 101, std::static_pointer_cast<arrow::Int64Scalar>(
                               scalar->values[1].scalar())
                               ->value);
  };
  ASSERT_NO_FATAL_FAILURE(check_rows(10));
  ASSERT_NO_FATAL_FAILURE(write_rows(30));
  std::filesystem::last_write_time(parquet_path,
                                   original_mtime + std::chrono::seconds(2));
  ASSERT_NO_FATAL_FAILURE(check_rows(30));
}

TEST_F(BufferPoolStoreTest, CacheKeyCopiesShareBudgetUntilLastOwner) {
  static_assert(std::is_nothrow_copy_constructible<ParquetBufferID>::value);
  static_assert(std::is_nothrow_copy_assignable<ParquetBufferID>::value);
  auto &pool = ailego::MemoryLimitPool::get_instance();
  const auto before = pool.metadata_used();
  ParquetBufferID escaped;
  size_t first = 0;
  {
    ParquetBufferID key(parquet_path + std::string(1024, 'x'), 0, 0);
    first = pool.metadata_used();
    EXPECT_GT(first, before + 1024);
    {
      auto copy = key;
      EXPECT_EQ(first, pool.metadata_used());
      EXPECT_EQ(key.filename().data(), copy.filename().data());
      auto moved = std::move(copy);
      EXPECT_EQ(first, pool.metadata_used());
      EXPECT_TRUE(ParquetBufferIDEqual{}(key, moved));
      escaped = moved;
    }
    EXPECT_EQ(first, pool.metadata_used());
  }
  EXPECT_EQ(first, pool.metadata_used());
  EXPECT_EQ(parquet_path + std::string(1024, 'x'), escaped.filename());
  escaped = {};
  EXPECT_EQ(before, pool.metadata_used());
}

TEST_F(BufferPoolStoreTest, CachedArrayShellsHaveSeparateMetadataAllowance) {
  detail::ParquetBufferLoader loader;
  detail::ParquetBufferPayload payload;
  size_t bytes = 0;
  ASSERT_TRUE(loader.load(ParquetBufferID(parquet_path, 0, 0), payload, bytes));
  EXPECT_GT(payload.metadata_memory.bytes(), 0u);
  EXPECT_GT(payload.arrow_refs.capacity(), 0u);
  EXPECT_GT(ailego::MemoryLimitPool::get_instance().external_used(),
            detail::GetParquetMemoryPool()->charged_bytes());
  loader.clear(payload);
  EXPECT_EQ(0u, payload.metadata_memory.bytes());
  EXPECT_EQ(0u, payload.arrow_refs.capacity());
  EXPECT_EQ(0u, ailego::PoolPayloadBudget::get_instance().external_used());
}

namespace {

// Model an evictable column, not an uncharged allocation. With no page
// buffers, background page-watermark eviction stays idle; only foreground
// admission should reclaim this payload. RAII also covers failed assertions.
class ReclaimableParquetBudget : public ailego::EvictableBlockOwner {
 public:
  ReclaimableParquetBudget() {
    memory_.add(1024 * 1024);
    data_.resize(1024 * 1024);
    auto &queue = ailego::BlockEvictionQueue::get_instance();
    queue.set_valid(this);
    registered_ = queue.add_single_block({0, 1, this, 0}, 0);
  }
  ~ReclaimableParquetBudget() override {
    ailego::BlockEvictionQueue::get_instance().set_invalid(this);
  }
  bool is_dead_block(ailego::eviction_key_t, ailego::version_t) override {
    return reclaimed_.load();
  }
  bool evict_block(ailego::eviction_key_t) override {
    std::vector<char>{}.swap(data_);
    memory_.retain_at_most(0);
    reclaimed_.store(true);
    return true;
  }
  bool registered() const {
    return registered_;
  }
  bool reclaimed() const {
    return reclaimed_.load();
  }

 private:
  ailego::MemoryReservation memory_;
  std::vector<char> data_;
  std::atomic<bool> reclaimed_{false};
  bool registered_{false};
};

}  // namespace

namespace {

struct ColdReadBarrier {
  void before_read() {
    std::unique_lock<std::mutex> lock(mutex);
    if (!armed || released) return;
    readers.insert(std::this_thread::get_id());
    cv.notify_all();
    cv.wait(lock, [&] { return released; });
  }
  std::mutex mutex;
  std::condition_variable cv;
  std::set<std::thread::id> readers;
  bool armed{false};
  bool released{false};
};

class BlockingReadFile : public arrow::io::RandomAccessFile {
 public:
  BlockingReadFile(std::shared_ptr<arrow::io::RandomAccessFile> source,
                   std::shared_ptr<ColdReadBarrier> barrier)
      : source_(std::move(source)), barrier_(std::move(barrier)) {}
  arrow::Status Close() override {
    return source_->Close();
  }
  bool closed() const override {
    return source_->closed();
  }
  arrow::Result<int64_t> Tell() const override {
    return source_->Tell();
  }
  arrow::Status Seek(int64_t position) override {
    return source_->Seek(position);
  }
  arrow::Result<int64_t> GetSize() override {
    return source_->GetSize();
  }
  arrow::Result<int64_t> Read(int64_t bytes, void *out) override {
    return source_->Read(bytes, out);
  }
  arrow::Result<std::shared_ptr<arrow::Buffer>> Read(int64_t bytes) override {
    return source_->Read(bytes);
  }
  arrow::Result<int64_t> ReadAt(int64_t position, int64_t bytes,
                                void *out) override {
    barrier_->before_read();
    return source_->ReadAt(position, bytes, out);
  }
  arrow::Result<std::shared_ptr<arrow::Buffer>> ReadAt(int64_t position,
                                                       int64_t bytes) override {
    barrier_->before_read();
    return source_->ReadAt(position, bytes);
  }

 private:
  std::shared_ptr<arrow::io::RandomAccessFile> source_;
  std::shared_ptr<ColdReadBarrier> barrier_;
};

}  // namespace

TEST_F(BufferPoolStoreTest, CacheKeyConstructionReclaimsOutsideCacheLock) {
  auto &pool = ailego::MemoryLimitPool::get_instance();
  ReclaimableParquetBudget cached;
  ASSERT_TRUE(cached.registered());
  ASSERT_EQ(0u, pool.stats().page_used);
  ailego::MemoryReservation pressure;
  pressure.add(pool.capacity() - pool.committed());
  ASSERT_FALSE(cached.reclaimed());
  ASSERT_NO_THROW(ParquetBufferID(parquet_path + std::string(1024, 'x'), 0, 0));
  EXPECT_TRUE(cached.reclaimed());
  EXPECT_LE(pool.committed(), pool.capacity());
}

TEST_F(BufferPoolStoreTest, SharedKeyCopiesDoNotReclaimAtFullBudget) {
  auto &pool = ailego::MemoryLimitPool::get_instance();
  ParquetBufferID key(parquet_path, 0, 0);
  ReclaimableParquetBudget cached;
  ASSERT_TRUE(cached.registered());
  ASSERT_EQ(0u, pool.stats().page_used);
  ailego::MemoryReservation pressure;
  pressure.add(pool.capacity() - pool.committed());
  ASSERT_EQ(pool.capacity(), pool.committed());

  auto copy = key;
  ParquetBufferID assigned;
  assigned = copy;
  ParquetBufferID hinted(parquet_path, 1, 0, key);
  EXPECT_EQ(key.filename().data(), assigned.filename().data());
  EXPECT_EQ(key.filename().data(), hinted.filename().data());
  EXPECT_TRUE(ParquetBufferIDEqual{}(key, copy));
  EXPECT_EQ(ParquetBufferIDHash{}(key), ParquetBufferIDHash{}(copy));
  EXPECT_FALSE(cached.reclaimed());
  EXPECT_EQ(pool.capacity(), pool.committed());

  // Cache node/bucket admission must still refuse without invoking an owner:
  // its caller may hold the same cache lock that owner needs to evict.
  ailego::CacheMetadataAllocator<char> allocator;
  EXPECT_THROW(allocator.allocate(64), std::bad_alloc);
  EXPECT_FALSE(cached.reclaimed());
}

TEST_F(BufferPoolStoreTest, CacheKeyRejectsPinnedBudgetAndCanRetry) {
  auto &pool = ailego::MemoryLimitPool::get_instance();
  ASSERT_EQ(0u, pool.stats().page_used);
  ailego::MemoryReservation pressure;
  pressure.add(pool.capacity() - pool.committed());
  const auto before = pool.metadata_used();
  EXPECT_THROW(ParquetBufferID(parquet_path, 0, 0), std::bad_alloc);
  EXPECT_EQ(before, pool.metadata_used());
  EXPECT_LE(pool.committed(), pool.capacity());
  // Allow the shared control block but not a long filename. Constructor
  // rollback must release the control block's reservation as well.
  pressure.retain_at_most(pressure.bytes() - 256);
  const auto partial = pool.metadata_used();
  EXPECT_THROW(ParquetBufferID(parquet_path + std::string(1024, 'x'), 0, 0),
               std::bad_alloc);
  EXPECT_EQ(partial, pool.metadata_used());
  pressure.retain_at_most(0);
  EXPECT_NO_THROW(ParquetBufferID(parquet_path, 0, 0));
}

TEST_F(BufferPoolStoreTest, ConcurrentKeyCopiesKeepLastOwnerBudget) {
  auto &pool = ailego::MemoryLimitPool::get_instance();
  const auto before = pool.metadata_used();
  ParquetBufferID key(parquet_path + std::string(1024, 'x'), 0, 0);
  ailego::MemoryReservation pressure;
  pressure.add(pool.capacity() - pool.committed());
  std::atomic<bool> start{false};
  std::atomic<bool> failed{false};
  std::array<std::thread, 4> workers;
  for (auto &worker : workers) {
    worker = std::thread([key, &start, &failed] {
      while (!start.load()) std::this_thread::yield();
      for (size_t i = 0; i < 256; ++i) {
        auto copy = key;
        ParquetBufferID assigned;
        assigned = copy;
        if (assigned.filename().data() != key.filename().data() ||
            !ParquetBufferIDEqual{}(assigned, key)) {
          failed.store(true);
        }
      }
    });
  }
  key = {};
  EXPECT_GT(pool.metadata_used(), before);
  start.store(true);
  for (auto &worker : workers) worker.join();
  EXPECT_FALSE(failed.load());
  EXPECT_EQ(before, pool.metadata_used());
}

TEST_F(BufferPoolStoreTest, PinnedViewReclaimsBudgetWithoutCopyingKeyStorage) {
  auto &pool = ailego::MemoryLimitPool::get_instance();
  ParquetBufferID key(parquet_path, 0, 0);
  auto handle = ParquetBufferPool::get_instance().acquire_buffer(key);
  ASSERT_NE(nullptr, handle.data());
  ReclaimableParquetBudget cached;
  ASSERT_TRUE(cached.registered());
  ASSERT_EQ(0u, pool.stats().page_used);
  ailego::MemoryReservation pressure;
  pressure.add(pool.capacity() - pool.used());
  ASSERT_EQ(pool.capacity(), pool.used());
  ASSERT_EQ(pool.capacity(), pool.committed());
  ASSERT_FALSE(cached.reclaimed());
  std::shared_ptr<arrow::ChunkedArray> view;
  ASSERT_NO_THROW(view = handle.data());
  ASSERT_NE(nullptr, view);
  EXPECT_TRUE(view->ValidateFull().ok());
  EXPECT_TRUE(cached.reclaimed());
  EXPECT_LE(pool.committed(), pool.capacity());
}

TEST_F(BufferPoolStoreTest, ForwardFetchReclaimsBudgetAtKeyBoundary) {
  BufferPoolForwardStore store(parquet_path);
  ASSERT_TRUE(store.open().ok());
  auto expected = store.fetch({"id"}, std::vector<int>{1});
  ASSERT_NE(nullptr, expected);
  // Cover result buffers separately, so the first reclaim opportunity is
  // the key/view path rather than an earlier Arrow output-builder reserve.
  ailego::MemoryReservation output_allowance;
  output_allowance.add(64 * 1024);
  detail::FetchMemoryPool output_pool(arrow::default_memory_pool(), 64 * 1024);
  auto &pool = ailego::MemoryLimitPool::get_instance();
  ReclaimableParquetBudget cached;
  ASSERT_TRUE(cached.registered());
  ASSERT_EQ(0u, pool.stats().page_used);
  ailego::MemoryReservation pressure;
  pressure.add(pool.capacity() - pool.used());
  ASSERT_EQ(pool.capacity(), pool.used());
  ASSERT_EQ(pool.capacity(), pool.committed());
  ASSERT_FALSE(cached.reclaimed());
  {
    auto actual = store.fetch_checked({"id"}, {1}, &output_pool);
    ASSERT_TRUE(actual.ok()) << actual.status().ToString();
    EXPECT_TRUE((*actual)->Equals(*expected));
    EXPECT_TRUE(cached.reclaimed());
    EXPECT_LE(pool.committed(), pool.capacity());
  }
  EXPECT_EQ(0, output_pool.bytes_allocated());
}

TEST_F(BufferPoolStoreTest, DecodedBufferOwnershipReclaimsCachedBudget) {
  arrow::Int64Builder builder;
  ASSERT_TRUE(builder.AppendValues({17, 23}).ok());
  std::shared_ptr<arrow::Array> array;
  ASSERT_TRUE(builder.Finish(&array).ok());
  detail::ParquetBufferPayload payload;
  payload.arrow = std::make_shared<arrow::ChunkedArray>(array);
  ReclaimableParquetBudget cached;
  ASSERT_TRUE(cached.registered());
  auto &pool = ailego::MemoryLimitPool::get_instance();
  ASSERT_EQ(0u, pool.stats().page_used);
  ailego::MemoryReservation pressure;
  pressure.add(pool.capacity() - pool.used());
  ASSERT_EQ(pool.capacity(), pool.committed());
  ASSERT_FALSE(cached.reclaimed());
  detail::ParquetBufferLoader loader;
  size_t bytes = 0;
  try {
    loader.retain_buffers(payload, bytes);
  } catch (const std::bad_alloc &) {
    ADD_FAILURE()
        << "Decoded buffer ownership rejected reclaimable cache budget";
    return;
  }
  EXPECT_TRUE(cached.reclaimed());
  EXPECT_GT(bytes, 0u);
  EXPECT_LE(pool.committed(), pool.capacity());
  EXPECT_EQ(23,
            std::static_pointer_cast<arrow::Int64Array>(payload.arrow->chunk(0))
                ->Value(1));
  array.reset();
  loader.clear(payload);
}

TEST_F(BufferPoolStoreTest,
       DecodedBufferOwnershipRejectsPinnedBudgetAndRetries) {
  arrow::StringBuilder builder;
  ASSERT_TRUE(builder.Append("first").ok());
  ASSERT_TRUE(builder.Append("second").ok());
  std::shared_ptr<arrow::Array> array;
  ASSERT_TRUE(builder.Finish(&array).ok());
  detail::ParquetBufferPayload payload;
  payload.arrow =
      std::make_shared<arrow::ChunkedArray>(arrow::ArrayVector{array, array});
  auto &pool = ailego::MemoryLimitPool::get_instance();
  ASSERT_EQ(0u, pool.stats().page_used);
  ailego::MemoryReservation pressure;
  pressure.add(pool.capacity() - pool.used());
  ASSERT_EQ(pool.capacity(), pool.committed());
  detail::ParquetBufferLoader loader;
  size_t bytes = 0;
  EXPECT_THROW(loader.retain_buffers(payload, bytes), std::bad_alloc);
  EXPECT_LE(pool.committed(), pool.capacity());
  EXPECT_TRUE(payload.arrow_refs.empty());
  pressure.retain_at_most(0);
  ASSERT_NO_THROW(loader.retain_buffers(payload, bytes));
  EXPECT_EQ(2u, payload.arrow_refs.size());
  EXPECT_EQ("second", std::static_pointer_cast<arrow::StringArray>(
                          payload.arrow->chunk(1))
                          ->GetString(1));
  array.reset();
  loader.clear(payload);
  EXPECT_EQ(0u, ailego::PoolPayloadBudget::get_instance().external_used());
}

TEST_F(BufferPoolStoreTest,
       EscapedArrayDataKeepsMetadataAndPinAfterRetirement) {
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  auto &cache = ParquetBufferPool::get_instance();
  std::shared_ptr<arrow::ArrayData> escaped;
  {
    ParquetBufferID key(parquet_path, 0, 0);
    auto handle = cache.acquire_buffer(key);
    const auto cached = pool.external_used();
    ASSERT_GT(cached, 0u);
    auto view = handle.data();
    ASSERT_NE(nullptr, view);
    EXPECT_GT(pool.external_used(), cached);
    escaped = view->chunk(0)->data();
    cache.retire_file(key);
  }
  EXPECT_GT(pool.external_used(), 0u);
  EXPECT_TRUE(arrow::MakeArray(escaped)->ValidateFull().ok());
  escaped.reset();
  EXPECT_EQ(0u, pool.external_used());
  EXPECT_EQ(0u, ParquetCacheEntryCount());
}

TEST_F(BufferPoolStoreTest, BufferedFilesUseSuppliedPoolForPathsAndUris) {
  auto owner = detail::GetParquetMemoryPool();
  arrow::ProxyMemoryPool memory_pool(owner.get());
  arrow::fs::LocalFileSystem fs;
  const auto absolute =
      std::filesystem::absolute(parquet_path).generic_u8string();
  auto uri = fs.MakeUri(absolute);
  ASSERT_TRUE(uri.ok()) << uri.status().ToString();
  for (const auto &path : {parquet_path, "./" + parquet_path, absolute, *uri,
                           *uri + "?use_mmap=true"}) {
    SCOPED_TRACE(path);
    std::shared_ptr<arrow::io::RandomAccessFile> file;
    std::string resolved;
    ASSERT_TRUE(CreateRandomAccessFileByUri(path, &file, &resolved,
                                            /*enable_mmap=*/false, &memory_pool)
                    .ok());
    EXPECT_FALSE(file->supports_zero_copy());
    auto result = file->ReadAt(0, 8);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    auto bytes = std::move(result).ValueOrDie();
    EXPECT_GE(memory_pool.bytes_allocated(), 8);
    ASSERT_TRUE(file->Close().ok());
    file.reset();
    EXPECT_EQ("PAR1",
              std::string(reinterpret_cast<const char *>(bytes->data()), 4));
    bytes.reset();
    EXPECT_EQ(0, memory_pool.bytes_allocated());
  }
}

TEST_F(BufferPoolStoreTest, BufferedFileReadRejectsExhaustedSuppliedPool) {
  auto owner = detail::GetParquetMemoryPool();
  detail::FetchMemoryPool memory_pool(owner.get(), 7);
  std::shared_ptr<arrow::io::RandomAccessFile> file;
  std::string resolved;
  ASSERT_TRUE(CreateRandomAccessFileByUri(parquet_path, &file, &resolved,
                                          /*enable_mmap=*/false, &memory_pool)
                  .ok());
  EXPECT_TRUE(file->ReadAt(0, 8).status().IsOutOfMemory());
  EXPECT_EQ(0, memory_pool.bytes_allocated());
  ASSERT_TRUE(file->Close().ok());
}

TEST_F(BufferPoolStoreTest, CheckedFetchPreservesResultAllocationFailure) {
  BufferPoolForwardStore store(parquet_path);
  ASSERT_TRUE(store.open().ok());
  auto owner = detail::GetParquetMemoryPool();
  detail::FetchMemoryPool exhausted(owner.get(), 0);
  auto failed = store.fetch_checked({"name"}, {0, 1}, &exhausted);
  EXPECT_TRUE(failed.status().IsOutOfMemory()) << failed.status().ToString();
  EXPECT_EQ(0, exhausted.bytes_allocated());
  auto recovered = store.fetch_checked({"name"}, {0, 1}, owner.get());
  ASSERT_TRUE(recovered.ok()) << recovered.status().ToString();
  EXPECT_EQ(2, (*recovered)->num_rows());
  auto invalid = store.fetch_checked({"name"}, {-1}, owner.get());
  EXPECT_TRUE(invalid.status().IsIndexError()) << invalid.status().ToString();
}

TEST_F(BufferPoolStoreTest, CheckedCachePreservesIoAndFormatFailures) {
  auto &cache = ParquetBufferPool::get_instance();
  auto missing = cache.acquire_buffer_checked(
      ParquetBufferID(parquet_path + ".missing", 0, 0));
  EXPECT_TRUE(missing.status().IsIOError()) << missing.status().ToString();
  auto invalid =
      cache.acquire_buffer_checked(ParquetBufferID(parquet_path, 0, -1));
  EXPECT_TRUE(invalid.status().IsInvalid()) << invalid.status().ToString();
  std::filesystem::resize_file(parquet_path,
                               std::filesystem::file_size(parquet_path) - 8);
  auto malformed =
      cache.acquire_buffer_checked(ParquetBufferID(parquet_path, 0, 0));
  EXPECT_FALSE(malformed.ok());
  EXPECT_FALSE(malformed.status().IsOutOfMemory())
      << malformed.status().ToString();
}

TEST_F(BufferPoolStoreTest, DefaultFileOpenPreservesMmapBehavior) {
  std::shared_ptr<arrow::io::RandomAccessFile> file;
  std::string resolved;
  ASSERT_TRUE(CreateRandomAccessFileByUri(parquet_path, &file, &resolved).ok());
  EXPECT_TRUE(file->supports_zero_copy());
  ASSERT_TRUE(file->Close().ok());
}

TEST_F(BufferPoolStoreTest, BufferedStoreReadsLocalFileUriWithMmapFlag) {
  arrow::fs::LocalFileSystem fs;
  auto uri =
      fs.MakeUri(std::filesystem::absolute(parquet_path).generic_u8string());
  ASSERT_TRUE(uri.ok()) << uri.status().ToString();
  BufferPoolForwardStore store(*uri + "?use_mmap=true");
  ASSERT_TRUE(store.open().ok());
  auto table = store.fetch({"id", "name", "score"}, {0, 1, 2});
  ASSERT_NE(nullptr, table);
  EXPECT_EQ(3, table->num_rows());
}

TEST_F(BufferPoolStoreTest, BudgetedFooterLeaseFollowsBorrowedMetadata) {
  auto file = arrow::io::ReadableFile::Open(parquet_path).ValueOrDie();
  auto &budget = detail::ParquetMetadataBudget::instance();
  auto reader = detail::OpenBudgetedParquetReader(
                    file, parquet::default_reader_properties())
                    .ValueOrDie();
  auto metadata = reader->metadata();
  const auto charge = budget.reserved();
  EXPECT_GE(charge, detail::ParquetMetadataBudget::estimate(metadata->size()));
  reader.reset();
  auto borrowed = detail::OpenBudgetedParquetReader(
                      file, parquet::default_reader_properties(), metadata)
                      .ValueOrDie();
  EXPECT_EQ(metadata.get(), borrowed->metadata().get());
  EXPECT_EQ(charge, budget.reserved());
  std::weak_ptr<parquet::FileMetaData> weak = metadata;
  metadata.reset();
  EXPECT_FALSE(weak.expired());
  borrowed.reset();
  EXPECT_TRUE(weak.expired());
  EXPECT_EQ(charge, budget.reserved());
  auto reopened = detail::OpenBudgetedParquetReader(
                      file, parquet::default_reader_properties())
                      .ValueOrDie();
  EXPECT_EQ(charge, budget.reserved());
}

TEST_F(BufferPoolStoreTest, MalformedFooterFailsBeforeAdmission) {
  auto &budget = detail::ParquetMetadataBudget::instance();
  const auto before = budget.reserved();
  std::filesystem::resize_file(parquet_path, 8);
  auto file = arrow::io::ReadableFile::Open(parquet_path).ValueOrDie();
  EXPECT_TRUE(detail::OpenBudgetedParquetReader(
                  file, parquet::default_reader_properties())
                  .status()
                  .IsInvalid());
  EXPECT_EQ(before, budget.reserved());
}

TEST_F(BufferPoolStoreTest, BorrowedMetadataSkipsFooterParsingOnEveryLoad) {
  auto reader = parquet::ParquetFileReader::OpenFile(parquet_path);
  auto metadata = reader->metadata();
  reader.reset();
  // Data pages are intact, but a new footer parse must fail. A valid, borrowed
  // footer still suffices to decode the same column after each cache eviction.
  std::filesystem::resize_file(parquet_path,
                               std::filesystem::file_size(parquet_path) - 8);
  ParquetBufferID source(parquet_path, 0, 0);
  source.metadata = metadata;
  detail::ParquetBufferLoader loader;
  detail::ParquetBufferPayload payload;
  size_t bytes = 0;
  for (size_t attempt = 0; attempt < 3; ++attempt) {
    ASSERT_TRUE(loader.load(ParquetBufferID(parquet_path, 0, 0, source),
                            payload, bytes));
    ASSERT_NE(nullptr, payload.arrow);
    EXPECT_EQ(metadata->RowGroup(0)->num_rows(), payload.arrow->length());
    loader.clear(payload);
  }
  metadata.reset();
  EXPECT_TRUE(source.metadata.expired());
  EXPECT_FALSE(loader
                   .load_checked(ParquetBufferID(parquet_path, 0, 0, source),
                                 payload, bytes)
                   .ok());
  loader.clear(payload);
}

TEST_F(BufferPoolStoreTest, EscapedNestedScalarKeepsParquetCachePinned) {
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  ASSERT_TRUE(store->open().ok());
  const int column = store->physic_schema()->GetFieldIndex("list_utf8");
  ASSERT_GE(column, 0);

  std::shared_ptr<arrow::Scalar> scalar;
  {
    auto handle = ParquetBufferPool::get_instance().acquire_buffer(
        ParquetBufferID(parquet_path, column, /*row_group=*/0));
    auto data = handle.data();
    ASSERT_NE(nullptr, data);
    auto scalar_result = data->GetScalar(0);
    ASSERT_TRUE(scalar_result.ok()) << scalar_result.status().ToString();
    scalar = scalar_result.ValueOrDie();
  }

  auto &memory_pool = ailego::PoolPayloadBudget::get_instance();
  const size_t pinned_bytes = memory_pool.external_used();
  ASSERT_GT(pinned_bytes, 0u);
  EXPECT_EQ(0u, ailego::BlockEvictionQueue::get_instance().batch_recycle(1));
  EXPECT_EQ(pinned_bytes, memory_pool.external_used());
  EXPECT_FALSE(scalar->ToString().empty());

  scalar.reset();
  EXPECT_EQ(1u, ailego::BlockEvictionQueue::get_instance().batch_recycle(1));
  EXPECT_EQ(0u, memory_pool.external_used());
}


TEST_F(BufferPoolStoreTest, ClosedReaderRetiresColumnsAfterEscapedViews) {
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  ASSERT_TRUE(store->open().ok());
  const size_t before = ParquetCacheEntryCount();
  const int column = store->physic_schema()->GetFieldIndex("list_utf8");
  ASSERT_GE(column, 0);
  std::shared_ptr<arrow::Scalar> scalar;
  {
    auto handle = ParquetBufferPool::get_instance().acquire_buffer(
        ParquetBufferID(parquet_path, column, 0));
    auto data = handle.data();
    ASSERT_NE(nullptr, data);
    auto result = data->GetScalar(0);
    ASSERT_TRUE(result.ok());
    scalar = *result;
  }
  const auto expected = scalar->ToString();
  store.reset();
  EXPECT_EQ(before + 1, ParquetCacheEntryCount());
  EXPECT_EQ(expected, scalar->ToString());
  scalar.reset();
  EXPECT_EQ(before, ParquetCacheEntryCount());
}

TEST_F(BufferPoolStoreTest,
       ConcurrentColdLoadsStayBoundedWithoutBlockingHotPins) {
  auto file = arrow::io::ReadableFile::Open(parquet_path);
  ASSERT_TRUE(file.ok());
  auto size = (*file)->GetSize();
  ASSERT_TRUE(size.ok());
  auto bytes = (*file)->Read(*size);
  ASSERT_TRUE(bytes.ok());
  auto barrier = std::make_shared<ColdReadBarrier>();
  std::shared_ptr<arrow::io::RandomAccessFile> input =
      std::make_shared<BlockingReadFile>(
          std::make_shared<arrow::io::BufferReader>(*bytes), barrier);
  auto &pool = ParquetBufferPool::get_instance();
  ParquetBufferID hot_key("gate-hot", input, 0, 0);
  auto warm = pool.acquire_buffer(hot_key);
  ASSERT_NE(warm.borrowed_data(), nullptr);
  std::vector<ParquetBufferID> keys;
  for (int i = 0; i < 8; ++i)
    keys.emplace_back("gate-cold-" + std::to_string(i), input, 0, 0);
  barrier->armed = true;
  std::atomic<int> ready{0};
  std::atomic<bool> correct{true};
  std::vector<std::thread> workers;
  for (const auto &key : keys)
    workers.emplace_back([&, key] {
      ready.fetch_add(1);
      try {
        auto handle = pool.acquire_buffer(key);
        const auto *column = handle.borrowed_data();
        if (!column || !column->ValidateFull().ok()) correct.store(false);
      } catch (...) {
        correct.store(false);
      }
    });
  while (ready.load() != 8) std::this_thread::yield();
  std::unique_lock<std::mutex> lock(barrier->mutex);
  const bool four_started =
      barrier->cv.wait_for(lock, std::chrono::seconds(5),
                           [&] { return barrier->readers.size() >= 4; });
  std::atomic<bool> hot_done{false};
  std::thread hot([&] {
    try {
      auto handle = pool.acquire_buffer(hot_key);
      if (!handle.borrowed_data()) correct.store(false);
    } catch (...) {
      correct.store(false);
    }
    hot_done.store(true);
    barrier->cv.notify_all();
  });
  const bool hot_unblocked = barrier->cv.wait_for(
      lock, std::chrono::seconds(1), [&] { return hot_done.load(); });
  const bool exceeded =
      barrier->cv.wait_for(lock, std::chrono::milliseconds(100),
                           [&] { return barrier->readers.size() > 4; });
  const auto initial_readers = barrier->readers.size();
  // Always unblock and join the workers before reporting an assertion failure.
  barrier->released = true;
  lock.unlock();
  barrier->cv.notify_all();
  hot.join();
  for (auto &worker : workers) worker.join();
  EXPECT_TRUE(four_started);
  EXPECT_FALSE(exceeded);
  EXPECT_EQ(initial_readers, 4u);
  EXPECT_TRUE(hot_unblocked);
  EXPECT_TRUE(correct.load());
}

TEST_F(BufferPoolStoreTest, ParquetFetch) {
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  EXPECT_TRUE(store->open().ok());
  TablePtr table = store->fetch({"id", "name", "score"}, {0, 1, 2});
  ASSERT_TRUE(table != nullptr);
  EXPECT_EQ(table->num_rows(), 3);
}

TEST_F(BufferPoolStoreTest, ParquetFetchNullAndEmptyStrings) {
  arrow::ArrayVector arrays;
  for (int pattern = 0; pattern < 3; ++pattern) {
    arrow::StringBuilder builder;
    for (int row = 0; row < 4; ++row) {
      if (pattern == 0 || (pattern == 2 && row % 2 == 0)) {
        ASSERT_TRUE(builder.AppendNull().ok());
      } else {
        ASSERT_TRUE(builder.Append("").ok());
      }
    }
    std::shared_ptr<arrow::Array> array;
    ASSERT_TRUE(builder.Finish(&array).ok());
    arrays.push_back(std::move(array));
  }
  auto schema = arrow::schema({arrow::field("all_null", arrow::utf8()),
                               arrow::field("all_empty", arrow::utf8()),
                               arrow::field("mixed", arrow::utf8())});
  auto expected = arrow::Table::Make(schema, arrays);
  auto output = arrow::io::FileOutputStream::Open(parquet_path);
  ASSERT_TRUE(output.ok()) << output.status().ToString();
  ASSERT_TRUE(parquet::arrow::WriteTable(*expected,
                                         arrow::default_memory_pool(), *output,
                                         /*chunk_size=*/2)
                  .ok());
  ASSERT_TRUE((*output)->Close().ok());

  for (int column = 0; column < 3; ++column) {
    SCOPED_TRACE(column);
    for (int row_group = 0; row_group < 2; ++row_group) {
      const ParquetBufferID buffer_id(parquet_path, column, row_group);
      SCOPED_TRACE(buffer_id.to_string());
      auto handle = ParquetBufferPool::get_instance().acquire_buffer(buffer_id);
      auto data = handle.data();
      ASSERT_NE(nullptr, data);
      auto status = data->ValidateFull();
      ASSERT_TRUE(status.ok()) << status.ToString();
      ASSERT_TRUE(
          data->Equals(expected->column(column)->Slice(row_group * 2, 2)))
          << "actual type=" << data->type()->ToString()
          << ", rows=" << data->length() << "; expected type=utf8, rows=2";
    }
  }

  BufferPoolForwardStore store(parquet_path);
  ASSERT_TRUE(store.open().ok());
  auto actual = store.fetch({"all_null", "all_empty", "mixed"},
                            std::vector<int>{0, 1, 2, 3});
  ASSERT_NE(nullptr, actual);
  EXPECT_TRUE(actual->Equals(*expected));
}


TEST_F(BufferPoolStoreTest, ParquetFetchWithSelectColumns) {
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  EXPECT_TRUE(store->open().ok());
  TablePtr table = store->fetch({"id", "name"}, {0, 1, 2});
  ASSERT_TRUE(table != nullptr);
  EXPECT_EQ(table->num_rows(), 3);
  EXPECT_EQ(table->num_columns(), 2);
}

TEST_F(BufferPoolStoreTest, ParquetFetchWithUID) {
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  EXPECT_TRUE(store->open().ok());
  auto table = store->fetch({USER_ID, "id", "name"}, {0, 1, 2});
  ASSERT_TRUE(table != nullptr);
  EXPECT_EQ(table->num_rows(), 3);
  EXPECT_EQ(table->num_columns(), 3);
}

TEST_F(BufferPoolStoreTest, ParquetFetchWithGlobalDocID) {
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  EXPECT_TRUE(store->open().ok());
  auto table = store->fetch({GLOBAL_DOC_ID, "id", "name"}, {0, 1, 2});
  ASSERT_TRUE(table != nullptr);
  EXPECT_EQ(table->num_rows(), 3);
  EXPECT_EQ(table->num_columns(), 3);
}

TEST_F(BufferPoolStoreTest, ParquetFetchWitEmptyColumns) {
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  EXPECT_TRUE(store->open().ok());
  TablePtr table = store->fetch({}, std::vector<int>{});
  EXPECT_EQ(table, nullptr);
}

TEST_F(BufferPoolStoreTest, ParquetFetchWitEmptyIndices) {
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  EXPECT_TRUE(store->open().ok());
  TablePtr table = store->fetch({"id", "name"}, std::vector<int>{});
  ASSERT_TRUE(table != nullptr);
  EXPECT_EQ(table->num_rows(), 0);
  EXPECT_EQ(table->num_columns(), 2);
}

TEST_F(BufferPoolStoreTest, ParquetFetchWithMoreIndices) {
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  EXPECT_TRUE(store->open().ok());
  TablePtr table = store->fetch({"id"}, {0, 1, 2, 3, 6, 2, 1, 7});
  ASSERT_TRUE(table != nullptr);
  EXPECT_EQ(table->num_rows(), 8);
  EXPECT_EQ(table->num_columns(), 1);
}

TEST_F(BufferPoolStoreTest, ParquetFetchWithInvalidIndices) {
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  EXPECT_TRUE(store->open().ok());
  TablePtr table = store->fetch({"id"}, {0, 1, 30});
  ASSERT_TRUE(table == nullptr);
}

TEST_F(BufferPoolStoreTest, ParquetFetchCheckOrderWithLocalRowIDMiddle) {
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  EXPECT_TRUE(store->open().ok());
  TablePtr table =
      store->fetch({"id", "name", LOCAL_ROW_ID, "score"}, {0, 3, 6, 1, 0});
  ASSERT_TRUE(table != nullptr);
  EXPECT_EQ(table->num_rows(), 5);
  EXPECT_EQ(table->num_columns(), 4);
  auto field = table->schema()->field(2);
  EXPECT_EQ(field->name(), LOCAL_ROW_ID);

  // Get data from the _zvec_row_id_ column for each row
  auto id_column = table->column(2);
  auto id_array =
      std::dynamic_pointer_cast<arrow::UInt64Array>(id_column->chunk(0));

  std::vector<int32_t> expected_ids = {0, 3, 6, 1, 0};
  std::vector<int32_t> actual_ids;

  for (int i = 0; i < id_array->length(); ++i) {
    actual_ids.push_back(id_array->Value(i));
  }

  EXPECT_EQ(actual_ids, expected_ids)
      << "ID column values don't match expected order";
}


TEST_F(BufferPoolStoreTest, ParquetFetchCheckOrderWithLocalRowIDEnd) {
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  EXPECT_TRUE(store->open().ok());
  TablePtr table =
      store->fetch({"id", "name", "score", LOCAL_ROW_ID}, {0, 3, 6, 1, 0});
  ASSERT_TRUE(table != nullptr);
  EXPECT_EQ(table->num_rows(), 5);
  EXPECT_EQ(table->num_columns(), 4);
  auto field = table->schema()->field(3);
  EXPECT_EQ(field->name(), LOCAL_ROW_ID);

  // Get data from the _zvec_row_id_ column for each row
  auto id_column = table->column(3);
  auto id_array =
      std::dynamic_pointer_cast<arrow::UInt64Array>(id_column->chunk(0));

  std::vector<int32_t> expected_ids = {0, 3, 6, 1, 0};
  std::vector<int32_t> actual_ids;

  for (int i = 0; i < id_array->length(); ++i) {
    actual_ids.push_back(id_array->Value(i));
  }

  EXPECT_EQ(actual_ids, expected_ids)
      << "ID column values don't match expected order";
}


TEST_F(BufferPoolStoreTest, ParquetScan) {
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  EXPECT_TRUE(store->open().ok());
  auto reader = store->scan({"id", "name", "score"});
  ASSERT_NE(nullptr, reader);
  {
    auto &pool = ailego::PoolPayloadBudget::get_instance();
    ailego::MemoryReservation pressure;
    pressure.add(pool.available());
    const size_t before = pool.external_used();
    auto denied = store->fetch_checked({"id"}, {0, 1},
                                       detail::GetParquetMemoryPool().get());
    EXPECT_TRUE(denied.status().IsOutOfMemory()) << denied.status().ToString();
    std::shared_ptr<arrow::RecordBatch> batch;
    EXPECT_TRUE(reader->ReadNext(&batch).IsOutOfMemory());
    EXPECT_EQ(nullptr, batch);
    EXPECT_EQ(before, pool.external_used());
  }
  auto recovered = store->fetch_checked({"id"}, {0, 1},
                                        detail::GetParquetMemoryPool().get());
  ASSERT_TRUE(recovered.ok()) << recovered.status().ToString();
  EXPECT_EQ(2, (*recovered)->num_rows());
  reader = store->scan({"id", "name", "score"});
  ASSERT_NE(nullptr, reader);
  int batch_count = 0;
  int total_rows = 0;
  while (true) {
    std::shared_ptr<arrow::RecordBatch> batch;
    auto status = reader->ReadNext(&batch);
    ASSERT_TRUE(status.ok());
    if (batch == nullptr) {
      break;
    }
    EXPECT_GT(batch->num_rows(), 0);
    EXPECT_EQ(batch->num_columns(), 3);
    batch_count++;
    total_rows += batch->num_rows();
  }
  EXPECT_GT(batch_count, 0);
  EXPECT_EQ(total_rows, 10);
}

TEST_F(BufferPoolStoreTest, ParquetScanWithSelectColumns) {
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  EXPECT_TRUE(store->open().ok());
  auto reader = store->scan({"id", "name"});
  int batch_count = 0;
  int total_rows = 0;
  while (true) {
    std::shared_ptr<arrow::RecordBatch> batch;
    auto status = reader->ReadNext(&batch);
    ASSERT_TRUE(status.ok());
    if (batch == nullptr) {
      break;
    }
    EXPECT_GT(batch->num_rows(), 0);
    EXPECT_EQ(batch->num_columns(), 2);
    batch_count++;
    total_rows += batch->num_rows();
  }
  EXPECT_GT(batch_count, 0);
  EXPECT_EQ(total_rows, 10);
}

TEST_F(BufferPoolStoreTest, ParquetScanWithInvalidColumn) {
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  EXPECT_TRUE(store->open().ok());
  auto reader = store->scan({"id", "unknown_column"});
  ASSERT_TRUE(reader == nullptr);
}


TEST_F(BufferPoolStoreTest, ParquetScanWithUserID) {
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  EXPECT_TRUE(store->open().ok());
  auto reader = store->scan({USER_ID, "id", "name", "score"});
  int batch_count = 0;
  int total_rows = 0;
  while (true) {
    std::shared_ptr<arrow::RecordBatch> batch;
    auto status = reader->ReadNext(&batch);
    ASSERT_TRUE(status.ok());
    if (batch == nullptr) {
      break;
    }
    EXPECT_GT(batch->num_rows(), 0);
    EXPECT_EQ(batch->num_columns(), 4);
    batch_count++;
    total_rows += batch->num_rows();
  }
  EXPECT_GT(batch_count, 0);
  EXPECT_EQ(total_rows, 10);
}

TEST_F(BufferPoolStoreTest, ParquetScanWithGlobalDocID) {
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  EXPECT_TRUE(store->open().ok());
  auto reader = store->scan({GLOBAL_DOC_ID, "id", "name", "score"});
  int batch_count = 0;
  int total_rows = 0;
  while (true) {
    std::shared_ptr<arrow::RecordBatch> batch;
    auto status = reader->ReadNext(&batch);
    ASSERT_TRUE(status.ok());
    if (batch == nullptr) {
      break;
    }
    EXPECT_GT(batch->num_rows(), 0);
    EXPECT_EQ(batch->num_columns(), 4);
    batch_count++;
    total_rows += batch->num_rows();
  }
  EXPECT_GT(batch_count, 0);
  EXPECT_EQ(total_rows, 10);
}

TEST_F(BufferPoolStoreTest, ParquetFetchSingleRow) {
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  EXPECT_TRUE(store->open().ok());

  ExecBatchPtr batch = store->fetch({"id", "name", "score"}, 0);
  ASSERT_TRUE(batch != nullptr);
  EXPECT_EQ(batch->length, 1);
  EXPECT_EQ(batch->values.size(), 3);

  auto name =
      std::dynamic_pointer_cast<arrow::StringScalar>(batch->values[1].scalar());
  ASSERT_NE(nullptr, name);
  ASSERT_NE(nullptr, name->value);
  const auto expected_name = name->value->ToString();
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  ASSERT_GT(pool.external_used(), 0u);
  EXPECT_GT(ailego::BlockEvictionQueue::get_instance().batch_recycle(1024), 0u);
  // The selected string owns budgeted bytes after the cache is recycled.
  EXPECT_GT(pool.external_used(), 0u);
  EXPECT_EQ(expected_name, name->value->ToString());

  auto id_scalar = batch->values[0].scalar();
  ASSERT_TRUE(id_scalar != nullptr);
  auto id_value = std::dynamic_pointer_cast<arrow::Int32Scalar>(id_scalar);
  ASSERT_TRUE(id_value != nullptr);
  EXPECT_EQ(id_value->value, 1);
  batch.reset();
  name.reset();
  EXPECT_EQ(0u, pool.external_used());
}

TEST_F(BufferPoolStoreTest, ParquetFetchSpecificRow) {
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  EXPECT_TRUE(store->open().ok());

  ExecBatchPtr batch = store->fetch({"id", "name", "score"}, 3);
  ASSERT_TRUE(batch != nullptr);
  EXPECT_EQ(batch->length, 1);
  EXPECT_EQ(batch->values.size(), 3);

  auto id_scalar = batch->values[0].scalar();
  ASSERT_TRUE(id_scalar != nullptr);
  auto id_value = std::dynamic_pointer_cast<arrow::Int32Scalar>(id_scalar);
  ASSERT_TRUE(id_value != nullptr);
  EXPECT_EQ(id_value->value, 4);
}

TEST_F(BufferPoolStoreTest, ParquetFetchSingleRowWithUserID) {
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  EXPECT_TRUE(store->open().ok());

  ExecBatchPtr batch = store->fetch({USER_ID, "id", "name"}, 1);
  ASSERT_TRUE(batch != nullptr);
  EXPECT_EQ(batch->length, 1);
  EXPECT_EQ(batch->values.size(), 3);

  auto user_id_scalar = batch->values[0].scalar();
  ASSERT_TRUE(user_id_scalar != nullptr);
  EXPECT_TRUE(std::dynamic_pointer_cast<arrow::StringScalar>(user_id_scalar) !=
              nullptr);
}

TEST_F(BufferPoolStoreTest, ParquetFetchSingleRowWithGlobalDocID) {
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  EXPECT_TRUE(store->open().ok());

  ExecBatchPtr batch = store->fetch({GLOBAL_DOC_ID, "id", "name"}, 4);
  ASSERT_TRUE(batch != nullptr);
  EXPECT_EQ(batch->length, 1);
  EXPECT_EQ(batch->values.size(), 3);

  auto global_doc_id_scalar = batch->values[0].scalar();
  ASSERT_TRUE(global_doc_id_scalar != nullptr);
  EXPECT_TRUE(std::dynamic_pointer_cast<arrow::UInt64Scalar>(
                  global_doc_id_scalar) != nullptr);
}

TEST_F(BufferPoolStoreTest, ParquetFetchSingleRowWithNegativeIndex) {
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  EXPECT_TRUE(store->open().ok());

  ExecBatchPtr batch = store->fetch({"id", "name"}, -1);
  EXPECT_EQ(batch, nullptr);
}

TEST_F(BufferPoolStoreTest, ParquetFetchSingleRowWithOutOfRangeIndex) {
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  EXPECT_TRUE(store->open().ok());

  ExecBatchPtr batch = store->fetch({"id", "name"}, 15);
  EXPECT_EQ(batch, nullptr);
}

TEST_F(BufferPoolStoreTest, ParquetFetchSingleRowWithInvalidColumn) {
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  EXPECT_TRUE(store->open().ok());

  ExecBatchPtr batch = store->fetch({"id", "invalid_column"}, 0);
  EXPECT_EQ(batch, nullptr);
}

TEST_F(BufferPoolStoreTest, ParquetFetchSingleRowWithEmptyColumns) {
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  EXPECT_TRUE(store->open().ok());

  ExecBatchPtr batch = store->fetch({}, 0);
  EXPECT_EQ(batch, nullptr);
}

TEST_F(BufferPoolStoreTest, AllDataTypeFetchSingleRow) {
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  EXPECT_TRUE(store->open().ok());

  ExecBatchPtr batch = store->fetch({"id", "list_int32"}, 2);
  ASSERT_TRUE(batch != nullptr);
  EXPECT_EQ(batch->length, 1);
  EXPECT_EQ(batch->values.size(), 2);

  auto id_scalar = batch->values[0].scalar();
  ASSERT_TRUE(id_scalar != nullptr);
  auto id_value = std::dynamic_pointer_cast<arrow::Int32Scalar>(id_scalar);
  ASSERT_TRUE(id_value != nullptr);
  EXPECT_EQ(id_value->value, 3);

  auto list_scalar = batch->values[1].scalar();
  ASSERT_TRUE(list_scalar != nullptr);
  auto list_value = std::dynamic_pointer_cast<arrow::ListScalar>(list_scalar);
  ASSERT_TRUE(list_value != nullptr);
  EXPECT_EQ(list_value->value->length(), 128);

  auto list_array =
      std::dynamic_pointer_cast<arrow::Int32Array>(list_value->value);
  ASSERT_TRUE(list_array != nullptr);
  for (int i = 0; i < 10 && i < list_array->length(); ++i) {
    EXPECT_EQ(list_array->Value(i), 2 * 10 + i);
  }
}

TEST_F(BufferPoolStoreTest, AllDataType) {
  auto mmap_store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  ASSERT_TRUE(mmap_store->open().ok());

  std::vector<std::string> columns = {"id", "list_int32"};
  std::vector<int> indices = {0, 3, 6, 1, 0};

  TablePtr mmap_table = mmap_store->fetch(columns, indices);
  ASSERT_TRUE(mmap_table != nullptr);
  EXPECT_EQ(mmap_table->num_rows(), 5);
  EXPECT_EQ(mmap_table->num_columns(), 2);

  for (size_t j = 0; j < columns.size(); ++j) {
    auto column = mmap_table->column(j);
    for (int k = 0; k < column->num_chunks(); ++k) {
      auto array = column->chunk(k);
      if (array->type()->id() == arrow::Type::INT32) {
        auto int_array = std::static_pointer_cast<arrow::Int32Array>(array);
        for (int i = 0; i < array->length(); ++i) {
          int32_t value = int_array->Value(i);
          EXPECT_EQ(value, indices[i] + 1);
        }
      } else if (array->type()->id() == arrow::Type::LIST) {
        auto list_array = std::static_pointer_cast<arrow::ListArray>(array);
        for (int i = 0; i < array->length(); ++i) {
          auto list_value = list_array->value_slice(i);
          auto list_value_array =
              std::static_pointer_cast<arrow::Int32Array>(list_value);
          EXPECT_EQ(list_value_array->length(), 128);
          for (int m = 0; m < list_value_array->length(); ++m) {
            int32_t value = list_value_array->Value(m);
            EXPECT_EQ(value, indices[i] * 10 + m);
          }
        }
      }
    }
  }
}

TEST_F(BufferPoolStoreTest, DeleteDestructs) {
  BufferPoolForwardStore *store = new BufferPoolForwardStore(parquet_path);
  delete store;
}

TEST_F(BufferPoolStoreTest, PhysicSchema) {
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  ASSERT_NE(store, nullptr);
  EXPECT_TRUE(store->open().ok());
  EXPECT_NE(store->physic_schema(), nullptr);
}

class BufferPoolScanBudgetTest : public testing::Test {
 protected:
  static constexpr int64_t kRows = 22003;
  static constexpr int64_t kRowsPerGroup = 10000;
  static constexpr int64_t kBatchRows = 8192;

  void SetUp() override {
    ASSERT_TRUE(detail::MemoryBudgetTestPeer::reset_idle_allowances());
    auto &pool = ailego::PoolPayloadBudget::get_instance();
    previous_capacity_ = pool.capacity();
    ailego::BlockEvictionQueue::get_instance().batch_recycle(1024);
    ASSERT_EQ(0u, pool.used());
    ASSERT_EQ(0, pool.init(1024 * 1024));

    arrow::UInt64Builder ids;
    arrow::UInt64Builder values;
    for (int64_t row = 0; row < kRows; ++row) {
      ASSERT_TRUE(ids.Append(static_cast<uint64_t>(row)).ok());
      ASSERT_TRUE(values.Append(static_cast<uint64_t>(row * 3 + 7)).ok());
    }
    std::shared_ptr<arrow::Array> id_array;
    std::shared_ptr<arrow::Array> value_array;
    ASSERT_TRUE(ids.Finish(&id_array).ok());
    ASSERT_TRUE(values.Finish(&value_array).ok());
    schema_ = arrow::schema({arrow::field(GLOBAL_DOC_ID, arrow::uint64()),
                             arrow::field("value", arrow::uint64())});
    auto table = arrow::Table::Make(schema_, {id_array, value_array});
    auto output = arrow::io::FileOutputStream::Open(parquet_path_);
    ASSERT_TRUE(output.ok()) << output.status().ToString();
    ASSERT_TRUE(parquet::arrow::WriteTable(*table, arrow::default_memory_pool(),
                                           *output, kRowsPerGroup)
                    .ok());
    ASSERT_TRUE((*output)->Close().ok());
  }

  void TearDown() override {
    EXPECT_TRUE(detail::MemoryBudgetTestPeer::reset_idle_allowances());
    auto &pool = ailego::PoolPayloadBudget::get_instance();
    pool.release_external(reservation_);
    ailego::BlockEvictionQueue::get_instance().batch_recycle(1024);
    EXPECT_EQ(0u, pool.used());
    EXPECT_EQ(0, pool.init(previous_capacity_));
    std::error_code error;
    std::filesystem::remove(parquet_path_, error);
    EXPECT_FALSE(error) << error.message();
  }

  void reserve_remaining_capacity() {
    auto &pool = ailego::PoolPayloadBudget::get_instance();
    ASSERT_LE(pool.used(), pool.capacity());
    const size_t bytes = pool.capacity() - pool.used();
    ASSERT_TRUE(pool.try_charge_external(bytes));
    reservation_ = bytes;
    ASSERT_EQ(pool.capacity(), pool.used());
  }

  void check_scan(const RecordBatchReaderPtr &reader, int columns) {
    ASSERT_NE(nullptr, reader);
    int64_t next_row = 0;
    int batches = 0;
    std::shared_ptr<arrow::RecordBatch> batch;
    while (next_row < kRows) {
      const auto status = reader->ReadNext(&batch);
      ASSERT_TRUE(status.ok()) << status.ToString();
      ASSERT_NE(nullptr, batch);
      ASSERT_TRUE(batch->ValidateFull().ok());
      ASSERT_EQ(columns, batch->num_columns());
      ASSERT_GT(batch->num_rows(), 0);
      ASSERT_LE(batch->num_rows(), kBatchRows);
      ASSERT_LE(next_row + batch->num_rows(), kRows);
      auto ids = std::static_pointer_cast<arrow::UInt64Array>(batch->column(0));
      std::shared_ptr<arrow::UInt64Array> values;
      if (columns == 2) {
        values = std::static_pointer_cast<arrow::UInt64Array>(batch->column(1));
      }
      for (int64_t row = 0; row < batch->num_rows(); ++row) {
        ASSERT_EQ(static_cast<uint64_t>(next_row + row), ids->Value(row));
        if (values) {
          ASSERT_EQ(static_cast<uint64_t>((next_row + row) * 3 + 7),
                    values->Value(row));
        }
      }
      next_row += batch->num_rows();
      ++batches;
    }
    EXPECT_EQ(5, batches);  // Two split row groups and a 2003-row tail.
    const auto previous_batch = batch;
    for (int repeat = 0; repeat < 2; ++repeat) {
      batch = previous_batch;
      EXPECT_TRUE(reader->ReadNext(&batch).ok());
      EXPECT_EQ(nullptr, batch);
    }
  }

  std::string parquet_path_ = "bufferpool_scan_budget_test.parquet";
  std::shared_ptr<arrow::Schema> schema_;
  size_t previous_capacity_{0};
  size_t reservation_{0};
};

TEST_F(BufferPoolScanBudgetTest, FullPoolRejectsDecodeAndRecoversAfterRelease) {
  BufferPoolForwardStore store(parquet_path_);
  ASSERT_TRUE(store.open().ok());
  const ParquetBufferID key(parquet_path_, 0, 0);
  auto reader = store.scan({GLOBAL_DOC_ID, "value"});
  ASSERT_NE(nullptr, reader);
  ASSERT_NO_FATAL_FAILURE(reserve_remaining_capacity());
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  const size_t external_before = pool.external_used();
  auto denied = ParquetBufferPool::get_instance().acquire_buffer(key);
  ASSERT_EQ(nullptr, denied.data());
  auto checked = ParquetBufferPool::get_instance().acquire_buffer_checked(key);
  EXPECT_TRUE(checked.status().IsOutOfMemory()) << checked.status().ToString();
  auto fetch = store.fetch_checked({"value"}, {0, 1},
                                   detail::GetParquetMemoryPool().get());
  EXPECT_TRUE(fetch.status().IsOutOfMemory()) << fetch.status().ToString();
  EXPECT_THROW(store.fetch({"value"}, 0), std::bad_alloc);
  // Opening a lazy reader may now reuse the store's immutable key without
  // allocating. Admission must still reject its first actual decode.
  auto denied_scan = store.scan({GLOBAL_DOC_ID, "value"});
  if (denied_scan) {
    std::shared_ptr<arrow::RecordBatch> batch;
    EXPECT_TRUE(denied_scan->ReadNext(&batch).IsOutOfMemory());
    EXPECT_EQ(nullptr, batch);
  }
  {
    std::shared_ptr<arrow::RecordBatch> batch;
    EXPECT_TRUE(reader->ReadNext(&batch).IsOutOfMemory());
    EXPECT_EQ(nullptr, batch);
  }
  EXPECT_EQ(external_before, pool.external_used());
  pool.release_external(reservation_);
  reservation_ = 0;
  auto recovered = store.fetch_checked({"value"}, {0, 1},
                                       detail::GetParquetMemoryPool().get());
  ASSERT_TRUE(recovered.ok()) << recovered.status().ToString();
  EXPECT_EQ(2, (*recovered)->num_rows());
  ASSERT_NO_FATAL_FAILURE(check_scan(store.scan({GLOBAL_DOC_ID, "value"}), 2));
}

TEST_F(BufferPoolScanBudgetTest, DocIdScanDoesNotPopulateAvailableCache) {
  BufferPoolForwardStore store(parquet_path_);
  ASSERT_TRUE(store.open().ok());
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  const size_t external_before = pool.external_used();
  ASSERT_NO_FATAL_FAILURE(check_scan(store.scan({GLOBAL_DOC_ID}), 1));
  EXPECT_EQ(external_before, pool.external_used());
}

TEST_F(BufferPoolScanBudgetTest, ColdScanDoesNotPopulatePointReadCache) {
  // The second scan deliberately keeps a point-read column pinned while
  // decoding two columns. Include both that overlap and retained metadata.
  ASSERT_EQ(0, ailego::PoolPayloadBudget::get_instance().init(2 * 1024 * 1024));
  BufferPoolForwardStore store(parquet_path_);
  ASSERT_TRUE(store.open().ok());
  auto &cache = ParquetBufferPool::get_instance();
  const auto entries = ParquetCacheEntryCount();
  ASSERT_NO_FATAL_FAILURE(check_scan(store.scan({GLOBAL_DOC_ID, "value"}), 2));
  EXPECT_EQ(entries, ParquetCacheEntryCount());

  // Point reads still admit columns. A subsequent scan may borrow that first
  // column, but must release its partial pins and stream on the next miss.
  auto handle = cache.acquire_buffer(ParquetBufferID(parquet_path_, 0, 0));
  auto column = handle.data();
  ASSERT_NE(nullptr, column);
  EXPECT_EQ(entries + 1, ParquetCacheEntryCount());
  ASSERT_NO_FATAL_FAILURE(check_scan(store.scan({GLOBAL_DOC_ID, "value"}), 2));
  EXPECT_EQ(entries + 1, ParquetCacheEntryCount());
  EXPECT_EQ(
      0u,
      std::static_pointer_cast<arrow::UInt64Array>(column->chunk(0))->Value(0));
}

TEST_F(BufferPoolScanBudgetTest, PartialCacheHitDoesNotBypassDecodeBudget) {
  BufferPoolForwardStore store(parquet_path_);
  ASSERT_TRUE(store.open().ok());
  auto first_column = ParquetBufferPool::get_instance().acquire_buffer(
      ParquetBufferID(parquet_path_, /*column=*/0, /*row_group=*/0));
  auto pinned = first_column.data();
  ASSERT_NE(nullptr, pinned);
  ASSERT_EQ(kRowsPerGroup, pinned->length());
  const ParquetBufferID missing_key(parquet_path_, 1, 0);
  auto reader = store.scan({GLOBAL_DOC_ID, "value"});
  ASSERT_NE(nullptr, reader);
  ASSERT_NO_FATAL_FAILURE(reserve_remaining_capacity());
  auto missing_column =
      ParquetBufferPool::get_instance().acquire_buffer(missing_key);
  ASSERT_EQ(nullptr, missing_column.data());
  {
    std::shared_ptr<arrow::RecordBatch> batch;
    EXPECT_TRUE(reader->ReadNext(&batch).IsOutOfMemory());
    EXPECT_EQ(nullptr, batch);
  }
  EXPECT_EQ(
      0u,
      std::static_pointer_cast<arrow::UInt64Array>(pinned->chunk(0))->Value(0));
}

TEST_F(BufferPoolScanBudgetTest, StreamingReadFailureClearsOutput) {
  auto input = arrow::io::ReadableFile::Open(parquet_path_);
  ASSERT_TRUE(input.ok()) << input.status().ToString();
  auto opened = parquet::arrow::OpenFile(*input, arrow::default_memory_pool());
  ASSERT_TRUE(opened.ok()) << opened.status().ToString();
  auto parquet_reader = std::move(*opened);
  ParquetRecordBatchReader reader(parquet_reader, {GLOBAL_DOC_ID}, schema_,
                                  parquet_path_, /*with_cache=*/false,
                                  /*stream_uncached=*/true);
  ASSERT_TRUE((*input)->Close().ok());
  std::shared_ptr<arrow::RecordBatch> batch =
      arrow::RecordBatch::Make(arrow::schema({}), 0, arrow::ArrayVector{});
  EXPECT_FALSE(reader.ReadNext(&batch).ok());
  EXPECT_EQ(nullptr, batch);
}

TEST_F(BufferPoolScanBudgetTest, UncachedDefaultPreservesRowGroupBoundaries) {
  auto input = arrow::io::ReadableFile::Open(parquet_path_);
  ASSERT_TRUE(input.ok()) << input.status().ToString();
  auto opened = parquet::arrow::OpenFile(*input, arrow::default_memory_pool());
  ASSERT_TRUE(opened.ok()) << opened.status().ToString();
  auto parquet_reader = std::move(*opened);
  parquet_reader->set_batch_size(kBatchRows);
  ParquetRecordBatchReader reader(parquet_reader, {GLOBAL_DOC_ID}, schema_,
                                  parquet_path_, /*with_cache=*/false);
  std::shared_ptr<arrow::RecordBatch> batch;
  int64_t next_row = 0;
  for (int64_t expected_rows :
       {kRowsPerGroup, kRowsPerGroup, kRows - 2 * kRowsPerGroup}) {
    ASSERT_TRUE(reader.ReadNext(&batch).ok());
    ASSERT_NE(nullptr, batch);
    ASSERT_TRUE(batch->ValidateFull().ok());
    ASSERT_EQ(expected_rows, batch->num_rows());
    ASSERT_EQ(1, batch->num_columns());
    auto ids = std::static_pointer_cast<arrow::UInt64Array>(batch->column(0));
    for (int64_t row = 0; row < expected_rows; ++row) {
      ASSERT_EQ(static_cast<uint64_t>(next_row + row), ids->Value(row));
    }
    next_row += expected_rows;
  }
  EXPECT_EQ(kRows, next_row);
  const auto previous_batch = batch;
  for (int repeat = 0; repeat < 2; ++repeat) {
    batch = previous_batch;
    EXPECT_TRUE(reader.ReadNext(&batch).ok());
    EXPECT_EQ(nullptr, batch);
  }
  // Arrow's default prefetch can retain the file after reader destruction.
  // Close the owned input before Windows removes the fixture in TearDown.
  ASSERT_TRUE((*input)->Close().ok());
}

TEST_F(BufferPoolScanBudgetTest, DefaultCacheFailureKeepsStreamingAcrossCalls) {
  auto input = arrow::io::ReadableFile::Open(parquet_path_);
  ASSERT_TRUE(input.ok()) << input.status().ToString();
  auto file_reader = parquet::ParquetFileReader::Open(*input);
  parquet::ArrowReaderProperties properties;
  properties.set_pre_buffer(false);
  properties.set_batch_size(kBatchRows);
  std::unique_ptr<parquet::arrow::FileReader> parquet_reader;
  ASSERT_TRUE(parquet::arrow::FileReader::Make(arrow::default_memory_pool(),
                                               std::move(file_reader),
                                               properties, &parquet_reader)
                  .ok());
  auto reader = std::make_shared<ParquetRecordBatchReader>(
      parquet_reader, std::vector<std::string>{GLOBAL_DOC_ID, "value"}, schema_,
      parquet_path_);
  ASSERT_NO_FATAL_FAILURE(reserve_remaining_capacity());
  ASSERT_NO_FATAL_FAILURE(check_scan(reader, 2));
}

TEST_F(BufferPoolScanBudgetTest,
       BudgetedScanPreservesRepeatedReorderedColumns) {
  BufferPoolForwardStore store(parquet_path_);
  ASSERT_TRUE(store.open().ok());
  auto reader = store.scan({"value", GLOBAL_DOC_ID, "value"});
  ASSERT_NE(nullptr, reader);
  const auto expected_schema =
      arrow::schema({schema_->field(1), schema_->field(0), schema_->field(1)});
  ASSERT_TRUE(reader->schema()->Equals(expected_schema));
  int64_t next_row = 0;
  int batches = 0;
  std::shared_ptr<arrow::RecordBatch> batch;
  while (next_row < kRows) {
    const auto status = reader->ReadNext(&batch);
    ASSERT_TRUE(status.ok()) << status.ToString();
    ASSERT_NE(nullptr, batch);
    ASSERT_TRUE(batch->ValidateFull().ok());
    ASSERT_TRUE(batch->schema()->Equals(reader->schema()));
    ASSERT_EQ(3, batch->num_columns());
    ASSERT_GT(batch->num_rows(), 0);
    ASSERT_LE(batch->num_rows(), kBatchRows);
    ASSERT_LE(next_row + batch->num_rows(), kRows);
    auto values =
        std::static_pointer_cast<arrow::UInt64Array>(batch->column(0));
    auto ids = std::static_pointer_cast<arrow::UInt64Array>(batch->column(1));
    auto repeated =
        std::static_pointer_cast<arrow::UInt64Array>(batch->column(2));
    for (int64_t row = 0; row < batch->num_rows(); ++row) {
      ASSERT_EQ(static_cast<uint64_t>(next_row + row), ids->Value(row));
      ASSERT_EQ(static_cast<uint64_t>((next_row + row) * 3 + 7),
                values->Value(row));
      ASSERT_EQ(values->Value(row), repeated->Value(row));
    }
    next_row += batch->num_rows();
    ++batches;
  }
  EXPECT_EQ(5, batches);
  EXPECT_TRUE(reader->ReadNext(&batch).ok());
  EXPECT_EQ(nullptr, batch);
}

class BufferPoolScalarFetchTest : public testing::Test {
 protected:
  static constexpr int64_t kRowsPerGroup = 2048;
  static constexpr int64_t kRows = 2 * kRowsPerGroup;

  static std::string text_for_row(int64_t row) {
    if (row % kRowsPerGroup == 7) return "";
    if (row % kRowsPerGroup == 8) return std::string("embedded\0nul", 12);
    return "row-" + std::to_string(row) + std::string(64, 'x');
  }

  void SetUp() override {
    ASSERT_TRUE(detail::MemoryBudgetTestPeer::reset_idle_allowances());
    auto &pool = ailego::PoolPayloadBudget::get_instance();
    previous_capacity_ = pool.capacity();
    ailego::BlockEvictionQueue::get_instance().batch_recycle(1024);
    ASSERT_EQ(0u, pool.used());
    ASSERT_EQ(0, pool.init(8 * 1024UL * 1024UL));

    arrow::Int64Builder ids;
    arrow::Int64Builder nullable_values;
    arrow::StringBuilder names;
    for (int64_t row = 0; row < kRows; ++row) {
      ASSERT_TRUE(ids.Append(row).ok());
      ASSERT_TRUE((row % kRowsPerGroup == 4
                       ? nullable_values.AppendNull()
                       : nullable_values.Append(row * 3 + 7))
                      .ok());
      ASSERT_TRUE((row % kRowsPerGroup == 3 ? names.AppendNull()
                                            : names.Append(text_for_row(row)))
                      .ok());
    }
    std::shared_ptr<arrow::Array> id_array, value_array, name_array;
    ASSERT_TRUE(ids.Finish(&id_array).ok());
    ASSERT_TRUE(nullable_values.Finish(&value_array).ok());
    ASSERT_TRUE(names.Finish(&name_array).ok());
    auto schema = arrow::schema({arrow::field("id", arrow::int64()),
                                 arrow::field("value", arrow::int64()),
                                 arrow::field("name", arrow::utf8())});
    auto table =
        arrow::Table::Make(schema, {id_array, value_array, name_array});
    auto output = arrow::io::FileOutputStream::Open(parquet_path_);
    ASSERT_TRUE(output.ok()) << output.status().ToString();
    ASSERT_TRUE(parquet::arrow::WriteTable(*table, arrow::default_memory_pool(),
                                           *output, kRowsPerGroup)
                    .ok());
    ASSERT_TRUE((*output)->Close().ok());
  }

  void TearDown() override {
    EXPECT_TRUE(detail::MemoryBudgetTestPeer::reset_idle_allowances());
    auto &pool = ailego::PoolPayloadBudget::get_instance();
    ailego::BlockEvictionQueue::get_instance().batch_recycle(1024);
    EXPECT_EQ(0u, pool.used());
    EXPECT_EQ(0, pool.init(previous_capacity_));
    std::error_code error;
    std::filesystem::remove(parquet_path_, error);
    EXPECT_FALSE(error) << error.message();
  }

  void warm_and_rechunk(int column, int row_group) {
    ParquetBufferID key(parquet_path_, column, row_group);
    auto &cache = ParquetBufferPoolTestPeer::cache_for(key);
    auto data = cache.acquire(key);
    ASSERT_NE(nullptr, data);
    ParquetBufferContextHandle pin(key, data);
    ASSERT_EQ(kRowsPerGroup, data->length());
    // Parquet may decode a small column as one chunk. Repartition its cached
    // view without copying or changing its buffers/accounting so every test
    // deterministically exercises chunk boundaries, including empty chunks.
    arrow::ArrayVector chunks;
    int64_t start = 0;
    for (int64_t end : std::array<int64_t, 5>{0, 1, 3, 7, kRowsPerGroup}) {
      auto slice = data->Slice(start, end - start);
      chunks.insert(chunks.end(), slice->chunks().begin(),
                    slice->chunks().end());
      start = end;
    }
    *data = arrow::ChunkedArray(std::move(chunks), data->type());
    ASSERT_GT(data->num_chunks(), 1);
  }

  void check_row(const ExecBatchPtr &batch, int64_t row) {
    ASSERT_NE(nullptr, batch);
    ASSERT_EQ(1, batch->length);
    ASSERT_EQ(3u, batch->values.size());
    auto id = std::dynamic_pointer_cast<arrow::Int64Scalar>(
        batch->values[0].scalar());
    auto value = std::dynamic_pointer_cast<arrow::Int64Scalar>(
        batch->values[1].scalar());
    auto name = std::dynamic_pointer_cast<arrow::StringScalar>(
        batch->values[2].scalar());
    ASSERT_NE(nullptr, id);
    ASSERT_NE(nullptr, value);
    ASSERT_NE(nullptr, name);
    EXPECT_TRUE(id->is_valid);
    EXPECT_EQ(row, id->value);
    EXPECT_EQ(row % kRowsPerGroup != 4, value->is_valid);
    if (value->is_valid) EXPECT_EQ(row * 3 + 7, value->value);
    EXPECT_EQ(row % kRowsPerGroup != 3, name->is_valid);
    if (name->is_valid) {
      ASSERT_NE(nullptr, name->value);
      EXPECT_EQ(text_for_row(row), name->value->ToString());
    }
  }

  std::string parquet_path_{"bufferpool_scalar_fetch_test.parquet"};
  size_t previous_capacity_{0};
};

TEST_F(BufferPoolScalarFetchTest, ResolvesChunkAndRowGroupBoundaries) {
  BufferPoolForwardStore store(parquet_path_);
  ASSERT_TRUE(store.open().ok());
  for (int group = 0; group < 2; ++group) {
    for (int column = 0; column < 3; ++column) {
      ASSERT_NO_FATAL_FAILURE(warm_and_rechunk(column, group));
    }
    for (int local : {0, 1, 2, 3, 4, 6, 7, 8, 2047}) {
      const int row = static_cast<int>(group * kRowsPerGroup + local);
      ASSERT_NO_FATAL_FAILURE(
          check_row(store.fetch({"id", "value", "name"}, row), row));
    }
  }
}

TEST_F(BufferPoolScalarFetchTest, WarmFetchDoesNotAllocateAWholeColumn) {
  BufferPoolForwardStore store(parquet_path_);
  ASSERT_TRUE(store.open().ok());
  for (int column = 0; column < 3; ++column) {
    ASSERT_NO_FATAL_FAILURE(warm_and_rechunk(column, /*row_group=*/1));
  }
  auto *arrow_pool = arrow::default_memory_pool();
  const int64_t allocated_before = arrow_pool->total_bytes_allocated();
  auto batch =
      store.fetch({"id", "value", "name"}, static_cast<int>(kRowsPerGroup + 8));
  const int64_t allocated =
      arrow_pool->total_bytes_allocated() - allocated_before;
  ASSERT_NO_FATAL_FAILURE(check_row(batch, kRowsPerGroup + 8));
  // This cumulative counter detects transient concatenation allocations even
  // after those buffers are freed. A numeric column alone is over 16 KiB.
  EXPECT_LT(allocated, 1024);
}

TEST_F(BufferPoolScalarFetchTest, ReturnedScalarsSurviveCacheEviction) {
  BufferPoolForwardStore store(parquet_path_);
  ASSERT_TRUE(store.open().ok());
  std::vector<ExecBatchPtr> retained;
  for (int row : {3, 4, 7, 8, 2049}) {
    auto batch = store.fetch({"id", "value", "name"}, row);
    ASSERT_NO_FATAL_FAILURE(check_row(batch, row));
    retained.push_back(std::move(batch));
  }
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  ASSERT_GT(pool.external_used(), 0u);
  const auto cached_bytes = pool.external_used();
  EXPECT_GT(ailego::BlockEvictionQueue::get_instance().batch_recycle(1024), 0u);
  // Selected string bytes are now owned by the managed Arrow pool. Their
  // charge must survive with the result, without retaining whole columns.
  EXPECT_GT(pool.external_used(), 0u);
  EXPECT_LT(pool.external_used(), cached_bytes);
  size_t index = 0;
  for (int row : {3, 4, 7, 8, 2049}) {
    ASSERT_NO_FATAL_FAILURE(check_row(retained[index++], row));
  }
  retained.clear();
  ailego::BlockEvictionQueue::get_instance().batch_recycle(1024);
  EXPECT_EQ(0u, pool.external_used());
}

TEST_F(BufferPoolScalarFetchTest, InvalidCachedColumnReturnsNull) {
  BufferPoolForwardStore store(parquet_path_);
  ASSERT_TRUE(store.open().ok());
  ParquetBufferID key(parquet_path_, /*column=*/0, /*row_group=*/0);
  auto data = ParquetBufferPoolTestPeer::cache_for(key).acquire(key);
  ASSERT_NE(nullptr, data);
  ParquetBufferContextHandle pin(key, data);
  auto original = data->Slice(0, data->length());
  *data = arrow::ChunkedArray(arrow::ArrayVector{}, data->type());
  EXPECT_EQ(nullptr, store.fetch({"id"}, 3));
  // The row is valid in file metadata but outside this damaged cached view.
  *data =
      arrow::ChunkedArray(original->Slice(0, 1)->chunks(), original->type());
  EXPECT_EQ(nullptr, store.fetch({"id"}, 3));
}

class BufferPoolLargeDecodeTest : public BufferPoolScalarFetchTest {
 protected:
  void SetUp() override {
    ASSERT_NO_FATAL_FAILURE(BufferPoolScalarFetchTest::SetUp());
    arrow::StringBuilder names;
    for (int row = 0; row < kLargeRows; ++row) {
      ASSERT_TRUE(names.Append(large_text(row)).ok());
    }
    std::shared_ptr<arrow::Array> values;
    ASSERT_TRUE(names.Finish(&values).ok());
    auto schema = arrow::schema({arrow::field("name", arrow::utf8())});
    auto table = arrow::Table::Make(schema, {values});
    auto output = arrow::io::FileOutputStream::Open(parquet_path_);
    ASSERT_TRUE(output.ok()) << output.status().ToString();
    ASSERT_TRUE(parquet::arrow::WriteTable(*table, arrow::default_memory_pool(),
                                           *output, kLargeRows)
                    .ok());
    ASSERT_TRUE((*output)->Close().ok());
  }

  static std::string large_text(int row) {
    return std::to_string(row) + std::string(512, 'x');
  }

  static constexpr int kLargeRows = 8192;
};

TEST_F(BufferPoolLargeDecodeTest, DecoderAndOutputUseDedicatedMemoryPool) {
  auto pool = detail::GetParquetMemoryPool();
  const auto mapped_before = pool->mapped_bytes();
  const auto allocations_before = pool->num_mapped_allocations();
  const auto total_before = pool->total_bytes_allocated();
  auto *default_pool = arrow::default_memory_pool();
  const auto default_before = default_pool->total_bytes_allocated();
  detail::ParquetBufferLoader loader;
  detail::ParquetBufferPayload payload;
  size_t bytes = 0;
  ASSERT_TRUE(
      loader.load(ParquetBufferID(parquet_path_, 0, 0), payload, bytes));
  ASSERT_NE(nullptr, payload.arrow);
  EXPECT_EQ(kLargeRows, payload.arrow->length());
  EXPECT_GT(bytes, detail::ParquetMemoryPool::kMappedAllocationThreshold);
  EXPECT_GT(pool->mapped_bytes(), mapped_before);
  EXPECT_GT(pool->num_mapped_allocations(), allocations_before);
  // Input, dictionaries, decoder workspaces and output must all bypass the
  // ordinary Arrow heap, including small buffers (zero-size sentinels aside).
  EXPECT_GT(pool->total_bytes_allocated(), total_before);
  EXPECT_EQ(default_before, default_pool->total_bytes_allocated());
  EXPECT_GE(ailego::PoolPayloadBudget::get_instance().external_used(),
            pool->charged_bytes() + payload.metadata_memory.bytes());
  loader.clear(payload);
  EXPECT_EQ(mapped_before, pool->mapped_bytes());
}

TEST_F(BufferPoolLargeDecodeTest, RepeatedEvictionReleasesLargeMappings) {
  auto pool = detail::GetParquetMemoryPool();
  const auto mapped_before = pool->mapped_bytes();
  BufferPoolForwardStore store(parquet_path_);
  ASSERT_TRUE(store.open().ok());
  const auto owned_before = pool->bytes_allocated();
  std::vector<ExecBatchPtr> retained;
  for (int row = 0; row < 12; ++row) {
    auto batch = store.fetch({"name"}, row);
    ASSERT_NE(nullptr, batch);
    retained.push_back(std::move(batch));
    EXPECT_GT(pool->mapped_bytes(), mapped_before);
    ailego::BlockEvictionQueue::get_instance().batch_recycle(1024);
    EXPECT_EQ(mapped_before, pool->mapped_bytes());
    // Keep the selected values budgeted, while rejecting retention of an
    // entire 8192-row decoded column (over 4 MiB).
    EXPECT_LE(pool->bytes_allocated() - owned_before,
              static_cast<int64_t>(retained.size() * 1024));
  }
  for (size_t row = 0; row < retained.size(); ++row) {
    auto scalar = std::dynamic_pointer_cast<arrow::StringScalar>(
        retained[row]->values[0].scalar());
    ASSERT_NE(nullptr, scalar);
    EXPECT_EQ(large_text(static_cast<int>(row)), scalar->view());
  }
  retained.clear();
  ailego::BlockEvictionQueue::get_instance().batch_recycle(1024);
  EXPECT_EQ(owned_before, pool->bytes_allocated());
  EXPECT_EQ(0u, ailego::PoolPayloadBudget::get_instance().external_used());
}

TEST_F(BufferPoolLargeDecodeTest, EscapedViewKeepsLargeMappingAlive) {
  auto pool = detail::GetParquetMemoryPool();
  const auto mapped_before = pool->mapped_bytes();
  std::shared_ptr<arrow::ChunkedArray> escaped;
  {
    auto handle = ParquetBufferPool::get_instance().acquire_buffer(
        ParquetBufferID(parquet_path_, 0, 0));
    escaped = handle.data();
  }
  ASSERT_NE(nullptr, escaped);
  EXPECT_GT(pool->mapped_bytes(), mapped_before);
  ailego::BlockEvictionQueue::get_instance().batch_recycle(1024);
  EXPECT_GT(pool->mapped_bytes(), mapped_before);
  {
    auto scalar = escaped->GetScalar(kLargeRows - 1);
    ASSERT_TRUE(scalar.ok()) << scalar.status().ToString();
    auto value = std::dynamic_pointer_cast<arrow::StringScalar>(*scalar);
    ASSERT_NE(nullptr, value);
    EXPECT_EQ(large_text(kLargeRows - 1), value->view());
  }
  escaped.reset();
  ailego::BlockEvictionQueue::get_instance().batch_recycle(1024);
  EXPECT_EQ(mapped_before, pool->mapped_bytes());
}

TEST_F(BufferPoolLargeDecodeTest,
       OversizedCachedColumnStillStreamsWithinBudget) {
  // Use small plain pages so the minimum decoder window fits; the full
  // decoded column deliberately does not. Budget enforcement must not turn
  // optional cache admission into a requirement to hold the entire column.
  arrow::StringBuilder names;
  for (int row = 0; row < kLargeRows; ++row) {
    ASSERT_TRUE(names.Append(large_text(row)).ok());
  }
  std::shared_ptr<arrow::Array> values;
  ASSERT_TRUE(names.Finish(&values).ok());
  auto table = arrow::Table::Make(
      arrow::schema({arrow::field("name", arrow::utf8())}), {values});
  auto output = arrow::io::FileOutputStream::Open(parquet_path_);
  ASSERT_TRUE(output.ok());
  parquet::WriterProperties::Builder properties;
  properties.disable_dictionary()->data_pagesize(64 * 1024)->write_batch_size(
      64);
  ASSERT_TRUE(parquet::arrow::WriteTable(*table, arrow::default_memory_pool(),
                                         *output, kLargeRows,
                                         properties.build())
                  .ok());
  ASSERT_TRUE((*output)->Close().ok());

  auto &budget = ailego::PoolPayloadBudget::get_instance();
  ASSERT_EQ(0, budget.init(2 * 1024 * 1024));
  auto allocator = detail::GetParquetMemoryPool();
  const auto before = allocator->charged_bytes();
  const auto metadata_before =
      detail::ParquetMetadataBudget::instance().reserved();
  const auto entries = ParquetCacheEntryCount();
  for (int attempt = 0; attempt < 3; ++attempt) {
    auto denied = ParquetBufferPool::get_instance().acquire_buffer(
        ParquetBufferID(parquet_path_, 0, 0));
    EXPECT_EQ(nullptr, denied.data());
    EXPECT_EQ(before, allocator->charged_bytes());
    EXPECT_EQ(entries, ParquetCacheEntryCount());
    // Failed decoding releases its payload. The parsed footer's high-water
    // allowance intentionally remains charged and is reused by the next try.
    EXPECT_EQ(
        detail::ParquetMetadataBudget::instance().reserved() - metadata_before,
        budget.used());
  }

  BufferPoolForwardStore store(parquet_path_);
  ASSERT_TRUE(store.open().ok());
  auto reader = store.scan({"name"});
  ASSERT_NE(nullptr, reader);
  std::shared_ptr<arrow::RecordBatch> batch;
  int row = 0;
  while (true) {
    auto status = reader->ReadNext(&batch);
    ASSERT_TRUE(status.ok()) << status.ToString();
    if (!batch) break;
    ASSERT_TRUE(batch->ValidateFull().ok());
    EXPECT_LE(budget.used(), budget.capacity());
    auto strings =
        std::static_pointer_cast<arrow::StringArray>(batch->column(0));
    for (int64_t i = 0; i < batch->num_rows(); ++i, ++row) {
      ASSERT_EQ(large_text(row), strings->GetString(i));
    }
  }
  EXPECT_EQ(kLargeRows, row);
  reader.reset();
  EXPECT_EQ(0u, budget.external_used());
  EXPECT_EQ(before, allocator->charged_bytes());
}

TEST_F(BufferPoolScalarFetchTest,
       CachedScanSlicesInsteadOfCopyingAndPinsBuffers) {
  ASSERT_NO_FATAL_FAILURE(warm_and_rechunk(0, 0));
  ASSERT_NO_FATAL_FAILURE(warm_and_rechunk(1, 0));
  auto allocator = detail::GetParquetMemoryPool();
  BufferPoolForwardStore store(parquet_path_);
  ASSERT_TRUE(store.open().ok());
  auto reader = store.scan({"id", "value", "id"});
  ASSERT_NE(nullptr, reader);
  auto handle = ParquetBufferPool::get_instance().acquire_buffer(
      ParquetBufferID(parquet_path_, 0, 0));
  auto column = handle.data();
  ASSERT_NE(nullptr, column);
  const auto allocations = allocator->num_allocations();
  auto *default_pool = arrow::default_memory_pool();
  const auto default_bytes = default_pool->total_bytes_allocated();
  std::shared_ptr<arrow::RecordBatch> batch;
  ASSERT_TRUE(reader->ReadNext(&batch).ok());
  ASSERT_NE(nullptr, batch);
  ASSERT_GT(batch->num_rows(), 0);
  EXPECT_EQ(column->chunk(1)->data()->buffers[1]->data(),
            batch->column(0)->data()->buffers[1]->data());
  EXPECT_EQ(batch->column(0)->data()->buffers[1]->data(),
            batch->column(2)->data()->buffers[1]->data());
  EXPECT_EQ(allocations, allocator->num_allocations());
  EXPECT_EQ(default_bytes, default_pool->total_bytes_allocated());
  reader.reset();
  ailego::BlockEvictionQueue::get_instance().batch_recycle(1024);
  EXPECT_EQ(
      0,
      std::static_pointer_cast<arrow::Int64Array>(batch->column(0))->Value(0));
}

TEST_F(BufferPoolStoreTest,
       ChunkedDecodePreservesNestedFieldsAndFollowingColumns) {
  arrow::Int64Builder numbers;
  arrow::StringBuilder labels;
  ASSERT_TRUE(numbers.AppendValues({10, 20}).ok());
  ASSERT_TRUE(labels.AppendValues({"first", "second"}).ok());
  std::shared_ptr<arrow::Array> number_array, label_array;
  ASSERT_TRUE(numbers.Finish(&number_array).ok());
  ASSERT_TRUE(labels.Finish(&label_array).ok());
  auto nested = arrow::StructArray::Make(
      {number_array, label_array}, std::vector<std::string>{"number", "label"});
  ASSERT_TRUE(nested.ok());
  auto schema = arrow::schema({arrow::field("nested", (*nested)->type()),
                               arrow::field("tail", arrow::int64())});
  auto table = arrow::Table::Make(schema, {*nested, number_array});
  auto output = arrow::io::FileOutputStream::Open(parquet_path);
  ASSERT_TRUE(output.ok());
  ASSERT_TRUE(parquet::arrow::WriteTable(*table, arrow::default_memory_pool(),
                                         *output, 2)
                  .ok());
  ASSERT_TRUE((*output)->Close().ok());
  BufferPoolForwardStore store(parquet_path);
  ASSERT_TRUE(store.open().ok());
  auto row = store.fetch({"nested", "tail"}, 1);
  ASSERT_NE(nullptr, row);
  auto scalar = (*nested)->GetScalar(1);
  ASSERT_TRUE(scalar.ok());
  EXPECT_TRUE(row->values[0].scalar()->Equals(**scalar));
  EXPECT_EQ(
      20, std::static_pointer_cast<arrow::Int64Scalar>(row->values[1].scalar())
              ->value);
  auto reader = store.scan({"tail", "nested", "tail"});
  ASSERT_NE(nullptr, reader);
  std::shared_ptr<arrow::RecordBatch> batch;
  ASSERT_TRUE(reader->ReadNext(&batch).ok());
  ASSERT_NE(nullptr, batch);
  EXPECT_TRUE(batch->column(0)->Equals(number_array));
  EXPECT_TRUE(batch->column(1)->Equals(*nested));
  EXPECT_TRUE(batch->column(2)->Equals(number_array));
}


TEST_F(BufferPoolStoreTest, BatchFetchNullableListsAcrossRowGroups) {
  arrow::Int32Builder ids;
  arrow::StringBuilder names;
  arrow::BinaryBuilder bytes;
  auto values = std::make_shared<arrow::Int32Builder>();
  arrow::ListBuilder items(arrow::default_memory_pool(), values);
  for (int i = 0; i < 12; ++i) {
    ASSERT_TRUE(ids.Append(i).ok());
    if (i % 3 == 0) {
      ASSERT_TRUE(names.AppendNull().ok());
      ASSERT_TRUE(bytes.AppendNull().ok());
      ASSERT_TRUE(items.AppendNull().ok());
    } else {
      ASSERT_TRUE(names.Append(std::string(i, 'a' + i)).ok());
      ASSERT_TRUE(bytes.Append(std::string("\0\xff", 2)).ok());
      ASSERT_TRUE(items.Append().ok());
      if (i % 3 == 1) {
        ASSERT_TRUE(values->Append(i).ok());
        ASSERT_TRUE(values->AppendNull().ok());
      }
    }
  }
  std::shared_ptr<arrow::Array> id_array, name_array, byte_array, item_array;
  ASSERT_TRUE(ids.Finish(&id_array).ok());
  ASSERT_TRUE(names.Finish(&name_array).ok());
  ASSERT_TRUE(bytes.Finish(&byte_array).ok());
  ASSERT_TRUE(items.Finish(&item_array).ok());
  auto source = arrow::Table::Make(
      arrow::schema({arrow::field("id", arrow::int32()),
                     arrow::field("name", arrow::utf8()),
                     arrow::field("bytes", arrow::binary()),
                     arrow::field("items", arrow::list(arrow::int32()))}),
      {id_array, name_array, byte_array, item_array});
  auto output = arrow::io::FileOutputStream::Open(parquet_path);
  ASSERT_TRUE(output.ok());
  ASSERT_TRUE(parquet::arrow::WriteTable(*source, arrow::default_memory_pool(),
                                         *output, 4)
                  .ok());
  ASSERT_TRUE((*output)->Close().ok());
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  ASSERT_TRUE(store->open().ok());
  const std::vector<std::string> columns{"items", "bytes", LOCAL_ROW_ID, "id",
                                         "name"};
  const std::vector<std::string> scalar_columns{"items", "bytes", "id", "name"};
  for (int row = 0; row < 12; ++row) {
    auto result = store->fetch(scalar_columns, row);
    ASSERT_NE(result, nullptr);
    ASSERT_EQ(result->values.size(), scalar_columns.size());
    for (size_t col = 0; col < scalar_columns.size(); ++col) {
      auto expected =
          source->GetColumnByName(scalar_columns[col])->GetScalar(row);
      ASSERT_TRUE(expected.ok());
      EXPECT_TRUE(result->values[col].scalar()->Equals(**expected));
    }
  }
  for (const std::vector<int> rows :
       {std::vector<int>{9, 0, 6, 3, 9, 4, 11, 1, 8},
        std::vector<int>{1, 3, 1, 0},
        std::vector<int>{0, 1, 3, 4, 4, 7, 8, 10, 11},
        std::vector<int>{11, 10, 8, 7, 4, 4, 3, 1, 0}}) {
    auto result = store->fetch(columns, rows);
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->ValidateFull().ok());
    ASSERT_EQ(result->num_rows(), rows.size());
    for (size_t col = 0; col < columns.size(); ++col) {
      for (size_t row = 0; row < rows.size(); ++row) {
        auto actual = result->column(col)->GetScalar(row);
        ASSERT_TRUE(actual.ok());
        if (columns[col] == LOCAL_ROW_ID) {
          EXPECT_TRUE((*actual)->Equals(arrow::UInt64Scalar(rows[row])));
        } else {
          auto expected =
              source->GetColumnByName(columns[col])->GetScalar(rows[row]);
          ASSERT_TRUE(expected.ok());
          EXPECT_TRUE((*actual)->Equals(**expected));
        }
      }
    }
  }
}

TEST_F(BufferPoolStoreTest, SelectedRowsPlanHandlesChangingColumnLayouts) {
  arrow::Int64Builder values;
  arrow::Int64Builder shifted_values;
  for (int64_t row = 0; row < 8; ++row) {
    ASSERT_TRUE(values.Append(row).ok());
    ASSERT_TRUE(shifted_values.Append(100 + row).ok());
  }
  auto array = values.Finish().ValueOrDie();
  auto shifted = shifted_values.Finish().ValueOrDie();
  const detail::SelectedRowsPlan::Rows rows{{0, 7}, {1, 0}, {2, 3},
                                            {3, 4}, {4, 7}, {5, 1}};
  detail::SelectedRowsPlan plan(rows);
  auto check = [&](const arrow::ArrayVector &chunks, int64_t base) {
    arrow::ChunkedArray column(chunks);
    ASSERT_TRUE(plan.prepare(column).ok());
    std::vector<int64_t> actual;
    ASSERT_TRUE(plan.visit<arrow::Int64Array>(
                        column,
                        [&](const arrow::Int64Array &chunk, int64_t row, int) {
                          actual.push_back(chunk.Value(row));
                          return arrow::Status::OK();
                        })
                    .ok());
    ASSERT_EQ(actual.size(), rows.size());
    for (size_t i = 0; i < rows.size(); ++i) {
      EXPECT_EQ(actual[i], base + rows[i].second);
    }
  };
  check({array->Slice(0, 3), array->Slice(3)}, 0);
  // Equal lengths reuse addressing but must bind the new column's data.
  check({shifted->Slice(0, 3), shifted->Slice(3)}, 100);
  // Equal total length and chunk count do not imply equal boundaries.
  check({shifted->Slice(0, 5), shifted->Slice(5)}, 100);
  check({array}, 0);
  check({shifted->Slice(0, 0), shifted->Slice(0, 2), shifted->Slice(2, 0),
         shifted->Slice(2), shifted->Slice(8, 0)},
        100);
  check({array->Slice(0, 2), array->Slice(2, 1), array->Slice(3)}, 0);
}

TEST_F(BufferPoolStoreTest, SelectedRowsPlanRechecksBoundsAfterLayoutChanges) {
  arrow::Int64Builder builder;
  ASSERT_TRUE(builder.AppendValues({0, 1, 2, 3, 4, 5}).ok());
  auto array = builder.Finish().ValueOrDie();
  const detail::SelectedRowsPlan::Rows rows{{0, 5}, {1, 0}, {2, 5}};
  detail::SelectedRowsPlan plan(rows);
  arrow::ChunkedArray full(array);
  arrow::ChunkedArray short_column(array->Slice(0, 5));
  ASSERT_TRUE(plan.prepare(full).ok());
  EXPECT_TRUE(plan.prepare(short_column).IsIndexError());
  ASSERT_TRUE(plan.prepare(full).ok());
  std::vector<int64_t> actual;
  ASSERT_TRUE(plan.visit<arrow::Int64Array>(
                      full,
                      [&](const arrow::Int64Array &chunk, int64_t row, int) {
                        actual.push_back(chunk.Value(row));
                        return arrow::Status::OK();
                      })
                  .ok());
  EXPECT_EQ(actual, (std::vector<int64_t>{5, 0, 5}));
  const detail::SelectedRowsPlan::Rows huge{{0, uint64_t(-1)}};
  detail::SelectedRowsPlan invalid(huge);
  EXPECT_TRUE(invalid.prepare(full).IsIndexError());
}

TEST_F(BufferPoolStoreTest, SelectedRowsPlanHandlesEmptySelectionsAndColumns) {
  const detail::SelectedRowsPlan::Rows rows;
  detail::SelectedRowsPlan plan(rows);
  arrow::ChunkedArray column(arrow::ArrayVector{}, arrow::int64());
  ASSERT_TRUE(plan.prepare(column).ok());
  int visits = 0;
  ASSERT_TRUE(plan.visit<arrow::Int64Array>(
                      column,
                      [&](const arrow::Int64Array &, int64_t, int) {
                        ++visits;
                        return arrow::Status::OK();
                      })
                  .ok());
  EXPECT_EQ(visits, 0);
  const detail::SelectedRowsPlan::Rows nonempty{{0, 0}};
  detail::SelectedRowsPlan invalid(nonempty);
  EXPECT_TRUE(invalid.prepare(column).IsIndexError());
}

TEST_F(BufferPoolStoreTest,
       TypedBatchAcrossLargeRowGroupsWithMixedNullColumns) {
  arrow::BooleanBuilder flags;
  arrow::Int32Builder i32;
  arrow::Int64Builder i64;
  arrow::UInt32Builder u32;
  arrow::UInt64Builder u64;
  arrow::FloatBuilder floats;
  arrow::DoubleBuilder doubles;
  arrow::StringBuilder strings;
  arrow::BinaryBuilder binary;
  arrow::StringBuilder nullable;
  arrow::StringBuilder all_null;
  constexpr int count = 130001;
  for (int row = 0; row < count; ++row) {
    ASSERT_TRUE(flags.Append(row % 2 != 0).ok());
    ASSERT_TRUE(i32.Append(-row).ok());
    ASSERT_TRUE(i64.Append(-(int64_t{1} << 40) - row).ok());
    ASSERT_TRUE(u32.Append(4000000000u + row).ok());
    ASSERT_TRUE(u64.Append((uint64_t{1} << 63) + row).ok());
    ASSERT_TRUE(floats.Append(row + 0.5f).ok());
    ASSERT_TRUE(doubles.Append(row + 0.125).ok());
    const auto text = std::to_string(row);
    ASSERT_TRUE(strings.Append(text).ok());
    ASSERT_TRUE(binary.Append(std::string("\0\xff", 2) + text).ok());
    if (row % 3 == 0) {
      ASSERT_TRUE(nullable.AppendNull().ok());
    } else {
      ASSERT_TRUE(nullable.Append(text).ok());
    }
    ASSERT_TRUE(all_null.AppendNull().ok());
  }
  arrow::ArrayVector arrays;
  arrow::FieldVector fields;
  std::vector<arrow::ArrayBuilder *> builders{
      &flags,   &i32,     &i64,    &u32,      &u64,     &floats,
      &doubles, &strings, &binary, &nullable, &all_null};
  for (size_t col = 0; col < builders.size(); ++col) {
    std::shared_ptr<arrow::Array> array;
    ASSERT_TRUE(builders[col]->Finish(&array).ok());
    if (col < 9) ASSERT_EQ(array->null_count(), 0);
    fields.push_back(arrow::field("n" + std::to_string(col), array->type()));
    arrays.push_back(std::move(array));
  }
  auto source = arrow::Table::Make(arrow::schema(fields), arrays);
  auto out = arrow::io::FileOutputStream::Open(parquet_path);
  ASSERT_TRUE(out.ok());
  ASSERT_TRUE(parquet::arrow::WriteTable(*source, arrow::default_memory_pool(),
                                         *out, 100000)
                  .ok());
  ASSERT_TRUE((*out)->Close().ok());
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  ASSERT_TRUE(store->open().ok());
  const std::vector<std::string> columns{"n8", "n3",  LOCAL_ROW_ID, "n9", "n0",
                                         "n7", "n10", "n1",         "n2", "n4",
                                         "n5", "n6",  "n7"};
  const std::vector<int> rows{99999,  0, 65536, 65535, 100000,
                              130000, 0, 66666, 99999, 100001};
  auto result = store->fetch(columns, rows);
  ASSERT_NE(result, nullptr);
  ASSERT_TRUE(result->ValidateFull().ok());
  ASSERT_EQ(result->num_rows(), rows.size());
  for (size_t col = 0; col < columns.size(); ++col) {
    for (size_t row = 0; row < rows.size(); ++row) {
      auto actual = result->column(col)->GetScalar(row);
      ASSERT_TRUE(actual.ok());
      if (columns[col] == LOCAL_ROW_ID) {
        EXPECT_TRUE((*actual)->Equals(arrow::UInt64Scalar(rows[row])));
      } else {
        auto expected =
            source->GetColumnByName(columns[col])->GetScalar(rows[row]);
        ASSERT_TRUE(expected.ok());
        EXPECT_TRUE((*actual)->Equals(**expected));
      }
    }
  }
  EXPECT_EQ(store->fetch(columns, std::vector<int>{0, count}), nullptr);
  EXPECT_EQ(store->fetch(columns, std::vector<int>{-1, 0}), nullptr);
}

TEST_F(BufferPoolStoreTest, SingleRowProjectionPreservesOrderAndDuplicates) {
  auto file = arrow::io::ReadableFile::Open(parquet_path);
  ASSERT_TRUE(file.ok());
  auto reader = parquet::arrow::OpenFile(*file, arrow::default_memory_pool());
  ASSERT_TRUE(reader.ok());
  std::shared_ptr<arrow::Table> source;
  ASSERT_TRUE((*reader)->ReadTable(&source).ok());
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  ASSERT_TRUE(store->open().ok());
  const std::vector<std::string> columns{"name",  GLOBAL_DOC_ID, "id", "name",
                                         USER_ID, "score",       "id"};
  for (int row : {0, 3, 9}) {
    auto result = store->fetch(columns, row);
    ASSERT_NE(result, nullptr);
    ASSERT_EQ(result->length, 1);
    ASSERT_EQ(result->values.size(), columns.size());
    for (size_t col = 0; col < columns.size(); ++col) {
      auto expected = source->GetColumnByName(columns[col])->GetScalar(row);
      ASSERT_TRUE(expected.ok());
      EXPECT_TRUE(result->values[col].scalar()->Equals(**expected));
    }
  }
  auto before = ailego::MemoryLimitPool::get_instance().used();
  EXPECT_EQ(store->fetch({"name", "unknown", "id"}, 0), nullptr);
  EXPECT_EQ(store->fetch({LOCAL_ROW_ID}, 0), nullptr);
  EXPECT_EQ(ailego::MemoryLimitPool::get_instance().used(), before);
}

TEST_F(BufferPoolStoreTest,
       OwnedScalarAndTypedBatchReadsMatchNullableSourceTypes) {
  arrow::BooleanBuilder flags;
  arrow::Int32Builder i32;
  arrow::Int64Builder i64;
  arrow::UInt32Builder u32;
  arrow::UInt64Builder u64;
  arrow::FloatBuilder floats;
  arrow::DoubleBuilder doubles;
  arrow::StringBuilder strings;
  arrow::BinaryBuilder binary;
  std::vector<arrow::ArrayBuilder *> builders{
      &flags, &i32, &i64, &u32, &u64, &floats, &doubles, &strings, &binary};
  for (int row = 0; row < 6; ++row) {
    if (row == 2) {
      for (auto *builder : builders) ASSERT_TRUE(builder->AppendNull().ok());
      continue;
    }
    ASSERT_TRUE(flags.Append(row % 2 != 0).ok());
    ASSERT_TRUE(i32.Append(-100 - row).ok());
    ASSERT_TRUE(i64.Append(-(int64_t{1} << 40) - row).ok());
    ASSERT_TRUE(u32.Append(4000000000u + row).ok());
    ASSERT_TRUE(u64.Append((uint64_t{1} << 63) + row).ok());
    ASSERT_TRUE(floats.Append(row + 0.5f).ok());
    ASSERT_TRUE(doubles.Append(row + 0.125).ok());
    ASSERT_TRUE(strings.Append(row ? std::string("a\0b", 3) : "").ok());
    ASSERT_TRUE(binary.Append(row ? std::string("\0\xff", 2) : "").ok());
  }
  arrow::ArrayVector arrays;
  arrow::FieldVector fields;
  std::vector<std::string> columns;
  for (size_t col = 0; col < builders.size(); ++col) {
    std::shared_ptr<arrow::Array> array;
    ASSERT_TRUE(builders[col]->Finish(&array).ok());
    columns.push_back("c" + std::to_string(col));
    fields.push_back(arrow::field(columns.back(), array->type()));
    arrays.push_back(std::move(array));
  }
  arrow::UInt64Builder doc_ids;
  arrow::StringBuilder primary_keys;
  for (int row = 0; row < 6; ++row) {
    ASSERT_TRUE(doc_ids.Append(100 + row).ok());
    ASSERT_TRUE(
        primary_keys.Append(std::string(256, 'p') + std::to_string(row)).ok());
  }
  std::shared_ptr<arrow::Array> id_array, pk_array;
  ASSERT_TRUE(doc_ids.Finish(&id_array).ok());
  ASSERT_TRUE(primary_keys.Finish(&pk_array).ok());
  columns.insert(columns.begin(), {GLOBAL_DOC_ID, USER_ID});
  arrays.insert(arrays.begin(), {id_array, pk_array});
  fields.insert(fields.begin(), {arrow::field(GLOBAL_DOC_ID, arrow::uint64()),
                                 arrow::field(USER_ID, arrow::utf8())});
  const auto path = std::filesystem::absolute(parquet_path).string();
  auto source = arrow::Table::Make(arrow::schema(fields), arrays);
  auto out = arrow::io::FileOutputStream::Open(path);
  ASSERT_TRUE(out.ok());
  ASSERT_TRUE(
      parquet::arrow::WriteTable(*source, arrow::default_memory_pool(), *out, 2)
          .ok());
  ASSERT_TRUE((*out)->Close().ok());
  auto store = std::make_shared<BufferPoolForwardStore>(path);
  ASSERT_TRUE(store->open().ok());
  for (int row : {5, 0, 2, 3, 1, 4, 5}) {
    auto result = store->fetch(columns, row);
    ASSERT_NE(result, nullptr);
    ASSERT_EQ(result->length, 1);
    ASSERT_EQ(result->values.size(), columns.size());
    for (size_t col = 0; col < columns.size(); ++col) {
      auto expected = source->column(col)->GetScalar(row);
      ASSERT_TRUE(expected.ok());
      ASSERT_TRUE(result->values[col].is_scalar());
      EXPECT_TRUE(result->values[col].scalar()->Equals(**expected));
    }
  }
  const auto memory_pool = detail::GetParquetMemoryPool();
  const auto before_scalar = memory_pool->bytes_allocated();
  auto owned_scalar = store->fetch({USER_ID, "c7", "c8"}, 1);
  ASSERT_NE(owned_scalar, nullptr);
  EXPECT_GE(memory_pool->bytes_allocated(), before_scalar + 257 + 3 + 2);
  owned_scalar.reset();
  EXPECT_EQ(memory_pool->bytes_allocated(), before_scalar);
  for (const std::vector<int> rows :
       {std::vector<int>{5, 0, 2, 3, 1, 4, 5}, std::vector<int>{2, 2, 0},
        std::vector<int>{4, 5, 4}}) {
    auto table = store->fetch(columns, rows);
    ASSERT_NE(table, nullptr);
    ASSERT_TRUE(table->ValidateFull().ok());
    ASSERT_EQ(table->num_rows(), rows.size());
    for (size_t col = 0; col < columns.size(); ++col) {
      for (size_t row = 0; row < rows.size(); ++row) {
        auto expected = source->column(col)->GetScalar(rows[row]);
        auto actual = table->column(col)->GetScalar(row);
        ASSERT_TRUE(expected.ok() && actual.ok());
        EXPECT_TRUE((*actual)->Equals(**expected));
      }
    }
  }
  std::vector<Doc::Ptr> retained;
  auto check_doc = [&](const Doc::Ptr &doc, int row) {
    ASSERT_NE(doc, nullptr);
    EXPECT_EQ(doc->doc_id(), 100 + row);
    EXPECT_EQ(doc->pk_ref(), std::string(256, 'p') + std::to_string(row));
    if (row == 2) {
      for (int col = 0; col < 9; ++col)
        EXPECT_FALSE(doc->has("c" + std::to_string(col)));
      return;
    }
    EXPECT_EQ(doc->get<bool>("c0"), row % 2 != 0);
    EXPECT_EQ(doc->get<int32_t>("c1"), -100 - row);
    EXPECT_EQ(doc->get<int64_t>("c2"), -(int64_t{1} << 40) - row);
    EXPECT_EQ(doc->get<uint32_t>("c3"), 4000000000u + row);
    EXPECT_EQ(doc->get<uint64_t>("c4"), (uint64_t{1} << 63) + row);
    EXPECT_EQ(doc->get<float>("c5"), row + 0.5f);
    EXPECT_EQ(doc->get<double>("c6"), row + 0.125);
    EXPECT_EQ(doc->get<std::string>("c7"), row ? std::string("a\0b", 3) : "");
    EXPECT_EQ(doc->get<std::string>("c8"), row ? std::string("\0\xff", 2) : "");
  };
  for (int row = 0; row < 6; ++row) {
    retained.push_back(store->fetch_doc(columns, source->schema(), row));
    check_doc(retained.back(), row);
  }
  auto file = arrow::io::ReadableFile::Open(path);
  ASSERT_TRUE(file.ok());
  ParquetBufferID context(path, *file, 0, 0);
  std::weak_ptr<arrow::Buffer> cached_bytes;
  {
    auto key = context;
    key.column = 1;
    auto handle =
        ParquetBufferPool::get_instance().acquire_buffer(std::move(key));
    ASSERT_NE(handle.data(), nullptr);
    cached_bytes = handle.borrowed_data()->chunk(0)->data()->buffers[2];
  }
  store.reset();
  for (int i = 0; i < 4; ++i)
    ailego::BlockEvictionQueue::get_instance().batch_recycle(1 << 20);
  EXPECT_TRUE(cached_bytes.expired());
  for (int row = 0; row < 6; ++row) check_doc(retained[row], row);
}

namespace {

template <typename BuilderType, typename ValueType>
void CheckOwnedArrayDocs(const std::string &path,
                         const std::vector<ValueType> &expected) {
  arrow::UInt64Builder ids;
  arrow::StringBuilder keys;
  auto values = std::make_shared<BuilderType>();
  arrow::ListBuilder lists(arrow::default_memory_pool(), values);
  for (int row = 0; row < 6; ++row) {
    ASSERT_TRUE(ids.Append(100 + row).ok());
    ASSERT_TRUE(keys.Append("pk" + std::to_string(row)).ok());
    if (row == 2) {
      ASSERT_TRUE(lists.AppendNull().ok());
      continue;
    }
    ASSERT_TRUE(lists.Append().ok());
    if (row == 0 || row == 3) {
      for (const auto &value : expected)
        ASSERT_TRUE(values->Append(value).ok());
    } else if (row == 5) {
      ASSERT_TRUE(values->AppendNull().ok());
    }
  }
  std::shared_ptr<arrow::Array> id_array, key_array, list_array;
  ASSERT_TRUE(ids.Finish(&id_array).ok());
  ASSERT_TRUE(keys.Finish(&key_array).ok());
  ASSERT_TRUE(lists.Finish(&list_array).ok());
  auto source = arrow::Table::Make(
      arrow::schema({arrow::field(GLOBAL_DOC_ID, arrow::uint64()),
                     arrow::field(USER_ID, arrow::utf8()),
                     arrow::field("items", list_array->type())}),
      {id_array, key_array, list_array});
  auto out = arrow::io::FileOutputStream::Open(path);
  ASSERT_TRUE(out.ok());
  ASSERT_TRUE(
      parquet::arrow::WriteTable(*source, arrow::default_memory_pool(), *out, 2)
          .ok());
  ASSERT_TRUE((*out)->Close().ok());
  auto store = std::make_shared<BufferPoolForwardStore>(path);
  ASSERT_TRUE(store->open().ok());
  const std::vector<std::string> columns{GLOBAL_DOC_ID, USER_ID, "items"};
  std::vector<Doc::Ptr> retained;
  auto check_doc = [&](const Doc::Ptr &doc, int row) {
    ASSERT_NE(doc, nullptr);
    EXPECT_TRUE(detail::DocumentMemoryAccess::budgeted(*doc));
    EXPECT_EQ(doc->doc_id(), 100 + row);
    EXPECT_EQ(doc->pk_ref(), "pk" + std::to_string(row));
    if (row == 2) {
      EXPECT_FALSE(doc->has("items"));
    } else {
      auto got = doc->get<std::vector<ValueType>>("items");
      ASSERT_TRUE(got.has_value());
      EXPECT_EQ(got.value(),
                (row == 0 || row == 3) ? expected : std::vector<ValueType>{});
    }
  };
  for (int row = 0; row < 5; ++row) {
    retained.push_back(store->fetch_doc(columns, source->schema(), row));
    check_doc(retained.back(), row);
  }
  // Null child elements are outside zvec's array value contract. Preserve
  // the existing Arrow scalar route rather than silently dropping a null.
  EXPECT_EQ(store->fetch_doc(columns, source->schema(), 5), nullptr);
  auto fallback = store->fetch(columns, 5);
  ASSERT_NE(fallback, nullptr);
  auto fallback_list = std::dynamic_pointer_cast<arrow::ListScalar>(
      fallback->values[2].scalar());
  ASSERT_NE(fallback_list, nullptr);
  EXPECT_EQ(fallback_list->value->null_count(), 1);
  auto file = arrow::io::ReadableFile::Open(path);
  ASSERT_TRUE(file.ok());
  std::weak_ptr<arrow::Buffer> cached_values;
  {
    ParquetBufferID key(path, *file, 2, 0);
    auto handle =
        ParquetBufferPool::get_instance().acquire_buffer(std::move(key));
    ASSERT_NE(handle.borrowed_data(), nullptr);
    const auto &list = static_cast<const arrow::ListArray &>(
        *handle.borrowed_data()->chunk(0));
    cached_values = list.values()->data()->buffers.back();
  }
  store.reset();
  for (int i = 0; i < 4; ++i)
    ailego::BlockEvictionQueue::get_instance().batch_recycle(1 << 20);
  EXPECT_TRUE(cached_values.expired());
  for (int row = 0; row < 5; ++row) check_doc(retained[row], row);
}

}  // namespace

TEST_F(BufferPoolStoreTest, DirectDocOwnsAllSupportedArrayTypes) {
  CheckOwnedArrayDocs<arrow::BooleanBuilder, bool>(parquet_path,
                                                   {true, false, true});
  CheckOwnedArrayDocs<arrow::Int32Builder, int32_t>(parquet_path, {-17, 0, 42});
  CheckOwnedArrayDocs<arrow::Int64Builder, int64_t>(
      parquet_path, {-(int64_t{1} << 40), 0, int64_t{1} << 40});
  CheckOwnedArrayDocs<arrow::UInt32Builder, uint32_t>(parquet_path,
                                                      {0, 4000000000u});
  CheckOwnedArrayDocs<arrow::UInt64Builder, uint64_t>(parquet_path,
                                                      {0, uint64_t{1} << 63});
  CheckOwnedArrayDocs<arrow::FloatBuilder, float>(parquet_path,
                                                  {-0.5f, 0, 1.25f});
  CheckOwnedArrayDocs<arrow::DoubleBuilder, double>(parquet_path,
                                                    {-0.125, 0, 1.5});
  CheckOwnedArrayDocs<arrow::StringBuilder, std::string>(
      parquet_path, {"", std::string("a\0b", 3), std::string(4096, 'x')});
  CheckOwnedArrayDocs<arrow::BinaryBuilder, std::string>(
      parquet_path, {"", std::string("\0\xff", 2), std::string(4096, '\xff')});
}

TEST_F(BufferPoolStoreTest,
       DirectDocDeclinesUnsupportedArrayTypesWithoutPinning) {
  arrow::UInt64Builder ids;
  arrow::StringBuilder keys;
  auto values = std::make_shared<arrow::Int16Builder>();
  arrow::ListBuilder lists(arrow::default_memory_pool(), values);
  ASSERT_TRUE(ids.Append(1).ok());
  ASSERT_TRUE(keys.Append("pk").ok());
  ASSERT_TRUE(lists.Append().ok());
  ASSERT_TRUE(values->Append(11).ok());
  std::shared_ptr<arrow::Array> id_array, key_array, list_array;
  ASSERT_TRUE(ids.Finish(&id_array).ok());
  ASSERT_TRUE(keys.Finish(&key_array).ok());
  ASSERT_TRUE(lists.Finish(&list_array).ok());
  auto source = arrow::Table::Make(
      arrow::schema({arrow::field(GLOBAL_DOC_ID, arrow::uint64()),
                     arrow::field(USER_ID, arrow::utf8()),
                     arrow::field("items", list_array->type())}),
      {id_array, key_array, list_array});
  auto out = arrow::io::FileOutputStream::Open(parquet_path);
  ASSERT_TRUE(out.ok());
  ASSERT_TRUE(
      parquet::arrow::WriteTable(*source, arrow::default_memory_pool(), *out, 1)
          .ok());
  ASSERT_TRUE((*out)->Close().ok());
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  ASSERT_TRUE(store->open().ok());
  const std::vector<std::string> columns{GLOBAL_DOC_ID, USER_ID, "items"};
  auto before = ailego::MemoryLimitPool::get_instance().used();
  EXPECT_EQ(store->fetch_doc(columns, source->schema(), 0), nullptr);
  EXPECT_EQ(store->fetch_doc(columns, source->schema(), -1), nullptr);
  EXPECT_EQ(store->fetch_doc(columns, source->schema(), 1), nullptr);
  EXPECT_EQ(ailego::MemoryLimitPool::get_instance().used(), before);
  auto legacy = store->fetch(columns, 0);
  ASSERT_NE(legacy, nullptr);
  auto scalar =
      std::dynamic_pointer_cast<arrow::ListScalar>(legacy->values[2].scalar());
  ASSERT_NE(scalar, nullptr);
  EXPECT_EQ(
      std::static_pointer_cast<arrow::Int16Array>(scalar->value)->Value(0), 11);
}

namespace {
arrow::Status WriteContextFile(const std::string &path, int64_t base) {
  arrow::Int64Builder values;
  for (int64_t row = 0; row < 8; ++row) {
    ARROW_RETURN_NOT_OK(values.Append(base + row));
  }
  ARROW_ASSIGN_OR_RAISE(auto array, values.Finish());
  auto table = arrow::Table::Make(
      arrow::schema({arrow::field("value", arrow::int64())}), {array});
  ARROW_ASSIGN_OR_RAISE(auto out, arrow::io::FileOutputStream::Open(path));
  ARROW_RETURN_NOT_OK(
      parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), out, 4));
  return out->Close();
}

void ExpectContextValue(const ExecBatchPtr &batch, int64_t value) {
  ASSERT_NE(batch, nullptr);
  ASSERT_EQ(batch->values.size(), 1u);
  EXPECT_EQ(
      std::static_pointer_cast<arrow::Int64Scalar>(batch->values[0].scalar())
          ->value,
      value);
}
}  // namespace


#if !defined(_WIN32) && !defined(_WIN64)
TEST_F(BufferPoolStoreTest, OpenedFileSurvivesReplacementAndEviction) {
  ASSERT_TRUE(WriteContextFile(parquet_path, 100).ok());
  auto original = std::make_shared<BufferPoolForwardStore>(parquet_path);
  ASSERT_TRUE(original->open().ok());
  auto retained = original->fetch({"value"}, 0);
  ExpectContextValue(retained, 100);

  const auto replacement_path = parquet_path + ".new";
  ASSERT_TRUE(WriteContextFile(replacement_path, 200).ok());
  std::filesystem::rename(replacement_path, parquet_path);
  auto replacement = std::make_shared<BufferPoolForwardStore>(parquet_path);
  ASSERT_TRUE(replacement->open().ok());

  // The old file's second row group has never been cached.
  ExpectContextValue(original->fetch({"value"}, 6), 106);
  ExpectContextValue(replacement->fetch({"value"}, 6), 206);
  auto scan = original->scan({"value"});
  ASSERT_NE(scan, nullptr);
  original.reset();
  size_t evicted = 0;
  for (int i = 0; i < 4; ++i) {
    evicted += ailego::BlockEvictionQueue::get_instance().batch_recycle(4096);
  }
  EXPECT_GT(evicted, 0u);
  ExpectContextValue(retained, 100);
  ExpectContextValue(replacement->fetch({"value"}, 0), 200);
  std::shared_ptr<arrow::RecordBatch> batch;
  int64_t rows = 0;
  while (true) {
    batch.reset();
    ASSERT_TRUE(scan->ReadNext(&batch).ok());
    if (!batch) break;
    auto array = std::static_pointer_cast<arrow::Int64Array>(batch->column(0));
    for (int64_t row = 0; row < array->length(); ++row) {
      EXPECT_EQ(array->Value(row), 100 + rows++);
    }
  }
  EXPECT_EQ(rows, 8);
}
#endif

TEST_F(BufferPoolStoreTest, DirectDocRetainsResultAdmissionAndOwnsFields) {
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  ASSERT_TRUE(store->open().ok());
  const std::vector<std::string> columns{GLOBAL_DOC_ID, USER_ID, "name"};
  auto schema = arrow::schema(SelectFields(store->physic_schema(), columns));
  auto expected = store->fetch(columns, 0);
  ASSERT_NE(expected, nullptr);
  const auto name = std::string(std::static_pointer_cast<arrow::StringScalar>(
                                    expected->values[2].scalar())
                                    ->view());
  auto doc = store->fetch_doc(columns, schema, 0);
  ASSERT_NE(doc, nullptr);
  EXPECT_TRUE(detail::DocumentMemoryAccess::budgeted(*doc));
  expected.reset();
  store.reset();
  ailego::BlockEvictionQueue::get_instance().batch_recycle(4096);
  EXPECT_EQ(doc->get<std::string>("name"), name);
}

TEST_F(BufferPoolStoreTest, OpenedFileKeysShareIdentityWithoutOwningFile) {
  auto first = arrow::io::ReadableFile::Open(parquet_path).ValueOrDie();
  auto second = arrow::io::ReadableFile::Open(parquet_path).ValueOrDie();
  ParquetBufferID key(parquet_path, first, 0, 0);
  ParquetBufferID other(parquet_path, second, 0, 0);
  EXPECT_TRUE(ParquetBufferIDEqual{}(key, other));
  EXPECT_EQ(ParquetBufferIDHash{}(key), ParquetBufferIDHash{}(other));
  {
    auto one = ParquetBufferPool::get_instance().acquire_buffer(key);
    auto two = ParquetBufferPool::get_instance().acquire_buffer(other);
    ASSERT_NE(one.borrowed_data(), nullptr);
    EXPECT_EQ(one.borrowed_data(), two.borrowed_data());
  }
  std::weak_ptr<arrow::io::RandomAccessFile> lifetime = first;
  first.reset();
  EXPECT_TRUE(lifetime.expired());
}

TEST_F(BufferPoolStoreTest, CachedLargeRowGroupKeepsScanSlicesAlive) {
  arrow::StringBuilder values;
  for (int row = 0; row < 10003; ++row) {
    ASSERT_TRUE((row % 7 == 0 ? values.AppendNull()
                              : values.Append(std::string(512, 'x') +
                                              std::to_string(row)))
                    .ok());
  }
  auto array = values.Finish().ValueOrDie();
  auto source = arrow::Table::Make(
      arrow::schema({arrow::field("value", arrow::utf8())}), {array});
  auto out = arrow::io::FileOutputStream::Open(parquet_path).ValueOrDie();
  ASSERT_TRUE(parquet::arrow::WriteTable(*source, arrow::default_memory_pool(),
                                         out, 10003)
                  .ok());
  ASSERT_TRUE(out->Close().ok());
  std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
  {
    auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
    ASSERT_TRUE(store->open().ok());
    // Budgeted scans reuse resident columns; cold scans deliberately stream
    // without admitting a whole row group into the point-read cache.
    ASSERT_NE(store->fetch({"value"}, 1), nullptr);
    auto reader = store->scan({"value"});
    std::shared_ptr<arrow::RecordBatch> batch;
    while (true) {
      ASSERT_TRUE(reader->ReadNext(&batch).ok());
      if (!batch) break;
      ASSERT_GT(batch->num_rows(), 0);
      EXPECT_LE(batch->num_rows(), 4096);
      batches.push_back(std::move(batch));
    }
    ASSERT_TRUE(reader->ReadNext(&batch).ok());
    EXPECT_EQ(batch, nullptr);
  }
  for (int i = 0; i < 4; ++i)
    ailego::BlockEvictionQueue::get_instance().batch_recycle(4096);
  // Budgeted decoding may already split a row group into finer chunks.
  ASSERT_GE(batches.size(), 3u);
  auto actual = arrow::Table::FromRecordBatches(batches).ValueOrDie();
  EXPECT_TRUE(actual->Equals(*source));
}
