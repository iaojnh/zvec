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
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <thread>
#include <arrow/api.h>
#include <arrow/io/file.h>
#include <arrow/result.h>
#include <arrow/table.h>
#include <gtest/gtest.h>
#include <parquet/arrow/writer.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/ailego/buffer/external_cache.h>
#include "db/index/storage/fetch_memory.h"
#include "../../../ailego/buffer/pool_payload_budget.h"
#define private public
#include "db/index/storage/parquet_buffer_pool.h"
#undef private
#include "db/index/storage/bufferpool_forward_store.h"
#include "db/index/storage/lazy_record_batch_reader.h"
#include "db/index/storage/parquet_memory_pool.h"
#include "db/index/storage/parquet_metadata_budget.h"
#include "db/index/storage/store_helper.h"
#include "utils/utils.h"
#include "../../common/memory_budget_test_helper.h"

using namespace zvec;

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
  wrong_source = source;
  wrong_source.filename += ".other";
  EXPECT_TRUE(
      ParquetBufferID(parquet_path, 1, 0, wrong_source).metadata.expired());

  reader.reset();
  EXPECT_TRUE(hinted.metadata.expired());
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
  auto &cache = ParquetBufferPool::get_instance().cache_;
  const size_t before = cache.entry_count();
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
  EXPECT_EQ(before + 1, cache.entry_count());
  EXPECT_EQ(expected, scalar->ToString());
  scalar.reset();
  EXPECT_EQ(before, cache.entry_count());
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

  auto id_scalar = batch->values[0].scalar();
  ASSERT_TRUE(id_scalar != nullptr);
  auto id_value = std::dynamic_pointer_cast<arrow::Int32Scalar>(id_scalar);
  ASSERT_TRUE(id_value != nullptr);
  EXPECT_EQ(id_value->value, 1);
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
  ASSERT_NO_FATAL_FAILURE(reserve_remaining_capacity());
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  const size_t external_before = pool.external_used();
  auto denied = ParquetBufferPool::get_instance().acquire_buffer(
      ParquetBufferID(parquet_path_, /*column=*/0, /*row_group=*/0));
  ASSERT_EQ(nullptr, denied.data());
  auto checked = ParquetBufferPool::get_instance().acquire_buffer_checked(
      ParquetBufferID(parquet_path_, /*column=*/0, /*row_group=*/0));
  EXPECT_TRUE(checked.status().IsOutOfMemory()) << checked.status().ToString();
  auto fetch = store.fetch_checked({"value"}, {0, 1},
                                   detail::GetParquetMemoryPool().get());
  EXPECT_TRUE(fetch.status().IsOutOfMemory()) << fetch.status().ToString();
  {
    auto reader = store.scan({GLOBAL_DOC_ID, "value"});
    ASSERT_NE(nullptr, reader);
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
  const auto entries = cache.cache_.entry_count();
  ASSERT_NO_FATAL_FAILURE(check_scan(store.scan({GLOBAL_DOC_ID, "value"}), 2));
  EXPECT_EQ(entries, cache.cache_.entry_count());

  // Point reads still admit columns. A subsequent scan may borrow that first
  // column, but must release its partial pins and stream on the next miss.
  auto handle = cache.acquire_buffer(ParquetBufferID(parquet_path_, 0, 0));
  auto column = handle.data();
  ASSERT_NE(nullptr, column);
  EXPECT_EQ(entries + 1, cache.cache_.entry_count());
  ASSERT_NO_FATAL_FAILURE(check_scan(store.scan({GLOBAL_DOC_ID, "value"}), 2));
  EXPECT_EQ(entries + 1, cache.cache_.entry_count());
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
  ASSERT_NO_FATAL_FAILURE(reserve_remaining_capacity());
  auto missing_column = ParquetBufferPool::get_instance().acquire_buffer(
      ParquetBufferID(parquet_path_, /*column=*/1, /*row_group=*/0));
  ASSERT_EQ(nullptr, missing_column.data());
  {
    auto reader = store.scan({GLOBAL_DOC_ID, "value"});
    ASSERT_NE(nullptr, reader);
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
  ASSERT_NO_FATAL_FAILURE(reserve_remaining_capacity());
  auto reader = std::make_shared<ParquetRecordBatchReader>(
      parquet_reader, std::vector<std::string>{GLOBAL_DOC_ID, "value"}, schema_,
      parquet_path_);
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
    auto &cache = ParquetBufferPool::get_instance().cache_;
    ParquetBufferID key(parquet_path_, column, row_group);
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
  EXPECT_GT(ailego::BlockEvictionQueue::get_instance().batch_recycle(1024), 0u);
  EXPECT_EQ(0u, pool.external_used());
  size_t index = 0;
  for (int row : {3, 4, 7, 8, 2049}) {
    ASSERT_NO_FATAL_FAILURE(check_row(retained[index++], row));
  }
}

TEST_F(BufferPoolScalarFetchTest, InvalidCachedColumnReturnsNull) {
  BufferPoolForwardStore store(parquet_path_);
  ASSERT_TRUE(store.open().ok());
  ParquetBufferID key(parquet_path_, /*column=*/0, /*row_group=*/0);
  auto data = ParquetBufferPool::get_instance().cache_.acquire(key);
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
  EXPECT_EQ(pool->charged_bytes(),
            ailego::PoolPayloadBudget::get_instance().external_used());
  loader.clear(payload);
  EXPECT_EQ(mapped_before, pool->mapped_bytes());
}

TEST_F(BufferPoolLargeDecodeTest, RepeatedEvictionReleasesLargeMappings) {
  auto pool = detail::GetParquetMemoryPool();
  const auto mapped_before = pool->mapped_bytes();
  BufferPoolForwardStore store(parquet_path_);
  ASSERT_TRUE(store.open().ok());
  std::vector<ExecBatchPtr> retained;
  for (int row = 0; row < 12; ++row) {
    auto batch = store.fetch({"name"}, row);
    ASSERT_NE(nullptr, batch);
    retained.push_back(std::move(batch));
    EXPECT_GT(pool->mapped_bytes(), mapped_before);
    ailego::BlockEvictionQueue::get_instance().batch_recycle(1024);
    EXPECT_EQ(mapped_before, pool->mapped_bytes());
    EXPECT_EQ(0u, ailego::PoolPayloadBudget::get_instance().external_used());
  }
  for (size_t row = 0; row < retained.size(); ++row) {
    auto scalar = std::dynamic_pointer_cast<arrow::StringScalar>(
        retained[row]->values[0].scalar());
    ASSERT_NE(nullptr, scalar);
    EXPECT_EQ(large_text(static_cast<int>(row)), scalar->view());
  }
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
  auto &cache = ParquetBufferPool::get_instance().cache_;
  const auto entries = cache.entry_count();
  for (int attempt = 0; attempt < 3; ++attempt) {
    auto denied = ParquetBufferPool::get_instance().acquire_buffer(
        ParquetBufferID(parquet_path_, 0, 0));
    EXPECT_EQ(nullptr, denied.data());
    EXPECT_EQ(before, allocator->charged_bytes());
    EXPECT_EQ(entries, cache.entry_count());
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
