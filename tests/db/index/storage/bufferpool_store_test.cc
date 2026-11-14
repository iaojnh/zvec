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
#include "../../../ailego/buffer/pool_payload_budget.h"
#include "../../common/memory_budget_test_helper.h"

using namespace zvec;

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
    for (int row = 0; row < rows; ++row) {
      ASSERT_TRUE(
          (row % 11 ? strings.Append(value(row)) : strings.AppendNull()).ok());
      ASSERT_TRUE(numbers.Append(row * 3).ok());
    }
    auto text = strings.Finish();
    auto numeric = numbers.Finish();
    ASSERT_TRUE(text.ok());
    ASSERT_TRUE(numeric.ok());
    auto table = arrow::Table::Make(
        arrow::schema({arrow::field("wide", arrow::utf8()),
                       arrow::field("number", arrow::int64())}),
        {*text, *numeric});
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
  auto reader = store.scan({"nested", "tail"});
  ASSERT_NE(nullptr, reader);
  std::shared_ptr<arrow::RecordBatch> batch;
  ASSERT_TRUE(reader->ReadNext(&batch).ok());
  ASSERT_NE(nullptr, batch);
  ASSERT_EQ(2, batch->num_columns());
  EXPECT_TRUE(batch->column(0)->Equals(*nested));
  EXPECT_TRUE(batch->column(1)->Equals(number_array));
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
