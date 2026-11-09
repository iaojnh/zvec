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

#include "db/common/rocksdb_context.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <filesystem>
#include <thread>
#include <gtest/gtest.h>
#include <rocksdb/convenience.h>
#include <rocksdb/filter_policy.h>
#include <rocksdb/memtablerep.h>
#include <rocksdb/merge_operator.h>
#include <rocksdb/slice_transform.h>
#include <rocksdb/table.h>
#include <rocksdb/write_batch.h>
#include <rocksdb/write_buffer_manager.h>
#include <zvec/db/config.h>
#include "db/common/file_helper.h"
#include "db/common/global_resource.h"
#include "memory_budget_test_helper.h"

namespace zvec {
namespace {

constexpr char kTestPath[] = "./test_rocksdb_memory_budget";
#if defined(__linux__) || defined(__APPLE__) || defined(_WIN32)
constexpr char kMemtableFactoryName[] = "ZvecMappedSkipList";
#else
constexpr char kMemtableFactoryName[] = "SkipListFactory";
#endif


class RocksdbContextMemoryTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    GlobalConfig::ConfigData config;
    config.memory_limit_bytes = 128ULL * 1024 * 1024;
    config.query_thread_count = 1;
    config.optimize_thread_count = 1;
    const auto status = GlobalConfig::Instance().initialize(config);
    ASSERT_TRUE(status.ok()) << status.message();
  }

  void SetUp() override {
    FileHelper::RemoveDirectory(kTestPath);
  }

  void TearDown() override {
    FileHelper::RemoveDirectory(kTestPath);
  }
};

class CountingSumMerge final : public rocksdb::MergeOperator {
 public:
  const char *Name() const override {
    return "CountingSumMerge";
  }
  bool FullMergeV2(const MergeOperationInput &input,
                   MergeOperationOutput *output) const override {
    size_t previous = max_operands_.load();
    while (previous < input.operand_list.size() &&
           !max_operands_.compare_exchange_weak(previous,
                                               input.operand_list.size())) {
    }
    uint64_t sum = 0;
    if (input.existing_value) {
      if (input.existing_value->size() != sizeof(sum)) return false;
      std::memcpy(&sum, input.existing_value->data(), sizeof(sum));
    }
    for (const auto &operand : input.operand_list) {
      uint64_t value = 0;
      if (operand.size() != sizeof(value)) return false;
      std::memcpy(&value, operand.data(), sizeof(value));
      sum += value;
    }
    output->new_value.assign(reinterpret_cast<const char *>(&sum), sizeof(sum));
    return true;
  }
  mutable std::atomic<size_t> max_operands_{0};
};

TEST_F(RocksdbContextMemoryTest, BufferedWritesSurviveFlushAndWritableReopen) {
  RocksdbContext writer;
  ASSERT_TRUE(writer.create(kTestPath).ok());
#if defined(_WIN32) || defined(__APPLE__) || defined(__linux__)
  EXPECT_EQ(0u, writer.create_opts_.writable_file_max_buffer_size);
  EXPECT_FALSE(writer.create_opts_.use_direct_io_for_flush_and_compaction);
#endif
  const std::string value(128 * 1024, 'x');
  for (size_t i = 0; i < 32; ++i) {
    ASSERT_TRUE(
        writer.db_->Put(writer.write_opts_, std::to_string(i), value).ok());
  }
  ASSERT_TRUE(writer.flush().ok());
  ASSERT_TRUE(writer.close().ok());

  RocksdbContext reopened;
  ASSERT_TRUE(reopened.open(kTestPath, false).ok());
#if defined(_WIN32) || defined(__APPLE__) || defined(__linux__)
  EXPECT_EQ(0u, reopened.create_opts_.writable_file_max_buffer_size);
#endif
  ASSERT_TRUE(reopened.db_->Put(reopened.write_opts_, "updated", value).ok());
  ASSERT_TRUE(reopened.flush().ok());
  ASSERT_TRUE(reopened.close().ok());

  RocksdbContext reader;
  ASSERT_TRUE(reader.open(kTestPath, true).ok());
  EXPECT_EQ(rocksdb::DBOptions().writable_file_max_buffer_size,
            reader.create_opts_.writable_file_max_buffer_size);
  for (size_t i = 0; i < 32; ++i) {
    std::string actual;
    ASSERT_TRUE(
        reader.db_->Get(reader.read_opts_, std::to_string(i), &actual).ok());
    EXPECT_EQ(value, actual);
  }
  std::string actual;
  ASSERT_TRUE(reader.db_->Get(reader.read_opts_, "updated", &actual).ok());
  EXPECT_EQ(value, actual);
  ASSERT_TRUE(reader.close().ok());
}

#if defined(_WIN32) || defined(__APPLE__) || defined(__linux__)
#endif

TEST_F(RocksdbContextMemoryTest, MetadataAdmissionRejectsBeforeCreatingFiles) {
  ASSERT_TRUE(detail::MemoryBudgetTestPeer::reset_idle_allowances());
  auto &pool = ailego::MemoryLimitPool::get_instance();
  detail::RetainedMetadataBudget::Lease blocker(
      detail::RetainedMetadataBudget::instance());
  blocker.acquire(pool.capacity() / 2);
  RocksdbContext denied;
  const auto status = denied.create(kTestPath);
  EXPECT_EQ(StatusCode::RESOURCE_EXHAUSTED, status.code());
  EXPECT_FALSE(std::filesystem::exists(std::string(kTestPath) + "/CURRENT"));
  blocker.retain_at_most(0);
  EXPECT_TRUE(denied.create(kTestPath).ok());
  EXPECT_TRUE(denied.close().ok());
}

TEST_F(RocksdbContextMemoryTest, ReopensLegacyFullFiltersAndHashMemtables) {
  // Write an SST with the previous options, then mix it with new partitioned
  // tables. Memtable choice must not require a stored-format migration.
  rocksdb::Options legacy;
  legacy.create_if_missing = true;
  legacy.memtable_factory.reset(rocksdb::NewHashSkipListRepFactory(4096));
  legacy.allow_concurrent_memtable_write = false;
  legacy.prefix_extractor.reset(rocksdb::NewCappedPrefixTransform(8));
  rocksdb::BlockBasedTableOptions table;
  table.filter_policy.reset(rocksdb::NewBloomFilterPolicy(10, false));
  table.block_cache = GlobalResource::Instance().rocksdb_block_cache();
  legacy.write_buffer_manager =
      GlobalResource::Instance().rocksdb_write_buffer_manager();
  legacy.table_factory.reset(rocksdb::NewBlockBasedTableFactory(table));
  rocksdb::DB *raw = nullptr;
  ASSERT_TRUE(rocksdb::DB::Open(legacy, kTestPath, &raw).ok());
  std::unique_ptr<rocksdb::DB> old(raw);
  for (size_t i = 0; i < 1024; ++i) {
    ASSERT_TRUE(
        old->Put(rocksdb::WriteOptions{}, "prefix00" + std::to_string(i), "old")
            .ok());
  }
  rocksdb::FlushOptions flush;
  flush.wait = true;
  ASSERT_TRUE(old->Flush(flush).ok());
  ASSERT_TRUE(old->Close().ok());
  old.reset();

  RocksdbContext writer;
  ASSERT_TRUE(writer.open({kTestPath, {}, nullptr, {}, true}, false).ok());
  for (size_t i = 1024; i < 2048; ++i) {
    ASSERT_TRUE(writer.put("prefix00" + std::to_string(i), "new").ok());
  }
  ASSERT_TRUE(writer.flush().ok());
  ASSERT_TRUE(writer.close().ok());
  RocksdbContext reader;
  ASSERT_TRUE(reader.open({kTestPath, {}, nullptr, {}, true}, true).ok());
  for (size_t i = 0; i < 2048; ++i) {
    std::string value;
    ASSERT_TRUE(
        reader.db_
            ->Get(reader.read_opts_, "prefix00" + std::to_string(i), &value)
            .ok());
    EXPECT_EQ(i < 1024 ? "old" : "new", value);
  }
  std::unique_ptr<rocksdb::Iterator> iter(
      reader.db_->NewIterator(reader.read_opts_));
  size_t count = 0;
  std::string previous;
  for (iter->SeekToFirst(); iter->Valid(); iter->Next()) {
    EXPECT_LT(previous, iter->key().ToString());
    previous = iter->key().ToString();
    ++count;
  }
  EXPECT_TRUE(iter->status().ok());
  EXPECT_EQ(2048u, count);
  iter.reset();
  ASSERT_TRUE(reader.close().ok());
}

}  // namespace
}  // namespace zvec
