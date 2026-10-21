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
#include <filesystem>
#include <gtest/gtest.h>
#include <rocksdb/convenience.h>
#include <rocksdb/filter_policy.h>
#include <rocksdb/memtablerep.h>
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

std::vector<std::string> column_family_names(size_t count) {
  std::vector<std::string> names{rocksdb::kDefaultColumnFamilyName};
  for (size_t i = 1; i < count; ++i) {
    names.push_back("cf" + std::to_string(i));
  }
  return names;
}

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

TEST_F(RocksdbContextMemoryTest, SharesWriteBudgetWithoutFtsIteratorCopies) {
  const std::vector<std::string> column_families = {
      "postings", "positions", "term_freq", "max_tf", "doc_len", "stat"};

  RocksdbContext writer;
  ASSERT_TRUE(writer
                  .create(RocksdbContext::Args{
                      kTestPath, column_families, nullptr, {}, true})
                  .ok());
  EXPECT_EQ(GlobalResource::Instance().rocksdb_write_buffer_manager(),
            writer.create_opts_.write_buffer_manager);
  EXPECT_EQ(nullptr, writer.create_opts_.statistics);
  EXPECT_STREQ("SkipListFactory", writer.create_opts_.memtable_factory->Name());
  EXPECT_NE(nullptr, writer.create_opts_.prefix_extractor);
  EXPECT_TRUE(writer.read_opts_.total_order_seek);
  ASSERT_TRUE(writer.close().ok());

  RocksdbContext reader;
  ASSERT_TRUE(reader
                  .open(RocksdbContext::Args{kTestPath, {}, nullptr, {}, true},
                        /*read_only=*/true)
                  .ok());
  EXPECT_EQ(GlobalResource::Instance().rocksdb_write_buffer_manager(),
            reader.create_opts_.write_buffer_manager);
  EXPECT_EQ(nullptr, reader.create_opts_.statistics);
  EXPECT_STREQ("SkipListFactory", reader.create_opts_.memtable_factory->Name());
  ASSERT_TRUE(reader.close().ok());
}

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

TEST_F(RocksdbContextMemoryTest, PartitionsAndChargesFilterConstruction) {
  RocksdbContext context;
  ASSERT_TRUE(context.create(kTestPath, {"postings", "positions"}).ok());
  for (auto *cf : context.cf_handles_) {
    const auto options = context.db_->GetOptions(cf);
    const auto *table =
        options.table_factory->GetOptions<rocksdb::BlockBasedTableOptions>();
    ASSERT_NE(nullptr, table);
    EXPECT_EQ(rocksdb::BlockBasedTableOptions::kTwoLevelIndexSearch,
              table->index_type);
    EXPECT_TRUE(table->partition_filters);
    EXPECT_EQ(4096u, table->metadata_block_size);
    EXPECT_GE(table->format_version, 5u);
    EXPECT_EQ(GlobalResource::Instance().rocksdb_block_cache(),
              table->block_cache);
    const auto charge = table->cache_usage_options.options_overrides.find(
        rocksdb::CacheEntryRole::kFilterConstruction);
    ASSERT_NE(table->cache_usage_options.options_overrides.end(), charge);
    EXPECT_EQ(rocksdb::CacheEntryRoleOptions::Decision::kEnabled,
              charge->second.charged);
  }
  ASSERT_TRUE(context.close().ok());
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

TEST_F(RocksdbContextMemoryTest, OversizedBatchDoesNotStallNextDatabase) {
  ASSERT_TRUE(FileHelper::CreateDirectory(kTestPath));
  RocksdbContext first;
  RocksdbContext second;
  ASSERT_TRUE(
      first.create(std::string(kTestPath) + "/first", {"one", "two"}).ok());
  ASSERT_TRUE(second.create(std::string(kTestPath) + "/second").ok());
  const auto manager = first.create_opts_.write_buffer_manager;
  ASSERT_NE(nullptr, manager);
  const std::string value(1024 * 1024, 'v');
  rocksdb::WriteBatch batch;
  const size_t count = manager->buffer_size() / value.size() + 2;
  for (size_t i = 0; i < count; ++i) {
    ASSERT_TRUE(
        batch.Put(first.get_cf(i % 2 ? "one" : "two"), std::to_string(i), value)
            .ok());
  }
  ASSERT_TRUE(first.write(&batch).ok());
  EXPECT_LT(manager->memory_usage(), manager->buffer_size());
  // Bound a regression's failure: without post-write draining this returns
  // Incomplete rather than leaving the test blocked on the shared manager.
  second.write_opts_.no_slowdown = true;
  EXPECT_TRUE(second.put("next", "database").ok());
  std::string loaded;
  EXPECT_TRUE(
      first.db_->Get(first.read_opts_, first.get_cf("two"), "0", &loaded).ok());
  EXPECT_EQ(value, loaded);
}

TEST_F(RocksdbContextMemoryTest, WritableFtsUsesBudgetedOrderedMemtables) {
  for (size_t cf_count : {1u, 7u, 12u}) {
    SCOPED_TRACE(cf_count);
    RocksdbContext writer;
    ASSERT_TRUE(
        writer
            .create(RocksdbContext::Args{
                kTestPath, column_family_names(cf_count), nullptr, {}, true})
            .ok());
    EXPECT_STREQ("SkipListFactory",
                 writer.create_opts_.memtable_factory->Name());
    for (auto *cf : writer.cf_handles_) {
      EXPECT_STREQ("SkipListFactory",
                   writer.db_->GetOptions(cf).memtable_factory->Name());
    }
    // Removing the copied sort structure must not change flush thresholds
    // or opt out of the process-wide memtable budget.
    EXPECT_EQ(128ULL * 1024 * 1024, writer.create_opts_.write_buffer_size);
    EXPECT_EQ(GlobalResource::Instance().rocksdb_write_buffer_manager(),
              writer.create_opts_.write_buffer_manager);
    ASSERT_TRUE(writer.close().ok());
    FileHelper::RemoveDirectory(kTestPath);
  }
}

TEST_F(RocksdbContextMemoryTest,
       ManyColumnFamiliesKeepOrderedMemtablesOnReopen) {
  for (size_t cf_count : {33u, 129u}) {
    SCOPED_TRACE(cf_count);
    RocksdbContext writer;
    ASSERT_TRUE(
        writer
            .create(RocksdbContext::Args{
                kTestPath, column_family_names(cf_count), nullptr, {}, true})
            .ok());
    EXPECT_STREQ("SkipListFactory",
                 writer.create_opts_.memtable_factory->Name());
    ASSERT_TRUE(writer.close().ok());

    // Reopening every persisted CF must not restore hash bucket arrays or
    // require an unbudgeted sorted copy when obtaining an iterator.
    RocksdbContext reopened;
    ASSERT_TRUE(
        reopened
            .open(RocksdbContext::Args{kTestPath, {}, nullptr, {}, true},
                  /*read_only=*/false)
            .ok());
    EXPECT_STREQ("SkipListFactory",
                 reopened.create_opts_.memtable_factory->Name());
    for (auto *cf : reopened.cf_handles_) {
      EXPECT_STREQ("SkipListFactory",
                   reopened.db_->GetOptions(cf).memtable_factory->Name());
    }
    ASSERT_TRUE(reopened.close().ok());
    FileHelper::RemoveDirectory(kTestPath);
  }
}

TEST_F(RocksdbContextMemoryTest, PrefixWritesSurviveFlushAndReadOnlyReopen) {
  const std::vector<std::string> names{"postings", "positions"};
  RocksdbContext writer;
  ASSERT_TRUE(
      writer.create(RocksdbContext::Args{kTestPath, names, nullptr, {}, true})
          .ok());
  for (const auto &name : names) {
    auto *cf = writer.get_cf(name);
    ASSERT_NE(nullptr, cf);
    for (size_t i = 0; i < 256; ++i) {
      // Repeated prefixes must preserve every distinct full key.
      const auto key = "prefix00" + std::to_string(i);
      ASSERT_TRUE(writer.db_->Put(writer.write_opts_, cf, key, "before").ok());
    }
  }
  ASSERT_TRUE(writer.flush().ok());
  for (const auto &name : names) {
    auto *cf = writer.get_cf(name);
    for (size_t i = 0; i < 256; ++i) {
      const auto key = "prefix00" + std::to_string(i);
      if (i % 2 == 0) {
        ASSERT_TRUE(writer.db_->Delete(writer.write_opts_, cf, key).ok());
      } else {
        ASSERT_TRUE(writer.db_->Put(writer.write_opts_, cf, key, "after").ok());
      }
    }
  }
  ASSERT_TRUE(writer.close().ok());

  RocksdbContext reader;
  ASSERT_TRUE(reader
                  .open(RocksdbContext::Args{kTestPath, {}, nullptr, {}, true},
                        /*read_only=*/true)
                  .ok());
  EXPECT_STREQ("SkipListFactory", reader.create_opts_.memtable_factory->Name());
  for (const auto &name : names) {
    auto *cf = reader.get_cf(name);
    ASSERT_NE(nullptr, cf);
    for (size_t i = 0; i < 256; ++i) {
      const auto key = "prefix00" + std::to_string(i);
      std::string value;
      const auto status = reader.db_->Get(reader.read_opts_, cf, key, &value);
      if (i % 2 == 0) {
        EXPECT_TRUE(status.IsNotFound()) << status.ToString();
      } else {
        ASSERT_TRUE(status.ok()) << status.ToString();
        EXPECT_EQ("after", value);
      }
    }
  }
  ASSERT_TRUE(reader.close().ok());
}

TEST_F(RocksdbContextMemoryTest, CompactionSharesCacheWithActiveWriteBuffers) {
  const std::vector<std::string> names{"postings", "positions", "term_freq",
                                       "max_tf",   "doc_len",   "stat"};
  RocksdbContext writer;
  ASSERT_TRUE(
      writer.create(RocksdbContext::Args{kTestPath, names, nullptr, {}, true})
          .ok());
  std::vector<rocksdb::ColumnFamilyHandle *> handles;
  for (const auto &name : names) {
    auto *cf = writer.get_cf(name);
    ASSERT_NE(nullptr, cf);
    handles.push_back(cf);
  }
  // Cross-CF writes exceed the shared memtable budget. Keep writing while
  // earlier SSTs are compacted, rather than testing only an empty-cache flush.
  constexpr size_t kRows = 2048;
  constexpr size_t kBatchRows = 64;
  for (size_t round = 0; round < 4; ++round) {
    const std::string value(1024, static_cast<char>('a' + round));
    for (size_t begin = 0; begin < kRows; begin += kBatchRows) {
      rocksdb::WriteBatch batch;
      for (size_t i = begin; i < begin + kBatchRows; ++i) {
        const auto key = "prefix00" + std::to_string(i);
        for (auto *cf : handles) {
          ASSERT_TRUE(batch.Put(cf, key, value).ok());
        }
      }
      const auto status = writer.db_->Write(writer.write_opts_, &batch);
      ASSERT_TRUE(status.ok()) << status.ToString();
    }
    // Flush one CF while the others still have active write buffers.
    const auto flush_status = writer.db_->Flush(writer.flush_opts_, handles[0]);
    ASSERT_TRUE(flush_status.ok()) << flush_status.ToString();
    rocksdb::CompactRangeOptions options;
    const auto compact_status =
        writer.db_->CompactRange(options, handles[0], nullptr, nullptr);
    ASSERT_TRUE(compact_status.ok()) << compact_status.ToString();
  }
  ASSERT_TRUE(writer.flush().ok());
  uint64_t background_errors = 0;
  ASSERT_TRUE(writer.db_->GetIntProperty("rocksdb.background-errors",
                                         &background_errors));
  EXPECT_EQ(0u, background_errors);
  ASSERT_TRUE(writer.close().ok());

  RocksdbContext reader;
  ASSERT_TRUE(reader
                  .open(RocksdbContext::Args{kTestPath, {}, nullptr, {}, true},
                        /*read_only=*/true)
                  .ok());
  for (const auto &name : names) {
    auto *cf = reader.get_cf(name);
    ASSERT_NE(nullptr, cf);
    for (size_t i = 0; i < kRows; ++i) {
      std::string value;
      const auto status = reader.db_->Get(
          reader.read_opts_, cf, "prefix00" + std::to_string(i), &value);
      ASSERT_TRUE(status.ok()) << status.ToString();
      EXPECT_EQ(std::string(1024, 'd'), value);
    }
  }
  ASSERT_TRUE(reader.close().ok());
}

}  // namespace
}  // namespace zvec
