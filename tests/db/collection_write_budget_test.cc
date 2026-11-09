// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <string>
#include <thread>
#include <ailego/pattern/defer.h>
#include <ailego/pattern/scope_guard.h>
#include <gtest/gtest.h>
#include <rocksdb/advanced_cache.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/ailego/buffer/memory_reservation.h>
#include <zvec/db/collection.h>
#include "common/memory_budget_test_helper.h"
#include "db/collection_fetch_internal.h"
#include "db/common/file_helper.h"
#include "db/common/global_resource.h"
#include "db/common/write_buffer_reservation.h"
#include "db/index/common/version_manager.h"
#include "db/index/storage/parquet_memory_pool.h"
#include "db/index/storage/wal/wal_file.h"
#include "../ailego/buffer/pool_payload_budget.h"

namespace zvec {
namespace {

constexpr size_t kMiB = 1024 * 1024;

class CollectionWriteBudgetTest : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    // These cases resize only the Buffer Pool to test write admission. Publish
    // immutable shared resources from the normal fixture budget first, so a
    // rejection case cannot leave a tiny RocksDB/WBM budget for later cases.
    ASSERT_EQ(0, ailego::PoolPayloadBudget::get_instance().init(64 * kMiB));
    ASSERT_EQ(0, GlobalResource::Instance().initialize());
  }

  void SetUp() override {
    ASSERT_TRUE(detail::MemoryBudgetTestPeer::reset_idle_allowances());
    ASSERT_EQ(0, pool_.init(64 * kMiB));
    root_ = std::filesystem::temp_directory_path() /
            ("zvec_write_budget_" +
             std::to_string(
                 std::chrono::steady_clock::now().time_since_epoch().count()));
    ASSERT_TRUE(std::filesystem::create_directory(root_));
  }
  void TearDown() override {
    EXPECT_TRUE(detail::MemoryBudgetTestPeer::reset_idle_allowances());
    // Unpinned Parquet read-cache entries intentionally outlive close(). Force
    // their eviction before checking that no non-evictable allowance leaked.
    if (pool_.try_charge_external(pool_.capacity())) {
      pool_.release_external(pool_.capacity());
    }
    EXPECT_EQ(0u, pool_.external_used());
    std::filesystem::remove_all(root_);
  }
  std::string path(const char *name) const {
    return (root_ / name).string();
  }
  size_t external_without_forward() const {
    // Isolate active writer leases. Arrow buffers, result backing and the idle
    // writer high water remain charged, but are not leaked active leases.
    const auto &writers = detail::SharedWriteBufferMemoryAllowance();
    const auto forward =
        detail::GetParquetMemoryPool()->charged_bytes() +
        detail::SharedResultMemoryAllowance().reserved_bytes() +
        writers.reserved_bytes() - writers.active_bytes();
    EXPECT_GE(pool_.external_used(), forward);
    return pool_.external_used() - std::min(pool_.external_used(), forward);
  }
  CollectionSchema schema() const {
    CollectionSchema result("write_budget");
    EXPECT_TRUE(result
                    .add_field(std::make_shared<FieldSchema>(
                        "payload", DataType::STRING, false))
                    .ok());
    result.set_max_doc_count_per_segment(1000);
    return result;
  }
  std::vector<Doc> docs(const std::string &pk,
                        const std::string &payload = "value") const {
    Doc doc;
    doc.set_pk(pk);
    EXPECT_TRUE(doc.set<std::string>("payload", payload));
    return {std::move(doc)};
  }
  ailego::PoolPayloadBudget &pool_ = ailego::PoolPayloadBudget::get_instance();
  std::filesystem::path root_;
};

TEST_F(CollectionWriteBudgetTest, RejectsBeforeWalReplayAndCanRetry) {
  auto created = Collection::CreateAndOpen(
      path("recovery"), schema(), CollectionOptions{false, false, kMiB});
  ASSERT_TRUE(created.has_value()) << created.error().message();
  ASSERT_TRUE(created.value()->close().ok());
  auto version = VersionManager::Recovery(path("recovery"));
  ASSERT_TRUE(version.has_value());
  const auto meta =
      version.value()->get_current_version().writing_segment_meta();
  ASSERT_TRUE(meta->writing_forward_block().has_value());
  const auto wal_path = FileHelper::MakeWalPath(
      path("recovery"), meta->id(), meta->writing_forward_block()->id());
  // Model a committed WAL record whose forward buffer was lost on process exit.
  WalFilePtr wal;
  ASSERT_EQ(0, WalFile::CreateAndOpen(wal_path, WalOptions{0, true}, &wal));
  Doc doc;
  doc.set_pk("recovered");
  doc.set_doc_id(0);
  doc.set_operator(Operator::INSERT);
  ASSERT_TRUE(doc.set<std::string>("payload", "from_wal"));
  const auto bytes = doc.serialize();
  ASSERT_EQ(0, wal->append(std::string(bytes.begin(), bytes.end())));
  ASSERT_EQ(0, wal->flush());
  ASSERT_EQ(0, wal->close());
  wal.reset();
  auto read_wal = [&] {
    std::ifstream file(wal_path, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(file), {});
  };
  const auto original_wal = read_wal();
  ASSERT_FALSE(original_wal.empty());
  WriteBufferReservation blocker;
  ASSERT_TRUE(blocker.acquire(12 * kMiB).ok());
  for (bool read_only : {false, true}) {
    auto denied = Collection::Open(path("recovery"), {read_only, true, kMiB});
    ASSERT_FALSE(denied.has_value());
    EXPECT_EQ(read_only ? StatusCode::INVALID_ARGUMENT
                        : StatusCode::RESOURCE_EXHAUSTED,
              denied.error().code());
    EXPECT_EQ(original_wal, read_wal());
    EXPECT_EQ(56 * kMiB, external_without_forward());
  }
  blocker.release();
  auto recovered = Collection::Open(path("recovery"), {false, true, kMiB});
  ASSERT_TRUE(recovered.has_value()) << recovered.error().message();
  EXPECT_GE(external_without_forward(), 12 * kMiB);
  {
    auto docs = recovered.value()->fetch({"recovered"});
    ASSERT_TRUE(docs.has_value()) << docs.error().message();
    ASSERT_EQ(1u, docs->count("recovered"));
    EXPECT_EQ("from_wal",
              docs->at("recovered")->get<std::string>("payload").value());
  }
  ASSERT_TRUE(recovered.value()->flush().ok());
  EXPECT_LT(external_without_forward(), 12 * kMiB);
  EXPECT_EQ(8 * kMiB,
            detail::SharedWriteBufferMemoryAllowance().reserved_bytes());
  // Record backing remains charged independently after the recovery lease
  // ends, rather than retaining four complete Arrow buffer equivalents.
  EXPECT_GE(detail::SharedResultMemoryAllowance().reserved_bytes(),
            2 * (bytes.size() + 1));
  EXPECT_LT(detail::SharedResultMemoryAllowance().reserved_bytes(), kMiB);
  ASSERT_TRUE(recovered.value()->close().ok());
}

TEST_F(CollectionWriteBudgetTest, ForwardAdmissionPrecedesWalAndMutation) {
  const auto collection_path = path("forward_admission");
  auto writer = Collection::CreateAndOpen(collection_path, schema(),
                                          {false, false, kMiB});
  ASSERT_TRUE(writer.has_value()) << writer.error().message();
  auto original = docs("original", "before");
  auto inserted = writer.value()->insert(original);
  ASSERT_TRUE(inserted.has_value());
  ASSERT_TRUE(inserted->front().ok());

  auto &pool = ailego::MemoryLimitPool::get_instance();
  const size_t pressure = pool.capacity() - pool.used() - 65536;
  ASSERT_TRUE(pool.try_charge_external(pressure));
  {
    AILEGO_DEFER([&]() { pool.release_external(pressure); });
    auto large = docs("denied", std::string(kMiB, 'x'));
    auto rejected = writer.value()->insert(large);
    ASSERT_TRUE(rejected.has_value());
    EXPECT_EQ(StatusCode::RESOURCE_EXHAUSTED, rejected->front().code());
    large.front().set_pk("original");
    rejected = writer.value()->update(large);
    ASSERT_TRUE(rejected.has_value());
    EXPECT_EQ(StatusCode::RESOURCE_EXHAUSTED, rejected->front().code());
    rejected = writer.value()->upsert(large);
    ASSERT_TRUE(rejected.has_value());
    EXPECT_EQ(StatusCode::RESOURCE_EXHAUSTED, rejected->front().code());
  }
  auto fetched = writer.value()->fetch({"original", "denied"});
  ASSERT_TRUE(fetched.has_value());
  ASSERT_NE(nullptr, fetched->at("original"));
  EXPECT_EQ("before", fetched->at("original")->get<std::string>("payload"));
  EXPECT_EQ(nullptr, fetched->at("denied"));
  fetched->clear();
  ASSERT_TRUE(writer.value()->flush().ok());
  ASSERT_TRUE(writer.value()->close().ok());
  auto reopened = Collection::Open(collection_path, {true, false, kMiB});
  ASSERT_TRUE(reopened.has_value());
  fetched = reopened.value()->fetch({"original", "denied"});
  ASSERT_TRUE(fetched.has_value());
  EXPECT_EQ("before", fetched->at("original")->get<std::string>("payload"));
  EXPECT_EQ(nullptr, fetched->at("denied"));
  fetched->clear();
  ASSERT_TRUE(reopened.value()->close().ok());
}

TEST_F(CollectionWriteBudgetTest, UpdateReadFailureKeepsWalAndAllowsRetry) {
  for (bool persisted : {false, true}) {
    SCOPED_TRACE(persisted);
    const auto collection_path =
        path(persisted ? "update_persisted" : "update_pending");
    auto collection_schema = schema();
    ASSERT_TRUE(collection_schema
                    .add_field(std::make_shared<FieldSchema>(
                        "emb", DataType::VECTOR_FP32, 4, false,
                        std::make_shared<FlatIndexParams>(MetricType::L2)))
                    .ok());
    auto writer = Collection::CreateAndOpen(collection_path, collection_schema,
                                            {false, false, 4 * kMiB});
    ASSERT_TRUE(writer.has_value()) << writer.error().message();
    const std::string original_payload(kMiB, 'a');
    const std::vector<float> vector{1, 2, 3, 4};
    auto original = docs("original", original_payload);
    ASSERT_TRUE(original.front().set("emb", vector));
    auto inserted = writer.value()->insert(original);
    ASSERT_TRUE(inserted.has_value());
    ASSERT_TRUE(inserted->front().ok());
    if (persisted) ASSERT_TRUE(writer.value()->flush().ok());
    // Keep the writer lease active so rejection exercises the document read,
    // rather than failing earlier when reacquiring the writer's reservation.
    auto anchor = original;
    anchor.front().set_pk("anchor");
    inserted = writer.value()->insert(anchor);
    ASSERT_TRUE(inserted.has_value());
    ASSERT_TRUE(inserted->front().ok());
    auto wal_sizes = [&] {
      std::map<std::string, uintmax_t> sizes;
      for (const auto &entry :
           std::filesystem::recursive_directory_iterator(collection_path)) {
        if (entry.path().extension() == ".wal") {
          sizes.emplace(entry.path().string(), entry.file_size());
        }
      }
      return sizes;
    };
    const auto before = wal_sizes();
    ASSERT_FALSE(before.empty());
    auto replacement = docs("original", "after");
    auto &pool = ailego::MemoryLimitPool::get_instance();
    ASSERT_GT(pool.capacity() - pool.used(), 65536u);
    const size_t pressure = pool.capacity() - pool.used() - 65536;
    ASSERT_TRUE(pool.try_charge_external(pressure));
    {
      AILEGO_DEFER([&]() { pool.release_external(pressure); });
      auto rejected = writer.value()->update(replacement);
      ASSERT_TRUE(rejected.has_value());
      EXPECT_EQ(StatusCode::RESOURCE_EXHAUSTED, rejected->front().code());
      EXPECT_EQ(before, wal_sizes());
    }
    {
      auto fetched = writer.value()->fetch({"original"});
      ASSERT_TRUE(fetched.has_value());
      ASSERT_NE(nullptr, fetched->at("original"));
      EXPECT_EQ(original_payload,
                fetched->at("original")->get<std::string>("payload"));
      EXPECT_EQ(vector,
                fetched->at("original")->get<std::vector<float>>("emb"));
    }
    auto updated = writer.value()->update(replacement);
    ASSERT_TRUE(updated.has_value());
    ASSERT_TRUE(updated->front().ok()) << updated->front().message();
    ASSERT_TRUE(writer.value()->flush().ok());
    ASSERT_TRUE(writer.value()->close().ok());
    auto reopened = Collection::Open(collection_path, {true, false, kMiB});
    ASSERT_TRUE(reopened.has_value());
    auto fetched = reopened.value()->fetch({"original"});
    ASSERT_TRUE(fetched.has_value());
    ASSERT_NE(nullptr, fetched->at("original"));
    EXPECT_EQ("after", fetched->at("original")->get<std::string>("payload"));
    EXPECT_EQ(vector, fetched->at("original")->get<std::vector<float>>("emb"));
    fetched->clear();
    ASSERT_TRUE(reopened.value()->close().ok());
  }
}

TEST_F(CollectionWriteBudgetTest, FailedFlushKeepsAllowanceUntilRetry) {
  const auto collection_path = path("flush_failure");
  auto writer = Collection::CreateAndOpen(collection_path, schema(),
                                          {false, false, kMiB});
  ASSERT_TRUE(writer.has_value());
  auto batch = docs("retained");
  auto inserted = writer.value()->insert(batch);
  ASSERT_TRUE(inserted.has_value());
  ASSERT_TRUE(inserted->front().ok());
  // A directory at the next manifest path makes Version::Save fail even on
  // hosts where changing file permissions would not prevent writes.
  const auto blocked_manifest =
      FileHelper::MakeFilePath(collection_path, FileID::MANIFEST_FILE, 1);
  ASSERT_TRUE(std::filesystem::create_directory(blocked_manifest));
  auto status = writer.value()->flush();
  EXPECT_FALSE(status.ok());
  EXPECT_GE(external_without_forward(), 12 * kMiB);
  ASSERT_TRUE(std::filesystem::remove(blocked_manifest));
  ASSERT_TRUE(writer.value()->flush().ok());
  EXPECT_LT(external_without_forward(), 12 * kMiB);
  ASSERT_TRUE(writer.value()->close().ok());
  auto reader = Collection::Open(collection_path, {true, false, kMiB});
  ASSERT_TRUE(reader.has_value());
  auto fetched = reader.value()->fetch({"retained"});
  ASSERT_TRUE(fetched.has_value());
  ASSERT_NE(nullptr, fetched->at("retained"));
  ASSERT_TRUE(reader.value()->close().ok());
}

TEST_F(CollectionWriteBudgetTest, FilterDeleteStagingFailureDoesNotDeleteDocs) {
  auto writer = Collection::CreateAndOpen(path("delete_denied"), schema(),
                                          {false, true, 128 * 1024});
  ASSERT_TRUE(writer) << writer.error().message();
  Doc doc;
  doc.set_pk("retained");
  ASSERT_TRUE(doc.set<std::string>("payload", "value"));
  std::vector<Doc> batch{doc};
  auto inserted = writer.value()->insert(batch);
  ASSERT_TRUE(inserted);
  ASSERT_TRUE(inserted->front().ok());
  const auto remaining = pool_.capacity() - pool_.used();
  ASSERT_TRUE(pool_.try_charge_external(remaining));
  const auto rejected = writer.value()->delete_by_filter("");
  pool_.release_external(remaining);
  ASSERT_EQ(StatusCode::RESOURCE_EXHAUSTED, rejected.code());
  auto fetched = writer.value()->fetch({"retained"});
  ASSERT_TRUE(fetched);
  EXPECT_NE(nullptr, fetched->at("retained"));
  ASSERT_TRUE(writer.value()->delete_by_filter("").ok());
  fetched = writer.value()->fetch({"retained"});
  ASSERT_TRUE(fetched);
  EXPECT_EQ(nullptr, fetched->at("retained"));
  ASSERT_TRUE(writer.value()->close().ok());
}

}  // namespace
}  // namespace zvec
