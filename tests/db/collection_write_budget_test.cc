// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <future>
#include <iterator>
#include <map>
#include <string>
#include <thread>
#include <ailego/pattern/defer.h>
#include <ailego/pattern/scope_guard.h>
#include <gtest/gtest.h>
#include <rocksdb/advanced_cache.h>
#include <rocksdb/env.h>
#include <rocksdb/write_buffer_manager.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/ailego/buffer/memory_reservation.h>
#include <zvec/db/collection.h>
#include "common/memory_budget_test_helper.h"
#include "db/collection_fetch_internal.h"
#include "db/common/file_helper.h"
#include "db/common/global_resource.h"
#include "db/common/rocksdb_context.h"
#include "db/common/write_buffer_reservation.h"
#include "db/index/common/id_map.h"
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

TEST_F(CollectionWriteBudgetTest, FailedMmapCreateCanRetryAtTheSamePath) {
  auto &pool = ailego::MemoryLimitPool::get_instance();
  const auto collection_path = path("failed_create") + "/child/./";
  const CollectionOptions options{false, true, kMiB};
  {
    ASSERT_GT(pool.available(), 64u << 10);
    const size_t pressure = pool.available() - (64u << 10);
    ASSERT_TRUE(pool.try_charge_external(pressure));
    AILEGO_DEFER([&] { pool.release_external(pressure); });
    const auto rejected =
        Collection::CreateAndOpen(collection_path, schema(), options);
    ASSERT_FALSE(rejected.has_value());
    EXPECT_EQ(StatusCode::RESOURCE_EXHAUSTED, rejected.error().code());
    EXPECT_FALSE(std::filesystem::exists(collection_path));
    EXPECT_FALSE(std::filesystem::exists(path("failed_create") + "/child"));
  }
  auto created = Collection::CreateAndOpen(collection_path, schema(), options);
  ASSERT_TRUE(created.has_value()) << created.error().message();
  EXPECT_TRUE(created.value()->close().ok());
  auto reopened = Collection::Open(collection_path, options);
  ASSERT_TRUE(reopened.has_value()) << reopened.error().message();
  EXPECT_TRUE(reopened.value()->close().ok());
}

TEST_F(CollectionWriteBudgetTest, LegacyUpsertReplayIgnoresAnAheadIdMap) {
  const CollectionOptions options{false, true, kMiB};
  for (const bool sealed : {false, true}) {
    SCOPED_TRACE(sealed);
    const auto collection_path = path(sealed ? "sealed" : "writing");
    {
      auto created = Collection::CreateAndOpen(collection_path, schema(), options);
      ASSERT_TRUE(created.has_value()) << created.error().message();
      auto previous = docs("pk", "committed");
      auto inserted = created.value()->insert(previous);
      ASSERT_TRUE(inserted.has_value() && inserted->front().ok());
      ASSERT_TRUE(created.value()->flush().ok());
      if (sealed) {
        auto iterator = created.value()->create_iterator({});
        ASSERT_TRUE(iterator.has_value()) << iterator.error().message();
      }
      ASSERT_TRUE(created.value()->close().ok());
    }
    const auto recovered_version = VersionManager::Recovery(collection_path);
    ASSERT_TRUE(recovered_version.has_value());
    const auto version = recovered_version.value()->get_current_version();
    ASSERT_EQ(sealed, !version.persisted_segment_metas().empty());
    const auto writing = version.writing_segment_meta();
    ASSERT_TRUE(writing->writing_forward_block().has_value());
    const auto block = writing->writing_forward_block().value();
    {
      IDMap future_map("write_budget");
      const auto id_path = FileHelper::MakeFilePath(
          collection_path, FileID::ID_FILE, version.id_map_path_suffix());
      ASSERT_TRUE(future_map.open(id_path, false, false).ok());
      ASSERT_TRUE(future_map.upsert("pk", block.min_doc_id() + 1).ok());
      ASSERT_TRUE(future_map.close().ok());
    }
    {
      WalFilePtr wal;
      WalOptions wal_options;
      wal_options.create_new = true;
      const auto wal_path =
          FileHelper::MakeWalPath(collection_path, writing->id(), block.id());
      ASSERT_EQ(0, WalFile::CreateAndOpen(wal_path, wal_options, &wal));
      for (const auto *payload : {"intermediate", "latest"}) {
        auto doc = docs("pk", payload).front();
        doc.set_operator(Operator::UPSERT);
        const auto bytes = doc.serialize();
        ASSERT_EQ(0, wal->append(std::string_view(
                         reinterpret_cast<const char *>(bytes.data()),
                         bytes.size())));
      }
      ASSERT_EQ(0, wal->close());
    }
    auto recovered = Collection::Open(collection_path, options);
    ASSERT_TRUE(recovered.has_value()) << recovered.error().message();
    const auto fetched = recovered.value()->fetch({"pk"});
    ASSERT_TRUE(fetched.has_value()) << fetched.error().message();
    ASSERT_NE(nullptr, fetched->at("pk"));
    EXPECT_EQ("latest", fetched->at("pk")->get<std::string>("payload"));
    SearchQuery query;
    query.topk_ = 10;
    query.filter_ = "payload != 'missing'";
    const auto matches = recovered.value()->query(query);
    ASSERT_TRUE(matches.has_value()) << matches.error().message();
    ASSERT_EQ(1u, matches->size());
    EXPECT_EQ("latest", matches->front()->get<std::string>("payload"));
    EXPECT_TRUE(recovered.value()->close().ok());
  }
}

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

TEST_F(CollectionWriteBudgetTest, FlushCanRetryAfterVectorReaderOpenFails) {
  const auto collection_path = path("vector_reopen_failure");
  auto collection_schema = schema();
  ASSERT_TRUE(collection_schema
                  .add_field(std::make_shared<FieldSchema>(
                      "emb", DataType::VECTOR_FP32, 4, false,
                      std::make_shared<FlatIndexParams>(MetricType::L2)))
                  .ok());
  auto writer = Collection::CreateAndOpen(collection_path, collection_schema,
                                          {false, false, kMiB});
  ASSERT_TRUE(writer.has_value()) << writer.error().message();
  const std::vector<float> vector{1, 2, 3, 4};
  auto first = docs("first", "before failed flush");
  ASSERT_TRUE(first.front().set("emb", vector));
  auto result = writer.value()->insert(first);
  ASSERT_TRUE(result.has_value());
  ASSERT_TRUE(result->front().ok()) << result->front().message();

  std::filesystem::path index_file;
  for (const auto &entry :
       std::filesystem::recursive_directory_iterator(collection_path)) {
    if (entry.path().extension() == ".proxima") index_file = entry.path();
  }
  ASSERT_FALSE(index_file.empty());
  auto hidden_index = index_file;
  hidden_index += ".hidden";
  std::error_code error;
  std::filesystem::rename(index_file, hidden_index, error);
  if (error) GTEST_SKIP() << "Cannot rename open vector index: " << error;
  const auto failed = writer.value()->flush();
  std::filesystem::rename(hidden_index, index_file);
  ASSERT_FALSE(failed.ok());

  {
    auto fetched = writer.value()->fetch({"first"});
    ASSERT_TRUE(fetched.has_value()) << fetched.error().message();
    ASSERT_NE(nullptr, fetched->at("first"));
    EXPECT_EQ("before failed flush",
              fetched->at("first")->get<std::string>("payload"));
    EXPECT_EQ(vector, fetched->at("first")->get<std::vector<float>>("emb"));
  }
  const auto retried = writer.value()->flush();
  ASSERT_TRUE(retried.ok()) << retried.message();
  auto second = docs("second");
  ASSERT_TRUE(second.front().set("emb", vector));
  result = writer.value()->insert(second);
  ASSERT_TRUE(result.has_value());
  ASSERT_TRUE(result->front().ok()) << result->front().message();
  ASSERT_TRUE(writer.value()->close().ok());
  auto reopened = Collection::Open(collection_path, {true, false, kMiB});
  ASSERT_TRUE(reopened.has_value()) << reopened.error().message();
  auto fetched = reopened.value()->fetch({"first", "second"});
  ASSERT_TRUE(fetched.has_value()) << fetched.error().message();
  ASSERT_NE(nullptr, fetched->at("first"));
  ASSERT_NE(nullptr, fetched->at("second"));
  EXPECT_EQ(vector, fetched->at("first")->get<std::vector<float>>("emb"));
  EXPECT_EQ(vector, fetched->at("second")->get<std::vector<float>>("emb"));
}

class RejectSstCreation : public rocksdb::EnvWrapper {
 public:
  RejectSstCreation() : rocksdb::EnvWrapper(rocksdb::Env::Default()) {}

  rocksdb::Status NewWritableFile(
      const std::string &filename,
      std::unique_ptr<rocksdb::WritableFile> *result,
      const rocksdb::EnvOptions &options) override {
    if (std::filesystem::path(filename).extension() == ".sst") {
      ++rejections;
      return rocksdb::Status::IOError("Injected SST creation failure");
    }
    return target()->NewWritableFile(filename, result, options);
  }

  std::atomic<int> rejections{0};
};

TEST_F(CollectionWriteBudgetTest, RocksWriteStatusReflectsCommittedMutation) {
  RejectSstCreation env;
  RocksdbContext ctx;
  RocksdbContext healthy;
  ctx.create_opts_.env = &env;
  ASSERT_TRUE(ctx.create(path("rocks_flush_failure")).ok());
  ASSERT_TRUE(healthy.create(path("rocks_healthy_writer")).ok());
  auto manager = ctx.create_opts_.write_buffer_manager;
  const size_t original_limit = manager->buffer_size();
  AILEGO_DEFER([&] { manager->SetBufferSize(original_limit); });
  constexpr size_t limit = 256 * 1024;
  ASSERT_LT(manager->memory_usage(), limit);
  manager->SetBufferSize(limit);

  const std::string value(2 * limit, 'v');
  const auto write = ctx.put("committed", value);
  EXPECT_GT(env.rejections.load(), 0);
  std::string actual;
  ASSERT_TRUE(ctx.db_->Get(ctx.read_opts_, "committed", &actual).ok());
  EXPECT_EQ(value, actual);
  EXPECT_TRUE(write.ok()) << write.ToString();
  EXPECT_FALSE(ctx.flush().ok());

  auto next = std::async(std::launch::async,
                         [&] { return healthy.put("next", "healthy"); });
  const auto completion = next.wait_for(std::chrono::seconds(5));
  if (completion != std::future_status::ready) {
    // Unblock a regressed shared stall so the assertion can be reported.
    manager->SetBufferSize(original_limit);
  }
  EXPECT_EQ(std::future_status::ready, completion);
  const auto result = next.get();
  EXPECT_TRUE(result.ok()) << result.ToString();
  EXPECT_TRUE(healthy.close().ok());
}

TEST_F(CollectionWriteBudgetTest, EmptyRocksColumnFamiliesDoNotStallWriters) {
  RocksdbContext ctx;
  ASSERT_TRUE(ctx.create(path("empty_rocks_cfs"), {"a", "b", "c", "d"}).ok());
  auto manager = ctx.create_opts_.write_buffer_manager;
  const size_t original_limit = manager->buffer_size();
  AILEGO_DEFER([&] { manager->SetBufferSize(original_limit); });
  manager->SetBufferSize(256 * 1024);
  ASSERT_FALSE(manager->ShouldStall());
  EXPECT_TRUE(ctx.put("key", "value").ok());
  EXPECT_TRUE(ctx.close().ok());
}

TEST_F(CollectionWriteBudgetTest, FlushCanRetryAfterManifestPublicationFails) {
  const auto collection_path = path("manifest_failure");
  auto writer = Collection::CreateAndOpen(collection_path, schema(),
                                          {false, false, kMiB});
  ASSERT_TRUE(writer.has_value()) << writer.error().message();
  auto first = docs("first");
  auto inserted = writer.value()->insert(first);
  ASSERT_TRUE(inserted.has_value());
  ASSERT_TRUE(inserted->front().ok());
  ASSERT_TRUE(writer.value()->flush().ok());

  auto second = docs("second");
  inserted = writer.value()->insert(second);
  ASSERT_TRUE(inserted.has_value());
  ASSERT_TRUE(inserted->front().ok());
  const auto removed = writer.value()->delete_({"first"});
  ASSERT_TRUE(removed.has_value());
  ASSERT_TRUE(removed->front().ok());

  const auto blocked =
      FileHelper::MakeFilePath(collection_path, FileID::MANIFEST_FILE, 2);
  ASSERT_TRUE(std::filesystem::create_directory(blocked));
  const auto failure = writer.value()->flush();
  auto third = docs("third");
  const auto blocked_write = writer.value()->insert(third);
  ASSERT_TRUE(std::filesystem::remove(blocked));
  ASSERT_FALSE(failure.ok());
  ASSERT_TRUE(blocked_write.has_value()) << blocked_write.error().message();
  ASSERT_FALSE(blocked_write->front().ok());
  {
    auto fetched = writer.value()->fetch({"first", "second"});
    ASSERT_TRUE(fetched.has_value()) << fetched.error().message();
    EXPECT_EQ(nullptr, fetched->at("first"));
    EXPECT_NE(nullptr, fetched->at("second"));
  }
  inserted = writer.value()->insert(third);
  ASSERT_TRUE(inserted.has_value());
  ASSERT_TRUE(inserted->front().ok()) << inserted->front().message();
  ASSERT_TRUE(writer.value()->close().ok());
  auto reopened = Collection::Open(collection_path, {true, false, kMiB});
  ASSERT_TRUE(reopened.has_value()) << reopened.error().message();
  auto fetched = reopened.value()->fetch({"first", "second", "third"});
  ASSERT_TRUE(fetched.has_value()) << fetched.error().message();
  EXPECT_EQ(nullptr, fetched->at("first"));
  EXPECT_NE(nullptr, fetched->at("second"));
  EXPECT_NE(nullptr, fetched->at("third"));
}

TEST_F(CollectionWriteBudgetTest, DeleteOnlyFlushIgnoresEmptyForwardWriter) {
  const auto collection_path = path("delete_only_flush");
  const CollectionOptions options{false, false, 64 * 1024};
  auto writer = Collection::CreateAndOpen(collection_path, schema(), options);
  ASSERT_TRUE(writer.has_value()) << writer.error().message();
  auto rows = docs("deleted");
  rows.push_back(docs("retained").front());
  auto inserted = writer.value()->insert(rows);
  ASSERT_TRUE(inserted.has_value());
  ASSERT_TRUE(inserted->at(0).ok());
  ASSERT_TRUE(inserted->at(1).ok());
  ASSERT_TRUE(writer.value()->flush().ok());

  auto version = VersionManager::Recovery(collection_path);
  ASSERT_TRUE(version.has_value());
  const auto meta = version.value()->get_current_version().writing_segment_meta();
  const auto writing_block = meta->writing_forward_block();
  ASSERT_TRUE(writing_block.has_value());
  const auto forward_path = FileHelper::MakeForwardBlockPath(
      collection_path, meta->id(), writing_block->id(), true);
  ASSERT_FALSE(std::filesystem::exists(forward_path));

  auto &budget = ailego::MemoryLimitPool::get_instance();
  ASSERT_GT(budget.capacity() - budget.used(), kMiB);
  const auto pressure = budget.capacity() - budget.used() - kMiB;
  ASSERT_TRUE(budget.try_charge_external(pressure));
  {
    AILEGO_DEFER([&] { budget.release_external(pressure); });
    auto oversized = docs("rejected", std::string(2 * kMiB, 'x'));
    auto rejected = writer.value()->insert(oversized);
    ASSERT_TRUE(rejected.has_value());
    EXPECT_EQ(StatusCode::RESOURCE_EXHAUSTED, rejected->front().code());
    EXPECT_TRUE(std::filesystem::exists(forward_path));
  }

  auto removed = writer.value()->delete_({"deleted"});
  ASSERT_TRUE(removed.has_value());
  ASSERT_TRUE(removed->front().ok());
  ASSERT_TRUE(writer.value()->flush().ok());
  ASSERT_TRUE(writer.value()->close().ok());

  auto reopened = Collection::Open(collection_path, options);
  ASSERT_TRUE(reopened.has_value()) << reopened.error().message();
  auto appended = docs("appended");
  inserted = reopened.value()->insert(appended);
  ASSERT_TRUE(inserted.has_value());
  ASSERT_TRUE(inserted->front().ok()) << inserted->front().message();
  auto fetched = reopened.value()->fetch({"deleted", "retained", "appended"});
  ASSERT_TRUE(fetched.has_value()) << fetched.error().message();
  EXPECT_EQ(nullptr, fetched->at("deleted"));
  ASSERT_NE(nullptr, fetched->at("retained"));
  ASSERT_NE(nullptr, fetched->at("appended"));
  EXPECT_LT(fetched->at("retained")->doc_id(), fetched->at("appended")->doc_id());
}

TEST_F(CollectionWriteBudgetTest, RolloverCanRetryAfterReplacementCreateFails) {
  const auto collection_path = path("rollover_failure");
  const auto small_segments = schema();
  auto writer = Collection::CreateAndOpen(collection_path, small_segments,
                                          {false, false, kMiB});
  ASSERT_TRUE(writer.has_value()) << writer.error().message();
  std::vector<Doc> rows;
  rows.reserve(small_segments.max_doc_count_per_segment());
  for (size_t i = 0; i < small_segments.max_doc_count_per_segment(); ++i) {
    Doc row;
    row.set_pk("row" + std::to_string(i));
    row.set("payload", std::string("original value"));
    rows.push_back(std::move(row));
  }
  const auto inserted = writer.value()->insert(rows);
  ASSERT_TRUE(inserted.has_value()) << inserted.error().message();
  ASSERT_EQ(rows.size(), inserted->size());
  for (const auto &status : *inserted)
    ASSERT_TRUE(status.ok()) << status.message();

  auto version = VersionManager::Recovery(collection_path);
  ASSERT_TRUE(version.has_value());
  const auto blocked = FileHelper::MakeSegmentPath(
      collection_path, version.value()->get_current_version().next_segment_id());
  ASSERT_TRUE(std::filesystem::create_directory(blocked));
  auto next = docs("next");
  const auto failure = writer.value()->insert(next);
  ASSERT_TRUE(std::filesystem::remove(blocked));
  ASSERT_FALSE(failure.has_value());
  {
    auto fetched = writer.value()->fetch({"row0", "next"});
    ASSERT_TRUE(fetched.has_value()) << fetched.error().message();
    ASSERT_NE(nullptr, fetched->at("row0"));
    EXPECT_EQ("original value", fetched->at("row0")->get<std::string>("payload"));
    EXPECT_EQ(nullptr, fetched->at("next"));
  }
  const auto retried = writer.value()->insert(next);
  ASSERT_TRUE(retried.has_value()) << retried.error().message();
  ASSERT_TRUE(retried->front().ok()) << retried->front().message();
  ASSERT_TRUE(writer.value()->close().ok());
  auto reopened = Collection::Open(collection_path, {true, false, kMiB});
  ASSERT_TRUE(reopened.has_value()) << reopened.error().message();
  auto fetched = reopened.value()->fetch({"row0", "next"});
  ASSERT_TRUE(fetched.has_value()) << fetched.error().message();
  ASSERT_NE(nullptr, fetched->at("row0"));
  ASSERT_NE(nullptr, fetched->at("next"));
  EXPECT_EQ(rows.size(), fetched->at("next")->doc_id());
}

TEST_F(CollectionWriteBudgetTest, RolloverRequiresPublishedManifestBeforeWrites) {
  const auto collection_path = path("rollover_manifest_failure");
  const auto small_segments = schema();
  auto writer = Collection::CreateAndOpen(collection_path, small_segments,
                                          {false, false, kMiB});
  ASSERT_TRUE(writer.has_value()) << writer.error().message();
  std::vector<Doc> rows(small_segments.max_doc_count_per_segment());
  for (size_t i = 0; i < rows.size(); ++i) {
    rows[i].set_pk("row" + std::to_string(i));
    rows[i].set("payload", std::string("original value"));
  }
  const auto inserted = writer.value()->insert(rows);
  ASSERT_TRUE(inserted.has_value()) << inserted.error().message();
  for (const auto &status : *inserted)
    ASSERT_TRUE(status.ok()) << status.message();
  // Manifest 1 contains the old writer's complete block. Block only the
  // subsequent publication that would make the replacement discoverable.
  ASSERT_TRUE(writer.value()->flush().ok());
  const auto blocked =
      FileHelper::MakeFilePath(collection_path, FileID::MANIFEST_FILE, 2);
  ASSERT_TRUE(std::filesystem::create_directory(blocked));
  auto next = docs("next");
  const auto rotation = writer.value()->insert(next);
  const auto blocked_insert = writer.value()->insert(next);
  const auto blocked_delete = writer.value()->delete_({"row0"});
  ASSERT_TRUE(std::filesystem::remove(blocked));
  EXPECT_FALSE(rotation.has_value());
  EXPECT_FALSE(blocked_insert.has_value());
  EXPECT_FALSE(blocked_delete.has_value());
  {
    auto fetched = writer.value()->fetch({"row0", "next"});
    ASSERT_TRUE(fetched.has_value()) << fetched.error().message();
    EXPECT_NE(nullptr, fetched->at("row0"));
    EXPECT_EQ(nullptr, fetched->at("next"));
  }

  const auto retried = writer.value()->insert(next);
  ASSERT_TRUE(retried.has_value()) << retried.error().message();
  ASSERT_TRUE(retried->front().ok()) << retried->front().message();
  auto version = VersionManager::Recovery(collection_path);
  ASSERT_TRUE(version.has_value()) << version.error().message();
  const auto writing_meta =
      version.value()->get_current_version().writing_segment_meta();
  EXPECT_EQ(1u, writing_meta->id());
  ASSERT_TRUE(writer.value()->close().ok());
  auto reopened = Collection::Open(collection_path, {true, false, kMiB});
  ASSERT_TRUE(reopened.has_value()) << reopened.error().message();
  auto fetched = reopened.value()->fetch({"row0", "next"});
  ASSERT_TRUE(fetched.has_value()) << fetched.error().message();
  EXPECT_NE(nullptr, fetched->at("row0"));
  ASSERT_NE(nullptr, fetched->at("next"));
  EXPECT_EQ(rows.size(), fetched->at("next")->doc_id());
}

TEST_F(CollectionWriteBudgetTest, BufferedVectorRefusalPrecedesWalAndMutation) {
  const auto collection_path = path("vector_write_admission");
  auto schema_with_vectors = schema();
  ASSERT_TRUE(schema_with_vectors
                  .add_field(std::make_shared<FieldSchema>(
                      "emb", DataType::VECTOR_FP32, 4096, true,
                      std::make_shared<FlatIndexParams>(MetricType::L2)))
                  .ok());
  auto writer = Collection::CreateAndOpen(collection_path, schema_with_vectors,
                                          {false, false, 8 * kMiB});
  ASSERT_TRUE(writer.has_value()) << writer.error().message();
  const std::vector<float> vector(4096, 1.0f);
  auto anchor = docs("anchor");
  ASSERT_TRUE(anchor.front().set("emb", vector));
  const auto inserted = writer.value()->insert(anchor);
  ASSERT_TRUE(inserted.has_value());
  ASSERT_TRUE(inserted->front().ok()) << inserted->front().message();

  auto version = VersionManager::Recovery(collection_path);
  ASSERT_TRUE(version.has_value());
  const auto meta = version.value()->get_current_version().writing_segment_meta();
  const auto wal = FileHelper::MakeWalPath(
      collection_path, meta->id(), meta->writing_forward_block()->id());
  const auto wal_size = std::filesystem::file_size(wal);
  auto next = docs("next");
  ASSERT_TRUE(next.front().set("emb", vector));

  auto &budget = ailego::MemoryLimitPool::get_instance();
  constexpr size_t step = 1024;
  ASSERT_GT(budget.capacity(), budget.external_used() + budget.metadata_used() +
                                   step);
  auto pressure = budget.capacity() - budget.external_used() -
                  budget.metadata_used() - step;
  // The effective hard pressure accounts for uncharged allocator fragments
  // and evicts unused index pages. No writer or WAL reserve is bypassed.
  while (pressure > step && !budget.try_charge_external(pressure))
    pressure -= step;
  ASSERT_GT(pressure, step);
  {
    AILEGO_DEFER([&] { budget.release_external(pressure); });
    auto refused = writer.value()->insert(next);
    ASSERT_TRUE(refused.has_value()) << refused.error().message();
    EXPECT_EQ(StatusCode::RESOURCE_EXHAUSTED, refused->front().code());
    EXPECT_NE(std::string::npos,
              refused->front().message().find("Failed to prepare vector insert"));
    EXPECT_EQ(wal_size, std::filesystem::file_size(wal));
  }
  {
    auto fetched = writer.value()->fetch({"anchor", "next"});
    ASSERT_TRUE(fetched.has_value()) << fetched.error().message();
    ASSERT_NE(nullptr, fetched->at("anchor"));
    EXPECT_EQ(nullptr, fetched->at("next"));
    EXPECT_EQ(vector, fetched->at("anchor")->get<std::vector<float>>("emb"));
  }
  const auto retried = writer.value()->insert(next);
  ASSERT_TRUE(retried.has_value()) << retried.error().message();
  ASSERT_TRUE(retried->front().ok()) << retried->front().message();
  ASSERT_TRUE(writer.value()->flush().ok());
}

TEST_F(CollectionWriteBudgetTest, BufferedVectorWriteHandlesQuantizedNullableGaps) {
  const auto collection_path = path("vector_write_gaps");
  auto schema_with_vectors = schema();
  ASSERT_TRUE(schema_with_vectors
                  .add_field(std::make_shared<FieldSchema>(
                      "emb", DataType::VECTOR_FP32, 4, true,
                      std::make_shared<FlatIndexParams>(MetricType::L2,
                                                        QuantizeType::FP16)))
                  .ok());
  ASSERT_TRUE(schema_with_vectors
                  .add_field(std::make_shared<FieldSchema>(
                      "other", DataType::VECTOR_FP32, 4, true,
                      std::make_shared<FlatIndexParams>(MetricType::L2)))
                  .ok());
  auto writer = Collection::CreateAndOpen(collection_path, schema_with_vectors,
                                          {false, false, kMiB});
  ASSERT_TRUE(writer.has_value()) << writer.error().message();
  auto rows = docs("gap");
  Doc next;
  next.set_pk("both");
  const std::vector<float> emb{1, 2, 3, 4};
  const std::vector<float> other{4, 3, 2, 1};
  ASSERT_TRUE(next.set("payload", std::string("both vectors")));
  ASSERT_TRUE(next.set("emb", emb));
  ASSERT_TRUE(next.set("other", other));
  rows.push_back(std::move(next));
  const auto inserted = writer.value()->insert(rows);
  ASSERT_TRUE(inserted.has_value()) << inserted.error().message();
  ASSERT_TRUE(inserted->at(0).ok()) << inserted->at(0).message();
  ASSERT_TRUE(inserted->at(1).ok()) << inserted->at(1).message();
  ASSERT_TRUE(writer.value()->close().ok());
  auto reopened = Collection::Open(collection_path, {true, false, kMiB});
  ASSERT_TRUE(reopened.has_value()) << reopened.error().message();
  auto fetched = reopened.value()->fetch({"gap", "both"});
  ASSERT_TRUE(fetched.has_value()) << fetched.error().message();
  ASSERT_NE(nullptr, fetched->at("gap"));
  ASSERT_NE(nullptr, fetched->at("both"));
  EXPECT_EQ(emb, fetched->at("both")->get<std::vector<float>>("emb"));
  EXPECT_EQ(other, fetched->at("both")->get<std::vector<float>>("other"));
  SearchQuery query;
  query.topk_ = 2;
  query.target_.field_name_ = "emb";
  query.target_.set_vector(std::string(reinterpret_cast<const char *>(emb.data()),
                                       emb.size() * sizeof(float)));
  auto matches = reopened.value()->query(query);
  ASSERT_TRUE(matches.has_value()) << matches.error().message();
  ASSERT_EQ(1u, matches->size());
  EXPECT_EQ("both", matches->front()->pk_ref());
}

TEST_F(CollectionWriteBudgetTest, WalRecoveryRemovesTornTailBeforeAppend) {
  for (const auto tail_bytes : {2, 6, 9}) {
    SCOPED_TRACE(tail_bytes);
    const auto wal_path = path(("torn_wal_" + std::to_string(tail_bytes)).c_str());
    WalFilePtr wal;
    ASSERT_EQ(0, WalFile::CreateAndOpen(wal_path, {0, true}, &wal));
    ASSERT_EQ(0, wal->append("first"));
    ASSERT_EQ(0, wal->flush());
    const auto first_end = std::filesystem::file_size(wal_path);
    ASSERT_EQ(0, wal->append("incomplete"));
    ASSERT_EQ(0, wal->close());
    std::filesystem::resize_file(wal_path, first_end + tail_bytes);

    ASSERT_EQ(0, WalFile::CreateAndOpen(wal_path, {0, false}, &wal));
    ASSERT_EQ(0, wal->prepare_for_read());
    EXPECT_EQ("first", wal->next());
    EXPECT_TRUE(wal->next().empty());
    EXPECT_TRUE(wal->read_status().ok());
    ASSERT_EQ(0, wal->close());

    ASSERT_EQ(0, WalFile::CreateAndOpen(wal_path, {0, false}, &wal));
    ASSERT_EQ(0, wal->append("second"));
    ASSERT_EQ(0, wal->prepare_for_read());
    EXPECT_EQ("first", wal->next());
    EXPECT_EQ("second", wal->next());
    EXPECT_TRUE(wal->next().empty());
    EXPECT_TRUE(wal->read_status().ok());
  }
}

}  // namespace
}  // namespace zvec
