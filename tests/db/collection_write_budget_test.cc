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
#include <gtest/gtest.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/db/collection.h>
#include "common/memory_budget_test_helper.h"
#include "db/common/file_helper.h"
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

TEST_F(CollectionWriteBudgetTest, RejectsBeforeCreatingCollectionFiles) {
  pool_.init(kMiB);
  auto result = Collection::CreateAndOpen(
      path("denied"), schema(), CollectionOptions{false, false, kMiB});
  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(StatusCode::RESOURCE_EXHAUSTED, result.error().code());
  EXPECT_FALSE(std::filesystem::exists(path("denied")));
}

TEST_F(CollectionWriteBudgetTest, DefaultBufferReturnsManagedPeakAfterFlush) {
  pool_.init(512 * kMiB);
  auto writer =
      Collection::CreateAndOpen(path("default"), schema(), {false, false});
  ASSERT_TRUE(writer.has_value()) << writer.error().message();
  EXPECT_EQ(0u, external_without_forward());
  auto batch = docs("written");
  auto inserted = writer.value()->insert(batch);
  ASSERT_TRUE(inserted.has_value());
  ASSERT_TRUE(inserted->front().ok());
  EXPECT_EQ(264 * kMiB, external_without_forward());
  ASSERT_TRUE(writer.value()->flush().ok());
  EXPECT_LT(external_without_forward(), kMiB);
  EXPECT_EQ(8 * kMiB,
            detail::SharedWriteBufferMemoryAllowance().reserved_bytes());
  EXPECT_GT(detail::SharedResultMemoryAllowance().reserved_bytes(), 0u);
  // A subsequent writer reuses the shared allowance; a transient admission
  // larger than the retained high water only keeps its excess while active.
  WriteBufferReservation other;
  ASSERT_TRUE(other.acquire(96 * kMiB).ok());
  other.release();
  EXPECT_EQ(8 * kMiB,
            detail::SharedWriteBufferMemoryAllowance().reserved_bytes());
  ASSERT_TRUE(writer.value()->close().ok());
  EXPECT_EQ(8 * kMiB,
            detail::SharedWriteBufferMemoryAllowance().reserved_bytes());
}

TEST_F(CollectionWriteBudgetTest, LiveWriteBuffersShareAndReturnBudget) {
  pool_.init(30 * kMiB);
  const CollectionOptions options{false, false, kMiB};
  auto first = Collection::CreateAndOpen(path("first"), schema(), options);
  ASSERT_TRUE(first.has_value()) << first.error().message();
  auto second = Collection::CreateAndOpen(path("second"), schema(), options);
  ASSERT_TRUE(second.has_value()) << second.error().message();
  EXPECT_EQ(0u, external_without_forward());
  auto first_docs = docs("first");
  auto second_docs = docs("second");
  auto inserted = first.value()->insert(first_docs);
  ASSERT_TRUE(inserted.has_value());
  ASSERT_TRUE(inserted->front().ok());
  inserted = second.value()->insert(second_docs);
  ASSERT_TRUE(inserted.has_value());
  ASSERT_TRUE(inserted->front().ok());
  EXPECT_EQ(24 * kMiB, external_without_forward());
  auto denied = Collection::CreateAndOpen(path("third"), schema(), options);
  ASSERT_FALSE(denied.has_value());
  EXPECT_EQ(StatusCode::RESOURCE_EXHAUSTED, denied.error().code());
  EXPECT_FALSE(std::filesystem::exists(path("third")));
  ASSERT_TRUE(first.value()->close().ok());
  EXPECT_EQ(12 * kMiB, external_without_forward());
  auto third = Collection::CreateAndOpen(path("third"), schema(), options);
  ASSERT_TRUE(third.has_value()) << third.error().message();
  EXPECT_EQ(12 * kMiB, external_without_forward());
  auto third_docs = docs("third");
  inserted = third.value()->insert(third_docs);
  ASSERT_TRUE(inserted.has_value());
  ASSERT_TRUE(inserted->front().ok());
  EXPECT_EQ(24 * kMiB, external_without_forward());
  ASSERT_TRUE(second.value()->close().ok());
  ASSERT_TRUE(third.value()->destroy().ok());
  EXPECT_EQ(0u, external_without_forward());
}

TEST_F(CollectionWriteBudgetTest, OrdinaryWriteBackingTracksDocumentSize) {
  auto writer = Collection::CreateAndOpen(path("ordinary_backing"), schema(),
                                          {false, false, kMiB});
  ASSERT_TRUE(writer.has_value()) << writer.error().message();
  auto &ordinary = detail::SharedResultMemoryAllowance();
  const size_t initial = ordinary.reserved_bytes();
  auto small = docs("small");
  auto inserted = writer.value()->insert(small);
  ASSERT_TRUE(inserted.has_value());
  ASSERT_TRUE(inserted->front().ok());
  // The small staging Doc is charged before its copy is published.
  EXPECT_GT(ordinary.active_bytes(), 0u);
  EXPECT_GT(ordinary.reserved_bytes(), initial);

  // A large Doc goes directly to charged Arrow, but its transient WAL vector
  // still needs its own ordinary-heap backing allowance.
  auto large = docs("large", std::string(kMiB, 'x'));
  const size_t wal_bytes = large.front().serialized_size();
  inserted = writer.value()->insert(large);
  ASSERT_TRUE(inserted.has_value());
  ASSERT_TRUE(inserted->front().ok()) << inserted->front().message();
  EXPECT_GE(ordinary.reserved_bytes(), 2 * wal_bytes);
  ASSERT_TRUE(writer.value()->flush().ok());
  ASSERT_TRUE(writer.value()->close().ok());
  EXPECT_EQ(0u, ordinary.active_bytes());
  EXPECT_GE(ordinary.reserved_bytes(), 2 * wal_bytes);
  EXPECT_EQ(8 * kMiB,
            detail::SharedWriteBufferMemoryAllowance().reserved_bytes());
}

TEST_F(CollectionWriteBudgetTest, ReadOnlyDoesNotReserveAndUsesPersistedMode) {
  auto created = Collection::CreateAndOpen(
      path("buffer"), schema(), CollectionOptions{false, false, kMiB});
  ASSERT_TRUE(created.has_value()) << created.error().message();
  ASSERT_TRUE(created.value()->close().ok());
  EXPECT_EQ(0u, external_without_forward());

  // The mmap hint cannot disable budgeting for an existing Buffer collection.
  WriteBufferReservation blocker;
  ASSERT_TRUE(blocker.acquire(12 * kMiB).ok());  // 56 of 64 MiB reserved.
  auto denied = Collection::Open(path("buffer"), {false, true, kMiB});
  ASSERT_FALSE(denied.has_value());
  EXPECT_EQ(StatusCode::RESOURCE_EXHAUSTED, denied.error().code());
  auto reader = Collection::Open(path("buffer"), {true, true, kMiB});
  ASSERT_TRUE(reader.has_value()) << reader.error().message();
  EXPECT_EQ(56 * kMiB, external_without_forward());
  ASSERT_TRUE(reader.value()->close().ok());
  blocker.release();
  auto writer = Collection::Open(path("buffer"), {false, true, kMiB});
  ASSERT_TRUE(writer.has_value()) << writer.error().message();
  EXPECT_EQ(0u, external_without_forward());
  auto batch = docs("written");
  auto inserted = writer.value()->insert(batch);
  ASSERT_TRUE(inserted.has_value());
  ASSERT_TRUE(inserted->front().ok());
  EXPECT_EQ(12 * kMiB, external_without_forward());
  ASSERT_TRUE(writer.value()->close().ok());
}

TEST_F(CollectionWriteBudgetTest, MmapDoesNotReserveBufferWriteAllowance) {
  // mmap still shares RocksDB instance metadata; only the Buffer writer
  // allowance is excluded by this test.
  ASSERT_EQ(0, pool_.init(16 * kMiB));
  auto writer = Collection::CreateAndOpen(path("mmap"), schema(),
                                          CollectionOptions{false, true});
  ASSERT_TRUE(writer.has_value()) << writer.error().message();
  EXPECT_EQ(0u, external_without_forward());
  ASSERT_TRUE(writer.value()->close().ok());
  auto reopened = Collection::Open(path("mmap"), {false, false});
  ASSERT_TRUE(reopened.has_value()) << reopened.error().message();
  EXPECT_EQ(0u, external_without_forward());
  ASSERT_TRUE(reopened.value()->close().ok());
}

TEST_F(CollectionWriteBudgetTest, FlushAndSegmentRotationReturnAllowance) {
  constexpr uint32_t kBufferSize = 128 * 1024;
  const size_t expected_reservation = 8 * kMiB + 4 * kBufferSize;
  auto writer = Collection::CreateAndOpen(path("roundtrip"), schema(),
                                          {false, false, kBufferSize});
  ASSERT_TRUE(writer.has_value()) << writer.error().message();
  EXPECT_EQ(0u, external_without_forward());
  for (int i = 0; i < 1050; ++i) {
    Doc doc;
    doc.set_pk(std::to_string(i));
    ASSERT_TRUE(doc.set<std::string>("payload", std::string(2048, 'x')));
    std::vector<Doc> docs{std::move(doc)};
    auto inserted = writer.value()->insert(docs);
    ASSERT_TRUE(inserted.has_value()) << inserted.error().message();
    ASSERT_EQ(1u, inserted->size());
    ASSERT_TRUE(inserted->front().ok()) << inserted->front().message();
    EXPECT_GE(external_without_forward(), expected_reservation);
    EXPECT_LT(external_without_forward(), expected_reservation * 2);
    if (i % 257 == 0) {
      ASSERT_TRUE(writer.value()->flush().ok());
      EXPECT_LT(external_without_forward(), expected_reservation);
    }
  }
  ASSERT_TRUE(writer.value()->flush().ok());
  // The active lease ends, but retained backing is shared across reopen.
  EXPECT_LT(external_without_forward(), expected_reservation);
  EXPECT_EQ(8 * kMiB,
            detail::SharedWriteBufferMemoryAllowance().reserved_bytes());
  ASSERT_TRUE(writer.value()->close().ok());
  EXPECT_EQ(0u, external_without_forward());
  auto reader = Collection::Open(path("roundtrip"), {true, false, kBufferSize});
  ASSERT_TRUE(reader.has_value()) << reader.error().message();
  EXPECT_LT(external_without_forward(), expected_reservation);
  for (int i = 0; i < 1050; ++i) {
    const auto key = std::to_string(i);
    auto docs = reader.value()->fetch({key});
    ASSERT_TRUE(docs.has_value()) << docs.error().message();
    ASSERT_EQ(1u, docs->size());
    ASSERT_NE(docs->end(), docs->find(key));
    EXPECT_EQ(std::string(2048, 'x'),
              docs->at(key)->get<std::string>("payload").value());
  }
  ASSERT_TRUE(reader.value()->close().ok());
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

TEST_F(CollectionWriteBudgetTest, ReadmissionFailureDoesNotMutateAndCanRetry) {
  const auto collection_path = path("readmission");
  auto writer = Collection::CreateAndOpen(collection_path, schema(),
                                          {false, false, kMiB});
  ASSERT_TRUE(writer.has_value()) << writer.error().message();
  auto batch = docs("existing", "original");
  auto inserted = writer.value()->insert(batch);
  ASSERT_TRUE(inserted.has_value());
  ASSERT_TRUE(inserted->front().ok());
  ASSERT_TRUE(writer.value()->flush().ok());
  EXPECT_LT(external_without_forward(), 12 * kMiB);

  auto version = VersionManager::Recovery(collection_path);
  ASSERT_TRUE(version.has_value());
  const auto meta =
      version.value()->get_current_version().writing_segment_meta();
  const auto wal_path = FileHelper::MakeWalPath(
      collection_path, meta->id(), meta->writing_forward_block()->id());
  ASSERT_FALSE(std::filesystem::exists(wal_path));
  WriteBufferReservation blocker;
  ASSERT_TRUE(blocker.acquire(12 * kMiB).ok());  // Leave less than 12 MiB.
  batch = docs("denied");
  auto rejected = writer.value()->insert(batch);
  ASSERT_FALSE(rejected.has_value());
  EXPECT_EQ(StatusCode::RESOURCE_EXHAUSTED, rejected.error().code());
  batch = docs("existing", "changed");
  rejected = writer.value()->update(batch);
  ASSERT_FALSE(rejected.has_value());
  EXPECT_EQ(StatusCode::RESOURCE_EXHAUSTED, rejected.error().code());
  rejected = writer.value()->upsert(batch);
  ASSERT_FALSE(rejected.has_value());
  EXPECT_EQ(StatusCode::RESOURCE_EXHAUSTED, rejected.error().code());
  rejected = writer.value()->delete_({"existing"});
  ASSERT_FALSE(rejected.has_value());
  EXPECT_EQ(StatusCode::RESOURCE_EXHAUSTED, rejected.error().code());
  EXPECT_EQ(StatusCode::RESOURCE_EXHAUSTED,
            writer.value()->delete_by_filter("").code());
  EXPECT_FALSE(std::filesystem::exists(wal_path));
  blocker.release();
  {
    auto fetched = writer.value()->fetch({"existing", "denied"});
    ASSERT_TRUE(fetched.has_value());
    ASSERT_NE(nullptr, fetched->at("existing"));
    EXPECT_EQ("original", fetched->at("existing")->get<std::string>("payload"));
    EXPECT_EQ(nullptr, fetched->at("denied"));
  }
  auto updated = writer.value()->update(batch);
  ASSERT_TRUE(updated.has_value());
  ASSERT_TRUE(updated->front().ok());
  EXPECT_GE(external_without_forward(), 12 * kMiB);
  ASSERT_TRUE(writer.value()->flush().ok());
  EXPECT_LT(external_without_forward(), 12 * kMiB);
  ASSERT_TRUE(writer.value()->close().ok());
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

TEST_F(CollectionWriteBudgetTest, OptimizeReturnsAllowanceAndAllowsMoreWrites) {
  auto writer = Collection::CreateAndOpen(path("optimized"), schema(),
                                          {false, false, kMiB});
  ASSERT_TRUE(writer.has_value());
  auto batch = docs("before");
  auto inserted = writer.value()->insert(batch);
  ASSERT_TRUE(inserted.has_value());
  ASSERT_TRUE(inserted->front().ok());
  EXPECT_GE(external_without_forward(), 12 * kMiB);
  // No explicit flush: optimize's seal/switch must also return the allowance.
  ASSERT_TRUE(writer.value()->optimize().ok());
  EXPECT_LT(external_without_forward(), 12 * kMiB);
  batch = docs("after");
  inserted = writer.value()->insert(batch);
  ASSERT_TRUE(inserted.has_value());
  ASSERT_TRUE(inserted->front().ok());
  EXPECT_GE(external_without_forward(), 12 * kMiB);
  ASSERT_TRUE(writer.value()->flush().ok());
  EXPECT_LT(external_without_forward(), 12 * kMiB);
  ASSERT_TRUE(writer.value()->close().ok());
}

TEST_F(CollectionWriteBudgetTest, ConcurrentOptimizeAndWritesPreserveDocs) {
  constexpr uint32_t kBufferSize = 128 * 1024;
  auto writer = Collection::CreateAndOpen(path("concurrent"), schema(),
                                          {false, false, kBufferSize});
  ASSERT_TRUE(writer.has_value());
  std::atomic<bool> started{false};
  std::atomic<bool> succeeded{true};
  std::thread producer([&] {
    started = true;
    for (int i = 0; i < 1100; ++i) {
      auto batch = docs(std::to_string(i));
      auto inserted = writer.value()->insert(batch);
      if (!inserted || !inserted->front().ok()) {
        succeeded = false;
        break;
      }
    }
  });
  while (!started.load()) std::this_thread::yield();
  for (int i = 0; i < 4; ++i) {
    EXPECT_TRUE(writer.value()->optimize().ok());
    EXPECT_TRUE(writer.value()->flush().ok());
  }
  producer.join();
  ASSERT_TRUE(succeeded.load());
  ASSERT_TRUE(writer.value()->flush().ok());
  EXPECT_LT(external_without_forward(), 8 * kMiB + 4 * kBufferSize);
  ASSERT_TRUE(writer.value()->close().ok());
  auto reader =
      Collection::Open(path("concurrent"), {true, false, kBufferSize});
  ASSERT_TRUE(reader.has_value());
  auto stats = reader.value()->stats();
  ASSERT_TRUE(stats.has_value());
  EXPECT_EQ(1100u, stats->doc_count);
  for (int i = 0; i < 1100; ++i) {
    auto fetched = reader.value()->fetch({std::to_string(i)});
    ASSERT_TRUE(fetched.has_value());
    ASSERT_NE(nullptr, fetched->at(std::to_string(i)));
  }
  ASSERT_TRUE(reader.value()->close().ok());
}

TEST_F(CollectionWriteBudgetTest, CompactedIdGapsSurviveBufferReopen) {
  constexpr uint32_t kBufferSize = 128 * 1024;
  auto writer = Collection::CreateAndOpen(path("gaps"), schema(),
                                          {false, false, kBufferSize});
  ASSERT_TRUE(writer.has_value()) << writer.error().message();
  std::vector<std::string> removed;
  for (int i = 0; i < 1100; ++i) {
    Doc doc;
    doc.set_pk(std::to_string(i));
    ASSERT_TRUE(doc.set<std::string>("payload", "value_" + std::to_string(i)));
    std::vector<Doc> docs{std::move(doc)};
    auto inserted = writer.value()->insert(docs);
    ASSERT_TRUE(inserted.has_value());
    ASSERT_TRUE(inserted->front().ok());
    if (i % 2 == 0) removed.push_back(std::to_string(i));
  }
  ASSERT_TRUE(writer.value()->delete_(removed).has_value());
  ASSERT_TRUE(writer.value()->optimize().ok());
  ASSERT_TRUE(writer.value()->close().ok());
  auto reader = Collection::Open(path("gaps"), {true, false, kBufferSize});
  ASSERT_TRUE(reader.has_value()) << reader.error().message();
  auto stats = reader.value()->stats();
  ASSERT_TRUE(stats.has_value()) << stats.error().message();
  EXPECT_EQ(550u, stats->doc_count);
  for (int i = 0; i < 1100; ++i) {
    auto key = std::to_string(i);
    auto fetched = reader.value()->fetch({key});
    ASSERT_TRUE(fetched.has_value());
    if (i % 2 == 0) {
      ASSERT_EQ(1u, fetched->size());
      EXPECT_EQ(nullptr, fetched->at(key));
    } else {
      ASSERT_EQ(1u, fetched->size());
      EXPECT_EQ("value_" + key,
                fetched->at(key)->get<std::string>("payload").value());
    }
  }
  // Exercise the merged batch-fetch path with compacted, non-contiguous IDs
  // in reverse order, spanning both a segment and a fetch microbatch boundary.
  std::vector<std::string> batch_ids;
  for (int i = 1099; i >= 0; --i) batch_ids.push_back(std::to_string(i));
  batch_ids.push_back("1");
  batch_ids.push_back("missing");
  auto batch = reader.value()->fetch(batch_ids);
  ASSERT_TRUE(batch.has_value()) << batch.error().message();
  ASSERT_EQ(1101u, batch->size());
  EXPECT_EQ(nullptr, batch->at("missing"));
  for (int i = 0; i < 1100; ++i) {
    const auto key = std::to_string(i);
    const auto &doc = batch->at(key);
    if (i % 2 == 0) {
      EXPECT_EQ(nullptr, doc);
    } else {
      ASSERT_NE(nullptr, doc);
      EXPECT_EQ("value_" + key, doc->get<std::string>("payload").value());
    }
  }
  ASSERT_TRUE(reader.value()->close().ok());
}

TEST_F(CollectionWriteBudgetTest,
       FilterDeleteAcrossSegmentsAndCompactedIdGaps) {
  for (bool mmap : {false, true}) {
    for (bool indexed : {false, true}) {
      auto fields = schema();
      ASSERT_TRUE(fields
                      .add_field(std::make_shared<FieldSchema>(
                          "bucket", DataType::INT32, false,
                          indexed ? std::make_shared<InvertIndexParams>(false)
                                  : nullptr))
                      .ok());
      const auto collection_path =
          path((std::to_string(mmap) + "_" + std::to_string(indexed)).c_str());
      auto writer = Collection::CreateAndOpen(collection_path, fields,
                                              {false, mmap, 128 * 1024});
      ASSERT_TRUE(writer) << writer.error().message();
      std::vector<std::string> removed;
      for (int i = 0; i < 2105; ++i) {
        Doc doc;
        doc.set_pk(std::to_string(i));
        ASSERT_TRUE(doc.set<std::string>("payload", "value"));
        ASSERT_TRUE(doc.set<int32_t>("bucket", i % 3));
        std::vector<Doc> batch{std::move(doc)};
        auto inserted = writer.value()->insert(batch);
        ASSERT_TRUE(inserted);
        ASSERT_TRUE(inserted->front().ok());
        if (i % 5 == 0) removed.push_back(std::to_string(i));
      }
      ASSERT_TRUE(writer.value()->delete_(removed));
      ASSERT_TRUE(writer.value()->optimize().ok());
      ASSERT_TRUE(writer.value()->close().ok());
      writer = Collection::Open(collection_path, {false, mmap, 128 * 1024});
      ASSERT_TRUE(writer) << writer.error().message();

      // Include the active writer as well as immutable compacted segments.
      Doc doc;
      doc.set_pk("new");
      ASSERT_TRUE(doc.set<std::string>("payload", "value"));
      ASSERT_TRUE(doc.set<int32_t>("bucket", 1));
      std::vector<Doc> batch{std::move(doc)};
      auto inserted = writer.value()->insert(batch);
      ASSERT_TRUE(inserted);
      ASSERT_TRUE(inserted->front().ok());
      EXPECT_FALSE(writer.value()->delete_by_filter("bucket =").ok());
      ASSERT_TRUE(writer.value()->delete_by_filter("bucket = 9").ok());
      ASSERT_TRUE(writer.value()->delete_by_filter("bucket = 1").ok());
      ASSERT_TRUE(writer.value()->delete_by_filter("bucket = 1").ok());
      ASSERT_TRUE(writer.value()->close().ok());
      writer = Collection::Open(collection_path, {false, mmap, 128 * 1024});
      ASSERT_TRUE(writer);
      size_t remaining = 0;
      for (int i = 0; i < 2105; ++i) {
        const auto key = std::to_string(i);
        auto fetched = writer.value()->fetch({key});
        ASSERT_TRUE(fetched);
        const bool present = i % 5 != 0 && i % 3 != 1;
        EXPECT_EQ(present, fetched->at(key) != nullptr) << key;
        remaining += present;
      }
      auto fetched = writer.value()->fetch({"new"});
      ASSERT_TRUE(fetched);
      EXPECT_EQ(nullptr, fetched->at("new"));
      auto stats = writer.value()->stats();
      ASSERT_TRUE(stats);
      EXPECT_EQ(remaining, stats->doc_count);
      ASSERT_TRUE(writer.value()->delete_by_filter("").ok());
      stats = writer.value()->stats();
      ASSERT_TRUE(stats);
      EXPECT_EQ(0u, stats->doc_count);
      ASSERT_TRUE(writer.value()->close().ok());
    }
  }
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
