// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <gtest/gtest.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/db/collection.h>
#include "db/common/file_helper.h"
#include "db/common/write_buffer_reservation.h"
#include "db/index/common/version_manager.h"
#include "db/index/storage/wal/wal_file.h"

namespace zvec {
namespace {

constexpr size_t kMiB = 1024 * 1024;

class CollectionWriteBudgetTest : public testing::Test {
 protected:
  void SetUp() override {
    pool_.init(64 * kMiB);
    root_ = std::filesystem::temp_directory_path() /
            ("zvec_write_budget_" +
             std::to_string(
                 std::chrono::steady_clock::now().time_since_epoch().count()));
    ASSERT_TRUE(std::filesystem::create_directory(root_));
  }
  void TearDown() override {
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
  CollectionSchema schema() const {
    CollectionSchema result("write_budget");
    EXPECT_TRUE(result
                    .add_field(std::make_shared<FieldSchema>(
                        "payload", DataType::STRING, false))
                    .ok());
    result.set_max_doc_count_per_segment(1000);
    return result;
  }
  ailego::MemoryLimitPool &pool_ = ailego::MemoryLimitPool::get_instance();
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

TEST_F(CollectionWriteBudgetTest, WritableCollectionsShareAndReturnBudget) {
  pool_.init(30 * kMiB);
  const CollectionOptions options{false, false, kMiB};
  auto first = Collection::CreateAndOpen(path("first"), schema(), options);
  ASSERT_TRUE(first.has_value()) << first.error().message();
  auto second = Collection::CreateAndOpen(path("second"), schema(), options);
  ASSERT_TRUE(second.has_value()) << second.error().message();
  EXPECT_EQ(24 * kMiB, pool_.external_used());
  auto denied = Collection::CreateAndOpen(path("third"), schema(), options);
  ASSERT_FALSE(denied.has_value());
  EXPECT_EQ(StatusCode::RESOURCE_EXHAUSTED, denied.error().code());
  EXPECT_FALSE(std::filesystem::exists(path("third")));
  ASSERT_TRUE(first.value()->close().ok());
  EXPECT_EQ(12 * kMiB, pool_.external_used());
  auto third = Collection::CreateAndOpen(path("third"), schema(), options);
  ASSERT_TRUE(third.has_value()) << third.error().message();
  EXPECT_EQ(24 * kMiB, pool_.external_used());
  ASSERT_TRUE(second.value()->close().ok());
  ASSERT_TRUE(third.value()->destroy().ok());
  EXPECT_EQ(0u, pool_.external_used());
}

TEST_F(CollectionWriteBudgetTest, ReadOnlyDoesNotReserveAndUsesPersistedMode) {
  auto created = Collection::CreateAndOpen(
      path("buffer"), schema(), CollectionOptions{false, false, kMiB});
  ASSERT_TRUE(created.has_value()) << created.error().message();
  ASSERT_TRUE(created.value()->close().ok());
  EXPECT_EQ(0u, pool_.external_used());

  // The mmap hint cannot disable budgeting for an existing Buffer collection.
  WriteBufferReservation blocker;
  ASSERT_TRUE(blocker.acquire(12 * kMiB).ok());  // 56 of 64 MiB reserved.
  auto denied = Collection::Open(path("buffer"), {false, true, kMiB});
  ASSERT_FALSE(denied.has_value());
  EXPECT_EQ(StatusCode::RESOURCE_EXHAUSTED, denied.error().code());
  auto reader = Collection::Open(path("buffer"), {true, true, kMiB});
  ASSERT_TRUE(reader.has_value()) << reader.error().message();
  EXPECT_EQ(56 * kMiB, pool_.external_used());
  ASSERT_TRUE(reader.value()->close().ok());
  blocker.release();
  auto writer = Collection::Open(path("buffer"), {false, true, kMiB});
  ASSERT_TRUE(writer.has_value()) << writer.error().message();
  EXPECT_EQ(12 * kMiB, pool_.external_used());
  ASSERT_TRUE(writer.value()->close().ok());
}

TEST_F(CollectionWriteBudgetTest, MmapDoesNotReserveBufferWriteAllowance) {
  pool_.init(kMiB);
  auto writer = Collection::CreateAndOpen(path("mmap"), schema(),
                                          CollectionOptions{false, true});
  ASSERT_TRUE(writer.has_value()) << writer.error().message();
  EXPECT_EQ(0u, pool_.external_used());
  ASSERT_TRUE(writer.value()->close().ok());
  auto reopened = Collection::Open(path("mmap"), {false, false});
  ASSERT_TRUE(reopened.has_value()) << reopened.error().message();
  EXPECT_EQ(0u, pool_.external_used());
  ASSERT_TRUE(reopened.value()->close().ok());
}

TEST_F(CollectionWriteBudgetTest, FlushAndSegmentRotationReuseAllowance) {
  constexpr uint32_t kBufferSize = 128 * 1024;
  const size_t expected_reservation = 8 * kMiB + 4 * kBufferSize;
  auto writer = Collection::CreateAndOpen(path("roundtrip"), schema(),
                                          {false, false, kBufferSize});
  ASSERT_TRUE(writer.has_value()) << writer.error().message();
  EXPECT_EQ(expected_reservation, pool_.external_used());
  for (int i = 0; i < 1050; ++i) {
    Doc doc;
    doc.set_pk(std::to_string(i));
    ASSERT_TRUE(doc.set<std::string>("payload", std::string(2048, 'x')));
    std::vector<Doc> docs{std::move(doc)};
    auto inserted = writer.value()->insert(docs);
    ASSERT_TRUE(inserted.has_value()) << inserted.error().message();
    ASSERT_EQ(1u, inserted->size());
    ASSERT_TRUE(inserted->front().ok()) << inserted->front().message();
    if (i % 257 == 0) ASSERT_TRUE(writer.value()->flush().ok());
  }
  ASSERT_TRUE(writer.value()->flush().ok());
  // Arrow output caches can also be charged; the writer allowance is still
  // present, and repeated flush/rotation must not multiply it.
  EXPECT_GE(pool_.external_used(), expected_reservation);
  EXPECT_LT(pool_.external_used(), expected_reservation * 2);
  ASSERT_TRUE(writer.value()->close().ok());
  EXPECT_EQ(0u, pool_.external_used());
  auto reader = Collection::Open(path("roundtrip"), {true, false, kBufferSize});
  ASSERT_TRUE(reader.has_value()) << reader.error().message();
  EXPECT_LT(pool_.external_used(), expected_reservation);
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
    EXPECT_EQ(StatusCode::RESOURCE_EXHAUSTED, denied.error().code());
    EXPECT_EQ(original_wal, read_wal());
    EXPECT_EQ(56 * kMiB, pool_.external_used());
  }
  blocker.release();
  auto recovered = Collection::Open(path("recovery"), {false, true, kMiB});
  ASSERT_TRUE(recovered.has_value()) << recovered.error().message();
  {
    auto docs = recovered.value()->fetch({"recovered"});
    ASSERT_TRUE(docs.has_value()) << docs.error().message();
    ASSERT_EQ(1u, docs->count("recovered"));
    EXPECT_EQ("from_wal",
              docs->at("recovered")->get<std::string>("payload").value());
  }
  ASSERT_TRUE(recovered.value()->close().ok());
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
  pool_.init(1);
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
  ASSERT_FALSE(writer.value()->delete_by_filter("").ok());
  auto fetched = writer.value()->fetch({"retained"});
  ASSERT_TRUE(fetched);
  EXPECT_NE(nullptr, fetched->at("retained"));
  pool_.init(64 * kMiB);
  ASSERT_TRUE(writer.value()->delete_by_filter("").ok());
  fetched = writer.value()->fetch({"retained"});
  ASSERT_TRUE(fetched);
  EXPECT_EQ(nullptr, fetched->at("retained"));
  ASSERT_TRUE(writer.value()->close().ok());
}

}  // namespace
}  // namespace zvec
