// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#if defined(__linux__)
#include <sys/resource.h>
#include <csignal>
#endif
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <ailego/io/file_lock.h>
#include <gtest/gtest.h>
#include <zvec/ailego/io/file.h>
#include <zvec/ailego/utility/file_helper.h>
#include <zvec/db/collection.h>
#include <zvec/db/config.h>
#include "db/common/file_helper.h"
#include "db/common/global_resource.h"
#include "db/index/common/id_map.h"
#include "db/index/common/version_manager.h"
#include "db/index/storage/wal/wal_file.h"
#include "../ailego/buffer/pool_payload_budget.h"

namespace zvec {
namespace {

constexpr size_t kMiB = 1024 * 1024;

#if GTEST_HAS_DEATH_TEST
TEST(CollectionStartupDeathTest, MissingManifestDoesNotFreezeConfiguration) {
  const auto previous_style = ::testing::FLAGS_gtest_death_test_style;
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  EXPECT_EXIT(
      {
        namespace fs = std::filesystem;
        const auto nonce =
            std::chrono::steady_clock::now().time_since_epoch().count();
        const auto path = fs::temp_directory_path() /
                          ("zvec_no_manifest_" + std::to_string(nonce));
        if (!fs::create_directory(path)) std::_Exit(1);
        std::ofstream(path / "LOCK").close();
        auto opened = Collection::Open(path.u8string(), {false, false, kMiB});
        fs::remove_all(path);
        if (opened || GlobalResource::Instance().initialized()) std::_Exit(2);
        GlobalConfig::ConfigData config;
        config.query_thread_count = config.query_thread_count == 1 ? 2 : 1;
        std::_Exit(GlobalConfig::Instance().initialize(config).ok() ? 0 : 3);
      },
      ::testing::ExitedWithCode(0), "");
  ::testing::FLAGS_gtest_death_test_style = previous_style;
}
#endif

class CollectionPublicationTest : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    // The BP write-buffer reservation includes headroom for the large WAL
    // documents exercised by this suite.
    ASSERT_EQ(0, ailego::PoolPayloadBudget::get_instance().init(256 * kMiB));
  }

  void SetUp() override {
    root_ = std::filesystem::temp_directory_path() /
            ("zvec_publication_" +
             std::to_string(
                 std::chrono::steady_clock::now().time_since_epoch().count()));
    ASSERT_TRUE(std::filesystem::create_directory(root_));
  }

  void TearDown() override {
    std::filesystem::remove_all(root_);
  }
  std::string path(const char *name) const {
    return (root_ / name).u8string();
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
  std::filesystem::path root_;
};

TEST_F(CollectionPublicationTest, PublishedCreateCanRetryFinalAttachment) {
  if (!ailego::StagedPath::supported()) GTEST_SKIP();
  const auto collection_path = path("published_create");
  const auto marker =
      ailego::FileHelper::PathJoin(collection_path, ".zvec-create-pending");
  const CollectionOptions options{false, true, kMiB};
  {
    auto created =
        Collection::CreateAndOpen(collection_path, schema(), options);
    ASSERT_TRUE(created.has_value()) << created.error().message();
    ASSERT_TRUE(created.value()->close().ok());
    ailego::File marker_file;
    ASSERT_TRUE(marker_file.create_exclusive(marker.c_str()));
  }
  const CollectionSchema other("wrong_schema");
  auto rejected = Collection::CreateAndOpen(collection_path, other, options);
  ASSERT_FALSE(rejected.has_value());
  EXPECT_EQ(StatusCode::INVALID_ARGUMENT, rejected.error().code());
  EXPECT_TRUE(ailego::File::IsExist(marker));

  auto retried = Collection::CreateAndOpen(collection_path, schema(), options);
  ASSERT_TRUE(retried.has_value()) << retried.error().message();
  EXPECT_FALSE(ailego::File::IsExist(marker));
  auto competing = Collection::Open(collection_path, options);
  EXPECT_FALSE(competing.has_value());
  EXPECT_TRUE(retried.value()->close().ok());
}

TEST_F(CollectionPublicationTest, WritableOpenCompletesPublishedCreation) {
  const auto collection_path = path("interrupted_create");
  const auto marker =
      ailego::FileHelper::PathJoin(collection_path, ".zvec-create-pending");
  const CollectionOptions options{false, true, kMiB};
  auto created = Collection::CreateAndOpen(collection_path, schema(), options);
  ASSERT_TRUE(created.has_value()) << created.error().message();
  ASSERT_TRUE(created.value()->close().ok());

  const std::string token = ".zvec-create-123-1";
  ASSERT_TRUE(std::filesystem::create_directory(root_ / token));
  {
    ailego::File lease;
    const auto owner_path = (root_ / token / "owner").string();
    ASSERT_TRUE(lease.create_exclusive(owner_path.c_str()));
    ASSERT_TRUE(ailego::FileLock::TryLock(lease.native_handle()));
    ailego::File marker_file;
    ASSERT_TRUE(marker_file.create_exclusive(marker.c_str()));
    ASSERT_EQ(token.size(), marker_file.write(token.data(), token.size()));
    marker_file.close();

    auto early_open = Collection::Open(collection_path, options);
    ASSERT_FALSE(early_open.has_value());
    EXPECT_EQ(StatusCode::FAILED_PRECONDITION, early_open.error().code());
    auto early_retry =
        Collection::CreateAndOpen(collection_path, schema(), options);
    ASSERT_FALSE(early_retry.has_value());
    EXPECT_EQ(StatusCode::FAILED_PRECONDITION, early_retry.error().code());
  }

  auto reopened = Collection::Open(collection_path, options);
  ASSERT_TRUE(reopened.has_value()) << reopened.error().message();
  EXPECT_FALSE(ailego::File::IsExist(marker));
  auto written = docs("published");
  auto inserted = reopened.value()->insert(written);
  ASSERT_TRUE(inserted.has_value() && inserted->front().ok());
  ASSERT_TRUE(reopened.value()->close().ok());
  EXPECT_FALSE(Collection::CreateAndOpen(collection_path, schema(), options)
                   .has_value());
}

#if defined(__linux__) && GTEST_HAS_DEATH_TEST
TEST_F(CollectionPublicationTest, FailedInitializationLeavesPathAvailable) {
  const auto previous_style = ::testing::FLAGS_gtest_death_test_style;
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  EXPECT_EXIT(
      {
        const auto collection_path = path("short_write");
        const CollectionOptions options(false, true, kMiB);
        struct rlimit original;
        if (getrlimit(RLIMIT_FSIZE, &original) != 0) std::_Exit(1);
        auto limited = original;
        limited.rlim_cur = 0;
        const auto previous_signal = std::signal(SIGXFSZ, SIG_IGN);
        if (previous_signal == SIG_ERR ||
            setrlimit(RLIMIT_FSIZE, &limited) != 0)
          std::_Exit(2);
        auto rejected =
            Collection::CreateAndOpen(collection_path, schema(), options);
        if (setrlimit(RLIMIT_FSIZE, &original) != 0) std::_Exit(3);
        std::signal(SIGXFSZ, previous_signal);
        if (rejected || std::filesystem::exists(collection_path)) std::_Exit(4);
        auto retry =
            Collection::CreateAndOpen(collection_path, schema(), options);
        if (!retry || !retry.value()->close().ok()) std::_Exit(5);
        std::filesystem::remove_all(root_);
        std::_Exit(0);
      },
      ::testing::ExitedWithCode(0), "");
  ::testing::FLAGS_gtest_death_test_style = previous_style;
}
#endif

TEST_F(CollectionPublicationTest, CreatorAndOpenerUseThePublishedPath) {
  for (bool mmap : {false, true}) {
    SCOPED_TRACE(mmap);
    const std::string collection_path = path(mmap ? "mmap" : "bp") + "/child";
    const CollectionOptions options{false, mmap, kMiB};
    auto created =
        Collection::CreateAndOpen(collection_path, schema(), options);
    ASSERT_TRUE(created.has_value()) << created.error().message();
    ASSERT_EQ(collection_path, created.value()->path().value());
    auto written = docs("published");
    auto inserted = created.value()->insert(written);
    ASSERT_TRUE(inserted.has_value() && inserted->front().ok());
    ASSERT_TRUE(created.value()->close().ok());
    auto reopened = Collection::Open(collection_path, options);
    ASSERT_TRUE(reopened.has_value()) << reopened.error().message();
    auto fetched = reopened.value()->fetch({"published"});
    ASSERT_TRUE(fetched.has_value()) << fetched.error().message();
    EXPECT_EQ(1u, fetched->size());
    EXPECT_TRUE(reopened.value()->close().ok());
  }
}

TEST_F(CollectionPublicationTest, AbandonedStageKeepsCompetingCollection) {
  if (!ailego::StagedPath::supported()) GTEST_SKIP();
  const auto target = path("published_winner");
  std::filesystem::path private_directory;
  {
    ailego::StagedPath stage;
    ASSERT_TRUE(stage.prepare(target));
    private_directory = std::filesystem::u8path(stage.path()).parent_path();
    ASSERT_TRUE(std::filesystem::create_directory(stage.path()));
    std::ofstream(stage.path() + "/unpublished") << "private";

    auto winner = Collection::CreateAndOpen(target, schema(), {false, true, kMiB});
    ASSERT_TRUE(winner.has_value()) << winner.error().message();
    auto seed = docs("winner");
    auto result = winner.value()->insert(seed);
    ASSERT_TRUE(result.has_value() && result->front().ok());
    ASSERT_TRUE(winner.value()->close().ok());
    EXPECT_FALSE(stage.publish());
  }
  EXPECT_FALSE(std::filesystem::exists(private_directory));
  auto reopened = Collection::Open(target, {false, true, kMiB});
  ASSERT_TRUE(reopened.has_value()) << reopened.error().message();
  auto docs = reopened.value()->fetch({"winner"});
  ASSERT_TRUE(docs.has_value());
  EXPECT_EQ(1u, docs->size());
  EXPECT_TRUE(reopened.value()->close().ok());
}

TEST_F(CollectionPublicationTest, LargeDocumentsSupportWritesAndReopen) {
  for (bool mmap : {false, true}) {
    SCOPED_TRACE(mmap);
    const auto collection_path = path(mmap ? "large_mmap" : "large_bp");
    const CollectionOptions options{false, mmap, 32 * kMiB};
    auto created =
        Collection::CreateAndOpen(collection_path, schema(), options);
    ASSERT_TRUE(created.has_value()) << created.error().message();
    for (size_t bytes : {4 * kMiB - 1024, 4 * kMiB + 1024}) {
      SCOPED_TRACE(bytes);
      auto written = docs("large", std::string(bytes, 'a'));
      // Check the encoded WAL size, including document metadata.
      EXPECT_EQ(bytes > 4 * kMiB,
                written.front().serialize().size() > 4 * kMiB);
      auto result = created.value()->insert(written);
      ASSERT_TRUE(result.has_value());
      ASSERT_TRUE(result->front().ok()) << result->front().message();
      written = docs("large", std::string(bytes, 'b'));
      result = created.value()->update(written);
      ASSERT_TRUE(result.has_value());
      ASSERT_TRUE(result->front().ok()) << result->front().message();
      written = docs("large", std::string(bytes, 'c'));
      result = created.value()->upsert(written);
      ASSERT_TRUE(result.has_value());
      ASSERT_TRUE(result->front().ok()) << result->front().message();
      ASSERT_TRUE(created.value()->close().ok());
      created = Collection::Open(collection_path, options);
      ASSERT_TRUE(created.has_value()) << created.error().message();
      auto fetched = created.value()->fetch({"large"});
      ASSERT_TRUE(fetched.has_value());
      ASSERT_EQ(1u, fetched->size());
      EXPECT_EQ(std::string(bytes, 'c'),
                fetched->at("large")->get<std::string>("payload").value());
      auto removed = created.value()->delete_({"large"});
      ASSERT_TRUE(removed.has_value());
      ASSERT_TRUE(removed->front().ok());
    }
    EXPECT_TRUE(created.value()->close().ok());
  }
}

TEST_F(CollectionPublicationTest, LargeWalDocumentCanRecoverAndAppend) {
  for (bool mmap : {false, true}) {
    SCOPED_TRACE(mmap);
    const auto collection_path = path(mmap ? "recover_mmap" : "recover_bp");
    const CollectionOptions options{false, mmap, 32 * kMiB};
    auto created =
        Collection::CreateAndOpen(collection_path, schema(), options);
    ASSERT_TRUE(created.has_value()) << created.error().message();
    ASSERT_TRUE(created.value()->close().ok());
    auto versions = VersionManager::Recovery(collection_path);
    ASSERT_TRUE(versions.has_value()) << versions.error().message();
    const auto meta = (*versions)->get_current_version().writing_segment_meta();
    ASSERT_NE(nullptr, meta);
    ASSERT_TRUE(meta->writing_forward_block().has_value());
    const auto wal_path = FileHelper::MakeWalPath(
        collection_path, meta->id(), meta->writing_forward_block()->id());
    auto written = docs("recovered", std::string(5 * kMiB, 'r'));
    written.front().set_operator(Operator::INSERT);
    const auto encoded = written.front().serialize();
    WalFilePtr wal;
    ASSERT_EQ(0, WalFile::CreateAndOpen(wal_path, {0, true}, &wal));
    ASSERT_EQ(0, wal->append(std::string(encoded.begin(), encoded.end())));
    ASSERT_EQ(0, wal->close());

    auto recovered = Collection::Open(collection_path, options);
    ASSERT_TRUE(recovered.has_value()) << recovered.error().message();
    auto next = docs("next");
    auto inserted = recovered.value()->insert(next);
    ASSERT_TRUE(inserted.has_value());
    ASSERT_TRUE(inserted->front().ok()) << inserted->front().message();
    ASSERT_TRUE(recovered.value()->close().ok());
    auto reopened = Collection::Open(collection_path, options);
    ASSERT_TRUE(reopened.has_value()) << reopened.error().message();
    auto fetched = reopened.value()->fetch({"recovered", "next"});
    ASSERT_TRUE(fetched.has_value());
    ASSERT_EQ(2u, fetched->size());
    EXPECT_EQ(std::string(5 * kMiB, 'r'),
              fetched->at("recovered")->get<std::string>("payload").value());
    EXPECT_TRUE(reopened.value()->close().ok());
  }
}

TEST_F(CollectionPublicationTest, CollectionRecoveryRepairsTailBeforeAppend) {
  for (bool mmap : {false, true}) {
    SCOPED_TRACE(mmap);
    for (bool corrupt : {false, true}) {
      SCOPED_TRACE(corrupt);
      const std::string name = std::string(mmap ? "mmap_" : "bp_") +
                               (corrupt ? "bad_crc" : "torn_payload");
      const auto collection_path = path(name.c_str());
      const CollectionOptions options{false, mmap, kMiB};
      auto created =
          Collection::CreateAndOpen(collection_path, schema(), options);
      ASSERT_TRUE(created.has_value()) << created.error().message();
      ASSERT_TRUE(created.value()->close().ok());
      auto versions = VersionManager::Recovery(collection_path);
      ASSERT_TRUE(versions.has_value()) << versions.error().message();
      const auto meta = (*versions)->get_current_version().writing_segment_meta();
      ASSERT_NE(nullptr, meta);
      ASSERT_TRUE(meta->writing_forward_block().has_value());
      const auto wal_path = FileHelper::MakeWalPath(
          collection_path, meta->id(), meta->writing_forward_block()->id());

      WalFilePtr writer;
      ASSERT_EQ(0, WalFile::CreateAndOpen(wal_path, {0, true}, &writer));
      auto encoded = docs("first").front().serialize();
      ASSERT_EQ(0, writer->append(std::string(encoded.begin(), encoded.end())));
      ASSERT_EQ(0, writer->flush());
      const auto complete = std::filesystem::file_size(wal_path);
      if (!corrupt) {
        ASSERT_EQ(0, writer->append("unfinished"));
      }
      ASSERT_EQ(0, writer->close());
      if (corrupt) {
        std::fstream wal(wal_path,
                         std::ios::in | std::ios::out | std::ios::binary);
        ASSERT_TRUE(wal.good());
        wal.seekg(-1, std::ios::end);
        char byte = 0;
        ASSERT_TRUE(wal.get(byte));
        wal.seekp(-1, std::ios::end);
        wal.put(byte ^ 1);
        wal.close();
        ASSERT_TRUE(wal.good());
        auto refused = Collection::Open(collection_path, options);
        ASSERT_FALSE(refused.has_value());
        EXPECT_EQ(StatusCode::INTERNAL_ERROR, refused.error().code());
        EXPECT_EQ(complete, std::filesystem::file_size(wal_path));
        continue;
      }
      std::filesystem::resize_file(wal_path, complete + 9);
      auto opened = Collection::Open(collection_path, options);
      ASSERT_TRUE(opened.has_value()) << opened.error().message();
      ASSERT_EQ(complete, std::filesystem::file_size(wal_path));
      auto next = docs("second");
      auto inserted = opened.value()->insert(next);
      ASSERT_TRUE(inserted.has_value() && inserted->front().ok());
      WalFilePtr reader;
      ASSERT_EQ(0, WalFile::CreateAndOpen(wal_path, {0, false, true}, &reader));
      ASSERT_EQ(0, reader->prepare_for_read());
      for (const char *id : {"first", "second"}) {
        const auto record = reader->next();
        ASSERT_FALSE(record.empty());
        const auto doc = Doc::deserialize(
            reinterpret_cast<const uint8_t *>(record.data()), record.size());
        ASSERT_NE(nullptr, doc);
        EXPECT_EQ(id, doc->pk_ref());
      }
      EXPECT_TRUE(reader->next().empty());
      EXPECT_TRUE(reader->read_status().ok());
      ASSERT_EQ(0, reader->close());
      EXPECT_TRUE(opened.value()->close().ok());
    }
  }
}

TEST_F(CollectionPublicationTest, FailedWalReplayPreservesIdMapAndRetries) {
  const auto contains = [](const IDMap &ids, const char *key,
                            uint64_t *doc_id = nullptr) {
    const auto found = ids.lookup(key, doc_id);
    EXPECT_TRUE(found.has_value());
    return found.has_value() && *found;
  };
  for (const bool mmap : {false, true}) {
    for (const int failure : {0, 2}) {
      SCOPED_TRACE(mmap);
      SCOPED_TRACE(failure);
      const auto collection_path =
          path((std::string(mmap ? "retry_mmap_" : "retry_bp_") +
                std::to_string(failure))
                   .c_str());
      const CollectionOptions options{false, mmap, kMiB};
      auto created =
          Collection::CreateAndOpen(collection_path, schema(), options);
      ASSERT_TRUE(created.has_value()) << created.error().message();
      auto seed = docs("stable", "original");
      auto inserted = created.value()->insert(seed);
      ASSERT_TRUE(inserted.has_value() && inserted->front().ok());
      seed = docs("doomed");
      inserted = created.value()->insert(seed);
      ASSERT_TRUE(inserted.has_value() && inserted->front().ok());
      ASSERT_TRUE(created.value()->close().ok());
      auto versions = VersionManager::Recovery(collection_path);
      ASSERT_TRUE(versions.has_value());
      const auto version = (*versions)->get_current_version();
      const auto meta = version.writing_segment_meta();
      ASSERT_NE(nullptr, meta);
      const auto wal_path = FileHelper::MakeWalPath(
          collection_path, meta->id(), meta->writing_forward_block()->id());
      const auto idmap_path = FileHelper::MakeFilePath(
          collection_path, FileID::ID_FILE, version.id_map_path_suffix());
      uint64_t stable_id = 0, doomed_id = 0;
      {
        auto ids =
            IDMap::CreateAndOpen("write_budget", idmap_path, false, true);
        ASSERT_NE(nullptr, ids);
        ASSERT_TRUE(contains(*ids, "stable", &stable_id));
        ASSERT_TRUE(contains(*ids, "doomed", &doomed_id));
        EXPECT_FALSE(contains(*ids, "first"));
      }
      const auto encoded = [&](const char *pk, const char *value,
                               Operator operation, uint64_t old_id = 0) {
        auto doc = docs(pk, value).front();
        doc.set_operator(operation);
        doc.set_doc_id(old_id);
        const auto bytes = doc.serialize();
        return std::string(bytes.begin(), bytes.end());
      };
      const std::vector<std::string> prefix{
          encoded("first", "one", Operator::UPSERT),
          encoded("first", "latest", Operator::UPSERT),
          encoded("stable", "updated", Operator::UPDATE, stable_id),
          encoded("doomed", "", Operator::DELETE, doomed_id)};
      const auto last = encoded("last", "value", Operator::INSERT);
      const auto write_wal = [&](bool corrupt) {
        ailego::File::Delete(wal_path);
        WalFilePtr writer;
        ASSERT_EQ(0, WalFile::CreateAndOpen(wal_path, {0, true}, &writer));
        for (const auto &record : prefix)
          ASSERT_EQ(0, writer->append(std::string(record)));
        std::string tail = last;
        if (corrupt && failure == 2)
          tail = encoded("last", "value", static_cast<Operator>(99));
        ASSERT_EQ(0, writer->append(std::move(tail)));
        ASSERT_EQ(0, writer->close());
        if (corrupt && failure == 0) {
          std::fstream stream(wal_path,
                              std::ios::in | std::ios::out | std::ios::binary);
          stream.seekg(-1, std::ios::end);
          char byte = 0;
          ASSERT_TRUE(stream.get(byte));
          stream.seekp(-1, std::ios::end);
          stream.put(byte ^ 1);
          stream.close();
          ASSERT_TRUE(stream.good());
        }
      };
      ASSERT_NO_FATAL_FAILURE(write_wal(true));
      for (int attempt = 0; attempt < 2; ++attempt) {
        auto refused = Collection::Open(collection_path, options);
        ASSERT_FALSE(refused.has_value());
        auto ids =
            IDMap::CreateAndOpen("write_budget", idmap_path, false, true);
        ASSERT_NE(nullptr, ids);
        uint64_t actual = 0;
        EXPECT_FALSE(contains(*ids, "first"));
        ASSERT_TRUE(contains(*ids, "stable", &actual));
        EXPECT_EQ(stable_id, actual);
        ASSERT_TRUE(contains(*ids, "doomed", &actual));
        EXPECT_EQ(doomed_id, actual);
      }
      ASSERT_NO_FATAL_FAILURE(write_wal(false));
      for (int attempt = 0; attempt < 2; ++attempt) {
        auto opened = Collection::Open(collection_path, options);
        ASSERT_TRUE(opened.has_value()) << opened.error().message();
        auto fetched =
            opened.value()->fetch({"first", "stable", "doomed", "last"});
        ASSERT_TRUE(fetched.has_value());
        ASSERT_NE(nullptr, fetched->at("first"));
        ASSERT_NE(nullptr, fetched->at("stable"));
        ASSERT_NE(nullptr, fetched->at("last"));
        EXPECT_EQ("latest",
                  fetched->at("first")->get<std::string>("payload").value());
        EXPECT_EQ("updated",
                  fetched->at("stable")->get<std::string>("payload").value());
        EXPECT_EQ(nullptr, fetched->at("doomed"));
        const auto stats = opened.value()->stats();
        ASSERT_TRUE(stats.has_value());
        EXPECT_EQ(3u, stats->doc_count);
        ASSERT_TRUE(opened.value()->close().ok());
      }
    }
  }
}

}  // namespace
}  // namespace zvec
