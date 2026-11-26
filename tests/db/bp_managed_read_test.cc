// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0

#include <arrow/io/api.h>
#include <arrow/util/thread_pool.h>
#include <algorithm>
#include <filesystem>
#include <ailego/pattern/scope_guard.h>
#include <gtest/gtest.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/db/collection.h>
#include <zvec/db/config.h>
#include <zvec/db/query.h>

namespace zvec {
namespace {

class BpManagedRead : public testing::TestWithParam<bool> {
 public:
  static void SetUpTestSuite() {
    ASSERT_TRUE(arrow::SetCpuThreadPoolCapacity(1).ok());
    ASSERT_TRUE(arrow::io::SetIOThreadPoolCapacity(1).ok());
    GlobalConfig::ConfigData config;
    config.memory_limit_bytes = 512UL << 20;
    config.query_thread_count = 1;
    config.optimize_thread_count = 1;
    auto configured = GlobalConfig::Instance().initialize(config);
    ASSERT_TRUE(configured.ok()) << configured.message();
  }
};

TEST_P(BpManagedRead, CollectionRequestsRetryAfterNativeAndRssPressure) {
  const bool multiple_segments = GetParam();
  const int segment_rows = MAX_DOC_COUNT_PER_SEGMENT_MIN_THRESHOLD;
  const std::string path = "bp_managed_read_collection";
  std::filesystem::remove_all(path);
  auto clean = ailego::ScopeGuard::Make(
      [&] { std::filesystem::remove_all(path); });
  CollectionSchema schema("bp_managed_read");
  schema.set_max_doc_count_per_segment(segment_rows);
  ASSERT_TRUE(schema.add_field(std::make_shared<FieldSchema>(
      "text", DataType::STRING, false, std::make_shared<FtsIndexParams>())).ok());
  ASSERT_TRUE(schema.add_field(
      std::make_shared<FieldSchema>("tag", DataType::INT32, false)).ok());
  for (const auto *field : {"warm", "payload", "cold"}) {
    ASSERT_TRUE(schema.add_field(
        std::make_shared<FieldSchema>(field, DataType::STRING, false)).ok());
  }
  auto created = Collection::CreateAndOpen(path, schema, {false, false});
  ASSERT_TRUE(created.has_value()) << created.error().message();
  auto collection = std::move(*created);
  const std::string payload(64 * 1024, 'z');
  std::vector<Doc> docs;
  const int rows = multiple_segments ? segment_rows + 4 : 8;
  for (int i = 0; i < rows; ++i) {
    const bool match = i < 4 || i >= rows - 4;
    auto &doc = docs.emplace_back();
    doc.set_pk("pk_" + std::to_string(i));
    ASSERT_TRUE(doc.set<std::string>("text", match ? "needle in parquet" : "other"));
    ASSERT_TRUE(doc.set<int32_t>("tag", match ? 7 : 8));
    ASSERT_TRUE(doc.set<std::string>("warm", "small"));
    ASSERT_TRUE(doc.set<std::string>("payload", match ? payload : "unused"));
    ASSERT_TRUE(doc.set<std::string>("cold", match ? payload : "unused"));
  }
  auto written = collection->insert(docs);
  ASSERT_TRUE(written.has_value()) << written.error().message();
  ASSERT_EQ(docs.size(), written->size());
  for (const auto &status : *written) {
    ASSERT_TRUE(status.ok()) << status.message();
  }
  ASSERT_TRUE(collection->flush().ok());
  ASSERT_TRUE(collection->optimize().ok());
  collection.reset();
  size_t parquet_files = 0;
  for (const auto &file : std::filesystem::recursive_directory_iterator(path)) {
    if (file.path().extension() == ".parquet") ++parquet_files;
  }
  ASSERT_GE(parquet_files, multiple_segments ? 2u : 1u);
  auto opened = Collection::Open(path, {true, false});
  ASSERT_TRUE(opened.has_value()) << opened.error().message();
  collection = std::move(*opened);

  auto scan = [&](const std::string &field, bool fts) {
    SearchQuery query;
    query.topk_ = 8;
    query.filter_ = "tag = 7";
    query.output_fields_ = std::vector<std::string>{field};
    if (fts) {
      FtsClause clause;
      clause.query_string_ = "needle";
      query.target_.field_name_ = "text";
      query.target_.clause_ = std::move(clause);
    }
    return collection->query(query);
  };
  const auto last_key = "pk_" + std::to_string(rows - 1);
  auto fetch = [&](const std::string &field) {
    return collection->fetch({"pk_0", last_key},
                             std::vector<std::string>{field}, false);
  };
  auto assert_success = [&](const std::string &field) {
    auto fetched = fetch(field);
    ASSERT_TRUE(fetched.has_value()) << fetched.error().message();
    ASSERT_EQ(2u, fetched->size());
    ASSERT_NE(nullptr, fetched->at("pk_0"));
    ASSERT_NE(nullptr, fetched->at(last_key));
    const std::string expected = field == "warm" ? "small" : payload;
    EXPECT_EQ(expected, fetched->at("pk_0")->get<std::string>(field));
    EXPECT_EQ(expected, fetched->at(last_key)->get<std::string>(field));
    for (bool fts : {false, true}) {
      SCOPED_TRACE(fts);
      auto result = scan(field, fts);
      ASSERT_TRUE(result.has_value()) << result.error().message();
      EXPECT_EQ(8u, result->size());
      for (const auto &doc : *result) {
        ASSERT_NE(nullptr, doc);
        EXPECT_EQ(expected, doc->get<std::string>(field));
      }
    }
  };

  // Initialize the managed executors while there is room for their stacks.
  ASSERT_NO_FATAL_FAILURE(assert_success("warm"));
  auto &pool = ailego::MemoryLimitPool::get_instance();
  constexpr size_t kHeadroom = 16UL << 20;
  {
    const size_t occupied = std::max(pool.used(), pool.committed());
    ASSERT_GT(pool.capacity(), occupied + kHeadroom);
    const size_t held = pool.capacity() - occupied - kHeadroom;
    ASSERT_TRUE(pool.try_charge_workspace_plan(held));
    auto release = ailego::ScopeGuard::Make(
        [&] { pool.release_workspace_plan(held); });
    ASSERT_NO_FATAL_FAILURE(assert_success("payload"));
  }

  const auto cap = pool.process_rss_limit();
  {
    auto restore = ailego::ScopeGuard::Make(
        [&] { EXPECT_TRUE(pool.configure_process_rss_limit(cap)); });
    ASSERT_TRUE(pool.configure_process_rss_limit(1));
    ASSERT_GT(pool.process_rss(), pool.process_rss_limit());
    const auto before = pool.external_used();
    auto expect_refused = [](const auto &result) {
      ASSERT_FALSE(result.has_value());
      EXPECT_EQ(StatusCode::RESOURCE_EXHAUSTED, result.error().code())
          << result.error().message();
    };
    for (const std::string field : {"cold", "payload"}) {
      SCOPED_TRACE(field);
      ASSERT_NO_FATAL_FAILURE(expect_refused(fetch(field)));
      EXPECT_LE(pool.external_used(), before);
      ASSERT_NO_FATAL_FAILURE(expect_refused(scan(field, false)));
      EXPECT_LE(pool.external_used(), before);
      ASSERT_NO_FATAL_FAILURE(expect_refused(scan(field, true)));
      EXPECT_LE(pool.external_used(), before);
    }
  }
  ASSERT_NO_FATAL_FAILURE(assert_success("payload"));
  ASSERT_NO_FATAL_FAILURE(assert_success("cold"));
}

INSTANTIATE_TEST_SUITE_P(SegmentLayouts, BpManagedRead, testing::Bool());

}  // namespace
}  // namespace zvec
