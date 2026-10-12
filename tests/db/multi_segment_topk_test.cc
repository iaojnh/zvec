// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <map>
#include <unordered_set>
#include <gtest/gtest.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/db/collection.h>
#include "db/common/global_resource.h"

namespace zvec {
namespace {
TEST(MultiSegmentTopKTest, LateFetchPreservesFieldsVectorsFiltersAndScores) {
  ailego::MemoryLimitPool::get_instance().init(128 * 1024 * 1024);
  for (bool mmap : {false, true}) {
    SCOPED_TRACE(mmap);
    auto root =
        std::filesystem::temp_directory_path() /
        ("zvec_topk_" +
         std::to_string(
             std::chrono::steady_clock::now().time_since_epoch().count()));
    struct Cleanup {
      std::filesystem::path path;
      ~Cleanup() {
        std::filesystem::remove_all(path);
      }
    } cleanup{root};
    auto flat_l2 = std::make_shared<FlatIndexParams>(MetricType::L2);
    auto flat_ip = std::make_shared<FlatIndexParams>(MetricType::IP);
    CollectionSchema schema(
        "topk",
        {std::make_shared<FieldSchema>("payload", DataType::STRING, true),
         std::make_shared<FieldSchema>(
             "bucket", DataType::INT32, false,
             std::make_shared<InvertIndexParams>(false)),
         std::make_shared<FieldSchema>("dense", DataType::VECTOR_FP32, 4, false,
                                       flat_l2),
         std::make_shared<FieldSchema>("ip", DataType::VECTOR_FP32, 4, false,
                                       flat_ip),
         std::make_shared<FieldSchema>("sparse", DataType::SPARSE_VECTOR_FP32,
                                       0, false, flat_ip)});
    schema.set_max_doc_count_per_segment(1000);
    auto collection = Collection::CreateAndOpen(root.string(), schema,
                                                {false, mmap, 128 * 1024});
    ASSERT_TRUE(collection) << collection.error().message();
    std::vector<std::string> deleted;
    for (int start = 0; start < 2105; start += 128) {
      std::vector<Doc> docs;
      for (int i = start; i < std::min(start + 128, 2105); ++i) {
        Doc doc;
        doc.set_pk(std::to_string(i));
        if (i % 7)
          ASSERT_TRUE(doc.set<std::string>(
              "payload", std::string(512, 'x') + std::to_string(i)));
        ASSERT_TRUE(doc.set<int32_t>("bucket", i % 11));
        float x = float(i % 1009) / 10.0f;
        ASSERT_TRUE(doc.set<std::vector<float>>("dense", {x, 1, 0, 0}));
        ASSERT_TRUE(doc.set<std::vector<float>>("ip", {x, 1, 0, 0}));
        using Sparse = std::pair<std::vector<uint32_t>, std::vector<float>>;
        ASSERT_TRUE(doc.set<Sparse>("sparse", {{0, 3}, {x, 1}}));
        docs.push_back(std::move(doc));
        if (i % 17 == 0) deleted.push_back(std::to_string(i));
      }
      auto inserted = collection.value()->insert(docs);
      ASSERT_TRUE(inserted) << inserted.error().message();
      for (const auto &status : *inserted)
        ASSERT_TRUE(status.ok()) << status.message();
    }
    auto removed = collection.value()->delete_(deleted);
    ASSERT_TRUE(removed);
    for (const auto &status : *removed) ASSERT_TRUE(status.ok());
    ASSERT_TRUE(collection.value()->flush().ok());
    for (const std::string field : {"dense", "ip", "sparse"}) {
      for (int filter : {0, 1, 2, 3}) {
        for (int topk : {1, 7, 257, 3000}) {
          SCOPED_TRACE(field + ":" + std::to_string(filter) + ":" +
                       std::to_string(topk));
          SearchQuery query;
          query.topk_ = topk;
          query.target_.field_name_ = field;
          if (field == "sparse") {
            const uint32_t index = 0;
            const float value = 1.0f;
            query.target_.set_sparse_vector(
                std::string(reinterpret_cast<const char *>(&index),
                            sizeof(index)),
                std::string(reinterpret_cast<const char *>(&value),
                            sizeof(value)));
          } else {
            const std::vector<float> target{1, 0, 0, 0};
            query.target_.set_vector(
                std::string(reinterpret_cast<const char *>(target.data()),
                            4 * sizeof(float)));
          }
          const std::string filters[]{"", "bucket >= 5", "bucket < 0",
                                      "payload IS NULL"};
          query.filter_ = filters[filter];
          query.output_fields_ = {"payload", "bucket", "dense", "sparse"};
          query.include_vector_ = true;
          query.include_doc_id_ = true;
          std::vector<float> expected;
          for (int i = 0; i < 2105; ++i) {
            if (i % 17 == 0 || (filter == 1 && i % 11 < 5) || filter == 2 ||
                (filter == 3 && i % 7 != 0))
              continue;
            float x = float(i % 1009) / 10.0f;
            expected.push_back(field == "dense" ? (x - 1) * (x - 1) + 1 : x);
          }
          if (field == "dense")
            std::sort(expected.begin(), expected.end());
          else
            std::sort(expected.begin(), expected.end(), std::greater<float>{});
          expected.resize(std::min(expected.size(), static_cast<size_t>(topk)));
          auto result = collection.value()->query(query);
          ASSERT_TRUE(result) << result.error().message();
          ASSERT_EQ(expected.size(), result->size());
          std::unordered_set<std::string> seen;
          for (size_t j = 0; j < result->size(); ++j) {
            const auto &doc = *(*result)[j];
            ASSERT_TRUE(seen.insert(doc.pk()).second);
            const int i = std::stoi(doc.pk());
            EXPECT_NE(0, i % 17);
            EXPECT_FLOAT_EQ(expected[j], doc.score());
            EXPECT_EQ(i % 11, *doc.get<int32_t>("bucket"));
            if (i % 7)
              EXPECT_EQ(std::string(512, 'x') + std::to_string(i),
                        *doc.get<std::string>("payload"));
            else
              EXPECT_FALSE(doc.get<std::string>("payload"));
            auto dense = doc.get<std::vector<float>>("dense");
            ASSERT_TRUE(dense);
            EXPECT_EQ((std::vector<float>{float(i % 1009) / 10.0f, 1, 0, 0}),
                      *dense);
            using Sparse = std::pair<std::vector<uint32_t>, std::vector<float>>;
            auto sparse = doc.get<Sparse>("sparse");
            ASSERT_TRUE(sparse);
            EXPECT_EQ((std::vector<uint32_t>{0, 3}), sparse->first);
          }
        }
      }
    }
    for (const std::string field : {"dense", "ip", "sparse"}) {
      for (uint32_t group_count : {1u, 3u, 20u}) {
        for (uint32_t per_group : {1u, 7u}) {
          SCOPED_TRACE(field + ":groups=" + std::to_string(group_count) +
                       ":per_group=" + std::to_string(per_group));
          GroupByVectorQuery query;
          query.target_.field_name_ = field;
          const float value = 1;
          const uint32_t index = 0;
          if (field == "sparse")
            query.target_.set_sparse_vector(
                std::string(reinterpret_cast<const char *>(&index), 4),
                std::string(reinterpret_cast<const char *>(&value), 4));
          else {
            const float target[]{1, 0, 0, 0};
            query.target_.set_vector(std::string(
                reinterpret_cast<const char *>(target), sizeof(target)));
          }
          query.group_by_field_name_ = "bucket";
          query.group_count_ = group_count;
          query.topk_per_group_ = per_group;
          query.filter_ = "bucket >= 5";
          query.output_fields_ = {"payload", "bucket", "dense"};
          query.include_vector_ = true;
          std::map<int, std::vector<float>> expected;
          for (int i = 0; i < 2105; ++i) {
            if (i % 17 == 0 || i % 11 < 5) continue;
            float x = float(i % 1009) / 10.0f;
            expected[i % 11].push_back(field == "dense" ? (x - 1) * (x - 1) + 1
                                                        : x);
          }
          auto better = [&](float a, float b) {
            return field == "dense" ? a < b : a > b;
          };
          std::vector<float> best;
          for (auto &entry : expected) {
            std::sort(entry.second.begin(), entry.second.end(), better);
            best.push_back(entry.second.front());
          }
          std::sort(best.begin(), best.end(), better);
          auto result = collection.value()->group_by_query(query);
          ASSERT_TRUE(result) << result.error().message();
          ASSERT_EQ(std::min(size_t(group_count), expected.size()),
                    result->size());
          for (size_t g = 0; g < result->size(); ++g) {
            const auto &group = (*result)[g];
            const int bucket = std::stoi(group.group_by_value_);
            ASSERT_EQ(per_group, group.docs_.size());
            EXPECT_FLOAT_EQ(best[g], group.docs_.front().score());
            for (size_t i = 0; i < group.docs_.size(); ++i) {
              const auto &doc = group.docs_[i];
              EXPECT_EQ(bucket, *doc.get<int32_t>("bucket"));
              // Index-level group_count truncates candidates before the DB
              // merge. Only compare all runner-ups to a global exact oracle
              // when every group is retained; compare the best score above
              // for small group_count (the pre-existing query semantics).
              if (group_count >= expected.size())
                EXPECT_FLOAT_EQ(expected.at(bucket)[i], doc.score());
              if (i)
                EXPECT_FALSE(better(doc.score(), group.docs_[i - 1].score()));
              const int id = std::stoi(doc.pk());
              const float x = float(id % 1009) / 10.0f;
              EXPECT_FLOAT_EQ(field == "dense" ? (x - 1) * (x - 1) + 1 : x,
                              doc.score());
              EXPECT_EQ(bucket, id % 11);
              auto dense = doc.get<std::vector<float>>("dense");
              ASSERT_TRUE(dense);
              EXPECT_EQ((std::vector<float>{x, 1, 0, 0}), *dense);
              if (id % 7) {
                auto payload = doc.get<std::string>("payload");
                ASSERT_TRUE(payload);
                EXPECT_EQ(std::string(512, 'x') + std::to_string(id), *payload);
              } else
                EXPECT_FALSE(doc.get<std::string>("payload"));
              EXPECT_NE(0, id % 17);
            }
          }
        }
      }
    }
    // Nested callers may already occupy every query-pool worker. The merge
    // must not enqueue more work on that same pool and synchronously wait.
    auto *pool = GlobalResource::Instance().query_thread_pool();
    auto group = pool->make_group();
    for (size_t worker = 0; worker < pool->count(); ++worker) {
      group->execute([&] {
        SearchQuery query;
        query.topk_ = 10;
        query.target_.field_name_ = "dense";
        query.target_.set_vector(std::string(4 * sizeof(float), '\0'));
        query.filter_ = "payload IS NULL";
        query.output_fields_ = std::vector<std::string>{};
        query.include_vector_ = false;
        auto result = collection.value()->query(query);
        ASSERT_TRUE(result) << result.error().message();
        EXPECT_EQ(10u, result->size());
      });
    }
    group->wait_finish();
    ASSERT_TRUE(collection.value()->close().ok());
  }
}
}  // namespace
}  // namespace zvec
