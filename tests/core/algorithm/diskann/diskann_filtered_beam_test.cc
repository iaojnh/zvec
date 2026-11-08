// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0

#include <filesystem>
#include <random>
#include <tuple>
#include <ailego/pattern/scope_guard.h>
#include <gtest/gtest.h>
#include <zvec/core/framework/index_framework.h>
#include "diskann_holder.h"
#include "diskann_params.h"
#include "../../../ailego/buffer/pool_payload_budget.h"

using namespace zvec::core;

class DiskAnnFilteredBeamTest
    : public testing::TestWithParam<std::tuple<bool, bool>> {
 protected:
  void SetUp() override {
    auto &pool = zvec::ailego::PoolPayloadBudget::get_instance();
    previous_capacity_ = pool.capacity();
    ASSERT_EQ(0u, pool.used());
    ASSERT_EQ(0, pool.init(64UL << 20));
  }
  void TearDown() override {
    auto &pool = zvec::ailego::PoolPayloadBudget::get_instance();
    EXPECT_EQ(0u, pool.used());
    EXPECT_EQ(0, pool.init(previous_capacity_));
  }
  size_t previous_capacity_{0};
};

TEST_P(DiskAnnFilteredBeamTest, DeletedCopiesLeaveRoomForTheLiveVector) {
  constexpr uint32_t dimension = 8;
  constexpr uint32_t count = 1024;
  constexpr uint32_t copies = 64;
  constexpr uint32_t target = count - 1;
  const std::string path = "diskann_filtered_beam_test.index";
  auto cleanup = zvec::ailego::ScopeGuard::Make([&] {
    std::error_code error;
    std::filesystem::remove(path, error);
  });
  IndexMeta meta(IndexMeta::DT_FP32, dimension);
  meta.set_metric("SquaredEuclidean", 0, zvec::ailego::Params());
  auto holder =
      std::make_shared<MultiPassIndexHolder<IndexMeta::DT_FP32>>(dimension);
  std::mt19937 generator(2026);
  std::normal_distribution<float> normal;
  zvec::ailego::NumericalVector<float> query(dimension);
  for (uint32_t d = 0; d < dimension; ++d) query[d] = 0;
  for (uint32_t id = 0; id < count; ++id) {
    zvec::ailego::NumericalVector<float> vector(dimension);
    for (uint32_t d = 0; d < dimension; ++d) {
      vector[d] = id >= count - copies ? query[d] : normal(generator);
    }
    ASSERT_TRUE(holder->emplace(id, vector));
  }
  auto builder = IndexFactory::CreateBuilder("DiskAnnBuilder");
  ASSERT_NE(builder, nullptr);
  zvec::ailego::Params params;
  params.set(PARAM_DISKANN_BUILDER_MAX_DEGREE, 32U);
  params.set(PARAM_DISKANN_BUILDER_LIST_SIZE, 128U);
  params.set(PARAM_DISKANN_BUILDER_THREAD_COUNT, 1U);
  ASSERT_EQ(0, builder->init(meta, params));
  ASSERT_EQ(0, builder->train(holder));
  ASSERT_EQ(0, builder->build(holder));
  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(dumper, nullptr);
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, builder->dump(dumper));
  ASSERT_EQ(0, dumper->close());
  builder.reset();
  holder.reset();

  auto searcher = IndexFactory::CreateSearcher("DiskAnnSearcher");
  ASSERT_NE(searcher, nullptr);
  zvec::ailego::Params search_params;
  search_params.set(PARAM_DISKANN_SEARCHER_LIST_SIZE, 16U);
  search_params.set(PARAM_DISKANN_SEARCHER_CACHE_NODE_NUM,
                    std::get<1>(GetParam()) ? 128U : 0U);
  ASSERT_EQ(0, searcher->init(search_params));
  auto storage = IndexFactory::CreateStorage(
      std::get<0>(GetParam()) ? "BufferReadStorage" : "FileReadStorage");
  ASSERT_NE(storage, nullptr);
  ASSERT_EQ(0, storage->open(path, false));
  ASSERT_EQ(0, searcher->load(storage, IndexMetric::Pointer()));
  auto context = searcher->create_context();
  ASSERT_NE(context, nullptr);
  context->set_topk(1);
  context->set_filter(
      [first_deleted = count - copies, live_target = target](uint64_t key) {
        return key >= first_deleted && key != live_target;
      });
  IndexQueryMeta query_meta(IndexMeta::DT_FP32, dimension);
  ASSERT_EQ(0, searcher->search_impl(query.data(), query_meta, context));
  ASSERT_EQ(1u, context->result().size());
  EXPECT_EQ(target, context->result()[0].key());
  EXPECT_FLOAT_EQ(0, context->result()[0].score());
}

INSTANTIATE_TEST_SUITE_P(StorageAndCache, DiskAnnFilteredBeamTest,
                         testing::Combine(testing::Bool(), testing::Bool()));
