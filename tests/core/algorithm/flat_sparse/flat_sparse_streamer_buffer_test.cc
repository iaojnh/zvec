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

#include <string>
#include <vector>
#include <ailego/pattern/scope_guard.h>
#include <ailego/utility/math_helper.h>
#include <ailego/utility/memory_helper.h>
#include <algorithm/flat/flat_utility.h>
#include <algorithm/flat_sparse/flat_sparse_utility.h>
#include <gtest/gtest.h>
#include <turbo/norm/norm2_matrix.h>
#include <zvec/core/framework/index_framework.h>
#include <zvec/core/framework/index_streamer.h>
#include "tests/test_util.h"
#include "../../../ailego/buffer/pool_payload_budget.h"

using namespace std;
using namespace testing;
using namespace zvec::ailego;

#if defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-result"
#endif

namespace zvec {
namespace core {

class FlatSparseStreamerTest : public testing::Test {
 protected:
  void SetUp() override;
  void TearDown() override;
  void generate_sparse_data(
      size_t cnt, uint32_t sparse_dim_count,
      std::vector<NumericalVector<uint32_t>> &sparse_indices_list,
      std::vector<NumericalVector<float>> &sparse_vec_list, bool norm);

  static std::string dir_;
  static shared_ptr<IndexMeta> index_meta_ptr_;
};

std::string FlatSparseStreamerTest::dir_("FlatSparseStreamerTest/");
shared_ptr<IndexMeta> FlatSparseStreamerTest::index_meta_ptr_;

void FlatSparseStreamerTest::generate_sparse_data(
    size_t cnt, uint32_t sparse_dim_count,
    std::vector<NumericalVector<uint32_t>> &sparse_indices_list,
    std::vector<NumericalVector<float>> &sparse_vec_list, bool norm) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(-1.0, 1.0);

  for (size_t i = 0; i < cnt; ++i) {
    // prepare sparse
    NumericalVector<uint32_t> sparse_indices(sparse_dim_count);
    NumericalVector<float> sparse_vec(sparse_dim_count);

    for (size_t j = 0; j < sparse_dim_count; ++j) {
      sparse_indices[j] = j * 20;
      sparse_vec[j] = dist(gen);
    }

    float norm;
    turbo::Norm2Matrix<float, 1>::Compute(sparse_vec.data(), sparse_dim_count,
                                          &norm);
    for (size_t j = 0; j < sparse_dim_count; ++j) {
      sparse_vec[j] = sparse_vec[j] / norm;
    }

    sparse_indices_list.push_back(sparse_indices);
    sparse_vec_list.push_back(sparse_vec);
  }
}

void FlatSparseStreamerTest::SetUp() {
  index_meta_ptr_.reset(new (nothrow) IndexMeta(IndexMeta::MetaType::MT_SPARSE,
                                                IndexMeta::DataType::DT_FP32));
  index_meta_ptr_->set_metric("InnerProductSparse", 0, ailego::Params());

  zvec::test_util::RemoveTestPath(dir_);
}

void FlatSparseStreamerTest::TearDown() {
  zvec::test_util::RemoveTestPath(dir_);
}

TEST_F(FlatSparseStreamerTest, PreparedAddCancelsBeforePublishAndReusesGap) {
  auto &pool = PoolPayloadBudget::get_instance();
  const size_t previous_capacity = pool.capacity();
  ASSERT_EQ(0, pool.init(8 * 1024 * 1024));
  auto restore =
      ScopeGuard::Make([&] { EXPECT_EQ(0, pool.init(previous_capacity)); });
  const IndexQueryMeta qmeta(IndexMeta::MT_SPARSE, IndexMeta::DT_FP32);
  const uint32_t dimension[] = {7};
  const float abandoned[] = {1.5f}, persisted[] = {4.25f};

  for (bool use_id_map : {false, true}) {
    SCOPED_TRACE(use_id_map);
    const auto path = dir_ + std::to_string(use_id_map) + "/index";
    Params params;
    params.set(PARAM_FLAT_USE_ID_MAP, use_id_map);
    params.set(PARAM_FLAT_SPARSE_STREAMER_MAX_DOC_CNT, uint32_t{128});
    params.set(PARAM_FLAT_SPARSE_STREAMER_MAX_DATA_CHUNK_CNT, uint32_t{4});
    params.set(PARAM_FLAT_SPARSE_STREAMER_OFFSET_CHUNK_SIZE, uint32_t{48});
    params.set(PARAM_FLAT_SPARSE_STREAMER_DATA_CHUNK_SIZE, uint32_t{64});
    auto storage = IndexFactory::CreateStorage("BufferStorage");
    ASSERT_NE(nullptr, storage);
    ASSERT_EQ(0, storage->init({}));
    ASSERT_EQ(0, storage->open(path, true));
    auto streamer = IndexFactory::CreateStreamer("FlatSparseStreamer");
    ASSERT_NE(nullptr, streamer);
    ASSERT_EQ(0, streamer->init(*index_meta_ptr_, params));
    ASSERT_EQ(0, streamer->open(storage));
    auto context = streamer->create_context();
    ASSERT_NE(nullptr, context);

    std::unique_ptr<IndexStreamer::PreparedAdd> pending;
    ASSERT_EQ(0, streamer->prepare_add_with_id(1, 1, dimension, abandoned,
                                               qmeta, context, &pending));
    ASSERT_NE(nullptr, pending);
    uint32_t count = 0;
    std::string indices, values;
    EXPECT_EQ(IndexError_NoExist,
              streamer->get_sparse_vector_by_id(1, &count, &indices, &values));
    EXPECT_EQ(0u, streamer->stats().added_count());
    pending.reset();

    ASSERT_EQ(0, streamer->prepare_add_with_id(2, 1, dimension, persisted,
                                               qmeta, context, &pending));
    ASSERT_NE(nullptr, pending);
    ASSERT_EQ(IndexError_NoExist,
              streamer->get_sparse_vector_by_id(2, &count, &indices, &values));
    {
      MemoryReservation pressure;
      pressure.add(pool.capacity() - pool.used());
      pending->commit();
    }
    EXPECT_EQ(1u, streamer->stats().added_count());
    pending.reset();

    auto verify = [&] {
      for (uint32_t id : {0, 1}) {
        EXPECT_EQ(IndexError_NoExist, streamer->get_sparse_vector_by_id(
                                          id, &count, &indices, &values));
      }
      indices.clear();
      values.clear();
      ASSERT_EQ(
          0, streamer->get_sparse_vector_by_id(2, &count, &indices, &values));
      EXPECT_EQ(1u, count);
      EXPECT_EQ(std::string(reinterpret_cast<const char *>(dimension),
                            sizeof(dimension)),
                indices);
      EXPECT_EQ(std::string(reinterpret_cast<const char *>(persisted),
                            sizeof(persisted)),
                values);
    };
    verify();
    ASSERT_EQ(0, streamer->flush(0));
    context.reset();
    ASSERT_EQ(0, streamer->close());
    ASSERT_EQ(0, storage->close());
    ASSERT_EQ(0, storage->open(path, false));
    ASSERT_EQ(0, streamer->open(storage));
    verify();
    ASSERT_EQ(0, streamer->close());
    ASSERT_EQ(0, storage->close());
  }
}

TEST_F(FlatSparseStreamerTest, TestGeneral) {
  IndexStreamer::Pointer write_streamer =
      IndexFactory::CreateStreamer("FlatSparseStreamer");
  ASSERT_TRUE(write_streamer != nullptr);

  size_t sparse_dim_count = 32;

  IndexMeta index_meta(IndexMeta::MetaType::MT_SPARSE,
                       IndexMeta::DataType::DT_FP32);
  index_meta.set_metric("InnerProductSparse", 0, ailego::Params());

  ailego::Params params;

  ailego::Params stg_params;
  auto write_storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_EQ(0, write_storage->init(stg_params));
  ASSERT_EQ(0, write_storage->open(dir_ + "Test/FlatSparseSearch", true));
  ASSERT_EQ(0, write_streamer->init(index_meta, params));
  ASSERT_EQ(0, write_streamer->open(write_storage));

  size_t cnt = 20000U;
  auto ctx = write_streamer->create_context();
  ASSERT_TRUE(!!ctx);

  std::vector<NumericalVector<uint32_t>> sparse_indices_list;
  std::vector<NumericalVector<float>> sparse_vec_list;

  generate_sparse_data(cnt, sparse_dim_count, sparse_indices_list,
                       sparse_vec_list, true);

  IndexQueryMeta qmeta(IndexMeta::MetaType::MT_SPARSE,
                       IndexMeta::DataType::DT_FP32);
  for (size_t i = 0; i < cnt; i++) {
    ASSERT_EQ(0, write_streamer->add_impl(
                     i, sparse_dim_count, sparse_indices_list[i].data(),
                     sparse_vec_list[i].data(), qmeta, ctx));
  }
  write_streamer->flush(0UL);
  write_streamer->close();
  write_streamer.reset();
  write_storage->close();
  write_storage.reset();

  IndexStreamer::Pointer read_streamer =
      IndexFactory::CreateStreamer("FlatSparseStreamer");
  ASSERT_EQ(0, read_streamer->init(*index_meta_ptr_, params));
  auto read_storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_NE(nullptr, read_storage);
  ASSERT_EQ(0, read_storage->init(stg_params));
  ASSERT_EQ(0, read_storage->open(dir_ + "Test/FlatSparseSearch", false));
  ASSERT_EQ(0, read_streamer->open(read_storage));

  auto linear_ctx = read_streamer->create_context();
  ASSERT_TRUE(!!linear_ctx);

  auto knn_ctx = read_streamer->create_context();
  ASSERT_TRUE(!!knn_ctx);

  // streamer->print_debug_info();
  size_t topk = 200;
  linear_ctx->set_topk(topk);
  knn_ctx->set_topk(topk);

  uint64_t knn_total_time = 0;
  uint64_t linear_total_time = 0;

  int total_hits = 0;
  int total_cnts = 0;
  int topk1_hits = 0;

  for (size_t i = 0; i < cnt; i += 100) {
    const auto &sparse_indices = sparse_indices_list[i];
    const auto &sparse_vec = sparse_vec_list[i];

    auto t1 = ailego::Realtime::MicroSeconds();

    ASSERT_EQ(
        0, read_streamer->search_impl(sparse_dim_count, sparse_indices.data(),
                                      sparse_vec.data(), qmeta, knn_ctx));

    auto t2 = ailego::Realtime::MicroSeconds();

    ASSERT_EQ(0, read_streamer->search_bf_impl(
                     sparse_dim_count, sparse_indices.data(), sparse_vec.data(),
                     qmeta, linear_ctx));

    auto t3 = ailego::Realtime::MicroSeconds();

    knn_total_time += t2 - t1;
    linear_total_time += t3 - t2;

    // std::cout << "i: " << i << std::endl;

    auto &knn_result = knn_ctx->result();
    ASSERT_EQ(topk, knn_result.size());
    topk1_hits += i == knn_result[0].key();

    auto &linear_result = linear_ctx->result();
    ASSERT_EQ(topk, linear_result.size());
    ASSERT_EQ(i, linear_result[0].key());

    for (size_t k = 0; k < topk; ++k) {
      total_cnts++;
      for (size_t j = 0; j < topk; ++j) {
        if (linear_result[j].key() == knn_result[k].key()) {
          total_hits++;
          break;
        }
      }
    }
  }
  float recall = total_hits * 1.0f / total_cnts;
  float topk1_recall = topk1_hits * 100.0f / cnt;
  // float cost = linearTotalTime * 1.0f / knnTotalTime;

  std::cout << "knnTotalTime=" << knn_total_time
            << " linearTotalTime=" << linear_total_time << std::endl;

#if 0
    printf("knnTotalTime=%zd linearTotalTime=%zd totalHits=%d totalCnts=%d "
           "R@%zd=%f R@1=%f cost=%f\n",
           knnTotalTime, linearTotalTime, totalHits, totalCnts, topk, recall,
           topk1Recall, cost);
#endif
  EXPECT_GT(recall, 0.80f);
  EXPECT_GT(topk1_recall, 0.80f);
  // EXPECT_GT(cost, 2.0f);
}

}  // namespace core
}  // namespace zvec

#if defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic pop
#endif
