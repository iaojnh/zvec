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

#include "diskann_builder.h"
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#include <chrono>
#include <cstring>
#include <future>
#include <limits>
#include <gtest/gtest.h>
#include <zvec/ailego/container/vector.h>
#include <zvec/core/framework/index_framework.h>
#include "diskann_builder_entity.h"
#include "diskann_context.h"
#include "diskann_params.h"

using namespace zvec::core;
using namespace zvec::ailego;
using namespace std;

constexpr size_t static dim = 64;

class DiskAnnBuilderTest : public testing::Test {
 protected:
  void SetUp(void) override;
  void TearDown(void) override;

  static std::string _dir;
  static shared_ptr<IndexMeta> _index_meta_ptr;
};

std::string DiskAnnBuilderTest::_dir("DiskAnnBuilderTest");
shared_ptr<IndexMeta> DiskAnnBuilderTest::_index_meta_ptr;

void DiskAnnBuilderTest::SetUp(void) {
  LoggerBroker::SetLevel(Logger::LEVEL_INFO);

  _index_meta_ptr.reset(new (nothrow)
                            IndexMeta(IndexMeta::DataType::DT_FP32, dim));
  _index_meta_ptr->set_metric("SquaredEuclidean", 0, Params());
}

void DiskAnnBuilderTest::TearDown(void) {
  char cmdBuf[100];
  snprintf(cmdBuf, 100, "rm -rf %s", _dir.c_str());
  system(cmdBuf);
}

TEST_F(DiskAnnBuilderTest, TestGeneral) {
  IndexBuilder::Pointer builder = IndexFactory::CreateBuilder("DiskAnnBuilder");
  ASSERT_NE(builder, nullptr);

  auto holder =
      make_shared<MultiPassIndexHolder<IndexMeta::DataType::DT_FP32>>(dim);
  size_t doc_cnt = 10000UL;
  for (size_t i = 0; i < doc_cnt; i++) {
    NumericalVector<float> vec(dim);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i;
    }
    ASSERT_TRUE(holder->emplace(i, vec));
  }

  Params params;

  params.set("zvec.diskann.builder.max_degree", 32);
  params.set("zvec.diskann.builder.list_size", 50);
  params.set("zvec.diskann.builder.max_pq_chunk_num", 32);
  params.set("zvec.diskann.builder.threads", 4);

  ASSERT_EQ(0, builder->init(*_index_meta_ptr, params));

  ASSERT_EQ(0, builder->train(holder));

  ASSERT_EQ(0, builder->build(holder));

  // Dump must use the vectors captured by build, not reread a holder that may
  // have changed after the graph and PQ data were finalized.
  NumericalVector<float> late_vector(dim, -1.0F);
  ASSERT_TRUE(holder->emplace(doc_cnt, late_vector));

  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(dumper, nullptr);

  string path = _dir + "/TestGeneral";
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, builder->dump(dumper));
  ASSERT_EQ(0, dumper->close());

  auto &stats = builder->stats();
  ASSERT_EQ(doc_cnt, stats.trained_count());
  ASSERT_EQ(doc_cnt, stats.built_count());
  ASSERT_EQ(doc_cnt, stats.dumped_count());
  ASSERT_EQ(0UL, stats.discarded_count());
  ASSERT_GT(stats.trained_costtime(), 0UL);
  ASSERT_GT(stats.built_costtime(), 0UL);
}

TEST_F(DiskAnnBuilderTest, RejectsDuplicateValidKeysAtDump) {
  DiskAnnBuilderEntity entity;
  ASSERT_EQ(0, entity.init(*_index_meta_ptr, 16, 32, 0.0, 1));

  std::vector<float> vector(dim, 1.0F);
  EXPECT_EQ(0, entity.add_vector(42, vector.data()));
  EXPECT_EQ(0, entity.add_vector(42, vector.data()));

  // Invalid keys represent empty/deleted slots and may legitimately repeat.
  EXPECT_EQ(0, entity.add_vector(kInvalidKey, vector.data()));
  EXPECT_EQ(0, entity.add_vector(kInvalidKey, vector.data()));
  EXPECT_EQ(4U, entity.doc_cnt());
  EXPECT_EQ(IndexError_InvalidArgument, entity.add_vector(7, nullptr));

  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(nullptr, dumper);
  ASSERT_EQ(0, dumper->create(_dir + "/DuplicateKeys"));
  EXPECT_EQ(IndexError_Exist, entity.dump_key_mapping_segment(dumper));
  EXPECT_EQ(0, dumper->close());

  DiskAnnBuilderEntity holes;
  ASSERT_EQ(0, holes.init(*_index_meta_ptr, 16, 32, 0.0, 1));
  ASSERT_EQ(0, holes.add_vector(kInvalidKey, vector.data()));
  ASSERT_EQ(0, holes.add_vector(kInvalidKey, vector.data()));
  auto holes_dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(nullptr, holes_dumper);
  ASSERT_EQ(0, holes_dumper->create(_dir + "/InvalidKeySlots"));
  EXPECT_EQ(0, holes.dump_key_mapping_segment(holes_dumper));
  EXPECT_EQ(0, holes_dumper->close());
}

TEST_F(DiskAnnBuilderTest, NeighborStorageIsNaturallyAligned) {
  DiskAnnBuilderEntity entity;
  ASSERT_EQ(0, entity.init(*_index_meta_ptr, 1, 1, 0.0, 1));

  std::vector<float> vector(dim, 1.0F);
  ASSERT_EQ(0, entity.add_vector(0, vector.data()));

  auto neighbors = entity.get_neighbors(0);
  ASSERT_NE(nullptr, neighbors.second);
  EXPECT_EQ(0U, reinterpret_cast<uintptr_t>(neighbors.second) %
                    alignof(diskann_id_t));

  ASSERT_EQ(0, entity.add_neighbor(0, 7));
  neighbors = entity.get_neighbors(0);
  ASSERT_EQ(1U, neighbors.first);
  EXPECT_EQ(7U, neighbors.second[0]);

  DiskAnnBuilderEntity slack_entity;
  ASSERT_EQ(0, slack_entity.init(*_index_meta_ptr, 100, 100, 0.0, 1));
  ASSERT_EQ(0, slack_entity.add_vector(0, vector.data()));
  for (diskann_id_t neighbor_id = 0; neighbor_id < 130; ++neighbor_id) {
    ASSERT_EQ(0, slack_entity.add_neighbor(0, neighbor_id));
  }
  EXPECT_EQ(IndexError_IndexFull, slack_entity.add_neighbor(0, 130));
}

TEST_F(DiskAnnBuilderTest, RejectsInvalidGraphAndSamplingParameters) {
  auto expect_invalid = [](const Params &params) {
    auto builder = IndexFactory::CreateBuilder("DiskAnnBuilder");
    ASSERT_NE(nullptr, builder);
    EXPECT_EQ(IndexError_InvalidArgument,
              builder->init(*_index_meta_ptr, params));
  };

  Params params;
  params.set(PARAM_DISKANN_BUILDER_MAX_DEGREE, 0U);
  expect_invalid(params);

  params.clear();
  params.set(PARAM_DISKANN_BUILDER_LIST_SIZE, 0U);
  expect_invalid(params);

  params.clear();
  params.set(PARAM_DISKANN_BUILDER_MAX_TRAIN_SAMPLE_COUNT, 0U);
  expect_invalid(params);

  params.clear();
  params.set(PARAM_DISKANN_BUILDER_TRAIN_SAMPLE_RATIO, 0.0);
  expect_invalid(params);

  params.clear();
  params.set(PARAM_DISKANN_BUILDER_TRAIN_SAMPLE_RATIO, 1.01);
  expect_invalid(params);

  params.clear();
  params.set(PARAM_DISKANN_BUILDER_TRAIN_SAMPLE_RATIO,
             std::numeric_limits<double>::quiet_NaN());
  expect_invalid(params);

  DiskAnnBuilderEntity entity;
  EXPECT_EQ(IndexError_InvalidArgument,
            entity.init(*_index_meta_ptr, 0, 1, 0.0, 1));
  EXPECT_EQ(IndexError_InvalidArgument,
            entity.init(*_index_meta_ptr, 1, 0, 0.0, 1));
  ASSERT_EQ(0, entity.init(*_index_meta_ptr, 1, 1, 0.0, 1));
  EXPECT_EQ(IndexError_InvalidLength, entity.reserve_space(0));
  EXPECT_EQ(IndexError_InvalidLength,
            entity.reserve_space((std::numeric_limits<size_t>::max)()));
}

TEST_F(DiskAnnBuilderTest, PqSamplingUsesRatioAndWholeDataset) {
  constexpr size_t kSamplingDim = 1;
  constexpr size_t kDocCount = 100;
  IndexMeta meta(IndexMeta::DataType::DT_FP32, kSamplingDim);
  meta.set_metric("SquaredEuclidean", 0, Params());
  auto holder = make_shared<MultiPassIndexHolder<IndexMeta::DataType::DT_FP32>>(
      kSamplingDim);
  for (size_t i = 0; i < kDocCount; ++i) {
    NumericalVector<float> vector(kSamplingDim, static_cast<float>(i));
    ASSERT_TRUE(holder->emplace(i, vector));
  }

  DiskAnnPqTrainer trainer(10, 0.5);
  std::string sample_data;
  size_t sample_size = 0;
  ASSERT_EQ(0,
            trainer.gen_random_sample(holder, meta, sample_data, sample_size));
  EXPECT_EQ(10U, sample_size);
  ASSERT_EQ(sample_size * sizeof(float), sample_data.size());

  std::string prefix(sample_data.size(), '\0');
  for (size_t i = 0; i < sample_size; ++i) {
    const float value = static_cast<float>(i);
    std::memcpy(prefix.data() + i * sizeof(float), &value, sizeof(value));
  }
  EXPECT_NE(prefix, sample_data)
      << "PQ sampling must not always select the first vectors";

  DiskAnnPqTrainer repeated_trainer(10, 0.5);
  std::string repeated_data;
  size_t repeated_size = 0;
  ASSERT_EQ(0, repeated_trainer.gen_random_sample(holder, meta, repeated_data,
                                                  repeated_size));
  EXPECT_EQ(sample_size, repeated_size);
  EXPECT_EQ(sample_data, repeated_data);

  DiskAnnPqTrainer ratio_limited_trainer(100, 0.05);
  ASSERT_EQ(0, ratio_limited_trainer.gen_random_sample(
                   holder, meta, sample_data, sample_size));
  EXPECT_EQ(5U, sample_size);

  DiskAnnPqTrainer invalid_trainer(0, 1.0);
  EXPECT_EQ(IndexError_InvalidArgument,
            invalid_trainer.gen_random_sample(holder, meta, sample_data,
                                              sample_size));
}

// Regression test: building a small DiskAnn index must complete quickly.
// A lost-wakeup bug in the condition-variable progress loops previously caused
// 15–30 second stalls during train/build on small datasets because
// notify_one() was either missing or racing against a wrong predicate.
TEST_F(DiskAnnBuilderTest, SmallDatasetBuildTime) {
  constexpr size_t kSmallDim = 4;
  constexpr size_t kSmallDocCnt = 12;

  auto meta = make_shared<IndexMeta>(IndexMeta::DataType::DT_FP32, kSmallDim);
  meta->set_metric("SquaredEuclidean", 0, Params());

  IndexBuilder::Pointer builder = IndexFactory::CreateBuilder("DiskAnnBuilder");
  ASSERT_NE(builder, nullptr);

  auto holder = make_shared<MultiPassIndexHolder<IndexMeta::DataType::DT_FP32>>(
      kSmallDim);
  for (size_t i = 0; i < kSmallDocCnt; ++i) {
    NumericalVector<float> vec(kSmallDim, static_cast<float>(i));
    ASSERT_TRUE(holder->emplace(i, vec));
  }

  Params params;
  params.set("zvec.diskann.builder.max_degree", 32);
  params.set("zvec.diskann.builder.list_size", 50);
  params.set("zvec.diskann.builder.max_pq_chunk_num", 2);
  params.set("zvec.diskann.builder.threads", 4);

  ASSERT_EQ(0, builder->init(*meta, params));

  auto t0 = std::chrono::steady_clock::now();
  ASSERT_EQ(0, builder->train(holder));
  ASSERT_EQ(0, builder->build(holder));
  auto t1 = std::chrono::steady_clock::now();

  auto elapsed_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
  // Before the fix, this took 15–30 seconds. After the fix, it should
  // complete in well under 5 seconds even on slow CI machines.
  EXPECT_LT(elapsed_ms, 5000)
      << "DiskAnn build with " << kSmallDocCnt << " vectors took " << elapsed_ms
      << " ms — likely a lost-wakeup regression in progress loops.";
}

TEST_F(DiskAnnBuilderTest, SingleDocumentIndexCanBeSearched) {
  auto builder = IndexFactory::CreateBuilder("DiskAnnBuilder");
  ASSERT_NE(nullptr, builder);

  auto holder =
      make_shared<MultiPassIndexHolder<IndexMeta::DataType::DT_FP32>>(dim);
  NumericalVector<float> vector(dim, 1.0F);
  ASSERT_TRUE(holder->emplace(7, vector));

  Params params;
  params.set(PARAM_DISKANN_BUILDER_MAX_DEGREE, 16);
  params.set(PARAM_DISKANN_BUILDER_LIST_SIZE, 32);
  params.set(PARAM_DISKANN_BUILDER_MAX_PQ_CHUNK_NUM, 1);
  params.set(PARAM_DISKANN_BUILDER_THREAD_COUNT, 1);
  ASSERT_EQ(0, builder->init(*_index_meta_ptr, params));
  ASSERT_EQ(0, builder->train(holder));
  ASSERT_EQ(0, builder->build(holder));

  const string path = _dir + "/SingleDocumentIndexCanBeSearched";
  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(nullptr, dumper);
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, builder->dump(dumper));
  ASSERT_EQ(0, dumper->close());

  auto storage = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_NE(nullptr, storage);
  ASSERT_EQ(0, storage->open(path, false));

  auto searcher = IndexFactory::CreateSearcher("DiskAnnSearcher");
  ASSERT_NE(nullptr, searcher);
  Params search_params;
  search_params.set(PARAM_DISKANN_SEARCHER_LIST_SIZE, 32);
  ASSERT_EQ(0, searcher->init(search_params));
  ASSERT_EQ(0, searcher->load(storage, IndexMetric::Pointer()));

  auto context = searcher->create_context();
  ASSERT_NE(nullptr, context);
  auto *diskann_context = dynamic_cast<DiskAnnContext *>(context.get());
  ASSERT_NE(nullptr, diskann_context);
  EXPECT_EQ(1U, diskann_context->list_size());

  context->set_topk(1);
  IndexQueryMeta query_meta(IndexMeta::DataType::DT_FP32, dim);
  ASSERT_EQ(0, searcher->search_impl(vector.data(), query_meta, context));
  ASSERT_EQ(1U, context->result().size());
  EXPECT_EQ(7U, context->result()[0].key());
}

TEST_F(DiskAnnBuilderTest, MemoryLimitCapsPqChunkCount) {
  constexpr size_t kTestDim = 8;
  constexpr size_t kDocCnt = 16;

  IndexMeta meta(IndexMeta::DataType::DT_FP32, kTestDim);
  meta.set_metric("SquaredEuclidean", 0, Params());

  auto holder =
      make_shared<MultiPassIndexHolder<IndexMeta::DataType::DT_FP32>>(kTestDim);
  for (size_t i = 0; i < kDocCnt; ++i) {
    NumericalVector<float> vec(kTestDim, static_cast<float>(i));
    ASSERT_TRUE(holder->emplace(i, vec));
  }

  Params params;
  params.set(PARAM_DISKANN_BUILDER_MAX_DEGREE, 16);
  params.set(PARAM_DISKANN_BUILDER_LIST_SIZE, 20);
  params.set(PARAM_DISKANN_BUILDER_MAX_PQ_CHUNK_NUM, 4);
  params.set(PARAM_DISKANN_BUILDER_THREAD_COUNT, 2);
  // 40 total bytes gives a two-byte PQ code budget per vector. Before the
  // fix this value was calculated and then discarded.
  params.set(PARAM_DISKANN_BUILDER_MEMORY_LIMIT,
             40.0 / (1024.0 * 1024.0 * 1024.0));

  auto builder = IndexFactory::CreateBuilder("DiskAnnBuilder");
  ASSERT_NE(builder, nullptr);
  ASSERT_EQ(0, builder->init(meta, params));
  ASSERT_EQ(0, builder->train(holder));
  ASSERT_EQ(0, builder->build(holder));

  const string path = _dir + "/MemoryLimitCapsPqChunkCount";
  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(dumper, nullptr);
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, builder->dump(dumper));
  ASSERT_EQ(0, dumper->close());

  auto storage = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_NE(storage, nullptr);
  ASSERT_EQ(0, storage->open(path, false));
  auto pq_meta_segment = storage->get(DiskAnnEntity::kDiskAnnPqMetaSegmentId);
  ASSERT_NE(pq_meta_segment, nullptr);
  const void *data = nullptr;
  ASSERT_EQ(sizeof(DiskAnnPqMeta),
            pq_meta_segment->read(0, &data, sizeof(DiskAnnPqMeta)));
  DiskAnnPqMeta pq_meta{};
  std::memcpy(&pq_meta, data, sizeof(pq_meta));
  EXPECT_EQ(2U, pq_meta.chunk_num);
}

TEST_F(DiskAnnBuilderTest, TestImplicitFactoryRegistration) {
  IndexBuilder::Pointer builder = IndexFactory::CreateBuilder("DiskAnnBuilder");
  ASSERT_NE(builder, nullptr)
      << "DiskAnnBuilder factory entry missing: DiskAnn must be available "
         "without any manual plugin load step.";

  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("DiskAnnStreamer");
  ASSERT_NE(streamer, nullptr)
      << "DiskAnnStreamer factory entry missing: DiskAnn must be available "
         "without any manual plugin load step.";
}
