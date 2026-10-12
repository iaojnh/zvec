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

#include <cstring>
#include <utility>
#include <vector>
#include <ailego/pattern/defer.h>
#include <gtest/gtest.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/ailego/buffer/vector_page_table.h>
#include <zvec/core/framework/index_memory.h>
#include "algorithm/cluster/cluster_params.h"
#include "algorithm/cluster/stratified_cluster_trainer.h"
#include "ivf_builder.h"
#include "ivf_searcher.h"

namespace zvec::core {
namespace {

class CountingCentroidTrainer : public StratifiedClusterTrainer {
 public:
  IndexBundle::Pointer indexes() const override {
    ++serialize_calls;
    return StratifiedClusterTrainer::indexes();
  }
  mutable size_t serialize_calls{0};
};

// Deliberately exposes only the public trainer interface, like an existing
// third-party trainer. Its output must remain usable without CentroidTrainer.
class LegacyTrainer : public IndexTrainer {
 public:
  explicit LegacyTrainer(IndexTrainer::Pointer trainer)
      : trainer_(std::move(trainer)) {}
  int init(const IndexMeta &meta, const ailego::Params &params) override {
    return trainer_->init(meta, params);
  }
  int cleanup() override {
    return trainer_->cleanup();
  }
  int train(IndexThreads::Pointer threads,
            IndexHolder::Pointer holder) override {
    return trainer_->train(std::move(threads), std::move(holder));
  }
  int load(IndexStorage::Pointer storage) override {
    return trainer_->load(std::move(storage));
  }
  int dump(const IndexDumper::Pointer &dumper) override {
    return trainer_->dump(dumper);
  }
  const IndexMeta &meta() const override {
    return trainer_->meta();
  }
  const Stats &stats() const override {
    return trainer_->stats();
  }
  IndexBundle::Pointer indexes() const override {
    return trainer_->indexes();
  }

 private:
  IndexTrainer::Pointer trainer_;
};

class InvalidCentroidTrainer : public LegacyTrainer, public CentroidTrainer {
 public:
  using LegacyTrainer::LegacyTrainer;
  const IndexCluster::CentroidList &centroids() const override {
    return tree;
  }
  IndexCluster::CentroidList tree;
};

class IVFCentroidTrainerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto &pool = ailego::MemoryLimitPool::get_instance();
    capacity_ = pool.capacity();
    ASSERT_EQ(0u, pool.used());
    ASSERT_EQ(0, pool.init(0));
  }
  void TearDown() override {
    auto &pool = ailego::MemoryLimitPool::get_instance();
    EXPECT_EQ(0u, pool.used());
    EXPECT_EQ(0, pool.init(capacity_));
  }

  using Results = std::vector<std::pair<uint64_t, float>>;

  void finish_and_query(IVFBuilder &builder, const IndexHolder::Pointer &holder,
                        const std::string &path, Results *results) {
    AILEGO_DEFER([&] { IndexMemory::Instance()->remove(path); });
    ASSERT_EQ(0, builder.build(threads_, holder));
    auto dumper = IndexFactory::CreateDumper("MemoryDumper");
    ASSERT_NE(nullptr, dumper);
    ASSERT_EQ(0, dumper->create(path));
    ASSERT_EQ(0, builder.dump(dumper));
    ASSERT_EQ(0, dumper->close());
    EXPECT_EQ(holder->count(), builder.stats().dumped_count());
    ASSERT_EQ(0, builder.cleanup());

    auto storage = IndexFactory::CreateStorage("MemoryReadStorage");
    ASSERT_NE(nullptr, storage);
    ASSERT_EQ(0, storage->init(ailego::Params()));
    ASSERT_EQ(0, storage->open(path, false));
    IVFSearcher searcher;
    ailego::Params params;
    params.set(PARAM_IVF_SEARCHER_SCAN_RATIO, 1.0);
    params.set(PARAM_IVF_SEARCHER_BRUTE_FORCE_THRESHOLD, 1);
    ASSERT_EQ(0, searcher.init(params));
    ASSERT_EQ(0, searcher.load(storage, nullptr));
    auto context = searcher.create_context();
    ASSERT_NE(nullptr, context);
    context->set_topk(10);
    IndexQueryMeta meta(holder->data_type(), holder->dimension());
    auto it = holder->create_iterator();
    for (size_t i = 0; i < 8; ++i, it->next()) {
      ASSERT_TRUE(it->is_valid());
      ASSERT_EQ(0, searcher.search_impl(it->data(), meta, context));
      const auto &docs = context->result(0);
      ASSERT_EQ(10u, docs.size());
      EXPECT_EQ(it->key(), docs.front().key());
      EXPECT_FLOAT_EQ(0.0f, docs.front().score());
      for (const auto &doc : docs) {
        results->emplace_back(doc.key(), doc.score());
      }
    }
  }

  template <IndexMeta::DataType Type, typename Scalar>
  void check_handoff(bool two_levels, bool quantize = false) {
    constexpr size_t dimension = 8;
    IndexMeta meta(Type, dimension);
    meta.set_metric("SquaredEuclidean", 0, ailego::Params());
    auto holder = std::make_shared<MultiPassIndexHolder<Type>>(dimension);
    for (size_t i = 0; i < 128; ++i) {
      ailego::NumericalVector<Scalar> row(dimension);
      for (size_t d = 0; d < dimension; ++d) {
        row[d] = static_cast<Scalar>(static_cast<float>((i + d * 7) % 128));
      }
      holder->emplace(1000 + i, std::move(row));
    }
    auto counting = std::make_shared<CountingCentroidTrainer>();
    IndexTrainer::Pointer trainer = counting;
    std::weak_ptr<IndexTrainer> lifetime = trainer;
    ailego::Params params;
    const char *count = two_levels ? "2*2" : "4";
    params.set(STRATIFIED_TRAINER_CLUSTER_COUNT, count);
    ailego::Params level;
    level.set(OPTKMEANS_CLUSTER_MAX_ITERATIONS, 2);
    params.set(STRATIFIED_TRAINER_PARAMS_IN_LEVEL_PREFIX + "1", level);
    params.set(STRATIFIED_TRAINER_PARAMS_IN_LEVEL_PREFIX + "2", level);
    ASSERT_EQ(0, trainer->init(meta, params));
    ASSERT_EQ(0, trainer->train(threads_, holder));
    ASSERT_FALSE(counting->centroids().empty());
    if (two_levels) {
      ASSERT_FALSE(counting->centroids().front().subitems().empty());
    }
    const void *original_data = counting->centroids().front().feature();
    IndexBundle::Pointer before;
    ASSERT_EQ(0, IndexCluster::Serialize(meta, counting->centroids(), &before));

    params.set(PARAM_IVF_BUILDER_CENTROID_COUNT, count);
    if (quantize) {
      params.set(PARAM_IVF_BUILDER_OPTIMIZER_QUANTIZER_CLASS,
                 "HalfFloatConverter");
    }
    IVFBuilder borrowed, legacy;
    ASSERT_EQ(0, borrowed.init(meta, params));
    ASSERT_EQ(0, borrowed.train(trainer));
    EXPECT_EQ(0u, counting->serialize_calls);
    EXPECT_EQ(original_data, counting->centroids().front().feature());
    ASSERT_EQ(0, legacy.init(meta, params));
    ASSERT_EQ(0, legacy.train(std::make_shared<LegacyTrainer>(trainer)));
    EXPECT_EQ(1u, counting->serialize_calls);
    EXPECT_EQ(legacy.centroid_index()->centroids_count(),
              borrowed.centroid_index()->centroids_count());

    IndexBundle::Pointer after;
    ASSERT_EQ(0, IndexCluster::Serialize(meta, counting->centroids(), &after));
    for (const char *name : {"IndexCluster.Centroids.Features",
                             "IndexCluster.Centroids.Indexes"}) {
      const auto old_data = before->get(name), new_data = after->get(name);
      ASSERT_EQ(old_data.size(), new_data.size());
      EXPECT_EQ(0, std::memcmp(old_data.buffer(), new_data.buffer(),
                               old_data.size()));
    }
    before.reset();
    after.reset();
    ASSERT_EQ(0, trainer->cleanup());
    EXPECT_TRUE(counting->centroids().empty());
    counting.reset();
    trainer.reset();
    ASSERT_TRUE(lifetime.expired());
    // Both coarse indexes must own their output, not the trainer or its tree.
    Results actual, expected;
    ASSERT_NO_FATAL_FAILURE(finish_and_query(
        borrowed, holder, "centroid_handoff_borrowed", &actual));
    ASSERT_NO_FATAL_FAILURE(
        finish_and_query(legacy, holder, "centroid_handoff_legacy", &expected));
    EXPECT_EQ(expected, actual);
  }

  size_t capacity_{0};
  IndexThreads::Pointer threads_ =
      std::make_shared<SingleQueueIndexThreads>(1, false);
};

TEST_F(IVFCentroidTrainerTest, BorrowedAndLegacyTreesMatchAfterTrainerRelease) {
  for (bool two_levels : {false, true}) {
    SCOPED_TRACE(two_levels);
    ASSERT_NO_FATAL_FAILURE(
        (check_handoff<IndexMeta::DT_FP32, float>(two_levels)));
    ASSERT_NO_FATAL_FAILURE(
        (check_handoff<IndexMeta::DT_FP16, ailego::Float16>(two_levels)));
    ASSERT_NO_FATAL_FAILURE(
        (check_handoff<IndexMeta::DT_INT8, int8_t>(two_levels)));
  }
}

TEST_F(IVFCentroidTrainerTest, QuantizedCoarseIndexAlsoOutlivesTrainer) {
  for (bool two_levels : {false, true}) {
    SCOPED_TRACE(two_levels);
    ASSERT_NO_FATAL_FAILURE(
        (check_handoff<IndexMeta::DT_FP32, float>(two_levels, true)));
  }
}

TEST_F(IVFCentroidTrainerTest, BorrowedHandoffWithSharedPool) {
  // Allow the source, partition and training scratch stores' fixed metadata,
  // plus a small page budget; do not assume platform-independent struct sizes.
  const size_t budget =
      3 * ailego::VecBufferPool::metadata_bytes_for_page_count(4096, true) +
      256 * 1024;
  ASSERT_EQ(0, ailego::MemoryLimitPool::get_instance().init(budget));
  ASSERT_NO_FATAL_FAILURE(
      (check_handoff<IndexMeta::DT_FP32, float>(true, true)));
}

TEST_F(IVFCentroidTrainerTest, FactoryTrainerExposesOptionalCapability) {
  auto trainer = IndexFactory::CreateTrainer("StratifiedClusterTrainer");
  ASSERT_NE(nullptr, trainer);
  auto *source = dynamic_cast<CentroidTrainer *>(trainer.get());
  ASSERT_NE(nullptr, source);
  EXPECT_TRUE(source->centroids().empty());
}

TEST_F(IVFCentroidTrainerTest,
       InvalidTreesFailWithoutSerializationOrPublication) {
  auto counting = std::make_shared<CountingCentroidTrainer>();
  IndexTrainer::Pointer trainer = counting;
  IndexMeta meta(IndexMeta::DT_FP32, 8);
  meta.set_metric("SquaredEuclidean", 0, ailego::Params());
  ailego::Params params;
  params.set(STRATIFIED_TRAINER_CLUSTER_COUNT, "1");
  params.set(PARAM_IVF_BUILDER_CENTROID_COUNT, "1");
  ASSERT_EQ(0, trainer->init(meta, params));
  auto invalid = std::make_shared<InvalidCentroidTrainer>(trainer);
  IVFBuilder builder;
  ASSERT_EQ(0, builder.init(meta, params));
  EXPECT_EQ(IndexError_InvalidFormat, builder.train(invalid));
  EXPECT_EQ(nullptr, builder.centroid_index());
  float row[8]{};
  invalid->tree.emplace_back(row, sizeof(float));
  EXPECT_EQ(IndexError_Mismatch, builder.train(invalid));
  invalid->tree.front().set_feature(row, sizeof(row));
  invalid->tree.front().mutable_subitems()->emplace_back(row, sizeof(float));
  EXPECT_EQ(IndexError_Mismatch, builder.train(invalid));
  EXPECT_EQ(nullptr, builder.centroid_index());
  EXPECT_EQ(0u, counting->serialize_calls);
  invalid->tree.front().mutable_subitems()->front().set_feature(row,
                                                                sizeof(row));
  EXPECT_EQ(0, builder.train(invalid));
}

TEST_F(IVFCentroidTrainerTest, LegacyEmptyBundleStillFails) {
  IndexTrainer::Pointer trainer = std::make_shared<CountingCentroidTrainer>();
  IndexMeta meta(IndexMeta::DT_FP32, 8);
  meta.set_metric("SquaredEuclidean", 0, ailego::Params());
  ailego::Params params;
  params.set(STRATIFIED_TRAINER_CLUSTER_COUNT, "1");
  params.set(PARAM_IVF_BUILDER_CENTROID_COUNT, "1");
  ASSERT_EQ(0, trainer->init(meta, params));
  IVFBuilder builder;
  ASSERT_EQ(0, builder.init(meta, params));
  EXPECT_NE(0, builder.train(std::make_shared<LegacyTrainer>(trainer)));
  EXPECT_EQ(nullptr, builder.centroid_index());
}

}  // namespace
}  // namespace zvec::core
