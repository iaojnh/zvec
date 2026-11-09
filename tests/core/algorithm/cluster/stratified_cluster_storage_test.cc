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

#include "stratified_cluster_storage_test_helper.h"

namespace zvec::core {
namespace {
TEST_F(StratifiedClusterStorageTest,
       StablePagedGroupsKeepDuplicateKeysAndEmptyBucket) {
  for (bool automatic : {false, true}) {
    calls = {};
    auto cluster = IndexFactory::CreateCluster("StratifiedCluster");
    ASSERT_EQ(0, cluster->init(meta(),
                               params("PartitionRecordingCluster", automatic)));
    cluster->suggest(8);
    auto source = std::make_shared<GeneratedStratifiedHolder>(meta(), 32);
    source->ordinal_values = true;
    auto *streaming = dynamic_cast<HolderCluster *>(cluster.get());
    ASSERT_NE(nullptr, streaming);
    IndexCluster::CentroidList cents;
    ASSERT_EQ(0, streaming->cluster_holder(threads, source, cents));
    ASSERT_EQ(3u, cents.size());
    EXPECT_TRUE(cents[2].subitems().empty());
    ASSERT_EQ(2u, calls.groups.size());
    std::vector<uint32_t> expected[2];
    for (uint32_t i = 0; i < 32; ++i) expected[i % 4 == 0 ? 1 : 0].push_back(i);
    EXPECT_EQ(expected[0], calls.groups[0]);
    EXPECT_EQ(expected[1], calls.groups[1]);
    EXPECT_EQ(
        (automatic ? std::vector<uint32_t>{6, 2} : std::vector<uint32_t>{2, 2}),
        calls.suggestions);
    for (const auto &centroid : cents) EXPECT_TRUE(centroid.similars().empty());
    EXPECT_EQ(1u, source->iterations);
    IndexCluster::CentroidList().swap(cents);
    EXPECT_EQ(0u, ailego::PoolPayloadBudget::get_instance().used());
  }
}

TEST_F(StratifiedClusterStorageTest,
       CorpusLargerThanPoolOnlyMountsBoundedBatches) {
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  const size_t metadata_bytes =
      ailego::PoolPayloadBudget::storage_metadata_bytes(4096);
  // Two 256 KiB label batches overlap while replacing the mounted batch.
  // Keep room for their directory, row scratch, and page reads as well.
  ASSERT_EQ(0, pool.init(2 * metadata_bytes + 768 * 1024));
  const auto metadata = meta(IndexMeta::DT_FP32, 128);
  auto cluster = IndexFactory::CreateCluster("StratifiedCluster");
  ASSERT_EQ(0, cluster->init(metadata, params()));
  auto source = std::make_shared<GeneratedStratifiedHolder>(
      metadata, pool.capacity() * 2 / metadata.element_size() + 1);
  source->ordinal_values = true;
  calls.retain_values = false;
  ASSERT_GT(source->count() * source->element_size(), pool.capacity());
  IndexCluster::CentroidList cents;
  ASSERT_EQ(0, dynamic_cast<HolderCluster *>(cluster.get())
                   ->cluster_holder(threads, source, cents));
  EXPECT_GT(calls.max_batch_bytes, 0u);
  EXPECT_LE(calls.max_batch_bytes, 256u * 1024);
  EXPECT_EQ(2u, calls.child_calls);
  EXPECT_EQ(1u, source->iterations);
  IndexCluster::CentroidList().swap(cents);
  EXPECT_EQ(0u, pool.used());
}

TEST_F(StratifiedClusterStorageTest,
       FailuresDoNotRetryOrPublishPartialCentroids) {
  for (int failure = 0; failure < 4; ++failure) {
    calls = {};
    auto &pool = ailego::PoolPayloadBudget::get_instance();
    ASSERT_EQ(0, pool.init(budget_));
    auto cluster = IndexFactory::CreateCluster("StratifiedCluster");
    ASSERT_EQ(0, cluster->init(meta(), params()));
    auto source = std::make_shared<GeneratedStratifiedHolder>(meta(), 32);
    source->ordinal_values = true;
    if (failure == 1) source->fail_at = 7;
    if (failure == 2) calls.reject_child = 2;
    if (failure == 3) calls.invalid_label = true;
    const float row[8]{42.0f};
    IndexCluster::CentroidList cents{IndexCluster::Centroid(row, sizeof(row))};
    ailego::MemoryReservation pressure;
    if (failure == 0) pressure.add(pool.capacity() - pool.used());
    const int ret = dynamic_cast<HolderCluster *>(cluster.get())
                        ->cluster_holder(threads, source, cents);
    EXPECT_NE(0, ret);
    EXPECT_NE(IndexError_NotImplemented, ret);
    if (failure == 1) EXPECT_EQ(IndexError_ReadData, ret);
    if (failure == 2) EXPECT_EQ(IndexError_NoMemory, ret);
    ASSERT_EQ(1u, cents.size());
    EXPECT_EQ(0, std::memcmp(row, cents[0].feature(), sizeof(row)));
    EXPECT_LE(source->iterations, 1u);
    pressure.retain_at_most(0);
    IndexCluster::CentroidList().swap(cents);
    EXPECT_EQ(0u, pool.used());
  }
}

TEST_F(StratifiedClusterStorageTest,
       UnsupportedChildAndUnbudgetedCallDoNotConsumeInput) {
  for (bool pooled : {false, true}) {
    ASSERT_EQ(0, ailego::PoolPayloadBudget::get_instance().init(
                     pooled ? 8 * 1024 * 1024 : 0));
    auto cluster = IndexFactory::CreateCluster("StratifiedCluster");
    ASSERT_EQ(0, cluster->init(meta(), params(pooled ? "RejectingSecondCluster"
                                                     : "OptKmeansCluster")));
    auto source = std::make_shared<GeneratedStratifiedHolder>(meta(), 32);
    IndexCluster::CentroidList cents;
    EXPECT_EQ(IndexError_NotImplemented,
              dynamic_cast<HolderCluster *>(cluster.get())
                  ->cluster_holder(threads, source, cents));
    EXPECT_EQ(0u, source->iterations);
    EXPECT_TRUE(cents.empty());
  }
}


TEST_F(StratifiedClusterStorageTest,
       RealTrainingMatchesLegacyForNumericPrecisions) {
  for (const auto type :
       {IndexMeta::DT_FP32, IndexMeta::DT_FP16, IndexMeta::DT_INT8}) {
    SCOPED_TRACE(type);
    const auto metadata = meta(type);
    GeneratedStratifiedHolder generator(metadata, 128);
    auto features = std::make_shared<CompactIndexFeatures>(metadata);
    std::vector<char> row(metadata.element_size());
    for (size_t i = 0; i < generator.count(); ++i) {
      generator.fill(i, row.data());
      features->emplace(row.data());
    }
    IndexCluster::CentroidList initial;
    for (size_t i = 0; i < 2; ++i) {
      initial.emplace_back(features->element(i), metadata.element_size());
      // Fix the second-level initialization too, avoiding random-seed effects.
      for (size_t j = 0; j < 2; ++j)
        initial.back().mutable_subitems()->emplace_back(
            features->element(i + j * 2), metadata.element_size());
    }
    auto legacy = IndexFactory::CreateCluster("StratifiedCluster");
    ASSERT_EQ(0, legacy->init(metadata, params("OptKmeansCluster")));
    ASSERT_EQ(0, legacy->mount(features));
    auto reference = initial;
    ASSERT_EQ(0, legacy->cluster(threads, reference));
    legacy.reset();
    auto buffered = IndexFactory::CreateCluster("StratifiedCluster");
    ASSERT_EQ(0, buffered->init(metadata, params("OptKmeansCluster")));
    auto source = std::make_shared<GeneratedStratifiedHolder>(metadata, 128);
    auto actual = initial;
    ASSERT_EQ(0, dynamic_cast<HolderCluster *>(buffered.get())
                     ->cluster_holder(threads, source, actual));
    ASSERT_EQ(reference.size(), actual.size());
    for (size_t i = 0; i < actual.size(); ++i) {
      EXPECT_EQ(reference[i].buffer(), actual[i].buffer());
      ASSERT_EQ(reference[i].subitems().size(), actual[i].subitems().size());
      for (size_t j = 0; j < actual[i].subitems().size(); ++j) {
        const auto &expected = reference[i].subitems()[j];
        const auto &value = actual[i].subitems()[j];
        EXPECT_EQ(expected.buffer(), value.buffer());
        EXPECT_EQ(expected.follows(), value.follows());
        EXPECT_DOUBLE_EQ(expected.score(), value.score());
      }
    }
    EXPECT_EQ(1u, source->iterations);
    // Trained output now owns budgeted storage until the consumer releases it.
    IndexCluster::CentroidList().swap(actual);
    IndexCluster::CentroidList().swap(reference);
    IndexCluster::CentroidList().swap(initial);
    buffered.reset();
    features.reset();
    EXPECT_EQ(0u, ailego::PoolPayloadBudget::get_instance().used());
  }
}

TEST_F(StratifiedClusterStorageTest,
       TrainerUsesPagedPathForFullAndSampledInput) {
  for (uint32_t sample : {0u, 64u, 512u}) {
    SCOPED_TRACE(sample);
    auto trainer = IndexFactory::CreateTrainer("StratifiedClusterTrainer");
    Params options, level;
    options.set(STRATIFIED_TRAINER_CLUSTER_COUNT, "2*2");
    options.set(STRATIFIED_TRAINER_SAMPLE_COUNT, sample);
    level.set(OPTKMEANS_CLUSTER_MAX_ITERATIONS, 3);
    options.set(STRATIFIED_TRAINER_PARAMS_IN_LEVEL_PREFIX + "1", level);
    options.set(STRATIFIED_TRAINER_PARAMS_IN_LEVEL_PREFIX + "2", level);
    ASSERT_EQ(0, trainer->init(meta(), options));
    auto workers = std::make_shared<SingleQueueIndexThreads>(2, false);
    for (size_t run = 0; run < 2; ++run) {
      auto source = std::make_shared<GeneratedStratifiedHolder>(meta(), 257);
      ASSERT_EQ(0, trainer->train(workers, source));
      const size_t expected = sample == 64 ? 64 : 257;
      EXPECT_EQ(expected, trainer->stats().trained_count());
      EXPECT_EQ(1u, source->iterations);
      IndexCluster::CentroidList cents;
      ASSERT_EQ(0,
                IndexCluster::Deserialize(meta(), trainer->indexes(), &cents));
      ASSERT_EQ(2u, cents.size());
      size_t follows = 0;
      for (const auto &parent : cents) {
        ASSERT_FALSE(parent.subitems().empty());
        for (const auto &child : parent.subitems()) {
          follows += child.follows();
          EXPECT_TRUE(std::isfinite(child.score()));
        }
      }
      EXPECT_EQ(expected, follows);
      EXPECT_GT(ailego::PoolPayloadBudget::get_instance().used(), 0u);
    }
    ASSERT_EQ(0, trainer->cleanup());
    EXPECT_EQ(0u, ailego::PoolPayloadBudget::get_instance().used());
  }
}

}  // namespace
}  // namespace zvec::core


namespace zvec::core {
namespace {
class RejectingSecondCluster : public IndexCluster {
 public:
  int init(const IndexMeta &, const ailego::Params &p) override {
    p.get("fail_init", &fail_init_);
    p.get("throw", &throw_);
    return fail_init_ ? IndexError_NoMemory : 0;
  }
  int cleanup() override {
    return 0;
  }
  int reset() override {
    return 0;
  }
  int update(const ailego::Params &) override {
    return 0;
  }
  void suggest(uint32_t) override {}
  int mount(IndexFeatures::Pointer) override {
    return 0;
  }
  int cluster(IndexThreads::Pointer, CentroidList &) override {
    if (throw_) throw std::bad_alloc();
    return IndexError_NoMemory;
  }
  int classify(IndexThreads::Pointer, CentroidList &) override {
    return 0;
  }
  int label(IndexThreads::Pointer, const CentroidList &,
            std::vector<uint32_t> *) override {
    return 0;
  }

 private:
  bool fail_init_{false}, throw_{false};
};
INDEX_FACTORY_REGISTER_CLUSTER(RejectingSecondCluster);
TEST(StratifiedClusterStorage,
     LegacySecondStagePreservesReturnedAndThrownErrors) {
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  const size_t previous = pool.capacity();
  ASSERT_EQ(0, pool.init(16 * 1024 * 1024));
  auto restore = ailego::ScopeGuard::Make([&] {
    EXPECT_EQ(0u, pool.used());
    EXPECT_EQ(0, pool.init(previous));
  });
  IndexMeta meta(IndexMeta::DT_FP32, 4);
  meta.set_metric("SquaredEuclidean", 0, ailego::Params{});
  std::vector<float> values(64, 1.0f);
  auto features = std::make_shared<CoherentIndexFeatures>(
      meta, values.data(), values.size() * sizeof(float));
  auto threads = std::make_shared<SingleQueueIndexThreads>(2, false);
  for (int mode = 0; mode < 3; ++mode) {
    ailego::Params first, second, params;
    first.set(KMEANS_CLUSTER_MAX_ITERATIONS, 1);
    second.set("fail_init", mode == 1);
    second.set("throw", mode == 2);
    params.set(STRATIFIED_CLUSTER_FIRST_CLASS, "KmeansCluster");
    params.set(STRATIFIED_CLUSTER_FIRST_COUNT, 2U);
    params.set(STRATIFIED_CLUSTER_FIRST_PARAMS, first);
    params.set(STRATIFIED_CLUSTER_SECOND_CLASS, "RejectingSecondCluster");
    params.set(STRATIFIED_CLUSTER_SECOND_PARAMS, second);
    auto cluster = IndexFactory::CreateCluster("StratifiedCluster");
    ASSERT_EQ(0, cluster->init(meta, params));
    ASSERT_EQ(0, cluster->mount(features));
    IndexCluster::CentroidList centers;
    EXPECT_EQ(IndexError_NoMemory, cluster->cluster(threads, centers));
  }
}
}  // namespace
}  // namespace zvec::core
