// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#include <thread>
#include <gtest/gtest.h>
#include <zvec/core/interface/index_factory.h>
#include <zvec/core/interface/index_param_builders.h>
#include "tests/ailego/buffer/pool_payload_budget.h"
#include "tests/test_util.h"

namespace zvec::core_interface {
namespace {

Index::Pointer MakeIndex(const std::string &path) {
  auto param = FlatIndexParamBuilder()
                   .with_metric_type(MetricType::kL2sq)
                   .with_data_type(DataType::DT_FP32)
                   .with_dimension(4)
                   .build();
  auto index = IndexFactory::CreateAndInitIndex(*param);
  if (!index || index->open(path, {StorageOptions::StorageType::kMMAP, true}))
    return nullptr;
  for (uint64_t i = 0; i < 64; ++i) {
    const float row[] = {static_cast<float>(i), 0, 0, 0};
    VectorData data;
    data.vector = DenseVector{row};
    if (index->add(data, i)) return nullptr;
  }
  if (index->train()) return nullptr;
  return index;
}

void CheckRetainedCandidates(bool refine) {
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  const size_t previous = pool.capacity();
  ASSERT_EQ(0, pool.init(64UL << 20));
  const size_t baseline = pool.external_used();
  std::thread worker([&] {
    const std::string path = "internal_ids_memory_test.index";
    const std::string reference_path = path + ".reference";
    test_util::RemoveTestFiles(path);
    test_util::RemoveTestFiles(reference_path);
    auto index = MakeIndex(path);
    ASSERT_NE(nullptr, index);
    auto reference = refine ? MakeIndex(reference_path) : nullptr;
    if (refine) ASSERT_NE(nullptr, reference);
    auto query = FlatQueryParamBuilder().with_topk(refine ? 16 : 64).build();
    if (refine) {
      auto refiner = std::make_shared<RefinerParam>();
      refiner->reference_index = reference;
      refiner->scale_factor_ = 4;
      query->refiner_param = std::move(refiner);
    }
    const float row[] = {0, 0, 0, 0};
    VectorData data;
    data.vector = DenseVector{row};
    int64_t ids[64];
    float scores[64];
    ASSERT_EQ(0, index->search_internal_ids(data, query, ids, nullptr));
    for (size_t i = 0; i < query->topk; ++i) EXPECT_EQ(i, ids[i]);
    const size_t first_query_memory = pool.external_used();
    if (refine) {
      // Force a larger retained coarse buffer, even though the index still
      // contains only 64 documents. Refusal must leave it usable for retry.
      query->refiner_param->scale_factor_ = 8;
    } else {
      // A smaller scored query must admit its new score buffer while keeping
      // the earlier IDs-only capacity charged independently.
      query->topk = 16;
    }
    {
      ailego::MemoryReservation pressure;
      pressure.add(pool.capacity() - pool.used() - 1);
      EXPECT_EQ(core::IndexError_NoMemory,
                index->search_internal_ids(data, query, ids, scores));
    }
    ASSERT_EQ(0, index->search_internal_ids(data, query, ids, scores));
    if (refine) {
      // All 64 matches and the reference's topk are unchanged. The extra
      // capacity belongs solely to the retained coarse candidate buffer.
      EXPECT_GE(pool.external_used(),
                first_query_memory + 64 * sizeof(uint64_t));
    }
    for (size_t i = 0; i < query->topk; ++i) {
      EXPECT_EQ(i, ids[i]);
      EXPECT_FLOAT_EQ(static_cast<float>(i * i), scores[i]);
    }
    query.reset();
    ASSERT_EQ(0, index->close());
    index.reset();
    if (reference) {
      ASSERT_EQ(0, reference->close());
      reference.reset();
    }
    // Closing the indexes drops their contexts, but these per-thread arrays
    // remain alive. Their reservation must remain alive with them.
    const size_t expected = refine ? 128 * sizeof(uint64_t)
                                   : 64 * sizeof(uint64_t) + 16 * sizeof(float);
    EXPECT_GE(pool.external_used(), baseline + expected);
    test_util::RemoveTestFiles(path);
    test_util::RemoveTestFiles(reference_path);
  });
  worker.join();
  EXPECT_EQ(baseline, pool.external_used());
  EXPECT_EQ(0, pool.init(previous));
}

TEST(InternalIdsMemory, CandidateKeysAndScoresRetainAdmissionAfterIndexClose) {
  CheckRetainedCandidates(false);
}

TEST(InternalIdsMemory, RefineCandidatesRetainAdmissionAfterIndexClose) {
  CheckRetainedCandidates(true);
}

}  // namespace
}  // namespace zvec::core_interface
