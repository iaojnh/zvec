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

#include <filesystem>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/core/interface/index_factory.h>
#include <zvec/core/interface/index_param_builders.h>
#include "../../ailego/buffer/pool_payload_budget.h"
#if RABITQ_SUPPORTED
#include "ailego/pattern/scope_guard.h"
#include "algorithm/ivf_rabitq/ivf_rabitq_streamer.h"
#include "interface/indexes/buffered_input.h"

namespace zvec::core_interface {
namespace {
class ObservedRabitq : public IVFRabitqIndex {
 public:
  using IVFRabitqIndex::add;
  using IVFRabitqIndex::merge;
  using IVFRabitqIndex::open;
  using IVFRabitqIndex::train;
  bool trained() const {
    return is_trained_;
  }
  int reject_first_reopen() {
    class Streamer : public core::IvfRabitqStreamer {
     public:
      int open(core::IndexStorage::Pointer storage) override {
        if (std::exchange(reject_, false)) return core::IndexError_NoMemory;
        return core::IvfRabitqStreamer::open(std::move(storage));
      }

     private:
      bool reject_{true};
    };
    auto streamer = std::make_shared<Streamer>();
    int ret = streamer->init(proxima_index_meta_, proxima_index_params_);
    if (ret == 0) streamer_ = std::move(streamer);
    return ret;
  }
  int initialize(const BaseIndexParam &param) {
    return init(param);
  }
  bool has_buffered_input() const {
    return buffered_input_ != nullptr;
  }
};

TEST(IvfRabitqBufferedInputTest, SparseIdsOverwriteTrainAndReopen) {
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  const size_t previous = pool.capacity();
  ASSERT_EQ(pool.used(), 0u);
  ASSERT_EQ(pool.init(16 * 1024 * 1024), 0);
  auto restore = ailego::ScopeGuard::Make([&]() {
    EXPECT_EQ(pool.used(), 0u);
    EXPECT_EQ(pool.init(previous), 0);
  });
  const std::string path = "ivf_rabitq_buffered_input.index";
  auto remove =
      ailego::ScopeGuard::Make([&]() { std::filesystem::remove(path); });
  constexpr uint32_t kDim = 128;
  auto param = IVFRabitqIndexParamBuilder()
                   .with_dimension(kDim)
                   .with_data_type(DataType::DT_FP32)
                   .with_metric_type(MetricType::kL2sq)
                   .with_n_list(4)
                   .with_sample_count(128)
                   .build();
  auto index = std::make_shared<ObservedRabitq>();
  ASSERT_EQ(index->initialize(*param), 0);
  ASSERT_EQ(index->open(path, {StorageOptions::StorageType::kBufferPool, true}),
            0);
  ASSERT_TRUE(index->has_buffered_input());
  std::vector<float> query(kDim);
  for (uint32_t i = 0; i < 259; ++i) {
    for (size_t d = 0; d < kDim; ++d)
      query[d] = float((i * 17 + d * 13) % 257) / 257.0f;
    ASSERT_EQ(index->add(VectorData{DenseVector{query.data()}}, i * 2), 0);
  }
  ASSERT_EQ(index->add(VectorData{DenseVector{query.data()}}, 0), 0);
  ASSERT_EQ(index->train(), 0);
  EXPECT_FALSE(index->has_buffered_input());
  EXPECT_NE(index->add(VectorData{DenseVector{query.data()}}, 1), 0);
  auto params = std::make_shared<IVFRabitqQueryParam>();
  params->topk = 5;
  params->nprobe = 4;
  SearchResult before;
  ASSERT_EQ(
      index->search(VectorData{DenseVector{query.data()}}, params, &before), 0);
  ASSERT_EQ(before.doc_list_.size(), 5u);
  for (const auto &doc : before.doc_list_) EXPECT_EQ(doc.key() % 2, 0u);
  ASSERT_EQ(index->close(), 0);
  index.reset();
  auto reopened = IndexFactory::CreateAndInitIndex(*param);
  ASSERT_NE(reopened, nullptr);
  ASSERT_EQ(
      reopened->open(path, {StorageOptions::StorageType::kMMAP, false, true}),
      0);
  SearchResult after;
  ASSERT_EQ(
      reopened->search(VectorData{DenseVector{query.data()}}, params, &after),
      0);
  ASSERT_EQ(after.doc_list_.size(), before.doc_list_.size());
  for (size_t i = 0; i < before.doc_list_.size(); ++i) {
    EXPECT_EQ(after.doc_list_[i].key(), before.doc_list_[i].key());
    EXPECT_FLOAT_EQ(after.doc_list_[i].score(), before.doc_list_[i].score());
  }
  ASSERT_EQ(reopened->close(), 0);
}

TEST(IvfRabitqBufferedInputTest, FailedMergeReopenCanResumeTraining) {
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  const size_t previous = pool.capacity();
  ASSERT_EQ(0, pool.init(32 * 1024 * 1024));
  const std::string source_path = "rabitq_merge_source.index";
  const std::string target_path = "rabitq_merge_target.index";
  auto restore = ailego::ScopeGuard::Make([&] {
    std::filesystem::remove(source_path);
    std::filesystem::remove(target_path);
    EXPECT_EQ(0u, pool.used());
    EXPECT_EQ(0, pool.init(previous));
  });
  constexpr size_t dim = 128;
  auto source_param = FlatIndexParamBuilder()
                          .with_dimension(dim)
                          .with_data_type(DataType::DT_FP32)
                          .with_metric_type(MetricType::kL2sq)
                          .build();
  auto source = IndexFactory::CreateAndInitIndex(*source_param);
  ASSERT_NE(nullptr, source);
  ASSERT_EQ(0, source->open(source_path,
                            {StorageOptions::StorageType::kBufferPool, true}));
  std::vector<float> row(dim);
  for (uint32_t i = 0; i < 64; ++i) {
    for (size_t j = 0; j < dim; ++j) row[j] = float((i * 7 + j) % 31) / 31;
    ASSERT_EQ(0, source->add(VectorData{DenseVector{row.data()}}, i));
  }
  auto target_param = IVFRabitqIndexParamBuilder()
                          .with_dimension(dim)
                          .with_data_type(DataType::DT_FP32)
                          .with_metric_type(MetricType::kL2sq)
                          .with_n_list(2)
                          .with_sample_count(64)
                          .build();
  auto target = std::make_shared<ObservedRabitq>();
  ASSERT_EQ(0, target->initialize(*target_param));
  ASSERT_EQ(0, target->open(target_path,
                            {StorageOptions::StorageType::kBufferPool, true}));
  ASSERT_EQ(0, target->reject_first_reopen());
  EXPECT_EQ(core::IndexError_NoMemory,
            target->merge({source}, {}, {2, nullptr}));
  EXPECT_FALSE(target->trained());
  ASSERT_EQ(0, target->train());
  EXPECT_TRUE(target->trained());
  auto query = std::make_shared<IVFRabitqQueryParam>();
  query->topk = 4;
  query->nprobe = 2;
  SearchResult result;
  EXPECT_EQ(
      0, target->search(VectorData{DenseVector{row.data()}}, query, &result));
  EXPECT_EQ(4u, result.doc_list_.size());
}
}  // namespace
}  // namespace zvec::core_interface
#endif
