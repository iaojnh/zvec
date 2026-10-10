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

#include <algorithm>
#include <cstring>
#include <limits>
#include <gtest/gtest.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/ailego/buffer/vector_page_table.h>
#include "ailego/pattern/defer.h"
#include "zvec/core/framework/index_factory.h"
#include "zvec/core/framework/index_memory.h"
#include "rabitq_converter.h"
#include "rabitq_utils.h"

namespace zvec::core {
namespace {

class TrainingInput : public IndexHolder {
 public:
  IndexMeta meta{IndexMeta::DT_FP32, 128};
  size_t error_at{std::numeric_limits<size_t>::max()};
  size_t reported_count{259};
  size_t iterations{0};
  size_t extra_dimensions{0};
  TrainingInput() {
    meta.set_metric("SquaredEuclidean", 0, ailego::Params());
  }
  size_t count() const override {
    return reported_count;
  }
  size_t dimension() const override {
    return meta.dimension() + extra_dimensions;
  }
  size_t element_size() const override {
    return dimension() * sizeof(float);
  }
  IndexMeta::DataType data_type() const override {
    return meta.data_type();
  }
  bool multipass() const override {
    return false;
  }

  class Iter : public IndexHolder::Iterator {
   public:
    explicit Iter(const TrainingInput *input)
        : input_(input), row_(input->dimension()) {}
    const void *data() const override {
      for (size_t d = 0; d < row_.size(); ++d) {
        row_[d] = float((ordinal_ * 17 + d * 13) % 257) / 257.0f;
      }
      return row_.data();
    }
    uint64_t key() const override {
      return ordinal_;
    }
    bool is_valid() const override {
      return ordinal_ < 259 && status() == 0;
    }
    int status() const override {
      return ordinal_ == input_->error_at ? IndexError_ReadData : 0;
    }
    void next() override {
      ++ordinal_;
    }

   private:
    const TrainingInput *input_;
    size_t ordinal_{0};
    mutable std::vector<float> row_;
  };
  Iterator::Pointer create_iterator() override {
    ++iterations;
    return std::make_unique<Iter>(this);
  }
};

class RabitqTrainingMemoryTest : public ::testing::TestWithParam<bool> {
 protected:
  void SetUp() override {
    auto &pool = ailego::MemoryLimitPool::get_instance();
    previous_capacity_ = pool.capacity();
    ASSERT_EQ(pool.used(), 0u);
    ASSERT_EQ(
        pool.init(
            GetParam()
                ? 2 * ailego::VecBufferPool::metadata_bytes_for_page_count(
                          4096, true) +
                      64 * 1024
                : 0),
        0);
  }
  void TearDown() override {
    auto &pool = ailego::MemoryLimitPool::get_instance();
    EXPECT_EQ(pool.used(), 0u);
    EXPECT_EQ(pool.init(previous_capacity_), 0);
  }

 private:
  size_t previous_capacity_{0};
};

TEST_P(RabitqTrainingMemoryTest,
       CentroidsMatchLegacySamplingIncludingFullCorpus) {
  auto threads = std::make_shared<SingleQueueIndexThreads>(1, false);
  for (size_t extra_dimensions : {size_t{0}, size_t{1}}) {
    SCOPED_TRACE(extra_dimensions);
    for (uint32_t requested : {0U, 64U, 259U, 300U}) {
      SCOPED_TRACE(requested);
      auto input = std::make_shared<TrainingInput>();
      input->extra_dimensions = extra_dimensions;
      const size_t count = requested == 0
                               ? input->count()
                               : std::min<size_t>(requested, input->count());
      auto sampler =
          std::make_shared<SampleIndexFeatures<CompactIndexFeatures>>(
              input->meta, count);
      for (auto iter = input->create_iterator(); iter->is_valid();
           iter->next()) {
        sampler->emplace(iter->data());
      }
      auto legacy = IndexFactory::CreateCluster("OptKmeansCluster");
      ASSERT_NE(legacy, nullptr);
      ASSERT_EQ(legacy->init(input->meta, ailego::Params()), 0);
      ASSERT_EQ(legacy->mount(sampler), 0);
      // A single centroid has a deterministic mean; independent randomized
      // KMC2 initializations must not be compared bit for bit for k > 1.
      legacy->suggest(1);
      IndexCluster::CentroidList expected;
      ASSERT_EQ(legacy->cluster(threads, expected), 0);

      RabitqConverter converter;
      ailego::Params params;
      params.set(PARAM_RABITQ_NUM_CLUSTERS, 1U);
      params.set(PARAM_RABITQ_SAMPLE_COUNT, requested);
      ASSERT_EQ(converter.init(input->meta, params), 0);
      input->iterations = 0;
      ASSERT_EQ(converter.train(input, threads), 0);
      EXPECT_EQ(input->iterations, 1u);
      EXPECT_EQ(converter.stats().trained_count(), count);

      const std::string id = "rabitq_training_memory_centroids";
      AILEGO_DEFER([&]() { IndexMemory::Instance()->remove(id); });
      auto dumper = IndexFactory::CreateDumper("MemoryDumper");
      ASSERT_NE(dumper, nullptr);
      ASSERT_EQ(dumper->init(ailego::Params()), 0);
      ASSERT_EQ(dumper->create(id), 0);
      ASSERT_EQ(converter.dump(dumper), 0);
      ASSERT_EQ(dumper->close(), 0);
      auto storage = IndexFactory::CreateStorage("MemoryReadStorage");
      ASSERT_NE(storage, nullptr);
      ASSERT_EQ(storage->open(id, false), 0);
      auto segment = storage->get(RABITQ_CONVERTER_SEG_ID);
      ASSERT_NE(segment, nullptr);
      RabitqConverterHeader header{};
      const void *data = nullptr;
      ASSERT_EQ(segment->read(0, &data, sizeof(header)), sizeof(header));
      std::memcpy(&header, data, sizeof(header));
      ASSERT_EQ(header.num_clusters, expected.size());
      size_t offset = sizeof(header) +
                      header.num_clusters * header.padded_dim * sizeof(float);
      std::vector<float> actual(input->meta.dimension());
      for (const auto &centroid : expected) {
        ASSERT_EQ(segment->read(offset, &data, input->meta.element_size()),
                  input->meta.element_size());
        std::memcpy(actual.data(), data, input->meta.element_size());
        EXPECT_EQ(std::memcmp(actual.data(), centroid.feature(),
                              input->meta.element_size()),
                  0);
        offset += input->meta.element_size();
      }
    }
  }
}

TEST_P(RabitqTrainingMemoryTest, DoesNotRetryAfterInputError) {
  for (uint32_t sample_count : {0U, 64U}) {
    for (size_t fail_at : {size_t{0}, size_t{11}, size_t{259}}) {
      auto input = std::make_shared<TrainingInput>();
      input->error_at = fail_at;
      RabitqConverter converter;
      ailego::Params params;
      params.set(PARAM_RABITQ_NUM_CLUSTERS, 4U);
      params.set(PARAM_RABITQ_SAMPLE_COUNT, sample_count);
      ASSERT_EQ(converter.init(input->meta, params), 0);
      auto threads = std::make_shared<SingleQueueIndexThreads>(1, false);
      EXPECT_EQ(converter.train(input, threads), IndexError_ReadData);
      EXPECT_EQ(input->iterations, 1u);
    }
  }
}

TEST_P(RabitqTrainingMemoryTest, FullTrainingValidatesReportedCount) {
  for (size_t count : {size_t{258}, size_t{260}}) {
    auto input = std::make_shared<TrainingInput>();
    input->reported_count = count;
    RabitqConverter converter;
    ailego::Params params;
    params.set(PARAM_RABITQ_NUM_CLUSTERS, 4U);
    ASSERT_EQ(converter.init(input->meta, params), 0);
    auto threads = std::make_shared<SingleQueueIndexThreads>(1, false);
    EXPECT_NE(converter.train(input, threads), 0);
    EXPECT_EQ(input->iterations, 1u);
  }
}

INSTANTIATE_TEST_SUITE_P(StorageModes, RabitqTrainingMemoryTest,
                         ::testing::Bool());

}  // namespace
}  // namespace zvec::core
