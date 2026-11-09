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
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/ailego/container/params.h>
#include <zvec/ailego/utility/float_helper.h>
#include <zvec/core/framework/index_holder.h>
#include <zvec/core/framework/index_meta.h>
#include "ailego/pattern/defer.h"
#include "quantizer/common/pq_training_samples.h"
#include "quantizer/pq_fast_quantizer/pq_fast_quantizer.h"
#include "quantizer/pq_int4_quantizer/pq_int4_quantizer.h"
#include "quantizer/pq_int8_quantizer/pq_int8_quantizer.h"
#include "../ailego/buffer/pool_payload_budget.h"

namespace zvec {
namespace turbo {

// Keep the production declaration unchanged in every translation unit. Using
// a private-to-public macro would change MSVC's mangled member-function names.
class PqInt8QuantizerTestAccess {
 public:
  static const std::vector<float> &dist_table(
      const PqInt8Quantizer &quantizer) {
    return quantizer.dist_table_;
  }

  static int compute_dist_table(PqInt8Quantizer &quantizer) {
    return quantizer.compute_dist_table();
  }
  static size_t paged_bytes(const PqInt8Quantizer &quantizer) {
    return quantizer.paged_dist_table_ ? quantizer.paged_dist_table_->size()
                                       : 0;
  }
};

namespace {

template <typename QuantizerType>
void check_small_codebook_budget() {
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  pool.init(4 * 1024 * 1024);
  auto reset = ailego::ScopeGuard::Make([&] { pool.init(0); });
  {
    QuantizerType quantizer;
    core::IndexMeta meta;
    meta.set_meta(core::IndexMeta::DT_FP32, 32);
    meta.set_metric("SquaredEuclidean", 0, ailego::Params());
    ailego::Params params;
    params.set("num_chunk", 8u);
    params.set("thread_count", 1u);
    params.set("markov_chain_length", 0u);
    ASSERT_EQ(0, quantizer.init(meta, params));
    const size_t initial = pool.external_used();
    EXPECT_GE(initial, 32u * 16 * sizeof(float));
    auto holder =
        std::make_shared<core::MultiPassIndexHolder<core::IndexMeta::DT_FP32>>(
            32);
    for (size_t i = 0; i < 128; ++i) {
      ailego::NumericalVector<float> vector(32);
      for (size_t d = 0; d < 32; ++d) vector[d] = float((i * 17 + d) % 31);
      holder->emplace(i, vector);
    }
    ASSERT_EQ(0, quantizer.train(holder));
    const auto resident = pool.external_used();
    EXPECT_GT(resident, initial);
    std::string before, after, table;
    ASSERT_EQ(0, quantizer.serialize(&before));
    std::vector<float> center(32, 0.0f);
    ASSERT_EQ(
        0, quantizer.build_centroid_distance_table(center.data(), 1, &table));
    EXPECT_EQ(resident, pool.external_used());
    ASSERT_EQ(0, quantizer.serialize(&after));
    EXPECT_EQ(before, after);
    ASSERT_EQ(0, quantizer.init(meta, params));
    EXPECT_EQ(initial, pool.external_used());
    ASSERT_EQ(0, quantizer.deserialize(before));
    ASSERT_EQ(
        0, quantizer.build_centroid_distance_table(center.data(), 1, &after));
    EXPECT_EQ(table, after);
  }
  EXPECT_EQ(0u, pool.external_used());
  pool.init(64 * 1024);
  {
    QuantizerType quantizer;
    core::IndexMeta meta;
    meta.set_meta(core::IndexMeta::DT_FP32, 2048);
    ailego::Params params;
    params.set("num_chunk", 128u);
    EXPECT_EQ(core::IndexError_NoMemory, quantizer.init(meta, params));
  }
  EXPECT_EQ(0u, pool.external_used());
}

template <typename QuantizerType>
void check_legacy_training_budget_failure() {
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  const size_t previous = pool.capacity();
  ASSERT_EQ(0, pool.init(4UL << 20));
  AILEGO_DEFER([&] { EXPECT_EQ(0, pool.init(previous)); });
  auto holder =
      std::make_shared<core::MultiPassIndexHolder<core::IndexMeta::DT_FP32>>(32);
  for (size_t i = 0; i < 512; ++i) {
    ailego::NumericalVector<float> row(32);
    for (size_t d = 0; d < 32; ++d) row[d] = float((i * 17 + d) % 31);
    holder->emplace(i, row);
  }
  QuantizerType quantizer;
  core::IndexMeta meta(core::IndexMeta::DT_FP32, 32);
  meta.set_metric("SquaredEuclidean", 0, ailego::Params());
  ailego::Params params;
  params.set("num_chunk", 8u);
  params.set("thread_count", 2u);
  params.set("markov_chain_length", 0u);
  params.set("precompute_sdc", false);
  params.set("build_sdc_table", false);
  ASSERT_EQ(0, quantizer.init(meta, params));
  quantizer.set_buffered_training(false);
  {
    ailego::MemoryReservation pressure;
    pressure.add(pool.capacity() - pool.used() - 256);
    const size_t blocked = pool.external_used();
    // The model is initialized and samples are available; deny the chunk
    // workers' KMeans workspace, then retry after releasing pressure.
    EXPECT_EQ(core::IndexError_NoMemory, quantizer.train(holder));
    EXPECT_EQ(blocked, pool.external_used());
  }
  ASSERT_EQ(0, quantizer.train(holder));
  std::string model;
  EXPECT_EQ(0, quantizer.serialize(&model));
  EXPECT_FALSE(model.empty());
}

TEST(PqModelMemoryTest, LegacyInt8WorkerBudgetFailureCanRetry) {
  check_legacy_training_budget_failure<PqInt8Quantizer>();
}
TEST(PqModelMemoryTest, FastQueryAdmitsBothLookupTableBuffers) {
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  const size_t previous = pool.capacity();
  ASSERT_EQ(0, pool.init(4 * 1024 * 1024));
  AILEGO_DEFER([&] { EXPECT_EQ(0, pool.init(previous)); });

  constexpr size_t kDim = 32;
  constexpr uint32_t kChunks = 8;
  PqFastQuantizer quantizer;
  core::IndexMeta meta(core::IndexMeta::DT_FP32, kDim);
  meta.set_metric("SquaredEuclidean", 0, ailego::Params());
  ailego::Params params;
  params.set("num_chunk", kChunks);
  params.set("thread_count", 1u);
  params.set("markov_chain_length", 0u);
  ASSERT_EQ(0, quantizer.init(meta, params));
  auto holder =
      std::make_shared<core::MultiPassIndexHolder<core::IndexMeta::DT_FP32>>(
          kDim);
  for (size_t i = 0; i < 64; ++i) {
    ailego::NumericalVector<float> row(kDim);
    for (size_t d = 0; d < kDim; ++d) row[d] = float((i * 17 + d) % 31);
    holder->emplace(i, row);
  }
  ASSERT_EQ(0, quantizer.train(holder));

  const std::vector<float> query(kDim, 0.0f);
  const size_t output_bytes = quantizer.quantized_query_vector_length();
  std::vector<uint8_t> expected(output_bytes);
  quantizer.quantize_query(query.data(), expected.data());
  const size_t baseline = pool.external_used();
  // First refuse the float table, then leave room for it but refuse the byte
  // table that overlaps it during packing. The packed output is caller-owned.
  const size_t float_bytes = kChunks * 16 * sizeof(float);
  for (size_t remaining : {size_t{1}, float_bytes + 64}) {
    std::vector<uint8_t> output(output_bytes, 0xa5);
    const auto untouched = output;
    {
      ailego::MemoryReservation pressure;
      pressure.add(pool.capacity() - pool.used() - remaining);
      const size_t blocked = pool.external_used();
      EXPECT_THROW(quantizer.quantize_query(query.data(), output.data()),
                   std::bad_alloc);
      EXPECT_EQ(untouched, output);
      EXPECT_EQ(blocked, pool.external_used());
    }
    EXPECT_NO_THROW(quantizer.quantize_query(query.data(), output.data()));
    EXPECT_EQ(expected, output);
    EXPECT_EQ(baseline, pool.external_used());
  }
}

TEST(PqInt8ModelMemoryTest, BudgetDenialIsReportedAndDoesNotLeak) {
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  pool.init(64 * 1024);
  {
    PqInt8Quantizer quantizer;
    core::IndexMeta meta;
    meta.set_meta(core::IndexMeta::DT_FP32, 768);
    ailego::Params params;
    params.set("num_chunk", 384u);
    EXPECT_EQ(core::IndexError_NoMemory, quantizer.init(meta, params));
    EXPECT_EQ(0u, pool.external_used());
    ASSERT_EQ(0, pool.init(8 * 1024 * 1024));
    params.set("build_sdc_table", std::string("invalid"));
    EXPECT_NE(0, quantizer.init(meta, params));
    EXPECT_EQ(0u, pool.external_used());
    params.set("build_sdc_table", false);
    ASSERT_EQ(0, quantizer.init(meta, params));
    EXPECT_GE(pool.external_used(), 768u * 256 * sizeof(float));
  }
  pool.init(0);
}

using PqTestAccess = PqInt8QuantizerTestAccess;

class PqInt8TrainingMemoryTest
    : public ::testing::TestWithParam<core::IndexMeta::DataType> {
 protected:
  static constexpr uint32_t kDimension = 17;
  static constexpr uint32_t kChunks = 4;
  static constexpr size_t kCount = 512;

  core::IndexMeta input_meta() const {
    core::IndexMeta meta;
    meta.set_meta(GetParam(), kDimension);
    meta.set_metric("SquaredEuclidean", 0, ailego::Params());
    return meta;
  }

  static ailego::Params training_params() {
    ailego::Params params;
    params.set("num_chunk", kChunks);
    params.set("thread_count", 2u);
    params.set("markov_chain_length", 0u);
    return params;
  }

  template <core::IndexMeta::DataType Type, typename T>
  static core::IndexHolder::Pointer make_typed_holder(bool varied) {
    auto holder =
        std::make_shared<core::MultiPassIndexHolder<Type>>(kDimension);
    std::mt19937 generator(42);
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    for (size_t i = 0; i < kCount; ++i) {
      ailego::NumericalVector<T> vector(kDimension);
      for (size_t d = 0; d < kDimension; ++d) {
        vector[d] = T(varied ? distribution(generator)
                             : static_cast<float>(d + 1) / 32.0f);
      }
      holder->emplace(i, vector);
    }
    return holder;
  }

  core::IndexHolder::Pointer make_holder(bool varied) const {
    if (GetParam() == core::IndexMeta::DT_FP16) {
      return make_typed_holder<core::IndexMeta::DT_FP16, ailego::Float16>(
          varied);
    }
    return make_typed_holder<core::IndexMeta::DT_FP32, float>(varied);
  }

  static std::vector<uint8_t> encode_all(
      const PqInt8Quantizer &quantizer,
      const core::IndexHolder::Pointer &holder) {
    const size_t length = quantizer.quantized_datapoint_vector_length();
    std::vector<uint8_t> codes(holder->count() * length);
    auto iterator = holder->create_iterator();
    for (size_t i = 0; iterator->is_valid(); iterator->next(), ++i) {
      quantizer.quantize_data(iterator->data(), codes.data() + i * length);
    }
    return codes;
  }

};


TEST_P(PqInt8TrainingMemoryTest, PagedSdcMatchesHeapAndHandleOwnsLifetime) {
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  const size_t previous = pool.capacity();
  auto restore = ailego::ScopeGuard::Make([&] {
    EXPECT_EQ(0u, pool.used());
    EXPECT_EQ(0, pool.init(previous));
  });
  for (const char *metric : {"SquaredEuclidean", "InnerProduct", "Cosine"}) {
    ASSERT_EQ(0u, pool.used());
    ASSERT_EQ(0, pool.init(0));
    PqInt8Quantizer quantizer;
    auto meta = input_meta();
    meta.set_metric(metric, 0, ailego::Params());
    auto params = training_params();
    ASSERT_EQ(0, quantizer.init(meta, params));
    auto holder = make_holder(true);
    ASSERT_EQ(0, quantizer.train(holder));
    auto codes = encode_all(quantizer, holder);
    const size_t stride = quantizer.quantized_datapoint_vector_length();
    std::vector<float> expected(32);
    for (size_t i = 0; i < expected.size(); ++i)
      expected[i] = quantizer.calc_distance_dp_dp(codes.data(),
                                                  codes.data() + i * stride);
    // Account for platform-dependent writable metadata (16 KiB macOS pages),
    // but leave fewer resident pages than the 1 MiB SDC table needs.
    const size_t budget =
        ailego::PoolPayloadBudget::storage_metadata_bytes(256) +
        32 * ailego::kVectorPageSize;
    ASSERT_EQ(0, pool.init(budget));
    ASSERT_EQ(0, PqTestAccess::compute_dist_table(quantizer));
    EXPECT_EQ(0u, PqTestAccess::dist_table(quantizer).capacity());
    EXPECT_EQ(kChunks * 256u * 256u * sizeof(float),
              PqTestAccess::paged_bytes(quantizer));
    auto handle = quantizer.sym_distance(codes.data(), core::IndexQueryMeta());
    std::vector<const void *> candidates(expected.size());
    std::vector<float> batch_distances(expected.size());
    for (size_t i = 0; i < candidates.size(); ++i)
      candidates[i] = codes.data() + i * stride;
    handle.batch(candidates.data(), candidates.size(), batch_distances.data());
    for (size_t i = 0; i < expected.size(); ++i) {
      const void *code = codes.data() + i * stride;
      EXPECT_NEAR(expected[i],
                  quantizer.calc_distance_dp_dp(codes.data(), code), 1e-5f);
      EXPECT_NEAR(expected[i], handle(code), 1e-5f);
      EXPECT_NEAR(expected[i], batch_distances[i], 1e-5f);
    }
    params.set("build_sdc_table", false);
    ASSERT_EQ(0, quantizer.init(meta, params));
    EXPECT_EQ(0u, PqTestAccess::paged_bytes(quantizer));
    EXPECT_NEAR(expected[1], handle(codes.data() + stride), 1e-5f);
    EXPECT_LE(pool.used(), pool.capacity());
  }
}

TEST_P(PqInt8TrainingMemoryTest, FusedQueryOwnsOneBudgetedReusableLut) {
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  const size_t previous = pool.capacity();
  auto restore = ailego::ScopeGuard::Make([&] {
    EXPECT_EQ(0u, pool.used());
    EXPECT_EQ(0, pool.init(previous));
  });
  ASSERT_EQ(0u, pool.used());
  ASSERT_EQ(0, pool.init(0));
  PqInt8Quantizer quantizer;
  auto params = training_params();
  params.set("build_sdc_table", false);
  ASSERT_EQ(0, quantizer.init(input_meta(), params));
  auto holder = make_holder(true);
  ASSERT_EQ(0, quantizer.train(holder));
  auto it = holder->create_iterator();
  std::string query(static_cast<const char *>(it->data()),
                    holder->element_size());
  auto codes = encode_all(quantizer, holder);
  std::string lut(quantizer.quantized_query_vector_length(), '\0');
  quantizer.quantize_query(query.data(), lut.data());
  auto legacy = quantizer.distance(lut.data(), core::IndexQueryMeta());
  DistanceImpl unbudgeted;
  ASSERT_EQ(0, quantizer.prepare_distance(query.data(), &unbudgeted));
  ASSERT_EQ(0, pool.init(1024 * 1024));
  {
    auto fused = std::move(unbudgeted);
    ASSERT_EQ(0, quantizer.prepare_distance(query.data(), &fused));
    EXPECT_EQ(lut.size(), fused.query_bytes());
    EXPECT_EQ(0, std::memcmp(lut.data(), fused.query_data(), lut.size()));
    EXPECT_EQ(std::string_view(lut), fused.query_storage());
    const size_t charged = pool.used();
    EXPECT_GE(charged, lut.size());
    EXPECT_LT(charged, 2 * lut.size());
    const void *original = fused.query_data();
    ASSERT_EQ(0, quantizer.prepare_distance(query.data(), &fused));
    EXPECT_EQ(original, fused.query_data());
    EXPECT_EQ(charged, pool.used());
    EXPECT_FLOAT_EQ(legacy(codes.data()), fused(codes.data()));
    auto retained = fused;
    it->next();
    ASSERT_EQ(0, quantizer.prepare_distance(it->data(), &fused));
    EXPECT_NE(retained.query_data(), fused.query_data());
    EXPECT_FLOAT_EQ(legacy(codes.data()), retained(codes.data()));
  }
  EXPECT_EQ(0u, pool.used());
  ASSERT_EQ(0, pool.init(1));
  DistanceImpl rejected;
  EXPECT_EQ(core::IndexError_NoMemory,
            quantizer.prepare_distance(query.data(), &rejected));
  EXPECT_FALSE(rejected.valid());
  EXPECT_EQ(0u, pool.used());
}

INSTANTIATE_TEST_SUITE_P(InputPrecisions, PqInt8TrainingMemoryTest,
                         ::testing::Values(core::IndexMeta::DT_FP16,
                                           core::IndexMeta::DT_FP32));

}  // namespace
}  // namespace turbo
}  // namespace zvec
