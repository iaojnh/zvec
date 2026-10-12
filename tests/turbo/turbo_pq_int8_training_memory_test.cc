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
#include "quantizer/pq_int8_quantizer/pq_int8_quantizer.h"

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

  void check_derived_sdc(bool varied);
};

TEST_P(PqInt8TrainingMemoryTest, DisabledSdcMatchesDefaultTraining) {
  PqInt8Quantizer with_sdc;
  PqInt8Quantizer without_sdc;
  auto params = training_params();
  ASSERT_EQ(0, with_sdc.init(input_meta(), params));
  params.set("build_sdc_table", false);
  ASSERT_EQ(0, without_sdc.init(input_meta(), params));

  // Identical vectors remove KMC2's random initialization as a confounder:
  // independent trainings must produce byte-identical codebooks and codes.
  auto holder = make_holder(false);
  ASSERT_EQ(0, with_sdc.train(holder));
  ASSERT_EQ(0, without_sdc.train(holder));
  EXPECT_EQ(kChunks * 256u * 256u, PqTestAccess::dist_table(with_sdc).size());
  EXPECT_EQ(0u, PqTestAccess::dist_table(without_sdc).size());
  EXPECT_EQ(0u, PqTestAccess::dist_table(without_sdc).capacity());

  std::string default_blob;
  std::string bounded_blob;
  ASSERT_EQ(0, with_sdc.serialize(&default_blob));
  ASSERT_EQ(0, without_sdc.serialize(&bounded_blob));
  EXPECT_EQ(default_blob, bounded_blob);
  EXPECT_EQ(encode_all(with_sdc, holder), encode_all(without_sdc, holder));
}

void PqInt8TrainingMemoryTest::check_derived_sdc(bool varied) {
  PqInt8Quantizer quantizer;
  auto params = training_params();
  params.set("build_sdc_table", false);
  ASSERT_EQ(0, quantizer.init(input_meta(), params));
  auto holder = make_holder(varied);
  ASSERT_EQ(0, quantizer.train(holder));
  ASSERT_EQ(0u, PqTestAccess::dist_table(quantizer).capacity());

  std::string before;
  ASSERT_EQ(0, quantizer.serialize(&before));
  auto codes = encode_all(quantizer, holder);
  auto iterator = holder->create_iterator();
  ASSERT_TRUE(iterator->is_valid());
  std::vector<float> lut(quantizer.quantized_query_vector_length() /
                         sizeof(float));
  quantizer.quantize_query(iterator->data(), lut.data());
  const float distance =
      quantizer.calc_distance_dp_query(codes.data(), lut.data());
  ASSERT_TRUE(std::isfinite(distance));
  if (!varied) {
    // Identical vectors deterministically leave empty clusters. Their NaN
    // centers exercise the same case that can occur with random training.
    ASSERT_TRUE(std::any_of(lut.begin(), lut.end(),
                            [](float value) { return std::isnan(value); }));
  }

  // Hold the trained codebook fixed and add only the skipped derivative.
  ASSERT_EQ(0, PqTestAccess::compute_dist_table(quantizer));
  ASSERT_EQ(kChunks * 256u * 256u, PqTestAccess::dist_table(quantizer).size());
  std::string after;
  ASSERT_EQ(0, quantizer.serialize(&after));
  EXPECT_EQ(before, after);
  EXPECT_EQ(codes, encode_all(quantizer, holder));
  std::vector<float> after_lut(lut.size());
  quantizer.quantize_query(iterator->data(), after_lut.data());
  // Empty clusters produce NaN LUT entries: NaN != NaN even when every bit
  // is unchanged. Compare representations, retaining a strict invariant for
  // both finite values and NaNs instead of relaxing floating-point tolerance.
  for (size_t i = 0; i < lut.size(); ++i) {
    EXPECT_EQ(0, std::memcmp(&lut[i], &after_lut[i], sizeof(float)))
        << "LUT entry " << i;
  }
  EXPECT_EQ(distance,
            quantizer.calc_distance_dp_query(codes.data(), after_lut.data()));
}

TEST_P(PqInt8TrainingMemoryTest, DerivedSdcDoesNotChangeEncodingState) {
  check_derived_sdc(true);
}

TEST_P(PqInt8TrainingMemoryTest, DerivedSdcPreservesEmptyClusterLutEntries) {
  check_derived_sdc(false);
}

TEST_P(PqInt8TrainingMemoryTest, ReinitReleasesTableAndRestoresDefault) {
  PqInt8Quantizer quantizer;
  auto params = training_params();
  auto holder = make_holder(false);
  ASSERT_EQ(0, quantizer.init(input_meta(), params));
  ASSERT_EQ(0, quantizer.train(holder));
  ASSERT_GT(PqTestAccess::dist_table(quantizer).capacity(), 0u);

  params.set("build_sdc_table", false);
  ASSERT_EQ(0, quantizer.init(input_meta(), params));
  EXPECT_EQ(0u, PqTestAccess::dist_table(quantizer).capacity());
  ASSERT_EQ(0, quantizer.train(holder));
  EXPECT_EQ(0u, PqTestAccess::dist_table(quantizer).capacity());

  ASSERT_EQ(0, quantizer.init(input_meta(), training_params()));
  ASSERT_EQ(0, quantizer.train(holder));
  EXPECT_EQ(kChunks * 256u * 256u, PqTestAccess::dist_table(quantizer).size());
}

TEST_P(PqInt8TrainingMemoryTest, PagedSdcMatchesHeapAndHandleOwnsLifetime) {
  auto &pool = ailego::MemoryLimitPool::get_instance();
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
        ailego::VecBufferPool::metadata_bytes_for_page_count(256, true) +
        32 * ailego::kVectorPageSize;
    ASSERT_EQ(0, pool.init(budget));
    ASSERT_EQ(0, PqTestAccess::compute_dist_table(quantizer));
    EXPECT_EQ(0u, PqTestAccess::dist_table(quantizer).capacity());
    EXPECT_EQ(kChunks * 256u * 256u * sizeof(float),
              PqTestAccess::paged_bytes(quantizer));
    auto handle = quantizer.sym_distance(codes.data(), core::IndexQueryMeta());
    for (size_t i = 0; i < expected.size(); ++i) {
      const void *code = codes.data() + i * stride;
      EXPECT_NEAR(expected[i],
                  quantizer.calc_distance_dp_dp(codes.data(), code), 1e-5f);
      EXPECT_NEAR(expected[i], handle(code), 1e-5f);
    }
    params.set("build_sdc_table", false);
    ASSERT_EQ(0, quantizer.init(meta, params));
    EXPECT_EQ(0u, PqTestAccess::paged_bytes(quantizer));
    EXPECT_NEAR(expected[1], handle(codes.data() + stride), 1e-5f);
    EXPECT_LE(pool.used(), pool.capacity());
  }
}

TEST_P(PqInt8TrainingMemoryTest, PooledTrainingBuildsSdcAndTinyBudgetFails) {
  auto &pool = ailego::MemoryLimitPool::get_instance();
  const size_t previous = pool.capacity();
  auto restore = ailego::ScopeGuard::Make([&] {
    EXPECT_EQ(0u, pool.used());
    EXPECT_EQ(0, pool.init(previous));
  });
  ASSERT_EQ(0, pool.init(32 * 1024 * 1024));
  auto holder = make_holder(true);
  std::string codebook;
  {
    PqInt8Quantizer quantizer;
    ASSERT_EQ(0, quantizer.init(input_meta(), training_params()));
    ASSERT_EQ(0, quantizer.train(holder));
    EXPECT_EQ(0u, PqTestAccess::dist_table(quantizer).capacity());
    EXPECT_GT(PqTestAccess::paged_bytes(quantizer), 0u);
    ASSERT_EQ(0, quantizer.serialize(&codebook));
  }
  ASSERT_EQ(0u, pool.used());
  ASSERT_EQ(0, pool.init(1));
  PqInt8Quantizer quantizer;
  ASSERT_EQ(0, quantizer.init(input_meta(), training_params()));
  ASSERT_EQ(0, quantizer.deserialize(codebook.data(), codebook.size()));
  EXPECT_EQ(core::IndexError_NoMemory,
            PqTestAccess::compute_dist_table(quantizer));
  EXPECT_EQ(0u, PqTestAccess::dist_table(quantizer).capacity());
  EXPECT_EQ(0u, PqTestAccess::paged_bytes(quantizer));
}

TEST_P(PqInt8TrainingMemoryTest, RejectsMalformedFlag) {
  PqInt8Quantizer quantizer;
  auto params = training_params();
  params.set("build_sdc_table", std::string("not-a-boolean"));
  EXPECT_NE(0, quantizer.init(input_meta(), params));
  params.set("build_sdc_table", false);
  EXPECT_EQ(0, quantizer.init(input_meta(), params));
}

TEST_P(PqInt8TrainingMemoryTest, FusedQueryOwnsOneBudgetedReusableLut) {
  auto &pool = ailego::MemoryLimitPool::get_instance();
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
