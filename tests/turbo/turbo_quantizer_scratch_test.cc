// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include "ailego/pattern/defer.h"
#include "quantizer/fp16_quantizer/fp16_quantizer.h"
#include "../ailego/buffer/pool_payload_budget.h"

namespace zvec::turbo {
namespace {
class CountingQuantizer : public Fp16Quantizer {
 public:
  float calc_distance_dp_query(const void *data,
                               const void *query) const override {
    max_charged =
        std::max(max_charged, ailego::PoolPayloadBudget::get_instance().used());
    return Fp16Quantizer::calc_distance_dp_query(data, query);
  }
  void calc_distance_dp_query_batch(const void *const *data, int count,
                                    const void *query,
                                    float *distances) const override {
    largest_batch = std::max(largest_batch, count);
    ++batches;
    max_charged =
        std::max(max_charged, ailego::PoolPayloadBudget::get_instance().used());
    Fp16Quantizer::calc_distance_dp_query_batch(data, count, query, distances);
  }
  mutable int largest_batch{0};
  mutable size_t batches{0};
  mutable size_t max_charged{0};
};

TEST(QuantizerScratch, BatchesAreBoundedAndMatchScalarDistances) {
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  const size_t previous = pool.capacity();
  ASSERT_EQ(0u, pool.used());
  AILEGO_DEFER([&] {
    EXPECT_EQ(0u, pool.used());
    pool.init(previous);
  });
  ASSERT_EQ(0, pool.init(128 * 1024));
  CountingQuantizer quantizer;
  IndexMeta meta(IndexMeta::DT_FP32, 64);
  meta.set_metric("SquaredEuclidean", 0, ailego::Params{});
  ASSERT_EQ(0, quantizer.init(meta, {}));
  std::vector<float> a(64, 0.5f), b(64, 1.0f);
  std::vector<const void *> data(10000);
  for (size_t i = 0; i < data.size(); ++i)
    data[i] = i % 2 ? a.data() : b.data();
  std::vector<float> output(data.size());
  quantizer.calc_distance_input_input_batch(
      data.data(), static_cast<int>(data.size()), a.data(), output.data());
  EXPECT_GT(quantizer.batches, 1u);
  EXPECT_LT(quantizer.largest_batch, static_cast<int>(data.size()));
  EXPECT_GT(quantizer.max_charged, 0u);
  EXPECT_LT(quantizer.max_charged, 66 * 1024u);
  EXPECT_EQ(0u, pool.used());
  for (size_t i = 0; i < data.size(); ++i)
    EXPECT_FLOAT_EQ(i % 2 ? 0.0f : 16.0f, output[i]);
  std::vector<uint8_t> query(quantizer.quantized_query_vector_length());
  quantizer.quantize_query(a.data(), query.data());
  quantizer.calc_distance_input_query_batch(
      data.data(), static_cast<int>(data.size()), query.data(), output.data());
  EXPECT_FLOAT_EQ(output[0],
                  quantizer.calc_distance_input_query(b.data(), query.data()));
  EXPECT_FLOAT_EQ(output[1],
                  quantizer.calc_distance_input_input(a.data(), a.data()));
  EXPECT_EQ(0u, pool.used());
}

TEST(QuantizerScratch, ShrinksToAvailableBudgetAndReleasesAfterFailure) {
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  const size_t previous = pool.capacity();
  ASSERT_EQ(0u, pool.used());
  AILEGO_DEFER([&] {
    EXPECT_EQ(0u, pool.used());
    pool.init(previous);
  });
  CountingQuantizer quantizer;
  IndexMeta meta(IndexMeta::DT_FP32, 64);
  meta.set_metric("SquaredEuclidean", 0, ailego::Params{});
  ASSERT_EQ(0, quantizer.init(meta, {}));
  std::vector<float> row(64, 0.5f), output(5000);
  std::vector<const void *> data(output.size(), row.data());
  ASSERT_EQ(0, pool.init(1));
  EXPECT_THROW(quantizer.calc_distance_input_input_batch(
                   data.data(), data.size(), row.data(), output.data()),
               std::bad_alloc);
  EXPECT_EQ(0u, pool.used());
  ASSERT_EQ(0, pool.init(1024));
  EXPECT_NO_THROW(quantizer.calc_distance_input_input_batch(
      data.data(), data.size(), row.data(), output.data()));
  EXPECT_LE(quantizer.max_charged, pool.capacity());
  EXPECT_EQ(0u, pool.used());
  for (float value : output) EXPECT_FLOAT_EQ(0.0f, value);
}

TEST(QuantizerScratch, LargeSingleVectorsAreChargedAndNotRetained) {
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  const size_t previous = pool.capacity();
  ASSERT_EQ(0u, pool.used());
  AILEGO_DEFER([&] {
    EXPECT_EQ(0u, pool.used());
    pool.init(previous);
  });
  CountingQuantizer quantizer;
  IndexMeta meta(IndexMeta::DT_FP32, 4096);
  ASSERT_EQ(0, quantizer.init(meta, {}));
  std::vector<float> row(4096, 0.5f);
  ASSERT_EQ(0, pool.init(8192));
  EXPECT_THROW(quantizer.calc_distance_input_input(row.data(), row.data()),
               std::bad_alloc);
  EXPECT_EQ(0u, pool.used());
  ASSERT_EQ(0, pool.init(32768));
  EXPECT_FLOAT_EQ(0.0f,
                  quantizer.calc_distance_input_input(row.data(), row.data()));
  EXPECT_GE(quantizer.max_charged, 16384u);
  EXPECT_EQ(0u, pool.used());
  EXPECT_NO_THROW(
      quantizer.calc_distance_input_query_batch(nullptr, 0, nullptr, nullptr));
}
}  // namespace
}  // namespace zvec::turbo

namespace zvec::turbo {
TEST(QuantizerScratch,
     DistancePayloadCapacityFollowsCopyAndRejectsBeforeGrowth) {
  auto &pool = zvec::ailego::PoolPayloadBudget::get_instance();
  const size_t previous = pool.capacity();
  ASSERT_EQ(0u, pool.used());
  AILEGO_DEFER([&] {
    EXPECT_EQ(0u, pool.used());
    pool.init(previous);
  });
  ASSERT_EQ(0, pool.init(128 * 1024));
  zvec::turbo::Fp16Quantizer quantizer;
  zvec::core::IndexMeta meta(zvec::core::IndexMeta::DT_FP32, 16384);
  meta.set_metric("SquaredEuclidean", 0, zvec::ailego::Params{});
  ASSERT_EQ(0, quantizer.init(meta, {}));
  std::vector<float> input(16384, 1.0f);
  std::string encoded;
  zvec::core::IndexQueryMeta encoded_meta;
  zvec::core::IndexQueryMeta query_meta(zvec::core::IndexMeta::DT_FP32, 16384);
  ASSERT_EQ(
      0, quantizer.quantize(input.data(), query_meta, &encoded, &encoded_meta));
  {
    auto distance = quantizer.distance(encoded.data(), encoded_meta);
    const size_t retained = pool.used();
    EXPECT_GE(retained, encoded.size());
    auto copy = distance;
    EXPECT_EQ(2 * retained, pool.used());
    zvec::ailego::MemoryReservation pressure;
    pressure.add(pool.capacity() - pool.used());
    EXPECT_THROW(quantizer.distance(encoded.data(), encoded_meta),
                 std::bad_alloc);
    EXPECT_FLOAT_EQ(0.0f, copy(encoded.data()));
  }
}

}  // namespace zvec::turbo
