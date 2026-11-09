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

#include "pq_buffered_training_test_helper.h"

namespace zvec::core_interface {
namespace {
class CountingPqHolder final : public core::IndexHolder {
 public:
  class Cursor : public Iterator {
   public:
    explicit Cursor(CountingPqHolder *owner) : owner_(owner) {}
    const void *data() const override {
      ++owner_->data_reads;
      for (size_t d = 0; d < 8; ++d) row_[d] = float(d + 1) / 32.0f;
      return row_;
    }
    uint64_t key() const override {
      return index_;
    }
    bool is_valid() const override {
      return index_ < owner_->actual_count;
    }
    int status() const override {
      return index_ == owner_->error_at ? core::IndexError_ReadData : 0;
    }
    void next() override {
      ++index_;
    }

   private:
    CountingPqHolder *owner_;
    size_t index_{0};
    mutable float row_[8]{};
  };
  size_t count() const override {
    return reported_count;
  }
  size_t dimension() const override {
    return 8;
  }
  size_t element_size() const override {
    return 8 * sizeof(float);
  }
  core::IndexMeta::DataType data_type() const override {
    return core::IndexMeta::DT_FP32;
  }
  bool multipass() const override {
    return false;
  }
  Iterator::Pointer create_iterator() override {
    return std::make_unique<Cursor>(this);
  }
  size_t actual_count{65543};
  size_t reported_count{65543};
  size_t error_at{std::numeric_limits<size_t>::max()};
  size_t data_reads{0};
};

template <class Quantizer>
void CheckSampledTraining(bool fail) {
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  const size_t budget =
      ailego::PoolPayloadBudget::storage_metadata_bytes(4096) +
      // This test selects 65,536 ordinals. Its mandatory shuffled-ordinal
      // workspace is now charged too, unlike evictable vector sample pages.
      8 * 1024 * 1024;
  std::string reference;
  for (size_t capacity : {size_t{0}, budget}) {
    ASSERT_EQ(pool.init(capacity), 0);
    auto holder = std::make_shared<CountingPqHolder>();
    if (fail) holder->error_at = holder->actual_count;
    core::IndexMeta meta(core::IndexMeta::DT_FP32, 8);
    meta.set_metric("SquaredEuclidean", 0, ailego::Params());
    ailego::Params params;
    params.set("num_chunk", 2u);
    params.set("thread_count", 2u);
    params.set("markov_chain_length", 0u);
    params.set("build_sdc_table", false);
    Quantizer quantizer;
    ASSERT_EQ(quantizer.init(meta, params), 0);
    if (fail) {
      EXPECT_NE(quantizer.train(holder), 0);
    } else {
      ASSERT_EQ(quantizer.train(holder), 0);
      std::string actual;
      ASSERT_EQ(quantizer.serialize(&actual), 0);
      if (capacity == 0)
        reference = std::move(actual);
      else
        EXPECT_EQ(reference, actual);
    }
    // Even a one-shot source larger than the training limit only copies
    // selected rows. Its reused cursor buffer must not be retained.
    EXPECT_EQ(holder->data_reads, 65536u);
    EXPECT_EQ(pool.used(), pool.external_used());
    EXPECT_LT(pool.used(), 16u * 1024);
  }
}

TEST_F(PqBufferedTrainingTest,
       Int4SamplesBeforeCopyingAndChecksDeferredErrors) {
  CheckSampledTraining<turbo::PqInt4Quantizer>(false);
  CheckSampledTraining<turbo::PqInt4Quantizer>(true);
}
TEST_F(PqBufferedTrainingTest,
       FastSamplesBeforeCopyingAndChecksDeferredErrors) {
  CheckSampledTraining<turbo::PqFastQuantizer>(false);
  CheckSampledTraining<turbo::PqFastQuantizer>(true);
}

TEST_F(PqBufferedTrainingTest, MetadataExhaustionDoesNotFallBackToHeap) {
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  // Enough for the three tiny codebooks, not writable page-table metadata.
  ASSERT_EQ(pool.init(64 * 1024), 0);
  auto input =
      std::make_shared<core::MultiPassIndexHolder<core::IndexMeta::DT_FP32>>(8);
  for (uint32_t i = 0; i < 512; ++i) {
    ailego::NumericalVector<float> row(8);
    ASSERT_TRUE(input->emplace(i, std::move(row)));
  }
  core::IndexMeta meta(core::IndexMeta::DT_FP32, 8);
  meta.set_metric("SquaredEuclidean", 0, ailego::Params());
  ailego::Params params;
  params.set("num_chunk", 2u);
  turbo::PqInt8Quantizer quantizer;
  ASSERT_EQ(quantizer.init(meta, params), 0);
  EXPECT_NE(quantizer.train(input), 0);
  turbo::PqInt4Quantizer int4;
  ASSERT_EQ(int4.init(meta, params), 0);
  EXPECT_NE(int4.train(input), 0);
  turbo::PqFastQuantizer fast;
  ASSERT_EQ(fast.init(meta, params), 0);
  EXPECT_NE(fast.train(input), 0);
  EXPECT_EQ(pool.used(), pool.external_used());
  EXPECT_LT(pool.used(), 64u * 1024);
}

TEST_F(PqBufferedTrainingTest, OpqDistinctRowsMatchAcrossPagedTileBoundaries) {
  static constexpr size_t kDim = 17;
  static constexpr size_t kRows = 4099;
  std::vector<float> original(kRows * kDim);
  for (size_t i = 0; i < original.size(); ++i)
    original[i] = float((i * 17 + i / kDim) % 257) / 128.0f - 1.0f;
  // A fixed, deterministic reconstruction isolates the workspace/rotation
  // changes from random codebook initialization, using distinct input rows.
  const auto reconstruct_sum = [](const float *rows, size_t count, float *out) {
    double error = 0;
    for (size_t i = 0; i < count * kDim; ++i) {
      out[i] = std::round(rows[i] * 4.0f) / 4.0f;
      const float difference = rows[i] - out[i];
      error += static_cast<double>(difference) * difference;
    }
    return error;
  };
  const auto reconstruct = [&](const float *rows, size_t count, float *out) {
    return static_cast<float>(reconstruct_sum(rows, count, out) / count);
  };
  ailego::Params params;
  params.set("rotate_type", std::string("opq"));
  params.set("opq_iter", 3u);
  std::vector<float> reference;
  size_t reference_rounds = 0;
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  const size_t budget =
      2 * ailego::PoolPayloadBudget::storage_metadata_bytes(4096) +
      // Three row tiles, the rotation solver and at least one I/O page must
      // coexist. Each full matrix still exceeds this workspace allowance.
      256 * 1024;
  static_assert(kRows * kDim * sizeof(float) > 256 * 1024);
  for (size_t capacity : {size_t{0}, budget}) {
    ASSERT_EQ(pool.init(capacity), 0);
    turbo::PqOpq opq;
    ASSERT_EQ(opq.init(kDim, turbo::DataType::kFp32, params), 0);
    auto data = original;
    size_t rounds = 0;
    ASSERT_EQ(opq.train(
                  data.data(), kRows, "test", [&](uint32_t) { ++rounds; },
                  reconstruct, reconstruct_sum),
              0);
    EXPECT_EQ(pool.used(), capacity ? kDim * kDim * sizeof(float) : 0u);
    if (!capacity) {
      reference = std::move(data);
      reference_rounds = rounds;
    } else {
      EXPECT_EQ(rounds, reference_rounds);
      for (size_t i = 0; i < data.size(); ++i)
        ASSERT_NEAR(data[i], reference[i], 1e-5f) << i;
    }
  }
}

TEST_F(PqBufferedTrainingTest, UnevenChunksAndRowTilesPreserveDistinctVectors) {
  constexpr size_t kDim = 17;
  auto input =
      std::make_shared<core::MultiPassIndexHolder<core::IndexMeta::DT_FP32>>(
          kDim);
  constexpr size_t kRows = 4099;
  constexpr size_t kWorkspaceBytes = 256 * 1024;
  static_assert(kRows * kDim * sizeof(float) > kWorkspaceBytes);
  for (size_t i = 0; i < kRows; ++i) {
    ailego::NumericalVector<float> row(kDim);
    for (size_t d = 0; d < kDim; ++d)
      row[d] = float((i % 16) * 32 + d) / 512.0f;
    ASSERT_TRUE(input->emplace(i, std::move(row)));
  }
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  const size_t metadata =
      ailego::PoolPayloadBudget::storage_metadata_bytes(4096);
  ASSERT_EQ(pool.init(metadata + kWorkspaceBytes), 0);
  core::IndexMeta meta(core::IndexMeta::DT_FP32, kDim);
  meta.set_metric("SquaredEuclidean", 0, ailego::Params());
  ailego::Params params;
  params.set("num_chunk", 4u);
  params.set("thread_count", 2u);
  params.set("build_sdc_table", false);
  turbo::PqInt8Quantizer quantizer;
  ASSERT_EQ(quantizer.init(meta, params), 0);
  ASSERT_EQ(quantizer.train(input), 0);
  std::vector<uint8_t> codes(quantizer.quantized_datapoint_vector_length());
  for (auto row = input->create_iterator(); row->is_valid(); row->next()) {
    quantizer.quantize_data(row->data(), codes.data());
    std::string decoded;
    const core::IndexQueryMeta qmeta(core::IndexMeta::DT_FP32, kDim);
    ASSERT_EQ(quantizer.dequantize(codes.data(), qmeta, &decoded), 0);
    ASSERT_EQ(decoded.size(), kDim * sizeof(float));
    for (size_t d = 0; d < kDim; ++d) {
      float value;
      std::memcpy(&value, decoded.data() + d * sizeof(float), sizeof(value));
      EXPECT_NEAR(value, static_cast<const float *>(row->data())[d], 0.001f);
    }
  }
}
}  // namespace
}  // namespace zvec::core_interface
