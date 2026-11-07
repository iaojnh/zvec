// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#include <array>
#include <random>
#include <gtest/gtest.h>
#include "ailego/pattern/defer.h"
#include "algorithm/hnsw_rabitq/paged_rabitq_rotator.h"
#include "algorithm/hnsw_rabitq/rabitq_converter.h"
#include "algorithm/hnsw_rabitq/rabitq_reformer.h"
#include "algorithm/hnsw_rabitq/rabitq_utils.h"
#include "utility/storage_segment_dumper.h"
#include "ivf_rabitq_reformer.h"
#include "../../../ailego/buffer/pool_payload_budget.h"

namespace zvec::core {
TEST(RabitqTrainingStorageTest, LoadingMatrixDoesNotRegenerateQr) {
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  const size_t previous = pool.capacity();
  auto restore = ailego::ScopeGuard::Make([&] {
    EXPECT_EQ(0u, pool.used());
    EXPECT_EQ(0, pool.init(previous));
  });
  const size_t budget =
      ailego::VecBufferPool::metadata_bytes_for_page_count(16, true) +
      8 * ailego::kVectorPageSize;
  ASSERT_EQ(0, pool.init(budget));
  std::unique_ptr<rabitqlib::Rotator<float>> rotator;
  ASSERT_EQ(0, MakeRabitqRotator(64, rabitqlib::RotatorType::MatrixRotator, 64,
                                 &rotator, false));
  ASSERT_EQ(0, LoadRabitqRotator(
                   rotator.get(), [](size_t offset, void *out, size_t bytes) {
                     auto *row = static_cast<float *>(out);
                     std::fill(row, row + bytes / sizeof(float), 0.0f);
                     row[offset / (64 * sizeof(float))] = 1;
                     return 0;
                   }));
  std::vector<float> input(64, 0.25f), output(64);
  rotator->rotate(input.data(), output.data());
  EXPECT_EQ(input, output);
  rotator.reset();
  ASSERT_EQ(0u, pool.used());
  // This budget fits the persisted matrix but not two QR workspace stores.
  EXPECT_EQ(IndexError_NoMemory,
            MakeRabitqRotator(64, rabitqlib::RotatorType::MatrixRotator, 64,
                              &rotator));
  EXPECT_EQ(nullptr, rotator);
}

TEST(RabitqTrainingStorageTest,
     PooledHandoffSurvivesConverterAndSourceRelease) {
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  const size_t previous = pool.capacity();
  ASSERT_EQ(0u, pool.used());
  auto restore = ailego::ScopeGuard::Make([&] {
    EXPECT_EQ(0u, pool.used());
    EXPECT_EQ(0, pool.init(previous));
  });
  ASSERT_EQ(0, pool.init(32 * 1024 * 1024));
  for (const char *type : {"fht", "matrix"}) {
    IndexStorage::Pointer storage;
    IndexMeta meta(IndexMeta::DT_FP32, 64);
    meta.set_metric("SquaredEuclidean", 0, ailego::Params());
    std::vector<float> query(64, 0.25f);
    {
      RabitqConverter converter;
      ailego::Params params;
      params.set(PARAM_RABITQ_NUM_CLUSTERS, size_t{16});
      params.set(PARAM_RABITQ_ROTATOR_TYPE, std::string(type));
      ASSERT_EQ(0, converter.init(meta, params));
      auto holder =
          std::make_shared<MultiPassIndexHolder<IndexMeta::DT_FP32>>(64);
      std::mt19937 generator(42);
      std::uniform_real_distribution<float> random(-1, 1);
      for (size_t i = 0; i < 256; ++i) {
        ailego::NumericalVector<float> vector(64);
        for (size_t d = 0; d < 64; ++d) vector[d] = random(generator);
        holder->emplace(i, vector);
      }
      ASSERT_EQ(0, converter.train(holder));
      ASSERT_EQ(0, converter.training_storage(&storage));
      ASSERT_NE(nullptr, storage->vec_buffer_pool());
    }
    IvfRabitqReformer ivf;
    ASSERT_EQ(0, ivf.init("SquaredEuclidean"));
    ASSERT_EQ(0, ivf.load(storage));
    auto hnsw = std::make_unique<RabitqReformer>();
    ailego::Params params;
    params.set(PARAM_RABITQ_METRIC_NAME, std::string("SquaredEuclidean"));
    ASSERT_EQ(0, hnsw->init(params));
    ASSERT_EQ(0, hnsw->load(storage));
    IndexQueryMeta qmeta(IndexMeta::DT_FP32, 64), outmeta;
    std::string reference;
    ASSERT_EQ(0, hnsw->convert(query.data(), qmeta, &reference, &outmeta));
    IndexStorage::Pointer copied;
    ASSERT_EQ(0, CreateTemporaryIndexStorage(&copied));
    ASSERT_EQ(0, hnsw->dump(copied));
    auto source = storage->get(RABITQ_CONVERTER_SEG_ID);
    auto destination = copied->get(RABITQ_CONVERTER_SEG_ID);
    ASSERT_EQ(source->data_size(), destination->data_size());
    std::array<char, 4096> a{}, b{};
    for (size_t offset = 0; offset < source->data_size();) {
      const size_t bytes = std::min(a.size(), source->data_size() - offset);
      ASSERT_EQ(bytes, source->fetch(offset, a.data(), bytes));
      ASSERT_EQ(bytes, destination->fetch(offset, b.data(), bytes));
      EXPECT_EQ(0, std::memcmp(a.data(), b.data(), bytes));
      offset += bytes;
    }
    source.reset();
    destination.reset();
    storage.reset();
    hnsw.reset();
    RabitqReformer reloaded;
    ASSERT_EQ(0, reloaded.init(params));
    ASSERT_EQ(0, reloaded.load(copied));
    std::string actual;
    ASSERT_EQ(0, reloaded.convert(query.data(), qmeta, &actual, &outmeta));
    EXPECT_EQ(reference, actual);
    uint32_t cluster = 0;
    ASSERT_EQ(0, ivf.find_nearest_centroid(query.data(), &cluster));
    uint32_t encoded_cluster = 0;
    std::memcpy(&encoded_cluster, reference.data(), sizeof(encoded_cluster));
    EXPECT_EQ(encoded_cluster, cluster);
    EXPECT_LE(pool.used(), pool.capacity());
  }
}
}  // namespace zvec::core
