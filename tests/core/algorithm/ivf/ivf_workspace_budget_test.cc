// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#include <array>
#include <filesystem>
#include <thread>
#include <gtest/gtest.h>
#include "ailego/algorithm/integer_quantizer.h"
#include "ailego/algorithm/kmeans.h"
#include "ailego/container/vector_array.h"
#include "algorithm/ivf/ivf_dumper.h"
#include "algorithm/ivf/ivf_entity.h"
#include "algorithm/ivf/ivf_searcher_context.h"
#include "algorithm/ivf_rabitq/ivf_rabitq_context.h"
#include "quantizer/rotator/rotator.h"
#include "../../../ailego/buffer/pool_payload_budget.h"

namespace zvec::core {
namespace {
class IVFWorkspaceBudgetTest : public testing::Test {
 protected:
  void SetUp() override {
    previous_ = pool.capacity();
    ASSERT_EQ(0, pool.init(32 * 1024 * 1024));
  }
  void TearDown() override {
    EXPECT_EQ(0u, pool.external_used());
    EXPECT_EQ(0, pool.init(previous_));
  }
  ailego::PoolPayloadBudget &pool = ailego::PoolPayloadBudget::get_instance();
  size_t previous_{0};
};

TEST_F(IVFWorkspaceBudgetTest, FhtMasksAreAdmittedAndReleasedWithRotator) {
  std::shared_ptr<Rotator> rotator;
  {
    ailego::MemoryReservation pressure;
    pressure.add(pool.capacity() - pool.used());
    EXPECT_EQ(IndexError_NoMemory, Rotator::create(&rotator, 4096));
    EXPECT_EQ(nullptr, rotator);
  }
  ASSERT_EQ(0, Rotator::create(&rotator, 4096));
  EXPECT_GE(pool.external_used(), 2048u);
  rotator.reset();
  EXPECT_EQ(0u, pool.external_used());
}

TEST_F(IVFWorkspaceBudgetTest, TrainingAndMetadataRejectBeforeAllocation) {
  ailego::NumericalVectorArray<float> centers(1536);
  IndexMeta meta(IndexMeta::DT_FP32, 32);
  IndexSearcher::Context::Pointer centroid_context;
  IVFSearcherContext context(nullptr, centroid_context);
  ailego::MemoryReservation pressure;
  pressure.add(pool.capacity() - pool.used());
  EXPECT_THROW(centers.resize(1024), std::bad_alloc);
  EXPECT_EQ(0u, centers.count());
  EXPECT_THROW((IVFDumper(meta, nullptr, 65536, 32)), std::bad_alloc);
  EXPECT_THROW(context.reset_results(100000), std::bad_alloc);
}

TEST_F(IVFWorkspaceBudgetTest, WorkerBudgetRejectionDoesNotEscapeThread) {
  using Algorithm = ailego::NumericalKmeans<float, IndexThreads>;
  class WorkerProbe : public Algorithm {
   public:
    WorkerProbe() : Algorithm(1, 1536) {}
    using Algorithm::cluster_matrix_features;
  } algorithm;
  ailego::MemoryReservation pressure;
  pressure.add(pool.capacity() - pool.used());
  std::thread worker([&] { algorithm.cluster_matrix_features(0, 1); });
  worker.join();
  EXPECT_EQ(Algorithm::kWorkspaceNoMemory, algorithm.matrix_status());
}

TEST_F(IVFWorkspaceBudgetTest, BlockTransposeChargesOldAndNewBuffers) {
  IndexMeta meta(IndexMeta::DT_FP32, 512);
  meta.set_major_order(IndexMeta::MO_COLUMN);
  IVFDumper::Block block;
  block.init(meta, 32);
  const size_t charged = pool.external_used();
  EXPECT_GE(charged, 32u * 512 * sizeof(float));
  std::array<float, 512> row{};
  for (size_t i = 0; i < 31; ++i)
    block.emplace(i, row.data(), IndexMeta::MO_ROW);
  {
    ailego::MemoryReservation pressure;
    pressure.add(pool.capacity() - pool.used());
    EXPECT_THROW(block.emplace(31, row.data(), IndexMeta::MO_ROW),
                 std::bad_alloc);
  }
  // A rejected dump may be retried from a fresh block without leaking scratch.
  EXPECT_EQ(charged, pool.external_used());
  block.clear();
  for (size_t i = 0; i < 32; ++i) {
    row.fill(static_cast<float>(i));
    block.emplace(i, row.data(), IndexMeta::MO_ROW);
  }
  EXPECT_EQ(31.0f, static_cast<const float *>(block.data())[31]);
}

TEST_F(IVFWorkspaceBudgetTest, HistogramIsChargedAndReleasedAfterTraining) {
  class Quantizer : public ailego::EntropyInt8Quantizer {
   public:
    size_t capacity() const {
      return histogram_.capacity();
    }
  } quantizer;
  quantizer.set_min(-1.0f);
  quantizer.set_max(1.0f);
  quantizer.set_non_bias(true);
  const float samples[] = {-1.0f, 0.0f, 1.0f};
  {
    ailego::MemoryReservation pressure;
    pressure.add(pool.capacity() - pool.used());
    EXPECT_THROW(quantizer.feed(samples, 3), std::bad_alloc);
    EXPECT_EQ(0u, quantizer.capacity());
  }
  ASSERT_TRUE(quantizer.feed(samples, 3));
  EXPECT_GE(pool.external_used(), quantizer.capacity() * sizeof(uint32_t));
  ASSERT_TRUE(quantizer.train());
  EXPECT_EQ(0u, quantizer.capacity());
  EXPECT_EQ(0u, pool.external_used());
}

TEST_F(IVFWorkspaceBudgetTest, QueryScratchReusesCapacityButRejectsGrowth) {
  IVFEntity::IVFReformerWrapper reformer;
  IndexMeta quantized(IndexMeta::DT_INT8, 1536);
  quantized.set_metric("InnerProduct", 0, ailego::Params{});
  ailego::Params params;
  params.set(INT8_QUANTIZER_REFORMER_BIAS, 0.0f);
  params.set(INT8_QUANTIZER_REFORMER_SCALE, 1.0f);
  quantized.set_reformer("Int8QuantizerReformer", 0, params);
  ASSERT_EQ(0, reformer.init(quantized));
  std::vector<float> queries(64 * 1536, 1.0f);
  IndexQueryMeta qmeta(IndexMeta::DT_FP32, 1536), output_meta;
  const void *out = nullptr;
  ASSERT_EQ(0,
            reformer.transform(queries.data(), qmeta, 32, &out, &output_meta));
  EXPECT_GE(pool.external_used(), 32u * 1536);
  ailego::MemoryReservation pressure;
  pressure.add(pool.capacity() - pool.used());
  const void *first = out;
  EXPECT_EQ(0,
            reformer.transform(queries.data(), qmeta, 32, &out, &output_meta));
  EXPECT_EQ(first, out);
  EXPECT_EQ(IndexError_NoMemory,
            reformer.transform(queries.data(), qmeta, 64, &out, &output_meta));
}

TEST_F(IVFWorkspaceBudgetTest, RetainedDenseBufferCopyAndMoveKeepCharge) {
  ailego::BudgetedStringBuffer first;
  first.reserve(16384);
  first.data.assign(16384, 'a');
  const size_t initial = pool.external_used();
  {
    auto copy = first;
    EXPECT_GE(pool.external_used(), initial + copy.data.capacity() + 1);
    const size_t before_move = pool.external_used();
    auto moved = std::move(copy);
    EXPECT_EQ(before_move, pool.external_used());
    EXPECT_EQ(first.data, moved.data);
  }
  EXPECT_EQ(initial, pool.external_used());
  ailego::MemoryReservation pressure;
  pressure.add(pool.capacity() - pool.used());
  EXPECT_THROW((ailego::BudgetedStringBuffer(first)), std::bad_alloc);
  EXPECT_EQ(16384u, first.data.size());
}

TEST_F(IVFWorkspaceBudgetTest, GroupKeysPayForStringCapacity) {
  IvfRabitqContext::GroupTopkHeaps heaps;
  std::string key(4096, 'x');
  heaps.try_emplace(key);
  EXPECT_GE(pool.external_used(), key.size());
  const size_t before = heaps.size();
  ailego::MemoryReservation pressure;
  pressure.add(pool.capacity() - pool.used());
  key[0] = 'y';
  EXPECT_THROW(heaps.try_emplace(key), std::bad_alloc);
  EXPECT_EQ(before, heaps.size());
}

TEST_F(IVFWorkspaceBudgetTest, ColumnFetchOwnsRowsAndPropagatesErrors) {
  const auto path = (std::filesystem::temp_directory_path() /
                     "zvec-ivf-owned-result-test.index")
                        .string();
  {
    auto file = IndexFactory::CreateDumper("FileDumper");
    ASSERT_NE(nullptr, file);
    ASSERT_EQ(0, file->init({}));
    ASSERT_EQ(0, file->create(path));
    IndexMeta meta(IndexMeta::DT_FP32, 32);
    meta.set_major_order(IndexMeta::MO_COLUMN);
    meta.set_metric("SquaredEuclidean", 0, ailego::Params{});
    IVFDumper dumper(meta, file, 1, 32);
    ASSERT_EQ(0, dumper.reserve_keys(32));
    for (size_t i = 0; i < 32; ++i) {
      std::array<float, 32> row;
      row.fill(static_cast<float>(i));
      ASSERT_EQ(0, dumper.dump_inverted_vector(0, i, row.data()));
    }
    ASSERT_EQ(0, dumper.dump_inverted_vector_finished());
    ASSERT_EQ(0, file->close());
  }
  IndexStorage::MemoryBlock first, second;
  {
    auto storage = IndexFactory::CreateStorage("BufferReadStorage");
    ASSERT_NE(nullptr, storage);
    ASSERT_EQ(0, storage->init({}));
    ASSERT_EQ(0, storage->open(path, false));
    auto entity = std::make_shared<IVFEntity>();
    ASSERT_EQ(0, entity->load(storage));
    ASSERT_EQ(0, entity->get_vector_by_key(0, first));
    ASSERT_EQ(0, entity->get_vector_by_key(1, second));
    EXPECT_NE(first.data(), second.data());
    EXPECT_EQ(0.0f, *static_cast<const float *>(first.data()));
    EXPECT_EQ(1.0f, *static_cast<const float *>(second.data()));
    IndexSearcher::Context::Pointer centroid_context;
    IVFSearcherContext context(entity, centroid_context);
    context.set_topk(2);
    context.set_fetch_vector(true);
    context.reset_results(1);
    context.mutable_result_heap().emplace(0, 0.0f);
    context.mutable_result_heap().emplace(1, 1.0f);
    ASSERT_EQ(0, context.topk_to_result(0));
    ASSERT_EQ(2u, context.result().size());
    EXPECT_EQ(0.0f, *static_cast<const float *>(context.result()[0].vector()));
    EXPECT_EQ(1.0f, *static_cast<const float *>(context.result()[1].vector()));
    context.reset_results(1);
    context.mutable_result_heap().clear();
    context.mutable_result_heap().emplace(1000, 0.0f);
    EXPECT_NE(0, context.topk_to_result(0));
    EXPECT_TRUE(context.result().empty());
    ailego::MemoryReservation pressure;
    pressure.add(pool.capacity() - pool.used());
    IndexStorage::MemoryBlock rejected;
    EXPECT_EQ(IndexError_NoMemory, entity->get_vector_by_key(0, rejected));
    EXPECT_EQ(nullptr, rejected.data());
  }
  std::filesystem::remove(path);
  EXPECT_EQ(0.0f, *static_cast<const float *>(first.data()));
  EXPECT_EQ(1.0f, *static_cast<const float *>(second.data()));
}
}  // namespace
}  // namespace zvec::core


namespace zvec::core {
namespace {
class ScaleFaultSegment : public IndexStorage::Segment {
 public:
  InvertedIntegerQuantizerParams params{4.0f, 0.0f};
  int failure{0};
  size_t data_size() const override {
    return sizeof(params);
  }
  size_t capacity() const override {
    return sizeof(params);
  }
  size_t padding_size() const override {
    return 0;
  }
  uint32_t data_crc() const override {
    return 0;
  }
  size_t fetch(size_t, void *out, size_t bytes) const override {
    if (failure == IndexError_NoMemory) throw std::bad_alloc();
    if (failure || bytes != sizeof(params)) return 0;
    std::memcpy(out, &params, bytes);
    return bytes;
  }
  size_t read(size_t, const void **, size_t) override {
    return 0;
  }
  size_t read(size_t, IndexStorage::MemoryBlock &, size_t) override {
    return 0;
  }
  size_t write(size_t, const void *, size_t) override {
    return 0;
  }
  size_t resize(size_t) override {
    return 0;
  }
  void update_data_crc(uint32_t) override {}
  Pointer clone() override {
    return std::make_shared<ScaleFaultSegment>(*this);
  }
};
TEST_F(IVFWorkspaceBudgetTest, QuantizationScaleReadErrorsAreNotDefaultValues) {
  class Entity : public IVFEntity {
   public:
    void set_scale(IndexStorage::Segment::Pointer segment) {
      integer_quantizer_params_ = std::move(segment);
    }
  } entity;
  auto segment = std::make_shared<ScaleFaultSegment>();
  entity.set_scale(segment);
  float value = -1.0f;
  ASSERT_EQ(0, entity.inverted_list_normalize_value(0, &value));
  EXPECT_FLOAT_EQ(0.25f, value);
  for (int failure : {int(IndexError_ReadData), int(IndexError_NoMemory)}) {
    segment->failure = failure;
    value = -1.0f;
    EXPECT_EQ(failure, entity.inverted_list_normalize_value(0, &value));
    EXPECT_FLOAT_EQ(-1.0f, value);
  }
  segment->failure = 0;
  EXPECT_EQ(0, entity.inverted_list_normalize_value(0, &value));
  EXPECT_FLOAT_EQ(0.25f, value);
}
}  // namespace
}  // namespace zvec::core
