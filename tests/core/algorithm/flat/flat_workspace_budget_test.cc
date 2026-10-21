// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#include <array>
#include <filesystem>
#include <gtest/gtest.h>
#include <zvec/core/framework/index_framework.h>
#include "ailego/pattern/defer.h"
#include "flat/flat_searcher.h"
#include "flat/flat_streamer_context.h"
#include "../../../ailego/buffer/pool_payload_budget.h"

namespace zvec::core {
namespace {
class FlatWorkspaceBudgetTest : public ::testing::Test {
 protected:
  void SetUp() override {
    previous_ = pool_.capacity();
    ASSERT_EQ(0u, pool_.used());
    ASSERT_EQ(0, pool_.init(8 * 1024 * 1024));
  }
  void TearDown() override {
    EXPECT_EQ(0u, pool_.used());
    EXPECT_EQ(0, pool_.init(previous_));
  }
  static IndexMeta meta(size_t dimension = 8) {
    IndexMeta value(IndexMeta::DT_FP32, dimension);
    value.set_metric("SquaredEuclidean", 0, ailego::Params{});
    return value;
  }
  ailego::PoolPayloadBudget &pool_ = ailego::PoolPayloadBudget::get_instance();
  size_t previous_{};
};

TEST_F(FlatWorkspaceBudgetTest, BatchCapacityIsRetainedAndGrowthIsRejected) {
  auto owner = IndexFactory::CreateStreamer("FlatStreamer");
  ASSERT_EQ(0, owner->init(meta(), {}));
  auto *flat = dynamic_cast<FlatStreamer<32> *>(owner.get());
  ASSERT_NE(nullptr, flat);
  FlatStreamerContext<32> context(flat);
  context.reset_results(1024);
  const size_t retained = pool_.used();
  EXPECT_GE(retained, 1024 * sizeof(IndexDocumentList));
  context.reset();
  EXPECT_EQ(retained, pool_.used());
  ailego::MemoryReservation pressure;
  pressure.add(pool_.capacity() - pool_.used());
  EXPECT_THROW(context.reset_results(2048), std::bad_alloc);
  pressure.retain_at_most(0);
  EXPECT_NO_THROW(context.reset_results(2048));
  EXPECT_TRUE(context.result(2047).empty());
}

TEST_F(FlatWorkspaceBudgetTest, ProviderRowCopyIsLazyChargedAndOwned) {
  const std::string path = "flat_provider_workspace.index";
  ASSERT_FALSE(std::filesystem::exists(path));
  AILEGO_DEFER([&] { std::filesystem::remove(path); });
  for (auto order : {IndexMeta::MO_ROW, IndexMeta::MO_COLUMN}) {
    auto metadata = meta(1536);
    metadata.set_major_order(order);
    auto holder =
        std::make_shared<MultiPassIndexHolder<IndexMeta::DT_FP32>>(1536);
    for (size_t i = 0; i < 64; ++i) {
      ailego::NumericalVector<float> row(1536);
      for (size_t d = 0; d < 1536; ++d) row[d] = static_cast<float>(i);
      ASSERT_TRUE(holder->emplace(i + 100, std::move(row)));
    }
    auto builder = IndexFactory::CreateBuilder("FlatBuilder");
    auto dumper = IndexFactory::CreateDumper("FileDumper");
    ASSERT_EQ(0, builder->init(metadata, {}));
    ASSERT_EQ(0, dumper->create(path));
    ASSERT_EQ(0, IndexBuilder::TrainBuildAndDump(builder, holder, dumper));
    ASSERT_EQ(0, dumper->close());
    builder.reset();
    holder.reset();
    auto storage = IndexFactory::CreateStorage("MMapFileReadStorage");
    ASSERT_EQ(0, storage->init({}));
    ASSERT_EQ(0, storage->open(path, false));
    auto searcher = IndexFactory::CreateSearcher("FlatSearcher");
    ASSERT_EQ(0, searcher->init({}));
    ASSERT_EQ(0, searcher->load(storage, nullptr));
    auto provider = searcher->create_provider();
    ASSERT_NE(nullptr, provider);
    {
      auto context = searcher->create_context();
      context->set_topk(4);
      std::vector<float> queries(2 * 1536, 0.0f);
      IndexQueryMeta query_meta(IndexMeta::DT_FP32, 1536);
      ASSERT_EQ(0,
                searcher->search_impl(queries.data(), query_meta, 2, context));
      ailego::MemoryReservation result_pressure;
      result_pressure.add(pool_.capacity() - pool_.used() -
                          (4 * sizeof(IndexDocument) + 512));
      EXPECT_THROW(context->result(1), std::bad_alloc);
      result_pressure.retain_at_most(0);
      EXPECT_EQ(4u, context->result(0).size());
      EXPECT_EQ(4u, context->result(1).size());
      context->set_group_params(1, 2);
      context->set_group_by([](uint64_t) { return std::string("group"); });
      context->set_fetch_vector(true);
      ASSERT_EQ(0, searcher->search_impl(queries.data(), query_meta, context));
      auto saved = context->group_result();
      ASSERT_EQ(1u, saved.size());
      ASSERT_EQ(2u, saved[0].docs().size());
      EXPECT_EQ(100u, saved[0].docs()[0].key());
      // A later scan must not overwrite the independently owned results.
      std::fill(queries.begin(), queries.end(), 63.0f);
      ASSERT_EQ(0, searcher->search_impl(queries.data(), query_meta, context));
      EXPECT_FLOAT_EQ(
          0.0f, static_cast<const float *>(saved[0].docs()[0].vector())[1535]);
      EXPECT_FLOAT_EQ(
          1.0f, static_cast<const float *>(saved[0].docs()[1].vector())[1535]);
    }
    const size_t before = pool_.used();
    ailego::MemoryReservation pressure;
    pressure.add(pool_.capacity() - pool_.used());
    // Both layouts export an owned row: mmap views cannot survive close().
    // Column-major rows no longer retain a whole transposed batch.
    IndexStorage::MemoryBlock rejected;
    EXPECT_EQ(IndexError_NoMemory, provider->get_vector(100, rejected));
    EXPECT_EQ(nullptr, rejected.data());
    EXPECT_EQ(nullptr, provider->get_vector(100));
    auto failed = provider->create_iterator();
    ASSERT_NE(nullptr, failed);
    if (order == IndexMeta::MO_COLUMN) {
      EXPECT_FALSE(failed->is_valid());
      EXPECT_EQ(IndexError_NoMemory, failed->status());
    } else {
      // An iterator's borrowed mmap view still needs no transpose workspace.
      EXPECT_TRUE(failed->is_valid());
      EXPECT_EQ(0, failed->status());
    }
    failed.reset();
    pressure.retain_at_most(0);
    const auto *row = static_cast<const float *>(provider->get_vector(107));
    ASSERT_NE(nullptr, row);
    EXPECT_FLOAT_EQ(7.0f, row[1535]);
    EXPECT_GE(pool_.used(), before + 1536 * sizeof(float));
    EXPECT_LT(pool_.used(), before + 32 * 1536 * sizeof(float));
    IndexStorage::MemoryBlock owned;
    ASSERT_EQ(0, provider->get_vector(107, owned));
    provider.reset();
    ASSERT_EQ(0, searcher->unload());
    ASSERT_EQ(0, storage->close());
    EXPECT_FLOAT_EQ(7.0f, static_cast<const float *>(owned.data())[1535]);
    owned.reset();
    searcher.reset();
    storage.reset();
  }
}

TEST_F(FlatWorkspaceBudgetTest, KeyAdmissionFailureDoesNotPublishOrBlockRetry) {
  const std::string path = "flat_key_workspace.index";
  ASSERT_FALSE(std::filesystem::exists(path));
  AILEGO_DEFER([&] { std::filesystem::remove(path); });
  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_EQ(0, storage->init({}));
  ASSERT_EQ(0, storage->open(path, true));
  IndexStreamer::Stats stats;
  FlatStreamerEntity entity(stats);
  *entity.mutable_meta() = meta();
  entity.set_block_vector_count(32);
  entity.set_linear_list_count(1);
  entity.set_segment_size(4096);
  ASSERT_EQ(0, entity.open(storage, meta()));
  std::array<float, 8> row{};
  ASSERT_EQ(0, entity.add(101, row.data(), sizeof(row)));
  ailego::MemoryReservation pressure;
  pressure.add(pool_.capacity() - pool_.used());
  EXPECT_EQ(IndexError_NoMemory, entity.add(202, row.data(), sizeof(row)));
  EXPECT_EQ(1u, entity.vector_count());
  EXPECT_EQ(nullptr, entity.get_vector_by_key(202));
  pressure.retain_at_most(0);
  EXPECT_EQ(0, entity.add(202, row.data(), sizeof(row)));
  EXPECT_EQ(2u, entity.vector_count());
}
}  // namespace
}  // namespace zvec::core

#include "tests/core/algorithm/flat/flat_fault_storage.h"
namespace zvec::core {
namespace {
TEST_F(FlatWorkspaceBudgetTest, StreamerReadFailuresAndExportedMmapOwnership) {
  const std::string path = "flat_read_failure.index";
  AILEGO_DEFER([&] { std::filesystem::remove(path); });
  auto storage = std::make_shared<test::FaultStorage>(
      IndexFactory::CreateStorage("MMapFileStorage"));
  ASSERT_EQ(0, storage->init({}));
  ASSERT_EQ(0, storage->open(path, true));
  auto owner = IndexFactory::CreateStreamer("FlatStreamer");
  ailego::Params params;
  params.set(PARAM_FLAT_USE_CONTIGUOUS_MEMORY, false);
  ASSERT_EQ(0, owner->init(meta(), params));
  ASSERT_EQ(0, owner->open(storage));
  auto context = owner->create_context();
  std::array<float, 8> row;
  row.fill(2.0f);
  IndexQueryMeta qmeta(IndexMeta::DT_FP32, row.size());
  ASSERT_EQ(0, owner->add_with_id_impl(0, row.data(), qmeta, context));
  context->set_topk(1);
  storage->fault->segment = FLAT_SEGMENT_FEATURES_SEG_ID + "1";
  for (bool grouped : {false, true}) {
    if (grouped) {
      context->set_group_params(1, 1);
      context->set_group_by([](uint64_t) { return std::string("group"); });
    }
    storage->fault->read = true;
    EXPECT_EQ(IndexError_ReadData, owner->search_bf_by_p_keys_impl(
                                       row.data(), {{0}}, qmeta, 1, context));
  }
  context->set_fetch_vector(true);
  ASSERT_EQ(0, owner->search_impl(row.data(), qmeta, context));
  auto result = context->group_result();
  ASSERT_EQ(1u, result.size());
  ASSERT_EQ(1u, result[0].docs().size());
  auto provider = owner->create_provider();
  IndexProvider::VectorBlocks batch;
  const uint64_t keys[]{0, 0};
  ASSERT_EQ(0, provider->get_vectors(keys, 2, batch));
  ASSERT_EQ(2u, batch.size());
  ASSERT_EQ(0, owner->close());
  ASSERT_EQ(0, storage->close());
  EXPECT_FLOAT_EQ(2.0f,
                  static_cast<const float *>(result[0].docs()[0].vector())[7]);
  EXPECT_FLOAT_EQ(2.0f, static_cast<const float *>(batch[1].data())[7]);
}

TEST_F(FlatWorkspaceBudgetTest, SegmentPublicationFailureCanRetryAndReopen) {
  const std::string path = "flat_segment_failure.index";
  AILEGO_DEFER([&] { std::filesystem::remove(path); });
  auto storage = std::make_shared<test::FaultStorage>(
      IndexFactory::CreateStorage("MMapFileStorage"));
  ASSERT_EQ(0, storage->init({}));
  ASSERT_EQ(0, storage->open(path, true));
  IndexStreamer::Stats stats;
  auto entity = std::make_unique<FlatStreamerEntity>(stats);
  *entity->mutable_meta() = meta();
  entity->set_block_vector_count(32);
  entity->set_linear_list_count(1);
  entity->set_segment_size(4096);
  ASSERT_EQ(0, entity->open(storage, meta()));
  std::array<float, 8> row{};
  storage->fault->segment = FLAT_LINEAR_META_SEG_ID;
  storage->fault->write = true;
  EXPECT_EQ(IndexError_WriteData,
            entity->add_vector_with_id(0, row.data(), sizeof(row)));
  EXPECT_EQ(0u, entity->vector_count());
  ASSERT_EQ(0, entity->add_vector_with_id(0, row.data(), sizeof(row)));
  ASSERT_EQ(1u, entity->vector_count());
  ASSERT_EQ(0, entity->flush(0));
  ASSERT_EQ(0, entity->close());
  ASSERT_EQ(0, storage->close());
  ASSERT_EQ(0, storage->open(path, true));
  ASSERT_EQ(0, entity->open(storage, meta()));
  EXPECT_EQ(1u, entity->vector_count());
  EXPECT_EQ(0u, entity->key(0));
}

TEST_F(FlatWorkspaceBudgetTest, ConvertedContextCapacitySurvivesRejection) {
  auto owner = IndexFactory::CreateStreamer("FlatStreamer");
  ASSERT_EQ(0, owner->init(meta(), {}));
  FlatStreamerContext<32> context(
      dynamic_cast<FlatStreamer<32> *>(owner.get()));
  auto *buffer = context.mutable_features(16384);
  buffer->assign(16384, 'q');
  const size_t retained = pool_.used();
  EXPECT_GE(retained, buffer->capacity());
  ailego::MemoryReservation pressure;
  pressure.add(pool_.capacity() - pool_.used());
  EXPECT_NO_THROW(context.mutable_features(16384));
  EXPECT_THROW(context.mutable_features(32768), std::bad_alloc);
  EXPECT_EQ(std::string(16384, 'q'), *buffer);
  pressure.retain_at_most(0);
  EXPECT_EQ(retained, pool_.used());
}
TEST_F(FlatWorkspaceBudgetTest, BatchOutputContainerOwnsBudgetAcrossCopyMove) {
  auto provider =
      std::make_shared<MultiPassIndexProvider<IndexMeta::DT_FP32>>(8);
  // Refusal must happen before attempting to look up these missing keys.
  const uint64_t keys[]{1, 2};
  std::vector<IndexStorage::MemoryBlock> legacy;
  IndexProvider::VectorBlocks owned;
  ailego::MemoryReservation pressure;
  pressure.add(pool_.available());
  EXPECT_EQ(IndexError_NoMemory, provider->get_vectors(keys, 2, legacy));
  EXPECT_EQ(IndexError_NoMemory, provider->get_vectors(keys, 2, owned));
  EXPECT_EQ(0u, legacy.capacity());
  EXPECT_EQ(0u, owned.capacity());
  pressure.retain_at_most(0);
  owned.resize(1024);
  const auto retained = pool_.used();
  EXPECT_GE(retained, 1024 * sizeof(IndexStorage::MemoryBlock));
  {
    auto copy = owned;
    EXPECT_GT(pool_.used(), retained);
    const auto copied = pool_.used();
    auto moved = std::move(copy);
    EXPECT_EQ(copied, pool_.used());
    EXPECT_EQ(1024u, moved.size());
  }
  EXPECT_EQ(retained, pool_.used());
  IndexProvider::VectorBlocks().swap(owned);
  EXPECT_EQ(0u, pool_.used());
}

}  // namespace
}  // namespace zvec::core
