// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#include <numeric>
#include <gtest/gtest.h>
#include "flat_sparse/flat_sparse_provider.h"
#include "flat_sparse/flat_sparse_search.h"
#include "../../../ailego/buffer/pool_payload_budget.h"

namespace zvec::core {
namespace {
class SparseRows : public FlatSparseEntity {
 public:
  SparseRows() {
    std::vector<uint32_t> indices(16384);
    std::iota(indices.begin(), indices.end(), 1);
    std::vector<float> values(indices.size(), 1.0f);
    SparseUtility::TransSparseFormat(indices.size(), indices.data(),
                                     values.data(), sizeof(float), row_);
  }
  uint32_t doc_cnt() const override {
    return 2;
  }
  uint32_t total_sparse_count() const override {
    return 32768;
  }
  node_id_t get_id(uint64_t key) const override {
    return key < 2 ? key : kInvalidNodeId;
  }
  uint64_t get_key(node_id_t id) const override {
    if (fail_key == id) throw ReadError{IndexError_ReadData};
    return id;
  }
  size_t sparse_unit_size() const override {
    return sizeof(float);
  }
  int get_sparse_vector_ptr_by_id(node_id_t, const void **data,
                                  uint32_t *length) const override {
    *data = row_.data();
    *length = row_.size();
    return 0;
  }
  int get_sparse_vector_ptr_by_id(node_id_t id,
                                  IndexStorage::MemoryBlock &block,
                                  uint32_t *length) const override {
    if (id == fail_payload) return IndexError_ReadData;
    block = IndexStorage::MemoryBlock::MakeBorrowedView(
        const_cast<char *>(row_.data()));
    *length = row_.size();
    return 0;
  }
  float get_search_distance(std::string_view query,
                            node_id_t id) const override {
    seen_counts.push_back(*reinterpret_cast<const uint32_t *>(query.data()));
    return static_cast<float>(id);
  }
  mutable std::vector<uint32_t> seen_counts;
  uint32_t fail_payload{kInvalidNodeId};
  uint32_t fail_key{kInvalidNodeId};

 private:
  std::string row_;
};
class FlatSparseWorkspaceBudgetTest : public ::testing::Test {
 protected:
  void SetUp() override {
    previous_ = pool_.capacity();
    ASSERT_EQ(0u, pool_.used());
    ASSERT_EQ(0, pool_.init(2 * 1024 * 1024));
  }
  void TearDown() override {
    EXPECT_EQ(0u, pool_.used());
    EXPECT_EQ(0, pool_.init(previous_));
  }
  ailego::PoolPayloadBudget &pool_ = ailego::PoolPayloadBudget::get_instance();
  size_t previous_{};
};

TEST_F(FlatSparseWorkspaceBudgetTest,
       QueryConversionRejectsThenReusesCapacity) {
  SparseRows rows;
  auto *owner = static_cast<const FlatSparseSearcher *>(nullptr);
  IndexContext::Pointer context(new FlatSparseContext(owner));
  context->set_topk(1);
  const uint32_t counts[]{16384, 3};
  std::vector<uint32_t> indices(16387);
  std::iota(indices.begin(), indices.begin() + 16384, 1);
  std::iota(indices.begin() + 16384, indices.end(), 1);
  std::vector<float> values(indices.size(), 1.0f);
  IndexQueryMeta query_meta(IndexMeta::MT_SPARSE, IndexMeta::DT_FP32);
  IndexMeta meta(IndexMeta::MT_SPARSE, IndexMeta::DT_FP32);
  ailego::MemoryReservation pressure;
  pressure.add(pool_.capacity() - pool_.used() - 32768);
  EXPECT_EQ(IndexError_NoMemory,
            FlatSearch(counts, indices.data(), values.data(), false, {},
                       query_meta, 2, meta, context, &rows));
  EXPECT_TRUE(rows.seen_counts.empty());
  pressure.retain_at_most(0);
  for (bool grouped : {false, true}) {
    if (grouped) {
      context->set_group_params(1, 1);
      context->set_group_by([](uint64_t key) { return std::to_string(key); });
    }
    rows.seen_counts.clear();
    ASSERT_EQ(0, FlatSearch(counts, indices.data(), values.data(), false, {},
                            query_meta, 2, meta, context, &rows));
    EXPECT_EQ((std::vector<uint32_t>{16384, 16384, 3, 3}), rows.seen_counts);
    auto *flat = static_cast<FlatSparseContext *>(context.get());
    const auto capacity = flat->query_buffer().capacity();
    EXPECT_GE(capacity, 16384 * 6u);
    const auto *buffer = flat->query_buffer().data();
    ASSERT_EQ(0, FlatSearch(counts, indices.data(), values.data(), false, {},
                            query_meta, 2, meta, context, &rows));
    EXPECT_EQ(buffer, flat->query_buffer().data());
    EXPECT_EQ(capacity, flat->query_buffer().capacity());
    EXPECT_GE(pool_.used(), capacity);
  }
}

TEST_F(FlatSparseWorkspaceBudgetTest,
       IteratorFailureRefundsAndRetryOwnsBuffers) {
  auto rows = std::make_shared<SparseRows>();
  IndexMeta meta(IndexMeta::MT_SPARSE, IndexMeta::DT_FP32);
  FlatSparseIndexProvider<SparseRows> provider(rows, meta, "SparseRows");
  const auto before = pool_.used();
  ailego::MemoryReservation pressure;
  pressure.add(pool_.capacity() - pool_.used() - 70000);
  // Enough for indices, but not both expanded arrays. Failure is distinct
  // from EOF; destroying the failed iterator refunds any partial workspace.
  {
    auto failed = provider.create_iterator();
    ASSERT_NE(nullptr, failed);
    EXPECT_FALSE(failed->is_valid());
    EXPECT_EQ(IndexError_NoMemory, failed->status());
  }
  EXPECT_EQ(pool_.capacity() - 70000, pool_.used());
  pressure.retain_at_most(0);
  auto iterator = provider.create_iterator();
  ASSERT_NE(nullptr, iterator);
  ASSERT_TRUE(iterator->is_valid());
  EXPECT_EQ(16384u, iterator->sparse_count());
  EXPECT_GE(pool_.used(), before + 2 * 16384 * sizeof(uint32_t));
  const size_t retained = pool_.used();
  iterator->next();
  EXPECT_TRUE(iterator->is_valid());
  EXPECT_EQ(retained, pool_.used());
  iterator.reset();
  EXPECT_EQ(before, pool_.used());
}
}  // namespace
}  // namespace zvec::core

namespace zvec::core {
namespace {
TEST_F(FlatSparseWorkspaceBudgetTest,
       FetchedPayloadOwnsChargeAcrossCopyMoveAndReuse) {
  std::vector<uint32_t> indices(16384);
  std::iota(indices.begin(), indices.end(), 1);
  std::vector<float> values(indices.size(), 1.0f);
  ailego::BudgetedString encoded;
  SparseUtility::TransSparseFormat(indices.size(), indices.data(),
                                   values.data(), sizeof(float), encoded);
  const size_t before = pool_.used();
  IndexSparseDocument result;
  SparseUtility::ReverseSparseFormat(encoded.data(), result, sizeof(float));
  EXPECT_EQ(16384u, result.sparse_count());
  const size_t payload = pool_.used() - before;
  EXPECT_GE(payload, indices.size() * (sizeof(float) + sizeof(uint32_t)));
  IndexSparseDocument copy(result);
  EXPECT_EQ(before + 2 * payload, pool_.used());
  IndexSparseDocument moved(std::move(copy));
  EXPECT_EQ(before + 2 * payload, pool_.used());
  ailego::MemoryReservation pressure;
  pressure.add(pool_.capacity() - pool_.used());
  EXPECT_THROW(IndexSparseDocument denied(result), std::bad_alloc);
  EXPECT_THROW(
      result.reserve_sparse_buffers(2 * indices.size() * sizeof(uint32_t),
                                    2 * values.size() * sizeof(float)),
      std::bad_alloc);
  EXPECT_EQ(16384u, result.sparse_count());
  EXPECT_EQ(indices.size() * sizeof(uint32_t), result.sparse_indices().size());
  pressure.retain_at_most(0);
  // Re-decoding the same record reuses capacity and does not append old data.
  SparseUtility::ReverseSparseFormat(encoded.data(), result, sizeof(float));
  EXPECT_EQ(before + 2 * payload, pool_.used());
  EXPECT_EQ(indices.size() * sizeof(uint32_t), result.sparse_indices().size());
  result = IndexSparseDocument();
  EXPECT_EQ(before + payload, pool_.used());
}
}  // namespace
}  // namespace zvec::core

namespace zvec::core {
namespace {
TEST_F(FlatSparseWorkspaceBudgetTest, IteratorReportsKeyAndPayloadFailures) {
  auto rows = std::make_shared<SparseRows>();
  IndexMeta meta(IndexMeta::MT_SPARSE, IndexMeta::DT_FP32);
  FlatSparseIndexProvider<SparseRows> provider(rows, meta, "SparseRows");
  for (bool key_error : {false, true}) {
    rows->fail_payload = key_error ? kInvalidNodeId : 1;
    rows->fail_key = key_error ? 1 : kInvalidNodeId;
    auto iterator = provider.create_iterator();
    ASSERT_TRUE(iterator->is_valid());
    EXPECT_EQ(0u, iterator->key());
    iterator->next();
    EXPECT_FALSE(iterator->is_valid());
    EXPECT_EQ(IndexError_ReadData, iterator->status());
    iterator->next();
    EXPECT_EQ(IndexError_ReadData, iterator->status());
  }
}
}  // namespace
}  // namespace zvec::core

#include <filesystem>
#include <zvec/core/framework/index_framework.h>
#include "ailego/pattern/defer.h"
namespace zvec::core {
namespace {
TEST_F(FlatSparseWorkspaceBudgetTest,
       BuilderRejectsPrematureEndAndConverterHandlesLargeRows) {
  // Dump buffers need more than the query-only fixture budget. The intended
  // failure below is the source read, after the first row was written.
  ASSERT_EQ(0, pool_.init(8 * 1024 * 1024));
  const std::string path = "flat_sparse_iterator_failure.index";
  AILEGO_DEFER([&] { std::filesystem::remove(path); });
  auto rows = std::make_shared<SparseRows>();
  rows->fail_payload = 1;
  IndexMeta meta(IndexMeta::MT_SPARSE, IndexMeta::DT_FP32);
  auto provider = std::make_shared<FlatSparseIndexProvider<SparseRows>>(
      rows, meta, "SparseRows");
  auto converter = IndexFactory::CreateConverter("HalfFloatSparseConverter");
  ASSERT_EQ(0, converter->init(meta, {}));
  ASSERT_EQ(0, converter->transform(provider));
  auto converted = converter->sparse_result();
  {
    auto iter = converted->create_iterator();
    ASSERT_TRUE(iter->is_valid());
    ASSERT_EQ(16384u, iter->sparse_count());
    const auto *data =
        static_cast<const ailego::Float16 *>(iter->sparse_data());
    EXPECT_FLOAT_EQ(1.0f, static_cast<float>(data[16383]));
    iter->next();
    EXPECT_FALSE(iter->is_valid());
    EXPECT_EQ(IndexError_ReadData, iter->status());
  }
  auto builder = IndexFactory::CreateBuilder("FlatSparseBuilder");
  ASSERT_EQ(0, builder->init(converter->meta(), {}));
  ASSERT_EQ(0, builder->train(converted));
  ASSERT_EQ(0, builder->build(converted));
  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_EQ(0, dumper->create(path));
  EXPECT_EQ(IndexError_ReadData, builder->dump(dumper));
  EXPECT_EQ(0u, builder->stats().dumped_count());
  // A failed dump must not be finalized as a complete index.
  EXPECT_NE(0, dumper->close());
}
TEST_F(FlatSparseWorkspaceBudgetTest,
       PublicOutputsChargeLifetimeAndLegacyCannotGrow) {
  auto rows = std::make_shared<SparseRows>();
  IndexMeta meta(IndexMeta::MT_SPARSE, IndexMeta::DT_FP32);
  IndexSparseProvider::Pointer provider =
      std::make_shared<FlatSparseIndexProvider<SparseRows>>(rows, meta,
                                                            "SparseRows");
  const auto baseline = pool_.used();
  IndexSparseDocument result;
  std::string indices, values;
  uint32_t count = 7;
  ailego::MemoryReservation pressure;
  pressure.add(pool_.available());
  EXPECT_EQ(IndexError_NoMemory, provider->get_sparse_vector(0, result));
  EXPECT_EQ(IndexError_NoMemory,
            provider->get_sparse_vector(0, &count, &indices, &values));
  EXPECT_TRUE(indices.empty());
  EXPECT_TRUE(values.empty());
  EXPECT_EQ(7u, count);
  pressure.retain_at_most(0);
  ASSERT_EQ(0, provider->get_sparse_vector(0, result));
  ASSERT_EQ(16384u, result.sparse_count());
  EXPECT_GE(pool_.used() - baseline, 2 * 16384 * sizeof(uint32_t));
  const auto retained = pool_.used();
  {
    auto copy = result;
    EXPECT_GT(pool_.used(), retained);
    const auto copied = pool_.used();
    auto moved = std::move(copy);
    EXPECT_EQ(copied, pool_.used());
    EXPECT_EQ(result.sparse_values(), moved.sparse_values());
  }
  EXPECT_EQ(retained, pool_.used());
  // The old API can fill caller-supplied storage without allocating.
  indices.reserve(16384 * sizeof(uint32_t));
  values.reserve(16384 * sizeof(float));
  pressure.add(pool_.available());
  EXPECT_EQ(0, provider->get_sparse_vector(0, result));
  EXPECT_EQ(0, provider->get_sparse_vector(0, &count, &indices, &values));
  EXPECT_EQ(result.sparse_indices(), indices);
  EXPECT_EQ(result.sparse_values(), values);
  provider.reset();
  pressure.retain_at_most(0);
  EXPECT_EQ(retained, pool_.used());
  result = IndexSparseDocument{};
  EXPECT_EQ(baseline, pool_.used());
}

}  // namespace
}  // namespace zvec::core
