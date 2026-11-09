// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#include <numeric>
#include <gtest/gtest.h>
#include "flat_sparse/flat_sparse_provider.h"
#include "flat_sparse/flat_sparse_search.h"
#include "../../../ailego/buffer/pool_payload_budget.h"
#include <filesystem>
#include <zvec/core/framework/index_framework.h>
#include "ailego/pattern/defer.h"
#include <cstring>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/ailego/buffer/vector_page_table.h>
#include "algorithm/flat/flat_utility.h"
#include "algorithm/flat_sparse/flat_sparse_streamer_entity.h"
#include "utility/sparse_utility.h"

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
}  // namespace
}  // namespace zvec::core

namespace zvec::core {
namespace {
class SparseFlowContext : public IndexContext {
 public:
  void set_topk(uint32_t) override {}
  const IndexDocumentList &result() const override {
    return results.at(0);
  }
  const IndexDocumentList &result(size_t i) const override {
    return results.at(i);
  }
  IndexDocumentList *mutable_result(size_t i) override {
    return &results.at(i);
  }
  void reset() override {
    results.clear();
  }
  std::vector<IndexDocumentList> results;
};

// Observe the actual framework handoff, without introducing another index's
// scratch allocations or training variability into this conversion test.
class SparseFlowSearcher : public IndexSearcher {
 public:
  const IndexMeta &meta() const override {
    return meta_;
  }
  const ailego::Params &params() const override {
    return params_;
  }
  const Stats &stats() const override {
    return stats_;
  }
  int cleanup() override {
    return 0;
  }
  int unload() override {
    return 0;
  }
  int load(IndexStorage::Pointer storage, IndexMetric::Pointer) override {
    return IndexHelper::DeserializeFromStorage(storage.get(), &meta_);
  }
  Context::Pointer create_context() const override {
    return std::make_unique<SparseFlowContext>();
  }
  int search_impl(const uint32_t *counts, const uint32_t *, const void *query,
                  const IndexQueryMeta &meta, uint32_t count,
                  Context::Pointer &context) const override {
    ++batches;
    last_input = query;
    if (meta.data_type() != IndexMeta::DT_FP16 || meta.unit_size() != 2)
      return IndexError_Mismatch;
    auto &results = static_cast<SparseFlowContext *>(context.get())->results;
    results.resize(count);
    for (auto &result : results) result.clear();
    const auto *values = static_cast<const ailego::Float16 *>(query);
    for (uint32_t i = 0; i < count; ++i) {
      float sum = 0;
      for (uint32_t j = 0; j < counts[i]; ++j) sum += float(*values++);
      results[i].emplace_back(counts[i], sum);
    }
    return 0;
  }
  int search_bf_impl(const uint32_t *counts, const uint32_t *indices,
                     const void *query, const IndexQueryMeta &meta,
                     uint32_t count, Context::Pointer &ctx) const override {
    return search_impl(counts, indices, query, meta, count, ctx);
  }
  int search_impl(uint32_t count, const uint32_t *indices, const void *query,
                  const IndexQueryMeta &meta,
                  Context::Pointer &ctx) const override {
    return search_impl(&count, indices, query, meta, 1, ctx);
  }
  int search_bf_impl(uint32_t count, const uint32_t *indices, const void *query,
                     const IndexQueryMeta &meta,
                     Context::Pointer &ctx) const override {
    return search_impl(&count, indices, query, meta, 1, ctx);
  }
  mutable size_t batches{0};
  mutable const void *last_input{nullptr};

 private:
  IndexMeta meta_;
  ailego::Params params_;
  Stats stats_;
};

}  // namespace
}  // namespace zvec::core

namespace zvec::core {
}  // namespace zvec::core

namespace zvec::core {
namespace {
}  // namespace
}  // namespace zvec::core

#include "algorithm/flat_sparse/flat_sparse_searcher_entity.h"
#include "tests/core/algorithm/flat/flat_fault_storage.h"
namespace zvec::core {
namespace {
TEST(FlatSparseMemoryTest,
     OffsetCommitFailureRollsBackPayloadAndPreservesOldRow) {
  const std::string path = "flat_sparse_record_failure.index";
  AILEGO_DEFER([&] { std::filesystem::remove(path); });
  auto storage = std::make_shared<test::FaultStorage>(
      IndexFactory::CreateStorage("MMapFileStorage"));
  ASSERT_EQ(0, storage->init({}));
  ASSERT_EQ(0, storage->open(path, true));
  IndexMeta meta(IndexMeta::MT_SPARSE, IndexMeta::DT_FP32);
  meta.set_metric("InnerProductSparse", 0, ailego::Params{});
  IndexStreamer::Stats stats;
  FlatSparseStreamerEntity entity(stats);
  ASSERT_EQ(0, entity.open(storage, meta));
  const uint32_t indices[]{1, 4, 7};
  const float values[]{1, 2, 3};
  std::string row;
  SparseUtility::TransSparseFormat(3, indices, values, sizeof(float), row);
  ASSERT_EQ(0, entity.add_vector_with_id(0, row, 3));
  auto data = storage->get(PARAM_FLAT_SPARSE_DATA_SEG_ID_PREFIX + "0");
  ASSERT_NE(nullptr, data);
  const auto committed_size = data->data_size();
  storage->fault->segment = PARAM_FLAT_SPARSE_OFFSET_SEG_ID_PREFIX + "0";
  std::string replacement;
  const float changed[]{9.0f};
  SparseUtility::TransSparseFormat(1, indices, changed, sizeof(float),
                                   replacement);
  for (uint32_t id : {0u, 1u}) {
    storage->fault->write = true;
    EXPECT_EQ(IndexError_WriteData,
              entity.add_vector_with_id(id, replacement, 1));
    EXPECT_EQ(1u, entity.doc_cnt());
    EXPECT_EQ(committed_size, data->data_size());
    std::string actual;
    ASSERT_EQ(0, entity.get_sparse_vector_by_key(0, &actual));
    EXPECT_EQ(row, actual.substr(0, row.size()));
  }
  storage->fault->write = true;
  EXPECT_EQ(IndexError_WriteData, entity.add(99, row, 3));
  EXPECT_EQ(committed_size, data->data_size());
  ASSERT_EQ(0, entity.add(99, row, 3));
  EXPECT_EQ(2u, entity.doc_cnt());
  ASSERT_EQ(0, entity.flush(0));
  ASSERT_EQ(0, entity.close());
  ASSERT_EQ(0, storage->close());
}

}  // namespace
}  // namespace zvec::core
