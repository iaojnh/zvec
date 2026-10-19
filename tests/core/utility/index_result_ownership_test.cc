// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#include <type_traits>
#include <gtest/gtest.h>
#include <zvec/core/framework/index_document.h>
#include "ailego/pattern/defer.h"
#include "utility/legacy_read_buffer.h"
#include "../../ailego/buffer/pool_payload_budget.h"

namespace zvec::core {
namespace {
TEST(IndexResultOwnership, LegacyReadCopiesKeepBudgetWithoutRetainingSource) {
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  const size_t old = pool.capacity();
  ASSERT_EQ(0u, pool.used());
  AILEGO_DEFER([&] {
    EXPECT_EQ(0u, pool.used());
    EXPECT_EQ(0, pool.init(old));
  });
  ASSERT_EQ(0, pool.init(64 * 1024));
  auto source = std::make_shared<std::string>(4096, 'x');
  std::weak_ptr<std::string> weak = source;
  auto block =
      IndexStorage::MemoryBlock::MakeSharedView(source->data(), source);
  LegacyReadBuffer buffer;
  const auto *copied =
      static_cast<const char *>(buffer.assign(block, source->size()));
  ASSERT_NE(nullptr, copied);
  EXPECT_GE(pool.used(), source->size());
  block.reset();
  source.reset();
  EXPECT_TRUE(weak.expired());
  EXPECT_EQ(std::string(4096, 'x'), std::string(copied, 4096));
  const size_t retained = pool.used();
  auto larger = std::make_shared<std::string>(128 * 1024, 'y');
  block = IndexStorage::MemoryBlock::MakeSharedView(larger->data(), larger);
  EXPECT_THROW(buffer.assign(block, larger->size()), std::bad_alloc);
  EXPECT_EQ(retained, pool.used());
  EXPECT_EQ(std::string(4096, 'x'), std::string(copied, 4096));
}

TEST(IndexResultOwnership, VectorBudgetFollowsReturnedAndCopiedDocuments) {
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  const size_t old = pool.capacity();
  ASSERT_EQ(0u, pool.used());
  AILEGO_DEFER([&] {
    EXPECT_EQ(0u, pool.used());
    pool.init(old);
  });
  ASSERT_EQ(0, pool.init(16 * 1024));
  const std::string source(4096, 'x');
  auto payload = std::allocate_shared<IndexDocument::VectorPayload>(
      ailego::BudgetedAllocator<IndexDocument::VectorPayload>{}, source.data(),
      source.size());
  const size_t payload_bytes = pool.used();
  IndexDocumentList context;
  context.emplace_back(7, 1, 7, payload);
  const char *data = context.front().vector_string().data();
  const size_t charged = pool.used();
  EXPECT_GT(charged, source.size());
  payload.reset();  // Candidate heaps no longer own the vector.
  EXPECT_EQ(charged, pool.used());
  IndexDocumentList output;
  output = std::move(context);
  EXPECT_TRUE(context.empty());
  IndexDocument retained(output.front());
  output.emplace_back(8, 2);  // Grow the result container.
  const size_t grown = pool.used();
  EXPECT_EQ(data, output.front().vector_string().data());
  EXPECT_EQ(data, retained.vector_string().data());
  output.clear();
  EXPECT_EQ(grown, pool.used());
  EXPECT_EQ(source, retained.vector_string());
  retained = IndexDocument{};
  EXPECT_EQ(grown - payload_bytes, pool.used());
  IndexDocumentList().swap(output);
  EXPECT_EQ(0u, pool.used());
}

TEST(IndexResultOwnership, BatchOutputCannotRecycleLivePayloadBudget) {
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  const size_t old = pool.capacity();
  ASSERT_EQ(0u, pool.used());
  AILEGO_DEFER([&] {
    EXPECT_EQ(0u, pool.used());
    pool.init(old);
  });
  ASSERT_EQ(0, pool.init(16 * 1024));
  const std::string source(3072, 'v');
  IndexDocumentList output;
  auto append = [&] {
    auto payload = std::allocate_shared<IndexDocument::VectorPayload>(
        ailego::BudgetedAllocator<IndexDocument::VectorPayload>{},
        source.data(), source.size());
    output.emplace_back(output.size(), 0, 0, payload);
  };
  EXPECT_THROW(
      {
        for (size_t i = 0; i < 32; ++i) append();
      },
      std::bad_alloc);
  EXPECT_FALSE(output.empty());
  EXPECT_LT(output.size(), 32u);
  EXPECT_LE(pool.used(), pool.capacity());
  for (const auto &doc : output) EXPECT_EQ(source, doc.vector_string());
  output.clear();
  EXPECT_GT(pool.used(), 0u);  // clear retains the result array's capacity.
  EXPECT_NO_THROW(append());
}

TEST(IndexResultOwnership, EmptyCapacityIsAdmittedAndSurvivesClearAndMove) {
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  const size_t old = pool.capacity();
  ASSERT_EQ(0u, pool.used());
  AILEGO_DEFER([&] {
    EXPECT_EQ(0u, pool.used());
    pool.init(old);
  });
  ASSERT_EQ(0, pool.init(16 * 1024));
  IndexDocumentList docs;
  docs.reserve(8);
  const size_t charged = pool.used();
  EXPECT_GE(charged, 8 * sizeof(IndexDocument));
  EXPECT_TRUE(docs.empty());
  docs.emplace_back(1, 2);
  docs.clear();
  EXPECT_EQ(charged, pool.used());
  IndexDocumentList output(std::move(docs));
  EXPECT_EQ(charged, pool.used());
  EXPECT_THROW(output.reserve(1000000), std::bad_alloc);
  EXPECT_EQ(charged, pool.used());
  EXPECT_GE(output.capacity(), 8u);
  EXPECT_NO_THROW(output.emplace_back(2, 3));
  IndexDocumentList().swap(output);
  EXPECT_EQ(0u, pool.used());
}

TEST(IndexResultOwnership, NestedGroupArraysRetainCapacityUntilReleased) {
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  const size_t old = pool.capacity();
  ASSERT_EQ(0u, pool.used());
  AILEGO_DEFER([&] {
    EXPECT_EQ(0u, pool.used());
    pool.init(old);
  });
  ASSERT_EQ(0, pool.init(16 * 1024));
  IndexGroupDocumentList groups(2);
  const size_t outer = pool.used();
  ASSERT_GT(outer, 0u);
  auto &docs = *groups[0].mutable_docs();
  docs.reserve(8);
  const size_t nested = pool.used();
  EXPECT_GT(nested, outer);
  docs.clear();
  EXPECT_EQ(nested, pool.used());
  IndexDocumentList retained(std::move(docs));
  groups.clear();
  EXPECT_EQ(nested, pool.used());
  IndexDocumentList().swap(retained);
  EXPECT_EQ(outer, pool.used());
  IndexGroupDocumentList().swap(groups);
  EXPECT_EQ(0u, pool.used());
}

TEST(IndexResultOwnership, SparseBuffersMoveThroughGrowthAndHandoff) {
  static_assert(std::is_nothrow_move_constructible_v<IndexDocument>);
  static_assert(std::is_nothrow_move_assignable_v<IndexDocument>);
  IndexSparseDocument sparse;
  *sparse.mutable_sparse_count() = 128;
  sparse.mutable_sparse_indices()->assign(512, 'i');
  sparse.mutable_sparse_values()->assign(512, 'v');
  const char *indices = sparse.sparse_indices().data();
  const char *values = sparse.sparse_values().data();
  IndexDocumentList context;
  context.emplace_back(1, 0, 0, nullptr, std::move(sparse));
  for (size_t i = 0; i < 128; ++i) context.emplace_back(i + 2, 0);
  EXPECT_EQ(indices, context.front().sparse_doc().sparse_indices().data());
  EXPECT_EQ(values, context.front().sparse_doc().sparse_values().data());
  IndexDocumentList output;
  output = std::move(context);
  EXPECT_TRUE(context.empty());
  EXPECT_EQ(values, output.front().sparse_doc().sparse_values().data());
  IndexDocument copy(output.front());
  EXPECT_NE(values, copy.sparse_doc().sparse_values().data());
  EXPECT_EQ(copy.sparse_doc().sparse_values(),
            output.front().sparse_doc().sparse_values());
}

TEST(IndexResultOwnership, ReplacingDocumentReleasesPreviousBackingOwner) {
  auto owner = std::make_shared<int>(42);
  std::weak_ptr<int> weak = owner;
  auto block = IndexStorage::MemoryBlock::MakeSharedView(owner.get(), owner);
  IndexDocument doc(1, 0, 0, std::move(block));
  owner.reset();
  const IndexDocument replacement(2, 0);
  doc = replacement;
  EXPECT_TRUE(weak.expired());
  EXPECT_EQ(nullptr, doc.vector());
}
}  // namespace
}  // namespace zvec::core
