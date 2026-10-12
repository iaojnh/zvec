// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#include <type_traits>
#include <gtest/gtest.h>
#include <zvec/core/framework/index_document.h>
#include "ailego/pattern/defer.h"

namespace zvec::core {
namespace {
TEST(IndexResultOwnership, VectorBudgetFollowsReturnedAndCopiedDocuments) {
  auto &pool = ailego::MemoryLimitPool::get_instance();
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
  EXPECT_EQ(data, output.front().vector_string().data());
  EXPECT_EQ(data, retained.vector_string().data());
  output.clear();
  EXPECT_EQ(charged, pool.used());
  EXPECT_EQ(source, retained.vector_string());
  retained = IndexDocument{};
  EXPECT_EQ(0u, pool.used());
}

TEST(IndexResultOwnership, BatchOutputCannotRecycleLivePayloadBudget) {
  auto &pool = ailego::MemoryLimitPool::get_instance();
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
  EXPECT_EQ(0u, pool.used());
  EXPECT_NO_THROW(append());
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
