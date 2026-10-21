// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <filesystem>
#include <gtest/gtest.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/ailego/buffer/vector_page_table.h>
#include <zvec/core/framework/index_framework.h>
#include "ailego/pattern/defer.h"
#include "algorithm/flat/flat_utility.h"
#include "algorithm/flat_sparse/flat_sparse_streamer_entity.h"
#include "utility/sparse_utility.h"
#include "../../../ailego/buffer/pool_payload_budget.h"

namespace zvec::core {
namespace {
TEST(FlatSparseMemoryTest, DirectIdsHolesUpdatesAndReopen) {
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  const size_t previous = pool.capacity();
  ASSERT_EQ(0u, pool.used());
  AILEGO_DEFER([&] {
    EXPECT_EQ(0u, pool.used());
    EXPECT_EQ(0, pool.init(previous));
  });
  const size_t budget =
      2 * ailego::VecBufferPool::metadata_bytes_for_page_count(4096, true) +
      64 * ailego::kVectorPageSize;
  ASSERT_EQ(0, pool.init(budget));
  for (const auto *mode : {"MMapFileStorage", "BufferStorage"}) {
    const std::string path = std::string("flat_sparse_direct_") + mode;
    ASSERT_FALSE(std::filesystem::exists(path));
    AILEGO_DEFER([&] { std::filesystem::remove(path); });
    IndexMeta meta(IndexMeta::MT_SPARSE, IndexMeta::DT_FP32);
    meta.set_metric("InnerProductSparse", 0, ailego::Params{});
    ailego::Params params;
    params.set(PARAM_FLAT_USE_ID_MAP, false);
    meta.set_streamer("FlatSparseStreamer", 0, params);
    const uint32_t indices[]{1, 4, 7};
    const float values[]{0.5f, 1.0f, 2.0f};
    std::string row;
    SparseUtility::TransSparseFormat(3, indices, values, sizeof(float), row);
    for (int round = 0; round < 2; ++round) {
      auto storage = IndexFactory::CreateStorage(mode);
      ASSERT_EQ(0, storage->init({}));
      ASSERT_EQ(0, storage->open(path, true));
      IndexStreamer::Stats stats;
      FlatSparseStreamerEntity entity(stats);
      ASSERT_EQ(0, entity.open(storage, meta));
      if (!round) {
        for (uint32_t id = 3; id < 3001; ++id)
          ASSERT_EQ(0, entity.add_vector_with_id(id, row, 3));
      } else {
        EXPECT_EQ(3001u, stats.loaded_count());
      }
      EXPECT_EQ(kInvalidNodeId, entity.get_id(0));
      EXPECT_EQ(kInvalidNodeId, entity.get_id(uint64_t{1} << 32));
      EXPECT_EQ(kInvalidNodeId, entity.get_id(3001));
      EXPECT_EQ(3000u, entity.get_id(3000));
      ASSERT_EQ(0, entity.add_vector_with_id(1, row, 3));
      auto clone = entity.clone();
      EXPECT_EQ(1u, clone->get_id(1));
      EXPECT_EQ(3000u, clone->get_id(3000));
      std::string actual;
      EXPECT_EQ(IndexError_NoMemory,
                clone->get_sparse_vector_by_key(3000, &actual));
      {
        IndexStorage::MemoryBlock encoded;
        uint32_t stored_length = 0;
        ASSERT_EQ(0, clone->get_sparse_vector_ptr_by_key(3000, encoded,
                                                         &stored_length));
        // The stored representation includes alignment padding.
        actual.reserve(stored_length);
      }
      ASSERT_EQ(0, clone->get_sparse_vector_by_key(3000, &actual));
      EXPECT_EQ(row, actual.substr(0, row.size()));
      clone.reset();
      // Holes have zero payload and retain their sentinel key in dumps.
      IndexStorage::MemoryBlock hole;
      uint32_t hole_length = 123;
      ASSERT_EQ(0, entity.get_sparse_vector_ptr_by_id(0, hole, &hole_length));
      EXPECT_EQ(0u, hole_length);
      const std::string dump_path = path + ".dump";
      ASSERT_FALSE(std::filesystem::exists(dump_path));
      AILEGO_DEFER([&] { std::filesystem::remove(dump_path); });
      auto dumper = IndexFactory::CreateDumper("FileDumper");
      ASSERT_EQ(0, dumper->create(dump_path));
      ASSERT_EQ(0, entity.dump(dumper));
      ASSERT_EQ(0, dumper->close());
      auto dumped = IndexFactory::CreateStorage("MMapFileReadStorage");
      ASSERT_EQ(0, dumped->init({}));
      ASSERT_EQ(0, dumped->open(dump_path, false));
      auto keys = dumped->get(PARAM_FLAT_SPARSE_DUMP_KEYS_SEG_ID);
      ASSERT_NE(nullptr, keys);
      uint64_t prefix[3]{};
      ASSERT_EQ(sizeof(prefix), keys->fetch(0, prefix, sizeof(prefix)));
      EXPECT_EQ(kInvalidKey, prefix[0]);
      EXPECT_EQ(1u, prefix[1]);
      EXPECT_EQ(kInvalidKey, prefix[2]);
      keys.reset();
      ASSERT_EQ(0, dumped->close());
      ASSERT_EQ(0, entity.flush(0));
      ASSERT_EQ(0, entity.close());
      ASSERT_EQ(0, storage->close());
    }
  }
}

TEST(FlatSparseMemoryTest, PagedDumpPreservesAllLegacySegments) {
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  const size_t previous = pool.capacity();
  ASSERT_EQ(0u, pool.used());
  AILEGO_DEFER([&] {
    EXPECT_EQ(0u, pool.used());
    EXPECT_EQ(0, pool.init(previous));
  });
  IndexMeta meta(IndexMeta::MT_SPARSE, IndexMeta::DT_FP32);
  meta.set_metric("InnerProductSparse", 0, ailego::Params{});
  auto holder =
      std::make_shared<MultiPassIndexSparseHolder<IndexMeta::DT_FP32>>();
  constexpr size_t count = 131075;
  for (size_t i = 0; i < count; ++i) {
    ailego::SparseVector<float> row;
    ailego::NumericalVector<uint32_t> indices(3);
    ailego::NumericalVector<float> values(3);
    for (size_t d = 0; d < 3; ++d) {
      indices[d] = static_cast<uint32_t>(1 + d * 3);
      values[d] = static_cast<float>(i + d);
    }
    row.add_sparses(indices, values);
    ASSERT_TRUE(holder->emplace((i * 17 % count) + 99, row));
  }
  std::vector<std::string> reference;
  // Two live writable stores (keys + offsets, or source + keys) need fixed
  // writeback metadata plus the explicitly enabled arbitrary-key map.
  // Allow up to 128 bytes per map entry, then 64 pages for the dump.
  const size_t budget =
      2 * ailego::VecBufferPool::metadata_bytes_for_page_count(4096, true) +
      64 * ailego::kVectorPageSize + count * 128;
  for (int mode = 0; mode < 4; ++mode) {
    SCOPED_TRACE(mode);
    ASSERT_EQ(0, pool.init(mode % 2 ? budget : 0));
    if (mode == 2) reference.clear();
    const std::string path = "flat_sparse_paged_dump_" + std::to_string(mode);
    const std::string stream_path = path + ".stream";
    ASSERT_FALSE(std::filesystem::exists(path));
    ASSERT_FALSE(std::filesystem::exists(stream_path));
    AILEGO_DEFER([&] {
      std::filesystem::remove(path);
      std::filesystem::remove(stream_path);
    });
    auto dumper = IndexFactory::CreateDumper("FileDumper");
    ASSERT_EQ(0, dumper->create(path));
    if (mode < 2) {
      auto builder = IndexFactory::CreateBuilder("FlatSparseBuilder");
      ASSERT_EQ(0, builder->init(meta, {}));
      ASSERT_EQ(0, builder->train(holder));
      ASSERT_EQ(0, builder->build(holder));
      ASSERT_EQ(0, builder->dump(dumper));
    } else {
      auto storage = IndexFactory::CreateStorage(mode == 2 ? "MMapFileStorage"
                                                           : "BufferStorage");
      ASSERT_EQ(0, storage->init({}));
      ASSERT_EQ(0, storage->open(stream_path, true));
      auto streamer = IndexFactory::CreateStreamer("FlatSparseStreamer");
      ASSERT_EQ(0, streamer->init(meta, {}));
      ASSERT_EQ(0, streamer->open(storage));
      auto context = streamer->create_context();
      auto it = holder->create_iterator();
      for (; it->is_valid(); it->next())
        ASSERT_EQ(0, streamer->add_impl(it->key(), it->sparse_count(),
                                        it->sparse_indices(), it->sparse_data(),
                                        IndexQueryMeta(IndexMeta::MT_SPARSE,
                                                       IndexMeta::DT_FP32),
                                        context));
      ASSERT_EQ(0, streamer->dump(dumper));
      context.reset();
      ASSERT_EQ(0, streamer->close());
      ASSERT_EQ(0, storage->close());
    }
    ASSERT_EQ(0, dumper->close());
    auto storage = IndexFactory::CreateStorage("MMapFileReadStorage");
    ASSERT_EQ(0, storage->init({}));
    ASSERT_EQ(0, storage->open(path, false));
    size_t index = 0;
    for (const auto &name : {PARAM_FLAT_SPARSE_DUMP_DATA_SEG_ID,
                             PARAM_FLAT_SPARSE_DUMP_OFFSET_SEG_ID,
                             PARAM_FLAT_SPARSE_DUMP_KEYS_SEG_ID,
                             PARAM_FLAT_SPARSE_DUMP_MAPPING_SEG_ID}) {
      auto segment = storage->get(name);
      ASSERT_NE(nullptr, segment);
      std::string bytes(segment->data_size(), '\0');
      ASSERT_EQ(bytes.size(), segment->fetch(0, bytes.data(), bytes.size()));
      if (mode % 2 == 0)
        reference.push_back(bytes);
      else
        EXPECT_TRUE(reference[index] == bytes) << "segment: " << name;
      ++index;
    }
    ASSERT_EQ(0, storage->close());
    EXPECT_EQ(0u, pool.used());
  }
}
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

TEST(FlatSparseMemoryTest, SearcherMetadataReadsAreScopedAndErrorsAreDistinct) {
  const std::string path = "flat_sparse_metadata_failure.index";
  AILEGO_DEFER([&] { std::filesystem::remove(path); });
  IndexMeta meta(IndexMeta::MT_SPARSE, IndexMeta::DT_FP32);
  meta.set_metric("InnerProductSparse", 0, ailego::Params{});
  auto holder =
      std::make_shared<MultiPassIndexSparseHolder<IndexMeta::DT_FP32>>();
  ailego::SparseVector<float> row;
  ailego::NumericalVector<uint32_t> indices(1);
  indices[0] = 1;
  ailego::NumericalVector<float> values(1);
  values[0] = 1;
  row.add_sparses(indices, values);
  ASSERT_TRUE(holder->emplace(37, row));
  auto builder = IndexFactory::CreateBuilder("FlatSparseBuilder");
  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_EQ(0, builder->init(meta, {}));
  ASSERT_EQ(0, builder->train(holder));
  ASSERT_EQ(0, builder->build(holder));
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, builder->dump(dumper));
  ASSERT_EQ(0, dumper->close());
  auto storage = std::make_shared<test::FaultStorage>(
      IndexFactory::CreateStorage("BufferReadStorage"));
  ASSERT_EQ(0, storage->init({}));
  ASSERT_EQ(0, storage->open(path, false));
  FlatSparseSearcherEntity entity;
  ASSERT_EQ(0, entity.load(storage, meta));
  const size_t block_reads = storage->fault->block_reads;
  for (size_t i = 0; i < 1000; ++i) {
    EXPECT_EQ(37u, entity.get_key(0));
    EXPECT_EQ(0u, entity.get_id(37));
  }
  EXPECT_EQ(block_reads, storage->fault->block_reads);
  storage->fault->segment = PARAM_FLAT_SPARSE_DUMP_KEYS_SEG_ID;
  storage->fault->read = true;
  EXPECT_THROW(entity.get_key(0), FlatSparseEntity::ReadError);
  storage->fault->read = true;
  EXPECT_THROW(entity.get_id(37), FlatSparseEntity::ReadError);
  EXPECT_EQ(kInvalidNodeId, entity.get_id(99));
  ASSERT_EQ(0, entity.unload());
  ASSERT_EQ(0, storage->close());
}
}  // namespace
}  // namespace zvec::core
