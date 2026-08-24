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

#include "diskann_searcher.h"
#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/ailego/container/vector.h>
#include <zvec/core/framework/index_factory.h>
#include "diskann_holder.h"
#include "diskann_params.h"
#include "diskann_util.h"

namespace zvec {
namespace core {

struct DiskAnnNodeLayoutForTest {
  uint64_t index_segment_offset{0};
  uint32_t node_per_sector{0};
  uint32_t max_node_size{0};
  uint32_t max_degree{0};
  uint64_t doc_count{0};
  diskann_id_t medoid{0};
};

class DiskAnnCacheTestPeer {
 public:
  static void set_parser_bounds(DiskAnnIndexer &indexer, uint32_t max_degree,
                                uint64_t doc_count) {
    indexer.max_degree_ = max_degree;
    indexer.doc_cnt_ = doc_count;
  }

  static int parse_node_neighbors(DiskAnnIndexer &indexer,
                                  const uint8_t *node_buf,
                                  diskann_id_t node_id,
                                  uint32_t &neighbor_count,
                                  diskann_id_t *neighbors) {
    return indexer.parse_node_neighbors(node_buf, node_id, neighbor_count,
                                        neighbors);
  }

  static DiskAnnNodeLayoutForTest layout(const DiskAnnSearcher &searcher) {
    const DiskAnnIndexer &indexer = *searcher.diskann_indexer_;
    return {indexer.index_segment_offset_, indexer.node_per_sector_,
            indexer.max_node_size_,       indexer.max_degree_,
            indexer.doc_cnt_,             indexer.medoid_};
  }
};

}  // namespace core
}  // namespace zvec

namespace {

using namespace zvec::ailego;
using namespace zvec::core;

void ExpectUnalignedNeighborParse(IndexMeta::DataType data_type,
                                  uint32_t dimension) {
  IndexMeta meta(data_type, dimension);
  DiskAnnIndexer indexer(meta);
  DiskAnnCacheTestPeer::set_parser_bounds(indexer, 2, 4);

  std::vector<uint8_t> storage(meta.element_size() + sizeof(uint32_t) +
                               2 * sizeof(diskann_id_t) + 4);
  size_t prefix = 0;
  while ((reinterpret_cast<uintptr_t>(storage.data() + prefix +
                                      meta.element_size()) %
          alignof(uint32_t)) == 0) {
    ++prefix;
  }
  uint8_t *node = storage.data() + prefix;
  EXPECT_NE(reinterpret_cast<uintptr_t>(node + meta.element_size()) %
                alignof(uint32_t),
            0U);

  const uint32_t stored_count = 2;
  const std::array<diskann_id_t, 2> stored_neighbors{1, 3};
  memcpy(node + meta.element_size(), &stored_count, sizeof(stored_count));
  memcpy(node + meta.element_size() + sizeof(stored_count),
         stored_neighbors.data(),
         stored_neighbors.size() * sizeof(diskann_id_t));

  uint32_t parsed_count = 0;
  std::array<diskann_id_t, 2> parsed_neighbors{};
  EXPECT_EQ(0, DiskAnnCacheTestPeer::parse_node_neighbors(
                   indexer, node, 0, parsed_count, parsed_neighbors.data()));
  EXPECT_EQ(parsed_count, stored_count);
  EXPECT_EQ(parsed_neighbors, stored_neighbors);
}

#if !defined(_WIN32) && !defined(_WIN64)
bool ReadExact(const std::string &path, uint64_t offset, void *data,
               size_t size) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    return false;
  }
  stream.seekg(static_cast<std::streamoff>(offset));
  stream.read(static_cast<char *>(data), static_cast<std::streamsize>(size));
  return static_cast<size_t>(stream.gcount()) == size;
}

bool WriteExact(const std::string &path, uint64_t offset, const void *data,
                size_t size) {
  std::fstream stream(path, std::ios::binary | std::ios::in | std::ios::out);
  if (!stream) {
    return false;
  }
  stream.seekp(static_cast<std::streamoff>(offset));
  stream.write(static_cast<const char *>(data),
               static_cast<std::streamsize>(size));
  stream.flush();
  return stream.good();
}
#endif

class DiskAnnNodeLayoutTest : public testing::Test {
 protected:
  void TearDown() override {
    std::error_code error;
    std::filesystem::remove(index_path_, error);
  }

  const std::string index_path_{"DiskAnnNodeLayoutTest.index"};
};

TEST(DiskAnnNodeParserTest, SafelyParsesUnalignedFp16AndInt8Records) {
  ExpectUnalignedNeighborParse(IndexMeta::DataType::DT_FP16, 3);
  ExpectUnalignedNeighborParse(IndexMeta::DataType::DT_INT8, 3);
}

TEST(DiskAnnNodeParserTest, RejectsMalformedCountAndNeighborId) {
  IndexMeta meta(IndexMeta::DataType::DT_FP16, 3);
  DiskAnnIndexer indexer(meta);
  DiskAnnCacheTestPeer::set_parser_bounds(indexer, 2, 4);

  std::vector<uint8_t> storage(1 + meta.element_size() + sizeof(uint32_t) +
                               3 * sizeof(diskann_id_t));
  uint8_t *node = storage.data() + 1;
  std::array<diskann_id_t, 2> output{};
  uint32_t parsed_count = 0;

  uint32_t stored_count = 3;
  memcpy(node + meta.element_size(), &stored_count, sizeof(stored_count));
  EXPECT_EQ(IndexError_InvalidFormat,
            DiskAnnCacheTestPeer::parse_node_neighbors(
                indexer, node, 0, parsed_count, output.data()));

  stored_count = 2;
  const std::array<diskann_id_t, 2> stored_neighbors{1, 4};
  memcpy(node + meta.element_size(), &stored_count, sizeof(stored_count));
  memcpy(node + meta.element_size() + sizeof(stored_count),
         stored_neighbors.data(),
         stored_neighbors.size() * sizeof(diskann_id_t));
  EXPECT_EQ(IndexError_InvalidFormat,
            DiskAnnCacheTestPeer::parse_node_neighbors(
                indexer, node, 0, parsed_count, output.data()));
}

TEST_F(DiskAnnNodeLayoutTest,
       OddDimensionFp16SearchRejectsMalformedRealtimeNeighbors) {
  constexpr uint32_t kDimension = 3;
  constexpr size_t kDocCount = 64;

  IndexMeta meta(IndexMeta::DataType::DT_FP16, kDimension);
  meta.set_metric("SquaredEuclidean", 0, Params());
  ASSERT_NE(meta.element_size() % alignof(uint32_t), 0U);

  auto holder =
      std::make_shared<MultiPassIndexHolder<IndexMeta::DataType::DT_FP16>>(
          kDimension);
  for (size_t i = 0; i < kDocCount; ++i) {
    NumericalVector<Float16> vector(kDimension);
    for (size_t d = 0; d < kDimension; ++d) {
      vector[d] = static_cast<float>(i + d) / 10.0f;
    }
    ASSERT_TRUE(holder->emplace(i, vector));
  }

  Params build_params;
  build_params.set(PARAM_DISKANN_BUILDER_MAX_DEGREE, 16);
  build_params.set(PARAM_DISKANN_BUILDER_LIST_SIZE, 32);
  build_params.set(PARAM_DISKANN_BUILDER_MAX_PQ_CHUNK_NUM, 1);
  build_params.set(PARAM_DISKANN_BUILDER_THREAD_COUNT, 2);

  auto builder = IndexFactory::CreateBuilder("DiskAnnBuilder");
  ASSERT_NE(builder, nullptr);
  ASSERT_EQ(0, builder->init(meta, build_params));
  ASSERT_EQ(0, builder->train(holder));
  ASSERT_EQ(0, builder->build(holder));

  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(dumper, nullptr);
  ASSERT_EQ(0, dumper->create(index_path_));
  ASSERT_EQ(0, builder->dump(dumper));
  ASSERT_EQ(0, dumper->close());

  Params search_params;
  search_params.set(PARAM_DISKANN_SEARCHER_LIST_SIZE, 32);
  search_params.set(PARAM_DISKANN_SEARCHER_CACHE_NODE_NUM, 0);
  auto searcher = IndexFactory::CreateSearcher("DiskAnnSearcher");
  ASSERT_NE(searcher, nullptr);
  ASSERT_EQ(0, searcher->init(search_params));

  auto storage = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_NE(storage, nullptr);
  ASSERT_EQ(0, storage->open(index_path_, false));
  ASSERT_EQ(0, searcher->load(storage, IndexMetric::Pointer()));

  NumericalVector<Float16> query(kDimension, static_cast<float>(12.1f));
  IndexQueryMeta query_meta(IndexMeta::DataType::DT_FP16, kDimension);
  auto run_search = [&]() -> int {
    auto context = searcher->create_context();
    if (!context) {
      return IndexError_NoMemory;
    }
    context->set_topk(5);
    return searcher->search_impl(query.data(), query_meta, context);
  };

  ASSERT_EQ(0, run_search());

#if !defined(_WIN32) && !defined(_WIN64)
  auto *diskann_searcher = dynamic_cast<DiskAnnSearcher *>(searcher.get());
  ASSERT_NE(diskann_searcher, nullptr);
  const DiskAnnNodeLayoutForTest layout =
      DiskAnnCacheTestPeer::layout(*diskann_searcher);
  ASSERT_GT(layout.max_degree, 0U);
  ASSERT_EQ(layout.doc_count, kDocCount);

  const uint64_t node_sector = DiskAnnUtil::get_node_sector(
      layout.node_per_sector, layout.max_node_size, DiskAnnUtil::kSectorSize,
      layout.medoid);
  const uint64_t node_offset =
      layout.node_per_sector == 0
          ? 0
          : static_cast<uint64_t>(layout.medoid % layout.node_per_sector) *
                layout.max_node_size;
  const uint64_t count_offset =
      layout.index_segment_offset + node_sector * DiskAnnUtil::kSectorSize +
      node_offset + meta.element_size();

  uint32_t original_count = 0;
  diskann_id_t original_neighbor = 0;
  ASSERT_TRUE(ReadExact(index_path_, count_offset, &original_count,
                        sizeof(original_count)));
  ASSERT_GT(original_count, 0U);
  ASSERT_TRUE(ReadExact(index_path_, count_offset + sizeof(original_count),
                        &original_neighbor, sizeof(original_neighbor)));

  const uint32_t invalid_count = layout.max_degree + 1;
  ASSERT_TRUE(WriteExact(index_path_, count_offset, &invalid_count,
                         sizeof(invalid_count)));
  EXPECT_EQ(IndexError_InvalidFormat, run_search());

  ASSERT_TRUE(WriteExact(index_path_, count_offset, &original_count,
                         sizeof(original_count)));
  const diskann_id_t invalid_neighbor =
      static_cast<diskann_id_t>(layout.doc_count);
  ASSERT_TRUE(WriteExact(index_path_, count_offset + sizeof(original_count),
                         &invalid_neighbor, sizeof(invalid_neighbor)));
  EXPECT_EQ(IndexError_InvalidFormat, run_search());

  ASSERT_TRUE(WriteExact(index_path_, count_offset + sizeof(original_count),
                         &original_neighbor, sizeof(original_neighbor)));
  EXPECT_EQ(0, run_search());
#endif
}

}  // namespace
