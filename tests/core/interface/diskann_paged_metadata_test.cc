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

#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <gtest/gtest.h>
#if DISKANN_SUPPORTED
#include "algorithm/diskann/diskann_dist_calculator.h"
#include "algorithm/diskann/diskann_searcher_entity.h"
#include "algorithm/diskann/diskann_util.h"

namespace zvec::core {
namespace {

class ObservedSegment : public IndexStorage::Segment {
 public:
  std::string bytes;
  mutable size_t reads{0};
  bool fail{false};
  std::shared_ptr<int> lease{std::make_shared<int>(0)};
  size_t data_size() const override {
    return bytes.size();
  }
  uint32_t data_crc() const override {
    return 0;
  }
  size_t padding_size() const override {
    return 0;
  }
  size_t capacity() const override {
    return bytes.size();
  }
  size_t fetch(size_t off, void *out, size_t len) const override {
    ++reads;
    if (fail || off > bytes.size() || len > bytes.size() - off) return 0;
    std::memcpy(out, bytes.data() + off, len);
    return len;
  }
  size_t read(size_t off, IndexStorage::MemoryBlock &out, size_t len) override {
    out.reset();
    ++reads;
    if (fail || off > bytes.size() || len > bytes.size() - off) return 0;
    if (len != 0) {
      out =
          IndexStorage::MemoryBlock::MakeSharedView(bytes.data() + off, lease);
    }
    return len;
  }
  size_t write(size_t, const void *, size_t) override {
    return 0;
  }
  size_t resize(size_t) override {
    return 0;
  }
  void update_data_crc(uint32_t) override {}
  Pointer clone() override {
    return nullptr;
  }
};

class ObservedStorage : public IndexStorage {
 public:
  explicit ObservedStorage(bool paged) {
    if (paged) {
      const std::string path = "diskann_paged_metadata_marker";
      {
        std::ofstream file(path, std::ios::binary);
        file << std::string(4096, '\0');
      }
      pool_ = std::make_shared<ailego::VecBufferPool>(path);
    }
  }
  ~ObservedStorage() override {
    if (pool_) {
      pool_.reset();
      std::remove("diskann_paged_metadata_marker");
    }
  }
  std::map<std::string, std::shared_ptr<ObservedSegment>> segments;
  void put(const std::string &id, const void *data, size_t len) {
    auto segment = std::make_shared<ObservedSegment>();
    segment->bytes.assign(static_cast<const char *>(data), len);
    segments[id] = std::move(segment);
  }
  int init(const ailego::Params &) override {
    return 0;
  }
  int cleanup() override {
    return 0;
  }
  int open(const std::string &, bool) override {
    return 0;
  }
  int flush() override {
    return 0;
  }
  int close() override {
    return 0;
  }
  int append(const std::string &, size_t) override {
    return IndexError_Unsupported;
  }
  void refresh(uint64_t) override {}
  uint64_t check_point() const override {
    return 0;
  }
  Segment::Pointer get(const std::string &id, int = -1) override {
    return segments.at(id);
  }
  bool has(const std::string &id) const override {
    return segments.count(id) != 0;
  }
  uint32_t magic() const override {
    return 0;
  }
  std::shared_ptr<ailego::VecBufferPool> vec_buffer_pool() const override {
    return pool_;
  }

 private:
  std::shared_ptr<ailego::VecBufferPool> pool_;
};

TEST(DiskAnnPagedMetadataTest, LazyReadsCloneLifetimeAndIoErrors) {
  constexpr size_t kDim = 8;
  constexpr uint32_t kChunks = 2;
  IndexMeta meta(IndexMeta::DT_FP32, kDim);
  meta.set_metric("SquaredEuclidean", 0, ailego::Params());
  // Import a deterministic legacy codebook; no training/SIMD requirements.
  std::vector<float> pivots(256 * kDim);
  for (size_t i = 0; i < pivots.size(); ++i) pivots[i] = float(i % 29) / 29.0f;
  std::vector<float> mean(kDim, 0.0f);
  const uint32_t offsets[]{0, 4, 8};
  std::string codebook(reinterpret_cast<const char *>(pivots.data()),
                       pivots.size() * sizeof(float));
  codebook.append(reinterpret_cast<const char *>(mean.data()),
                  mean.size() * sizeof(float));
  codebook.append(reinterpret_cast<const char *>(offsets), sizeof(offsets));
  auto quantizer = DiskAnnUtil::create_quantizer_from_meta_buffer(
      codebook, meta, kChunks, true);
  ASSERT_NE(quantizer, nullptr);
  auto measure = IndexFactory::CreateMetric("SquaredEuclidean");
  ASSERT_NE(measure, nullptr);
  ASSERT_EQ(measure->init(meta, ailego::Params()), 0);

  for (bool paged : {false, true}) {
    SCOPED_TRACE(paged);
    auto storage = std::make_shared<ObservedStorage>(paged);
    DiskAnnMetaHeader header;
    header.doc_cnt = 3;
    header.ndims = kDim;
    DiskAnnPqMeta pq_meta;
    pq_meta.chunk_num = kChunks;
    pq_meta.quantizer_meta_buffer_size = sizeof(turbo::QuantizerSerHeader);
    const diskann_key_t keys[]{70, 10, 40};
    const diskann_id_t mapping[]{1, 2, 0};
    const uint8_t codes[]{1, 2, 3, 4, 5, 6};
    const uint32_t no_entrypoints = 0;
    storage->put(DiskAnnEntity::kDiskAnnMetaSegmentId, &header, sizeof(header));
    storage->put(DiskAnnEntity::kDiskAnnPqMetaSegmentId, &pq_meta,
                 sizeof(pq_meta));
    storage->put(DiskAnnEntity::kDiskAnnPqDataSegmentId, codes, sizeof(codes));
    storage->put(DiskAnnEntity::kDiskAnnKeySegmentId, keys, sizeof(keys));
    storage->put(DiskAnnEntity::kDiskAnnKeyMappingSegmentId, mapping,
                 sizeof(mapping));
    storage->put(DiskAnnEntity::kDiskAnnEntryPointSegmentId, &no_entrypoints,
                 sizeof(no_entrypoints));
    storage->put(DiskAnnEntity::kDiskAnnVectorSegmentId, "", 0);
    DiskAnnSearcherEntity entity;
    ASSERT_EQ(entity.load(meta, storage), 0);
    auto pq_segment =
        storage->segments.at(DiskAnnEntity::kDiskAnnPqDataSegmentId);
    auto key_segment =
        storage->segments.at(DiskAnnEntity::kDiskAnnKeySegmentId);
    auto mapping_segment =
        storage->segments.at(DiskAnnEntity::kDiskAnnKeyMappingSegmentId);
    EXPECT_EQ(pq_segment->reads, paged ? 0u : 1u);
    EXPECT_EQ(key_segment->reads, paged ? 0u : 1u);
    EXPECT_EQ(mapping_segment->reads, paged ? 0u : 1u);
    auto clone =
        std::dynamic_pointer_cast<DiskAnnSearcherEntity>(entity.clone());
    ASSERT_NE(clone, nullptr);
    entity.clear();
    storage.reset();
    for (diskann_id_t i = 0; i < 3; ++i) {
      diskann_key_t key = kInvalidKey;
      diskann_id_t id = kInvalidId;
      ASSERT_EQ(clone->read_key(i, &key), 0);
      EXPECT_EQ(key, keys[i]);
      ASSERT_EQ(clone->find_id(key, &id), 0);
      EXPECT_EQ(id, i);
    }
    diskann_id_t missing = 0;
    ASSERT_EQ(clone->find_id(99, &missing), 0);
    EXPECT_EQ(missing, kInvalidId);

    DistCalculator dc(clone.get(), meta, measure);
    dc.bind_quantizer(quantizer.get(), clone->pq_codes(), kChunks,
                      clone->paged_pq_codes());
    const float query[kDim]{};
    dc.reset_query(query);
    dc.quantize_query(query);
    const diskann_id_t ids[]{2, 0, 2, 1};
    float distances[4]{};
    dc.quantized_dist(ids, 4, distances);
    ASSERT_FALSE(dc.error());
    std::vector<float> lut(quantizer->quantized_query_vector_length() /
                           sizeof(float));
    quantizer->quantize_query(query, lut.data());
    for (size_t i = 0; i < 4; ++i)
      EXPECT_FLOAT_EQ(distances[i], quantizer->calc_distance_dp_query(
                                        codes + ids[i] * kChunks, lut.data()));
    EXPECT_EQ(pq_segment->lease.use_count(),
              1);  // No page pin retained between hops.

    for (auto mode : {VisitFilter::BitMap, VisitFilter::ByteMap,
                      VisitFilter::EpochBitMap}) {
      SCOPED_TRACE(mode);
      VisitFilter visited;
      ASSERT_EQ(visited.init(mode, 3, 3, 0.01f), 0);
      visited.set_visited(0);
      std::vector<diskann_id_t> unvisited;
      std::vector<float> selected_distances;
      const size_t before = pq_segment->reads;
      dc.quantized_dist_unvisited(ids, 4, visited, unvisited,
                                  selected_distances);
      ASSERT_FALSE(dc.error());
      ASSERT_EQ(unvisited, (std::vector<diskann_id_t>{2, 1}));
      ASSERT_EQ(selected_distances.size(), 2u);
      EXPECT_FLOAT_EQ(selected_distances[0], distances[0]);
      EXPECT_FLOAT_EQ(selected_distances[1], distances[3]);
      EXPECT_EQ(pq_segment->reads - before, paged ? 2u : 0u);
      EXPECT_EQ(pq_segment->lease.use_count(), 1);

      // No read (or quantizer invocation) when every neighbor was visited.
      pq_segment->fail = true;
      dc.quantized_dist_unvisited(ids, 4, visited, unvisited,
                                  selected_distances);
      EXPECT_FALSE(dc.error());
      EXPECT_TRUE(unvisited.empty());
      EXPECT_TRUE(selected_distances.empty());
      EXPECT_EQ(pq_segment->reads - before, paged ? 2u : 0u);

      if (paged) {
        // A fresh query still surfaces read failures and releases every pin.
        visited.clear();
        dc.quantized_dist_unvisited(ids, 4, visited, unvisited,
                                    selected_distances);
        EXPECT_EQ(dc.error_code(), IndexError_ReadData);
        EXPECT_EQ(pq_segment->lease.use_count(), 1);
        dc.clear();
      }
      pq_segment->fail = false;
      visited.clear();
      const diskann_id_t invalid = 3;
      dc.quantized_dist_unvisited(&invalid, 1, visited, unvisited,
                                  selected_distances);
      EXPECT_EQ(dc.error_code(), IndexError_InvalidFormat);
      dc.clear();
    }

    if (paged) {
      pq_segment->fail = true;
      dc.quantized_dist(ids, 4, distances);
      EXPECT_EQ(dc.error_code(), IndexError_ReadData);
      EXPECT_EQ(pq_segment->lease.use_count(), 1);
      key_segment->fail = true;
      diskann_key_t key;
      EXPECT_EQ(clone->read_key(0, &key), IndexError_ReadData);
      EXPECT_EQ(clone->find_id(40, &missing), IndexError_ReadData);
      key_segment->fail = false;
      mapping_segment->fail = true;
      EXPECT_EQ(clone->find_id(40, &missing), IndexError_ReadData);
    }
  }
}

}  // namespace
}  // namespace zvec::core
#endif
