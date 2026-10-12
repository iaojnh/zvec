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

#include "hnsw_sparse_entity.h"
#include <numeric>
#include "utility/graph_segment_writer.h"

namespace zvec {
namespace core {

const std::string HnswSparseEntity::kSparseGraphHeaderSegmentId =
    "sparse_graph.header";
const std::string HnswSparseEntity::kSparseGraphFeaturesSegmentId =
    "sparse_graph.features";
const std::string HnswSparseEntity::kSparseGraphKeysSegmentId =
    "sparse_graph.keys";
const std::string HnswSparseEntity::kSparseGraphNeighborsSegmentId =
    "sparse_graph.neighbors";
const std::string HnswSparseEntity::kSparseGraphOffsetsSegmentId =
    "sparse_graph.offsets";
const std::string HnswSparseEntity::kSparseGraphMappingSegmentId =
    "sparse_graph.mapping";
const std::string HnswSparseEntity::kSparseHnswHeaderSegmentId =
    "sparse_hnsw.header";
const std::string HnswSparseEntity::kSparseHnswNeighborsSegmentId =
    "sparse_hnsw.neighbors";
const std::string HnswSparseEntity::kSparseHnswOffsetsSegmentId =
    "sparse_hnsw.offsets";
const std::string HnswSparseEntity::kSparseGraphVectorsSegmentId =
    "sparse_graph.vectors";
const std::string HnswSparseEntity::kSparseGraphVectorMetaSegmentId =
    "sparse_graph.vector_meta";

int HnswSparseEntity::CalcAndAddPadding(const IndexDumper::Pointer &dumper,
                                        size_t data_size,
                                        size_t *padding_size) {
  *padding_size = AlignSize(data_size) - data_size;
  if (*padding_size == 0) {
    return 0;
  }

  std::string padding(*padding_size, '\0');
  if (dumper->write(padding.data(), *padding_size) != *padding_size) {
    LOG_ERROR("Append padding failed, size %lu", *padding_size);
    return IndexError_WriteData;
  }
  return 0;
}

int64_t HnswSparseEntity::dump_segment(const IndexDumper::Pointer &dumper,
                                       const std::string &segment_id,
                                       const void *data, size_t size) const {
  size_t len = dumper->write(data, size);
  if (len != size) {
    LOG_ERROR("Dump segment %s data failed, expect: %lu, actual: %lu",
              segment_id.c_str(), size, len);
    return IndexError_WriteData;
  }

  size_t padding_size = AlignSize(size) - size;
  if (padding_size > 0) {
    std::string padding(padding_size, '\0');
    if (dumper->write(padding.data(), padding_size) != padding_size) {
      LOG_ERROR("Append padding failed, size %lu", padding_size);
      return IndexError_WriteData;
    }
  }

  uint32_t crc = ailego::Crc32c::Hash(data, size);
  int ret = dumper->append(segment_id, size, padding_size, crc);
  if (ret != 0) {
    LOG_ERROR("Dump segment %s meta failed, ret=%d", segment_id.c_str(), ret);
    return ret;
  }

  return len + padding_size;
}

int64_t HnswSparseEntity::dump_header(const IndexDumper::Pointer &dumper,
                                      const HNSWSparseHeader &hd) const {
  //! dump basic graph header. header is aligned and does not need padding
  int64_t graph_hd_size = dump_segment(dumper, kSparseGraphHeaderSegmentId,
                                       &hd.graph, hd.graph.size);
  if (graph_hd_size < 0) {
    return graph_hd_size;
  }

  //! dump basic graph header. header is aligned and does not need padding
  int64_t hnsw_hd_size =
      dump_segment(dumper, kSparseHnswHeaderSegmentId, &hd.hnsw, hd.hnsw.size);
  if (hnsw_hd_size < 0) {
    return hnsw_hd_size;
  }

  return graph_hd_size + hnsw_hd_size;
}

int64_t HnswSparseEntity::dump_segments(
    const IndexDumper::Pointer &dumper,
    const std::function<level_t(node_id_t)> &get_level) const {
  HNSWSparseHeader dump_hd(header());

  dump_hd.graph.node_size = sparse_meta_size();

  std::vector<node_id_t> n2o_mapping;  // map new id to origin id
  std::vector<node_id_t> o2n_mapping;  // map origin id to new id
  // Preserve node order; reshuffling was never implemented.
  if (!o2n_mapping.empty()) {
    dump_hd.hnsw.entry_point = o2n_mapping[entry_point()];
  }

  //! Dump header
  int64_t hd_size = dump_header(dumper, dump_hd);
  if (hd_size < 0) {
    return hd_size;
  }

  //! Dump vectors
  int64_t sparse_vector_meta_size =
      dump_sparse_vector_meta(dumper, n2o_mapping);
  if (sparse_vector_meta_size < 0) {
    return sparse_vector_meta_size;
  }

  int64_t sparse_vecs_size = dump_sparse_vector(dumper, n2o_mapping);
  if (sparse_vecs_size < 0) {
    return sparse_vecs_size;
  }

  //! Dump neighbors
  auto neighbors_size =
      dump_neighbors(dumper, get_level, n2o_mapping, o2n_mapping);
  if (neighbors_size < 0) {
    return neighbors_size;
  }
  //! free memory
  n2o_mapping = std::vector<node_id_t>();
  o2n_mapping = std::vector<node_id_t>();

  int64_t keys_size = DumpGraphRecords<key_t>(
      dumper, kSparseGraphKeysSegmentId, doc_cnt(), [&](size_t i, key_t *key) {
        *key = get_key(i);
        return 0;
      });
  if (keys_size < 0) return keys_size;
  int64_t mapping_size =
      DumpGraphKeyMapping(dumper, kSparseGraphMappingSegmentId, doc_cnt(),
                          [&](size_t i) { return get_key(i); });
  if (mapping_size < 0) {
    return mapping_size;
  }

  return hd_size + keys_size + sparse_vector_meta_size + sparse_vecs_size +
         neighbors_size + mapping_size;
}


int64_t HnswSparseEntity::dump_sparse_vector_meta(
    const IndexDumper::Pointer &dumper,
    const std::vector<node_id_t> &reorder_mapping) const {
  const void *data = nullptr;
  uint32_t crc = 0U;
  size_t dump_size = 0UL;

  uint64_t sparse_data_offset = 0UL;
  uint64_t sparse_data_len = 0UL;

  //! dump vectors
  for (node_id_t id = 0; id < doc_cnt(); ++id) {
    data = get_vector_meta(reorder_mapping.empty() ? id : reorder_mapping[id]);
    if (ailego_unlikely(!data)) {
      return IndexError_ReadData;
    }

    const char *data_ptr = reinterpret_cast<const char *>(data);
    sparse_data_len = *((uint32_t *)(data_ptr + sizeof(uint64_t)));

    size_t len = dumper->write(&sparse_data_offset, sizeof(uint64_t));
    if (len != sizeof(uint64_t)) {
      LOG_ERROR("Dump sparse data offset failed, write=%zu expect=%zu", len,
                sizeof(uint64_t));
      return IndexError_WriteData;
    }

    crc = ailego::Crc32c::Hash(&sparse_data_offset, sizeof(uint64_t), crc);
    dump_size += sizeof(uint64_t);

    len = dumper->write(&sparse_data_len, sizeof(uint64_t));
    if (len != sizeof(uint64_t)) {
      LOG_ERROR("Dump sparse data len failed, write=%zu expect=%zu", len,
                sizeof(uint64_t));
      return IndexError_WriteData;
    }

    crc = ailego::Crc32c::Hash(&sparse_data_len, sizeof(uint64_t), crc);
    dump_size += sizeof(uint64_t);

    sparse_data_offset += sparse_data_len;
  }

  int ret =
      dumper->append(kSparseGraphVectorMetaSegmentId, dump_size, 0UL, crc);
  if (ret != 0) {
    LOG_ERROR("Dump vectors segment meta failed, ret %d", ret);
    return ret;
  }

  return dump_size;
}

int64_t HnswSparseEntity::dump_sparse_vector(
    const IndexDumper::Pointer &dumper,
    const std::vector<node_id_t> &reorder_mapping) const {
  uint32_t crc = 0U;
  size_t data_size = 0UL;
  const void *data = nullptr;

  uint64_t sparse_data_len = 0UL;
  uint32_t sparse_chunk_index = 0U;
  uint32_t sparse_chunk_offset = 0U;

  //! dump vectors
  for (node_id_t id = 0; id < doc_cnt(); ++id) {
    data = get_vector_meta(reorder_mapping.empty() ? id : reorder_mapping[id]);
    if (ailego_unlikely(!data)) {
      return IndexError_ReadData;
    }

    const char *data_ptr = reinterpret_cast<const char *>(data);

    sparse_data_len = *((uint32_t *)(data_ptr + sizeof(uint64_t)));

    uint64_t sparse_offset = *((uint64_t *)(data_ptr));

    const void *sparse = get_sparse_data(sparse_offset, sparse_data_len);
    if (ailego_unlikely(sparse == nullptr)) {
      LOG_ERROR("Get nullptr sparse, chunk index=%u, chunk offset=%u, len=%zu",
                sparse_chunk_index, sparse_chunk_offset,
                (size_t)sparse_data_len);
      return IndexError_ReadData;
    }

    size_t len = dumper->write(sparse, sparse_data_len);
    if (len != sparse_data_len) {
      LOG_ERROR("Dump sparse data failed, write=%zu expect=%zu", len,
                (size_t)sparse_data_len);
      return IndexError_WriteData;
    }

    crc = ailego::Crc32c::Hash(sparse, sparse_data_len, crc);
    data_size += sparse_data_len;
  }

  int ret = dumper->append(kSparseGraphVectorsSegmentId, data_size, 0UL, crc);
  if (ret != 0) {
    LOG_ERROR("Dump vectors segment meta failed, ret %d", ret);
    return ret;
  }

  return data_size;
}

int64_t HnswSparseEntity::dump_graph_neighbors(
    const IndexDumper::Pointer &dumper,
    const std::vector<node_id_t> &reorder_mapping,
    const std::vector<node_id_t> &neighbor_mapping) const {
  size_t offset = 0;
  uint32_t crc = 0;
  std::vector<node_id_t> mapping(l0_neighbor_cnt());

  uint32_t min_neighbor_count = 10000;
  uint32_t max_neighbor_count = 0;
  size_t sum_neighbor_count = 0;

  for (node_id_t id = 0; id < doc_cnt(); ++id) {
    const Neighbors neighbors =
        get_neighbors(0, reorder_mapping.empty() ? id : reorder_mapping[id]);
    ailego_assert_with(!!neighbors.data, "invalid neighbors");
    ailego_assert_with(neighbors.size() <= l0_neighbor_cnt(),
                       "invalid neighbors");

    uint32_t neighbor_count = neighbors.size();
    if (neighbor_count < min_neighbor_count) {
      min_neighbor_count = neighbor_count;
    }
    if (neighbor_count > max_neighbor_count) {
      max_neighbor_count = neighbor_count;
    }
    sum_neighbor_count += neighbor_count;

    size_t size = neighbors.size() * sizeof(node_id_t);
    const node_id_t *data = &neighbors[0];
    if (!neighbor_mapping.empty()) {
      for (node_id_t i = 0; i < neighbors.size(); ++i) {
        mapping[i] = neighbor_mapping[neighbors[i]];
      }
      data = mapping.data();
    }
    if (dumper->write(data, size) != size) {
      LOG_ERROR("Dump graph neighbor id=%u failed, size %lu", id, size);
      return IndexError_WriteData;
    }
    crc = ailego::Crc32c::Hash(data, size, crc);
    offset += size;
  }

  uint32_t average_neighbor_count = 0;
  if (doc_cnt() > 0) {
    average_neighbor_count = sum_neighbor_count / doc_cnt();
  }
  LOG_INFO(
      "Dump hnsw graph: min_neighbor_count[%u] max_neighbor_count[%u] "
      "average_neighbor_count[%u]",
      min_neighbor_count, max_neighbor_count, average_neighbor_count);

  size_t padding_size = 0;
  int ret = CalcAndAddPadding(dumper, offset, &padding_size);
  if (ret != 0) {
    return ret;
  }
  ret =
      dumper->append(kSparseGraphNeighborsSegmentId, offset, padding_size, crc);
  if (ret != 0) {
    LOG_ERROR("Dump segment %s failed, ret %d",
              kSparseGraphNeighborsSegmentId.c_str(), ret);
    return ret;
  }

  //! dump level 0 neighbors meta
  // The graph is unchanged throughout dump. Recompute offsets using bounded
  // output tiles, rather than retaining an array proportional to doc count.
  size_t meta_offset = 0;
  auto len = DumpGraphRecords<SparseGraphNeighborMeta>(
      dumper, kSparseGraphOffsetsSegmentId, doc_cnt(),
      [&](size_t id, SparseGraphNeighborMeta *record) -> int {
        const auto neighbors = get_neighbors(
            0, reorder_mapping.empty() ? id : reorder_mapping[id]);
        if (!neighbors.data && neighbors.size()) return IndexError_ReadData;
        record->offset = meta_offset;
        record->neighbor_cnt = neighbors.size();
        meta_offset += neighbors.size() * sizeof(node_id_t);
        return 0;
      });
  if (len < 0) {
    return len;
  }

  return len + offset + padding_size;
}

int64_t HnswSparseEntity::dump_upper_neighbors(
    const IndexDumper::Pointer &dumper,
    const std::function<level_t(node_id_t)> &get_level,
    const std::vector<node_id_t> &reorder_mapping,
    const std::vector<node_id_t> &neighbor_mapping) const {
  size_t offset = 0;
  uint32_t crc = 0;
  std::vector<node_id_t> buffer(upper_neighbor_cnt() + 1);
  size_t buffer_bytes = buffer.size() * sizeof(node_id_t);
  for (node_id_t id = 0; id < doc_cnt(); ++id) {
    node_id_t new_id = reorder_mapping.empty() ? id : reorder_mapping[id];
    auto level = get_level(new_id);
    if (level < 0) return level;
    if (level == 0) {
      continue;
    }
    ailego_assert_with((size_t)level < kMaxGraphLayers, "invalid level");
    for (level_t cur_level = 1; cur_level <= level; ++cur_level) {
      const Neighbors neighbors = get_neighbors(cur_level, new_id);
      ailego_assert_with(!!neighbors.data, "invalid neighbors");
      ailego_assert_with(neighbors.size() <= neighbor_cnt(cur_level),
                         "invalid neighbors");
      memset(buffer.data(), 0, buffer_bytes);
      buffer[0] = neighbors.size();
      if (neighbor_mapping.empty()) {
        memcpy(&buffer[1], &neighbors[0], neighbors.size() * sizeof(node_id_t));
      } else {
        for (node_id_t i = 0; i < neighbors.size(); ++i) {
          buffer[i + 1] = neighbor_mapping[neighbors[i]];
        }
      }
      if (dumper->write(buffer.data(), buffer_bytes) != buffer_bytes) {
        LOG_ERROR("Dump graph neighbor id=%u failed, size %lu", id,
                  buffer_bytes);
        return IndexError_WriteData;
      }
      crc = ailego::Crc32c::Hash(buffer.data(), buffer_bytes, crc);
      offset += buffer_bytes;
    }
  }
  size_t padding_size = 0;
  int ret = CalcAndAddPadding(dumper, offset, &padding_size);
  if (ret != 0) {
    return ret;
  }

  ret =
      dumper->append(kSparseHnswNeighborsSegmentId, offset, padding_size, crc);
  if (ret != 0) {
    LOG_ERROR("Dump segment %s failed, ret %d",
              kSparseHnswNeighborsSegmentId.c_str(), ret);
    return ret;
  }

  //! dump level 0 neighbors meta
  size_t meta_offset = 0;
  auto len = DumpGraphRecords<HnswSparseNeighborMeta>(
      dumper, kSparseHnswOffsetsSegmentId, doc_cnt(),
      [&](size_t id, HnswSparseNeighborMeta *record) -> int {
        auto level =
            get_level(reorder_mapping.empty() ? id : reorder_mapping[id]);
        if (level < 0) return level;
        if (static_cast<size_t>(level) >= kMaxGraphLayers)
          return IndexError_InvalidFormat;
        record->level = level;
        record->offset = level ? meta_offset : 0;
        meta_offset += static_cast<size_t>(level) * (upper_neighbor_cnt() + 1) *
                       sizeof(node_id_t);
        return 0;
      });
  if (len < 0) {
    return len;
  }

  return len + offset + padding_size;
}

}  // namespace core
}  // namespace zvec
