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
#include "vamana_entity.h"
#include <zvec/ailego/hash/crc32c.h>
#include "utility/graph_segment_writer.h"

namespace zvec {
namespace core {

const std::string VamanaEntity::kGraphHeaderSegmentId = "vamana.graph.header";
const std::string VamanaEntity::kGraphFeaturesSegmentId =
    "vamana.graph.features";
const std::string VamanaEntity::kGraphKeysSegmentId = "vamana.graph.keys";
const std::string VamanaEntity::kGraphNeighborsSegmentId =
    "vamana.graph.neighbors";
const std::string VamanaEntity::kGraphOffsetsSegmentId = "vamana.graph.offsets";
const std::string VamanaEntity::kGraphMappingSegmentId = "vamana.graph.mapping";
const std::string VamanaEntity::kGraphNeighborDistsSegmentId =
    "vamana.graph.neighbor_dists";

int VamanaEntity::CalcAndAddPadding(const IndexDumper::Pointer &dumper,
                                    size_t data_size, size_t *padding_size) {
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

int64_t VamanaEntity::dump_segment(const IndexDumper::Pointer &dumper,
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

int64_t VamanaEntity::dump_header(const IndexDumper::Pointer &dumper,
                                  const VamanaHeader &hd) const {
  return dump_segment(dumper, kGraphHeaderSegmentId, &hd.graph, hd.graph.size);
}

int64_t VamanaEntity::dump_vectors(const IndexDumper::Pointer &dumper) const {
  GraphSegmentWriter writer(dumper);
  for (node_id_t i = 0; i < doc_cnt(); ++i) {
    IndexStorage::MemoryBlock block;
    int ret = get_vector(i, block);
    if (ret != 0) return ret;
    if (!block.data()) return IndexError_ReadData;
    ret = writer.write(block.data(), vector_size());
    if (ret != 0) return ret;
  }
  return writer.finish(kGraphFeaturesSegmentId);
}

int64_t VamanaEntity::dump_neighbors(const IndexDumper::Pointer &dumper) const {
  GraphSegmentWriter writer(dumper);
  std::vector<node_id_t> row(static_cast<size_t>(max_degree()) + 1);
  static_assert(sizeof(NeighborsHeader) == sizeof(node_id_t),
                "Keep the on-disk neighbor header layout");
  for (node_id_t i = 0; i < doc_cnt(); ++i) {
    const auto neighbors = get_neighbors(i);
    if (neighbors.size() > max_degree() ||
        (!neighbors.data && neighbors.size()))
      return IndexError_InvalidFormat;
    std::fill(row.begin(), row.end(), 0);
    row[0] = neighbors.size();
    for (size_t j = 0; j < neighbors.size(); ++j) row[j + 1] = neighbors[j];
    int ret = writer.write(row.data(), row.size() * sizeof(node_id_t));
    if (ret != 0) return ret;
  }
  return writer.finish(kGraphNeighborsSegmentId);
}

int64_t VamanaEntity::dump_neighbor_dists(
    const IndexDumper::Pointer &dumper) const {
  if (!dist_storage_loaded()) return 0;
  GraphSegmentWriter writer(dumper);
  const std::vector<dist_t> zeros(max_degree(), 0);
  for (node_id_t i = 0; i < doc_cnt(); ++i) {
    const dist_t *distances = get_neighbor_dists(i);
    int ret = writer.write(distances ? distances : zeros.data(),
                           zeros.size() * sizeof(dist_t));
    if (ret != 0) return ret;
  }
  return writer.finish(kGraphNeighborDistsSegmentId);
}

int64_t VamanaEntity::dump_segments(const IndexDumper::Pointer &dumper) const
    try {
  // Node order is already the persisted order. No identity permutations or
  // corpus-sized key/vector/graph copies are necessary.
  int64_t total = dump_header(dumper, header_);
  if (total < 0) return total;
  int64_t size = dump_vectors(dumper);
  if (size < 0) return size;
  total += size;
  size = dump_neighbors(dumper);
  if (size < 0) return size;
  total += size;
  size = DumpGraphRecords<key_t>(dumper, kGraphMappingSegmentId, doc_cnt(),
                                 [&](size_t i, key_t *key) {
                                   *key = get_key(i);
                                   return 0;
                                 });
  if (size < 0) return size;
  total += size;
  size = dump_neighbor_dists(dumper);
  return size < 0 ? size : total + size;
} catch (const std::bad_alloc &) {
  return IndexError_NoMemory;
} catch (const std::exception &) {
  return IndexError_Runtime;
}

}  // namespace core
}  // namespace zvec
