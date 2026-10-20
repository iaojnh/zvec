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
#pragma once

#include <algorithm>
#include <cmath>
#include <zvec/ailego/hash/crc32c.h>
#include <zvec/core/framework/disk_format.h>
#include <zvec/core/framework/index_error.h>
#include <zvec/core/framework/index_storage.h>

namespace zvec {
namespace core {

// Logical address geometry is established once, when the index is created.
// Reopening never derives it from segment capacity, OS pages or new options.
struct StreamerLayout {
  enum Algorithm : uint32_t {
    Hnsw = 1,
    HnswSparse = 2,
    Vamana = 3,
    HnswRabitq = 4
  };
  DiskUInt32 version;
  DiskUInt32 crc;
  DiskUInt32 algorithm;
  DiskUInt32 node_size;
  DiskUInt32 upper_neighbor_size;
  DiskUInt32 scaling_factor;
  DiskUInt32 node_count;
  DiskUInt32 node_chunk_size;
  DiskUInt32 upper_chunk_size;
  DiskUInt32 sparse_chunk_size;
  DiskUInt32 node_mask_bits;
  DiskUInt32 upper_mask_bits;
  DiskUInt32 hash_slot_items;
  DiskUInt32 hash_mask_bits;
  uint8_t reserved[72];

  static constexpr const char *kSegment = "zvec.streamer.layout.v3";

  uint32_t checksum() const {
    auto copy = *this;
    copy.crc = 0;
    return ailego::Crc32c::Hash(&copy, sizeof(copy), 0);
  }

  static uint32_t MaskBits(uint32_t count) {
    uint32_t bits = 0;
    for (uint64_t capacity = 1; capacity < count; capacity <<= 1) {
      ++bits;
    }
    return bits;
  }

  bool valid(Algorithm kind, uint32_t node_bytes, uint32_t upper_bytes,
             uint32_t scaling) const {
    if (version != 1 || crc != checksum() || algorithm != kind ||
        node_size != node_bytes || upper_neighbor_size != upper_bytes ||
        scaling_factor != scaling || node_bytes == 0 || node_count == 0 ||
        node_mask_bits >= 32 || (uint64_t{1} << node_mask_bits) != node_count ||
        node_chunk_size == 0 || node_chunk_size % kDiskAlignment != 0) {
      return false;
    }
    uint64_t aligned = 0;
    if (!AlignDiskSize(uint64_t{node_count} * node_bytes, &aligned) ||
        aligned != node_chunk_size) {
      return false;
    }
    if (kind != Vamana &&
        (upper_bytes == 0 || scaling == 0 || upper_chunk_size < upper_bytes ||
         upper_chunk_size % kDiskAlignment != 0 || upper_mask_bits >= 28 ||
         upper_mask_bits != MaskBits(upper_chunk_size / upper_bytes) ||
         hash_slot_items == 0 ||
         (hash_slot_items & (hash_slot_items - 1)) != 0 ||
         uint64_t{hash_slot_items} * 8 < upper_chunk_size ||
         hash_mask_bits >= 32)) {
      return false;
    }
    if (kind != Vamana) {
      const uint64_t slots = uint64_t{1}
                             << MaskBits((uint64_t{upper_chunk_size} + 7) / 8);
      const uint64_t range = slots * scaling / 2;
      if (slots != hash_slot_items || range == 0 || range > UINT32_MAX ||
          (uint64_t{1} << hash_mask_bits) > range ||
          (uint64_t{1} << (hash_mask_bits + 1)) <= range) {
        return false;
      }
    }
    return kind != HnswSparse ||
           (sparse_chunk_size != 0 && sparse_chunk_size % kDiskAlignment == 0);
  }

  static int Bind(const IndexStorage::Pointer &storage, Algorithm kind,
                  uint32_t requested_chunk_size, uint32_t node_bytes,
                  uint32_t upper_bytes, uint32_t scaling, uint32_t max_levels,
                  StreamerLayout *layout) {
    auto segment = storage->get(kSegment);
    if (segment) {
      IndexStorage::MemoryBlock block;
      if (segment->data_size() != sizeof(*layout) ||
          segment->read(0, block, sizeof(*layout)) != sizeof(*layout)) {
        return IndexError_InvalidFormat;
      }
      std::memcpy(layout, block.data(), sizeof(*layout));
      return layout->valid(kind, node_bytes, upper_bytes, scaling)
                 ? 0
                 : IndexError_Mismatch;
    }

    // A missing layout in an existing graph is corruption, not an invitation
    // to derive new addressing parameters from the current configuration.
    if (storage->has("HnswT2S0") || storage->has("HnswT1S0")) {
      return IndexError_InvalidFormat;
    }

    if (!node_bytes || !requested_chunk_size ||
        (kind != Vamana && (!upper_bytes || !scaling))) {
      return IndexError_InvalidArgument;
    }
    *layout = {};
    layout->version = 1;
    layout->algorithm = kind;
    layout->node_size = node_bytes;
    layout->upper_neighbor_size = upper_bytes;
    layout->scaling_factor = scaling;
    const uint32_t bits =
        MaskBits(std::max<uint32_t>(1, requested_chunk_size / node_bytes));
    if (bits >= 32) {
      return IndexError_InvalidArgument;
    }
    uint64_t aligned = 0;
    if (!AlignDiskSize((uint64_t{1} << bits) * node_bytes, &aligned) ||
        aligned > UINT32_MAX) {
      return IndexError_InvalidArgument;
    }
    layout->node_count = uint32_t{1} << bits;
    layout->node_mask_bits = bits;
    layout->node_chunk_size = aligned;
    if (kind != Vamana) {
      const double ratio =
          std::sqrt(double(node_bytes) * scaling / upper_bytes);
      const double upper =
          std::max(double(upper_bytes) * max_levels, aligned / ratio);
      if (!std::isfinite(upper) || upper > UINT32_MAX ||
          !AlignDiskSize(static_cast<uint64_t>(upper), &aligned) ||
          aligned > UINT32_MAX) {
        return IndexError_InvalidArgument;
      }
      layout->upper_chunk_size = aligned;
      layout->upper_mask_bits = MaskBits(aligned / upper_bytes);
      // The upper-level address table has two uint32 fields per slot and
      // an expansion ratio of two. Persist its geometry as well.
      const uint32_t slot_bits = MaskBits((aligned + 7) / 8);
      layout->hash_slot_items = uint32_t{1} << slot_bits;
      uint64_t range = uint64_t{layout->hash_slot_items} * scaling / 2;
      if (range == 0 || range > UINT32_MAX) {
        return IndexError_InvalidArgument;
      }
      uint32_t range_bits = 0;
      while ((uint64_t{1} << (range_bits + 1)) <= range) ++range_bits;
      layout->hash_mask_bits = range_bits;
    }
    if (kind == HnswSparse) {
      if (!AlignDiskSize(requested_chunk_size, &aligned) ||
          aligned > UINT32_MAX) {
        return IndexError_InvalidArgument;
      }
      layout->sparse_chunk_size = aligned;
    }
    layout->crc = layout->checksum();
    if (!layout->valid(kind, node_bytes, upper_bytes, scaling)) {
      return IndexError_InvalidArgument;
    }
    int ret = storage->append(kSegment, kDiskAlignment);
    if (ret != 0) {
      return ret;
    }
    segment = storage->get(kSegment);
    if (!segment ||
        segment->write(0, layout, sizeof(*layout)) != sizeof(*layout)) {
      return IndexError_WriteData;
    }
    return 0;
  }
};

static_assert(sizeof(StreamerLayout) == 128 &&
                  offsetof(StreamerLayout, reserved) == 56,
              "Streamer layout must have a fixed, padding-free encoding");

}  // namespace core
}  // namespace zvec
