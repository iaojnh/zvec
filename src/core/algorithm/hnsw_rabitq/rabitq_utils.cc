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

#include "rabitq_utils.h"
#include <string>
#include <zvec/ailego/hash/crc32c.h>
#include "zvec/ailego/logger/logger.h"
#include "zvec/core/framework/index_error.h"
#include "paged_rabitq_rotator.h"

namespace zvec {
namespace core {

int dump_rabitq_centroids(
    const IndexDumper::Pointer &dumper, size_t dimension, size_t padded_dim,
    size_t ex_bits, size_t num_clusters, rabitqlib::RotatorType rotator_type,
    const std::function<int(size_t, void *, size_t)> &rotated_centroids,
    const std::function<int(size_t, void *, size_t)> &centroids,
    const std::unique_ptr<rabitqlib::Rotator<float>> &rotator,
    size_t *out_dumped_size) {
  auto align_size = [](size_t size) -> size_t {
    return (size + 0x1F) & (~0x1F);
  };

  uint32_t crc = 0;
  size_t dumped_size = 0;

  // Write header
  RabitqConverterHeader header;
  header.dim = static_cast<uint32_t>(dimension);
  header.padded_dim = static_cast<uint32_t>(padded_dim);
  header.num_clusters = static_cast<uint32_t>(num_clusters);
  header.ex_bits = static_cast<uint8_t>(ex_bits);
  header.rotator_type = static_cast<uint8_t>(rotator_type);
  header.rotator_size = static_cast<uint32_t>(rotator->dump_bytes());
  size_t size = dumper->write(&header, sizeof(header));
  if (size != sizeof(header)) {
    LOG_ERROR("Failed to write header: written=%zu, expected=%zu", size,
              sizeof(header));
    return IndexError_WriteData;
  }
  crc = ailego::Crc32c::Hash(&header, sizeof(header), crc);
  dumped_size += size;

  std::array<char, 65536> buffer;
  for (size_t which = 0; which < 2; ++which) {
    const size_t table_size =
        num_clusters * (which == 0 ? padded_dim : dimension) * sizeof(float);
    const auto &read = which == 0 ? rotated_centroids : centroids;
    for (size_t offset = 0; offset < table_size;) {
      size_t bytes = std::min(buffer.size(), table_size - offset);
      int ret = read(offset, buffer.data(), bytes);
      if (ret != 0) return ret;
      if (dumper->write(buffer.data(), bytes) != bytes)
        return IndexError_WriteData;
      crc = ailego::Crc32c::Hash(buffer.data(), bytes, crc);
      offset += bytes;
      dumped_size += bytes;
    }
  }
  for (size_t offset = 0; offset < rotator->dump_bytes();) {
    size_t bytes = std::min(buffer.size(), rotator->dump_bytes() - offset);
    int ret = ReadRabitqRotator(rotator.get(), offset, buffer.data(), bytes);
    if (ret != 0) return ret;
    if (dumper->write(buffer.data(), bytes) != bytes)
      return IndexError_WriteData;
    crc = ailego::Crc32c::Hash(buffer.data(), bytes, crc);
    offset += bytes;
    dumped_size += bytes;
  }

  // Write padding
  size_t padding_size = align_size(dumped_size) - dumped_size;
  if (padding_size > 0) {
    std::string padding(padding_size, '\0');
    if (dumper->write(padding.data(), padding_size) != padding_size) {
      LOG_ERROR("Append padding failed, size %lu", padding_size);
      return IndexError_WriteData;
    }
  }

  int ret =
      dumper->append(RABITQ_CONVERTER_SEG_ID, dumped_size, padding_size, crc);
  if (ret != 0) {
    LOG_ERROR("Dump segment %s meta failed, ret=%d",
              RABITQ_CONVERTER_SEG_ID.c_str(), ret);
    return ret;
  }

  if (out_dumped_size) {
    *out_dumped_size = dumped_size;
  }
  return 0;
}

}  // namespace core
}  // namespace zvec
