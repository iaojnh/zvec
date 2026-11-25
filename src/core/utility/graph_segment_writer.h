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
#include <array>
#include <cstring>
#include <limits>
#include <numeric>
#include <type_traits>
#include "paged_record_sort.h"

namespace zvec::core {

// Graph segments use 32-byte padding and a CRC of data only. Buffering a fixed
// tile avoids both corpus-sized materialization and tiny writes per record.
class GraphSegmentWriter {
 public:
  explicit GraphSegmentWriter(const IndexDumper::Pointer &dumper)
      : dumper_(dumper) {}

  int write(const void *data, size_t bytes) {
    if (!dumper_ || (!data && bytes)) return IndexError_InvalidArgument;
    if (bytes >
        static_cast<size_t>(std::numeric_limits<int64_t>::max()) - 31 - size_)
      return IndexError_Overflow;
    auto *source = static_cast<const uint8_t *>(data);
    while (bytes) {
      const size_t n = std::min(bytes, buffer_.size() - pending_);
      std::memcpy(buffer_.data() + pending_, source, n);
      source += n;
      bytes -= n;
      size_ += n;
      pending_ += n;
      if (pending_ == buffer_.size()) {
        int ret = flush();
        if (ret != 0) return ret;
      }
    }
    return 0;
  }

  int64_t finish(const std::string &id) {
    if (!dumper_) return IndexError_InvalidArgument;
    int ret = flush();
    if (ret != 0) return ret;
    const size_t padding = (32 - size_ % 32) % 32;
    const uint8_t zeros[32]{};
    if (padding && dumper_->write(zeros, padding) != padding)
      return IndexError_WriteData;
    ret = dumper_->append(id, size_, padding, crc_);
    return ret != 0 ? ret : static_cast<int64_t>(size_ + padding);
  }

 private:
  int flush() {
    if (pending_) {
      if (dumper_->write(buffer_.data(), pending_) != pending_)
        return IndexError_WriteData;
      crc_ = ailego::Crc32c::Hash(buffer_.data(), pending_, crc_);
      pending_ = 0;
    }
    return 0;
  }
  IndexDumper::Pointer dumper_;
  std::array<uint8_t, 64 * 1024> buffer_;
  size_t pending_{0};
  size_t size_{0};
  uint32_t crc_{0};
};

template <typename Record, typename Generate>
int64_t DumpGraphRecords(const IndexDumper::Pointer &dumper,
                         const std::string &id, size_t count,
                         Generate &&generate) {
  static_assert(std::is_trivially_copyable<Record>::value, "POD records only");
  GraphSegmentWriter writer(dumper);
  for (size_t i = 0; i < count; ++i) {
    Record record{};
    int ret = generate(i, &record);
    if (ret == 0) ret = writer.write(&record, sizeof(record));
    if (ret != 0) return ret;
  }
  return writer.finish(id);
}

template <typename ReadKey>
int64_t DumpGraphKeyMapping(const IndexDumper::Pointer &dumper,
                            const std::string &id, size_t count,
                            ReadKey &&read_key) try {
  if (count > std::numeric_limits<uint32_t>::max()) return IndexError_Overflow;
  // Retain a cheap in-memory sort without a configured pool, but avoid a
  // second full key array. Equal keys use ordinal order on both backends.
  if (!ailego::MemoryLimitPool::get_instance().capacity()) {
    std::vector<uint32_t> order(count);
    std::iota(order.begin(), order.end(), 0U);
    std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
      const auto ka = read_key(a), kb = read_key(b);
      return ka != kb ? ka < kb : a < b;
    });
    return DumpGraphRecords<uint32_t>(dumper, id, count,
                                      [&](size_t i, uint32_t *out) {
                                        *out = order[i];
                                        return 0;
                                      });
  }
  PagedBuildBuffer mapping;
  int ret = BuildPagedKeyMapping(
      count,
      [&](size_t i, uint64_t *key) {
        *key = read_key(i);
        return 0;
      },
      &mapping);
  if (ret != 0) return ret;
  GraphSegmentWriter writer(dumper);
  std::array<uint32_t, 4096> batch;
  for (size_t first = 0; first < count; first += batch.size()) {
    const size_t bytes =
        std::min(batch.size(), count - first) * sizeof(uint32_t);
    ret = mapping.read(first * sizeof(uint32_t), batch.data(), bytes);
    if (ret == 0) ret = writer.write(batch.data(), bytes);
    if (ret != 0) return ret;
  }
  return writer.finish(id);
} catch (const std::bad_alloc &) {
  return IndexError_NoMemory;
} catch (const std::exception &) {
  return IndexError_Runtime;
}

}  // namespace zvec::core
