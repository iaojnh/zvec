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
#pragma once

#include <array>
#include <limits>
#include <zvec/core/framework/index_dumper.h>
#include "utility/appendable_build_buffer.h"
#include "flat_sparse_index_format.h"
#include "flat_sparse_utility.h"

namespace zvec::core {

// Preserve the legacy packed (uint64 offset, uint32 length) wire format.
// A one-pass holder cannot regenerate offsets after emitting vector data.
class FlatSparseDumpOffsets {
 public:
  int append(uint64_t offset, uint32_t length) {
    if (data_.size() > std::numeric_limits<size_t>::max() - kRecordSize)
      return IndexError_IndexFull;
    std::array<char, kRecordSize> record{};
    std::memcpy(record.data(), &offset, sizeof(offset));
    std::memcpy(record.data() + sizeof(offset), &length, sizeof(length));
    const size_t pos = data_.size();
    int ret = data_.resize(pos + record.size());
    if (ret != 0) return ret;
    return data_.write(pos, record.data(), record.size());
  }

  int dump(IndexDumper *dumper) const {
    std::array<char, kRecordSize * 1024> batch{};
    for (size_t pos = 0; pos < data_.size();) {
      const size_t bytes = std::min(batch.size(), data_.size() - pos);
      int ret = data_.read(pos, batch.data(), bytes);
      if (ret != 0) return ret;
      if (dumper->write(batch.data(), bytes) != bytes)
        return IndexError_WriteData;
      pos += bytes;
    }
    return dumper->append(PARAM_FLAT_SPARSE_DUMP_OFFSET_SEG_ID, data_.size(), 0,
                          0);
  }

 private:
  static constexpr size_t kRecordSize = sizeof(uint64_t) + sizeof(uint32_t);
  AppendableBuildBuffer data_;
};
}  // namespace zvec::core
