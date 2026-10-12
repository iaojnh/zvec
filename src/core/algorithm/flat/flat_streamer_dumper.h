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

#include "flat_dump_keys.h"
#include "flat_streamer.h"
#include "flat_utility.h"

namespace zvec {
namespace core {

template <size_t BATCH_SIZE>
class FlatStreamerDumper {
 public:
  typedef std::unique_ptr<FlatStreamerDumper> Pointer;

  FlatStreamerDumper(const FlatStreamer<BATCH_SIZE> *owner) {
    owner_ = owner;
    dump_size_ = 0;
  }

  int dump(const IndexDumper::Pointer &dumper) {
    ailego::ElapsedTime stamp;

    FlatDumpKeys keys;
    if (owner_->meta().major_order() == IndexMeta::MO_COLUMN) {
      int error_code = this->write_column_index(dumper.get(), &keys);
      if (error_code != 0) {
        return error_code;
      }
    } else {
      int error_code = this->write_row_index(dumper.get(), &keys);
      if (error_code != 0) {
        return error_code;
      }
    }

    int error_code = keys.write_keys(dumper.get());
    if (error_code != 0) {
      return error_code;
    }

    error_code = keys.write_mapping(dumper.get());
    if (error_code != 0) {
      return error_code;
    }

    dump_size_ += ailego_align(keys.size() * sizeof(uint64_t), 32) +
                  ailego_align(keys.size() * sizeof(uint32_t), 32);
    error_code = IndexHelper::SerializeToDumper(owner_->meta(), dumper.get());
    if (error_code != 0) {
      return error_code;
    }
    LOG_DEBUG("dumped_count: %zu, costtime: %zu", keys.size(),
              (size_t)stamp.milli_seconds());
    return 0;
  }

  size_t dump_size() {
    return dump_size_;
  }

 private:
  int write_column_index(IndexDumper *dumper, FlatDumpKeys *keys) {
    switch (IndexMeta::AlignSizeof(owner_->meta().data_type())) {
      case 2:
        return this->write_column_index<uint16_t>(dumper, keys);
      case 4:
        return this->write_column_index<uint32_t>(dumper, keys);
      case 8:
        return this->write_column_index<uint64_t>(dumper, keys);
      default:
        ailego_check_with(0, "BAD CASE");
    }
    return IndexError_Runtime;
  }

  template <typename T>
  int write_column_index(IndexDumper *dumper, FlatDumpKeys *keys) {
    auto iter = owner_->entity().creater_iterator();
    if (!iter) {
      LOG_ERROR("Failed to create iterator");
      return IndexError_Runtime;
    }

    // Write features
    size_t element_size = owner_->meta().element_size();
    size_t block_size = element_size * BATCH_SIZE;
    std::string block1, block2;
    block1.reserve(block_size);
    block2.resize(block_size);

    for (; iter->is_valid(); iter->next()) {
      const void *data = iter->data();
      if (iter->status() != 0) return iter->status();
      if (!data) return IndexError_ReadData;
      block1.append(reinterpret_cast<const char *>(data), element_size);
      int key_ret = keys->append(iter->key());
      if (key_ret != 0) return key_ret;

      if (block1.size() == block_size) {
        ailego::MatrixHelper::Transpose<T, BATCH_SIZE>(
            block1.data(), element_size / sizeof(T), (void *)block2.data());

        if (dumper->write(block2.data(), block_size) != block_size) {
          LOG_ERROR("Failed to write data into dumper %s",
                    dumper->name().c_str());
          return IndexError_WriteData;
        }
        block1.clear();
        dump_size_ += block_size;
      }
    }

    if (!block1.empty()) {
      if (dumper->write(block1.data(), block1.size()) != block1.size()) {
        LOG_ERROR("Failed to write data into dumper %s",
                  dumper->name().c_str());
        return IndexError_WriteData;
      }
      dump_size_ += block1.size();
    }

    if (iter->status() != 0) return iter->status();

    // Write the padding if need
    size_t features_size = keys->size() * element_size;
    size_t features_padding_size =
        ailego_align(features_size, 32) - features_size;
    if (features_padding_size) {
      std::string padding(features_padding_size, '\0');

      if (dumper->write(padding.data(), padding.size()) != padding.size()) {
        LOG_ERROR("Failed to write data into dumper %s",
                  dumper->name().c_str());
        return IndexError_WriteData;
      }
      dump_size_ += padding.size();
    }
    return dumper->append(FLAT_SEGMENT_FEATURES_SEG_ID, features_size,
                          features_padding_size, 0);
  }

  int write_row_index(IndexDumper *dumper, FlatDumpKeys *keys) {
    auto iter = owner_->entity().creater_iterator();
    if (!iter) {
      LOG_ERROR("Failed to create iterator");
      return IndexError_Runtime;
    }

    // Write features
    size_t element_size = owner_->meta().element_size();
    for (; iter->is_valid(); iter->next()) {
      const void *data = iter->data();
      if (iter->status() != 0) return iter->status();
      if (!data) return IndexError_ReadData;
      if (dumper->write(data, element_size) != element_size) {
        LOG_ERROR("Failed to write data into dumper %s",
                  dumper->name().c_str());
        return IndexError_WriteData;
      }
      dump_size_ += element_size;
      int key_ret = keys->append(iter->key());
      if (key_ret != 0) return key_ret;
    }

    if (iter->status() != 0) return iter->status();

    // Write the padding if need
    size_t features_size = keys->size() * element_size;
    size_t features_padding_size =
        ailego_align(features_size, 32) - features_size;
    if (features_padding_size) {
      std::string padding(features_padding_size, '\0');

      if (dumper->write(padding.data(), padding.size()) != padding.size()) {
        LOG_ERROR("Failed to write data into dumper %s",
                  dumper->name().c_str());
        return IndexError_WriteData;
      }
      dump_size_ += padding.size();
    }
    return dumper->append(FLAT_SEGMENT_FEATURES_SEG_ID, features_size,
                          features_padding_size, 0);
  }


 private:
  const FlatStreamer<BATCH_SIZE> *owner_{nullptr};
  size_t dump_size_{0};
};

}  // namespace core
}  // namespace zvec
