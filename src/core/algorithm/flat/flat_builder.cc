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

#include "flat_builder.h"
#include "flat_utility.h"

namespace zvec {
namespace core {

template <size_t BATCH_SIZE>
int FlatBuilder<BATCH_SIZE>::init(const IndexMeta &meta,
                                  const ailego::Params &params) {
  return this->init_impl(meta, params, true);
}

template <size_t BATCH_SIZE>
int FlatBuilder<BATCH_SIZE>::init_impl(const IndexMeta &meta,
                                       const ailego::Params &params,
                                       bool verify_metric) {
  meta_ = meta;

  // Set the major order
  bool column_major_order = false;
  if (params.get(PARAM_FLAT_COLUMN_MAJOR_ORDER, &column_major_order)) {
    meta_.set_major_order(column_major_order ? IndexMeta::MO_COLUMN
                                             : IndexMeta::MO_ROW);
  }

  // Verify column major order
  if (meta_.major_order() != IndexMeta::MO_ROW) {
    IndexMeta::DataType dt = meta_.data_type();

    bool support_column_major = false;
    if ((dt != IndexMeta::DataType::DT_FP32 &&
         dt != IndexMeta::DataType::DT_FP16 &&
         dt != IndexMeta::DataType::DT_INT8 && dt != IndexMeta::DT_INT4 &&
         dt != IndexMeta::DT_BINARY32 && dt != IndexMeta::DT_BINARY64) ||
        (meta_.unit_size() != IndexMeta::UnitSizeof(dt))) {
      if (meta_.major_order() == IndexMeta::MO_COLUMN) {
        LOG_ERROR("Unsupported type %d with unit size %u.", dt,
                  meta_.unit_size());
        return IndexError_Unsupported;
      } else {
        support_column_major = false;
      }
    }
    if (meta_.element_size() % IndexMeta::AlignSizeof(dt) != 0) {
      if (meta_.major_order() == IndexMeta::MO_COLUMN) {
        LOG_ERROR("Unsupported type %d with dimension %u.", dt,
                  meta_.dimension());
        return IndexError_Unsupported;
      } else {
        support_column_major = false;
      }
    }

    if (meta_.major_order() == IndexMeta::MO_UNDEFINED &&
        support_column_major) {
      meta_.set_major_order(IndexMeta::MO_COLUMN);
    }
  }

  if (verify_metric && !VerifyMetric(meta_)) {
    LOG_ERROR("Invalid index measure %s.", meta_.metric_name().c_str());
    return IndexError_InvalidArgument;
  }

  std::string tag = std::to_string(BATCH_SIZE);
  ailego::Params searcher_params;
  searcher_params.set(PARAM_FLAT_BATCH_SIZE, BATCH_SIZE);
  meta_.set_searcher("FlatSearcher" + tag, 0, searcher_params);
  meta_.set_builder("FlatBuilder" + tag, 0, params);
  return 0;
}

template <size_t BATCH_SIZE>
int FlatBuilder<BATCH_SIZE>::init(
    const IndexMeta &meta, const ailego::Params &params,
    const std::shared_ptr<zvec::turbo::Quantizer> &quantizer) {
  if (!quantizer) {
    return this->init(meta, params);
  }

  // The flat builder computes no distances while building; the quantizer
  // only requires the row major layout.
  bool column_major_order = false;
  params.get(PARAM_FLAT_COLUMN_MAJOR_ORDER, &column_major_order);
  if (column_major_order || meta.major_order() == IndexMeta::MO_COLUMN) {
    LOG_ERROR("Quantizer distance does not support column index.");
    return IndexError_Unsupported;
  }

  IndexMeta row_meta = meta;
  row_meta.set_major_order(IndexMeta::MO_ROW);
  return this->init_impl(row_meta, params, false);
}

template <size_t BATCH_SIZE>
int FlatBuilder<BATCH_SIZE>::build(IndexThreads::Pointer,
                                   IndexHolder::Pointer holder) {
  ailego::ElapsedTime stamp;
  if (!holder->is_matched(meta_)) {
    LOG_ERROR("The holder is unmatched with initialized meta.");
    return IndexError_Mismatch;
  }

  holder_ = std::move(holder);
  stats_.set_built_count(holder_->count());
  stats_.set_built_costtime(stamp.milli_seconds());
  return 0;
}

template <size_t BATCH_SIZE>
int FlatBuilder<BATCH_SIZE>::dump(const IndexDumper::Pointer &dumper) try {
  ailego::ElapsedTime stamp;
  if (!holder_) {
    return IndexError_NoReady;
  }

  FlatDumpKeys keys;
  if (meta_.major_order() == IndexMeta::MO_COLUMN) {
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

  error_code = IndexHelper::SerializeToDumper(meta_, dumper.get());
  if (error_code != 0) {
    return error_code;
  }

  stats_.set_dumped_count(keys.size());
  stats_.set_dumped_costtime(stamp.milli_seconds());
  return 0;
} catch (const std::bad_alloc &) {
  return IndexError_NoMemory;
}

template <size_t BATCH_SIZE>
template <typename T>
int FlatBuilder<BATCH_SIZE>::write_column_index(IndexDumper *dumper,
                                                FlatDumpKeys *keys) {
  auto iter = holder_->create_iterator();
  if (!iter) {
    LOG_ERROR("Failed to create iterator of holder");
    return IndexError_Runtime;
  }

  // Write features
  size_t element_size = holder_->element_size();
  size_t block_size = element_size * BATCH_SIZE;
  ailego::BudgetedString block1, block2;
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
    }
  }

  if (!block1.empty()) {
    if (dumper->write(block1.data(), block1.size()) != block1.size()) {
      LOG_ERROR("Failed to write data into dumper %s", dumper->name().c_str());
      return IndexError_WriteData;
    }
  }

  if (iter->status() != 0) return iter->status();

  // Write the padding if need
  size_t features_size = keys->size() * element_size;
  size_t features_padding_size =
      ailego_align(features_size, 32) - features_size;
  if (features_padding_size) {
    ailego::BudgetedString padding(features_padding_size, '\0');

    if (dumper->write(padding.data(), padding.size()) != padding.size()) {
      LOG_ERROR("Failed to write data into dumper %s", dumper->name().c_str());
      return IndexError_WriteData;
    }
  }
  return dumper->append(FLAT_SEGMENT_FEATURES_SEG_ID, features_size,
                        features_padding_size, 0);
}

template <size_t BATCH_SIZE>
int FlatBuilder<BATCH_SIZE>::write_row_index(IndexDumper *dumper,
                                             FlatDumpKeys *keys) {
  auto iter = holder_->create_iterator();
  if (!iter) {
    LOG_ERROR("Failed to create iterator of holder");
    return IndexError_Runtime;
  }

  // Write features
  size_t element_size = holder_->element_size();
  for (; iter->is_valid(); iter->next()) {
    const void *data = iter->data();
    if (iter->status() != 0) return iter->status();
    if (!data) return IndexError_ReadData;
    if (dumper->write(data, element_size) != element_size) {
      LOG_ERROR("Failed to write data into dumper %s", dumper->name().c_str());
      return IndexError_WriteData;
    }
    int key_ret = keys->append(iter->key());
    if (key_ret != 0) return key_ret;
  }

  if (iter->status() != 0) return iter->status();

  // Write the padding if need
  size_t features_size = keys->size() * element_size;
  size_t features_padding_size =
      ailego_align(features_size, 32) - features_size;
  if (features_padding_size) {
    ailego::BudgetedString padding(features_padding_size, '\0');

    if (dumper->write(padding.data(), padding.size()) != padding.size()) {
      LOG_ERROR("Failed to write data into dumper %s", dumper->name().c_str());
      return IndexError_WriteData;
    }
  }
  return dumper->append(FLAT_SEGMENT_FEATURES_SEG_ID, features_size,
                        features_padding_size, 0);
}

INDEX_FACTORY_REGISTER_BUILDER_ALIAS(LinearBuilder, FlatBuilder<32>);
INDEX_FACTORY_REGISTER_BUILDER_ALIAS(FlatBuilder, FlatBuilder<32>);
INDEX_FACTORY_REGISTER_BUILDER_ALIAS(FlatBuilder16, FlatBuilder<16>);
INDEX_FACTORY_REGISTER_BUILDER_ALIAS(FlatBuilder32, FlatBuilder<32>);

}  // namespace core
}  // namespace zvec
