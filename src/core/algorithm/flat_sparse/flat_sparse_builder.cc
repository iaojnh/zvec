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

#include "flat_sparse_builder.h"
#include <cstddef>
#include <cstdint>
#include <utility>
#include <utility/sparse_utility.h>
#include <zvec/ailego/logger/logger.h>
#include <zvec/ailego/utility/time_helper.h>
#include <zvec/core/framework/index_error.h>
#include "flat_sparse_dump_offsets.h"
#include "flat_sparse_index_format.h"
#include "flat_sparse_utility.h"

namespace zvec {
namespace core {

FlatSparseBuilder::FlatSparseBuilder() = default;

int FlatSparseBuilder::init(const IndexMeta &meta,
                            const ailego::Params & /*params*/) {
  LOG_INFO("Begin FlatSparseBuilder::init");

  meta_ = meta;

  state_ = BUILD_STATE_INITED;
  LOG_INFO("End FlatSparseBuilder::init");
  return 0;
}

int FlatSparseBuilder::cleanup() {
  LOG_INFO("Begin FlatSparseBuilder::cleanup");

  stats_.clear_attributes();
  stats_.set_trained_count(0UL);
  stats_.set_built_count(0UL);
  stats_.set_dumped_count(0UL);
  stats_.set_discarded_count(0UL);
  stats_.set_trained_costtime(0UL);
  stats_.set_built_costtime(0UL);
  stats_.set_dumped_costtime(0UL);
  state_ = BUILD_STATE_INIT;

  LOG_INFO("End FlatSparseBuilder::cleanup");

  return 0;
}

int FlatSparseBuilder::train(IndexThreads::Pointer,
                             IndexSparseHolder::Pointer /*holder*/) {
  if (state_ != BUILD_STATE_INITED) {
    LOG_ERROR("Init the builder before FlatSparseBuilder::train");
    return IndexError_NoReady;
  }

  LOG_INFO("Begin FlatSparseBuilder::train");

  stats_.set_trained_count(0UL);
  stats_.set_trained_costtime(0UL);
  state_ = BUILD_STATE_TRAINED;

  LOG_INFO("End FlatSparseBuilder::train");

  return 0;
}

int FlatSparseBuilder::train(const IndexTrainer::Pointer & /*trainer*/) {
  if (state_ != BUILD_STATE_INITED) {
    LOG_ERROR("Init the builder before FlatSparseBuilder::train");
    return IndexError_NoReady;
  }

  LOG_INFO("Begin FlatSparseBuilder::train by trainer");

  stats_.set_trained_count(0UL);
  stats_.set_trained_costtime(0UL);
  state_ = BUILD_STATE_TRAINED;

  LOG_INFO("End FlatSparseBuilder::train by trainer");

  return 0;
}

int FlatSparseBuilder::build(IndexThreads::Pointer,
                             IndexSparseHolder::Pointer holder) {
  LOG_INFO("Begin FlatSparseBuilder::build");

  ailego::ElapsedTime stamp;
  if (!holder) {
    LOG_ERROR("Input holder is nullptr while building index");
    return IndexError_InvalidArgument;
  }

  if (!holder->is_matched(meta_)) {
    LOG_ERROR("Input holder doesn't match index meta while building index");
    return IndexError_Mismatch;
  }

  holder_ = std::move(holder);

  stats_.set_built_count(holder_->count());
  stats_.set_built_costtime(stamp.milli_seconds());
  state_ = BUILD_STATE_BUILT;

  LOG_INFO("End FlatSparseBuilder::build");
  return 0;
}

int FlatSparseBuilder::dump(const IndexDumper::Pointer &dumper) try {
  if (state_ != BUILD_STATE_BUILT || !holder_) {
    LOG_INFO("Build the index before FlatSparseBuilder::dump");
    return IndexError_NoReady;
  }

  LOG_INFO("Begin FlatSparseBuilder::dump");

  auto start_time = ailego::Monotime::MilliSeconds();

  int ret = IndexHelper::SerializeToDumper(meta_, dumper.get());
  if (ret != 0) {
    LOG_ERROR("Failed to serialize meta into dumper.");
    return ret;
  }

  uint32_t dump_count;
  ret = do_dump(dumper, &dump_count);
  if (ret != 0) {
    LOG_ERROR("Failed to dump index");
    return ret;
  }

  holder_ = nullptr;
  stats_.set_dumped_count(dump_count);
  stats_.set_dumped_costtime(ailego::Monotime::MilliSeconds() - start_time);

  LOG_INFO("End FlatSparseBuilder::dump");
  return 0;
} catch (const std::bad_alloc &) {
  return IndexError_NoMemory;
}

int FlatSparseBuilder::do_dump(const IndexDumper::Pointer &dumper,
                               uint32_t *dump_count) {
  // bf meta
  int ret = dump_meta(dumper.get());
  if (ret != 0) {
    LOG_ERROR("Failed to dump meta");
    return ret;
  }

  FlatDumpKeys keys;
  ret = dump_vector_and_offset(dumper.get(), &keys);
  if (ret != 0) {
    LOG_ERROR("Failed to dump offset data");
    return ret;
  }

  ret = keys.write_keys(dumper.get(), PARAM_FLAT_SPARSE_DUMP_KEYS_SEG_ID);
  if (ret != 0) {
    LOG_ERROR("Failed to dump keys");
    return ret;
  }

  ret = keys.write_mapping(dumper.get(), PARAM_FLAT_SPARSE_DUMP_MAPPING_SEG_ID);
  if (ret != 0) {
    LOG_ERROR("Failed to dump mapping");
    return ret;
  }

  *dump_count = keys.size();

  return 0;
}

int FlatSparseBuilder::dump_meta(IndexDumper *dumper) {
  FlatSparseMeta meta;
  meta.create_time = ailego::Realtime::Seconds();
  meta.update_time = ailego::Realtime::Seconds();
  meta.doc_cnt = holder_->count();

  if (dumper->write(&meta, sizeof(meta)) != sizeof(meta)) {
    LOG_ERROR("Failed to write meta");
    return IndexError_WriteData;
  }

  size_t meta_padding_size = ailego_align(sizeof(meta), 32) - sizeof(meta);
  if (meta_padding_size) {
    std::string padding(meta_padding_size, '\0');
    if (dumper->write(padding.data(), meta_padding_size) != meta_padding_size) {
      LOG_ERROR("Failed to write meta padding");
      return IndexError_WriteData;
    }
  }
  return dumper->append(PARAM_FLAT_SPARSE_META_SEG_ID, sizeof(meta),
                        meta_padding_size, 0);
}

int FlatSparseBuilder::dump_vector_and_offset(IndexDumper *dumper,
                                              FlatDumpKeys *keys) {
  // iterate the holder
  auto iter = holder_->create_iterator();
  if (!iter) {
    LOG_ERROR("Failed to create iterator");
    return IndexError_Runtime;
  }

  uint64_t written_length{0U};

  FlatSparseDumpOffsets offsets;
  while (iter->is_valid()) {
    const uint64_t key = iter->key();
    if (iter->status() != 0) return iter->status();
    int ret = keys->append(key);
    if (ret != 0) return ret;

    uint32_t length;
    if (write_vector_data(iter->sparse_count(), iter->sparse_indices(),
                          iter->sparse_data(), dumper, &length) != 0) {
      return IndexError_WriteData;
    }

    ret = offsets.append(written_length, length);
    if (ret != 0) return ret;
    written_length += length;
    iter->next();
  }

  if (iter->status() != 0) return iter->status();

  if (dumper->append(PARAM_FLAT_SPARSE_DUMP_DATA_SEG_ID, written_length, 0,
                     0) != 0) {
    LOG_ERROR("Failed to append offset data");
    return IndexError_WriteData;
  }

  LOG_DEBUG("Data total written: %zu", (size_t)written_length);

  return offsets.dump(dumper);
}

int FlatSparseBuilder::write_vector_data(const uint32_t sparse_count,
                                         const uint32_t *sparse_indices,
                                         const void *sparse_vec,
                                         IndexDumper *dumper,
                                         uint32_t *length) {
  ailego::BudgetedString sparse_buffer;

  SparseUtility::TransSparseFormat(sparse_count, sparse_indices, sparse_vec,
                                   meta_.unit_size(), sparse_buffer);

  if (dumper->write(sparse_buffer.data(), sparse_buffer.size()) !=
      sparse_buffer.size()) {
    LOG_ERROR("Failed to write sparse data");
    return IndexError_WriteData;
  }

  *length = sparse_buffer.size();

  return 0;
}

INDEX_FACTORY_REGISTER_BUILDER(FlatSparseBuilder);

}  // namespace core
}  // namespace zvec
