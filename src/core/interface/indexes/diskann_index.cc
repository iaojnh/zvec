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

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <ailego/pattern/defer.h>
#include <zvec/ailego/io/file.h>
#if defined(_WIN32) || defined(_WIN64)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#include <zvec/core/interface/index.h>
#if DISKANN_SUPPORTED
#include "algorithm/diskann/diskann_params.h"
#include "holder_builder.h"
#endif

namespace zvec::core_interface {

#if DISKANN_SUPPORTED
namespace {

std::string MakeSnapshotTemporaryPath(const std::string &file_path) {
  static std::atomic<uint64_t> sequence{0};
  const auto timestamp =
      std::chrono::steady_clock::now().time_since_epoch().count();
  return file_path + ".merge-" + std::to_string(timestamp) + "-" +
         std::to_string(sequence.fetch_add(1, std::memory_order_relaxed)) +
         ".tmp";
}

bool ReplaceFileAtomically(const std::string &source,
                           const std::string &destination) {
#if defined(_WIN32) || defined(_WIN64)
  const auto source_path = ailego::FileHelper::PathFromUtf8(source);
  const auto destination_path = ailego::FileHelper::PathFromUtf8(destination);
  return ::MoveFileExW(source_path.c_str(), destination_path.c_str(),
                       MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0;
#else
  return ailego::File::Rename(source, destination);
#endif
}

}  // namespace
#endif

#if !DISKANN_SUPPORTED

int DiskAnnIndex::CreateAndInitStreamer(const BaseIndexParam &param) {
  (void)param;
  LOG_ERROR("DiskAnn is not supported on this platform");
  return core::IndexError_Unsupported;
}

int DiskAnnIndex::open(const std::string &file_path,
                       StorageOptions storage_options) {
  (void)file_path;
  (void)storage_options;
  LOG_ERROR("DiskAnn is not supported on this platform");
  return core::IndexError_Unsupported;
}

int DiskAnnIndex::GenerateHolder() {
  LOG_ERROR("DiskAnn is not supported on this platform");
  return core::IndexError_Unsupported;
}

int DiskAnnIndex::CommitBuiltSnapshot() {
  LOG_ERROR("DiskAnn is not supported on this platform");
  return core::IndexError_Unsupported;
}

int DiskAnnIndex::add(const VectorData &vector, uint32_t doc_id) {
  (void)vector;
  (void)doc_id;
  LOG_ERROR("DiskAnn is not supported on this platform");
  return core::IndexError_Unsupported;
}

int DiskAnnIndex::train() {
  LOG_ERROR("DiskAnn is not supported on this platform");
  return core::IndexError_Unsupported;
}

int DiskAnnIndex::_dense_fetch(const uint32_t doc_id,
                               VectorDataBuffer *vector_data_buffer) {
  (void)doc_id;
  (void)vector_data_buffer;
  LOG_ERROR("DiskAnn is not supported on this platform");
  return core::IndexError_Unsupported;
}

int DiskAnnIndex::_prepare_for_search(
    const VectorData &query, const BaseIndexQueryParam::Pointer &search_param,
    core::IndexContext::Pointer &context) {
  (void)query;
  (void)search_param;
  (void)context;
  LOG_ERROR("DiskAnn is not supported on this platform");
  return core::IndexError_Unsupported;
}

int DiskAnnIndex::merge(const std::vector<Index::Pointer> &indexes,
                        const IndexFilter &filter,
                        const MergeOptions &options) {
  (void)indexes;
  (void)filter;
  (void)options;
  LOG_ERROR("DiskAnn is not supported on this platform");
  return core::IndexError_Unsupported;
}

#else

int DiskAnnIndex::CreateAndInitStreamer(const BaseIndexParam &param) {
  if (is_sparse_) {
    LOG_ERROR("Failed to create streamer. Sparse is not Supported.");
    return core::IndexError_Unsupported;
  }

  param_ = dynamic_cast<const DiskAnnIndexParam &>(param);
  if (param_.max_degree <= 0 || param_.list_size <= 0 ||
      param_.pq_chunk_num < 0) {
    LOG_ERROR(
        "Invalid DiskAnn parameters: max_degree=%d list_size=%d "
        "pq_chunk_num=%d",
        param_.max_degree, param_.list_size, param_.pq_chunk_num);
    return core::IndexError_InvalidArgument;
  }
  param_.max_degree = std::min(100, param_.max_degree);
  param_.list_size = std::min(100, param_.list_size);
  param_.pq_chunk_num = std::min(1024, param_.pq_chunk_num);
  proxima_index_params_.set(core::PARAM_DISKANN_BUILDER_MAX_DEGREE,
                            param_.max_degree);
  proxima_index_params_.set(core::PARAM_DISKANN_BUILDER_LIST_SIZE,
                            param_.list_size);
  proxima_index_params_.set(core::PARAM_DISKANN_BUILDER_MAX_PQ_CHUNK_NUM,
                            param_.pq_chunk_num);
  builder_ = core::IndexFactory::CreateBuilder("DiskAnnBuilder");
  streamer_ = core::IndexFactory::CreateStreamer("DiskAnnStreamer");

  if (ailego_unlikely(!builder_ || !streamer_)) {
    LOG_ERROR(
        "Failed to create DiskAnnBuilder/DiskAnnStreamer: DiskAnn factory "
        "entries are not registered. This usually means the DiskAnn shared "
        "module could not be located next to the hosting binary.");
    return core::IndexError_Runtime;
  }

  IndexMeta real_meta;
  if (converter_) {
    real_meta = converter_->meta();
  } else {
    real_meta = proxima_index_meta_;
  }

  const int builder_ret = builder_->init(real_meta, proxima_index_params_);
  const int streamer_ret = streamer_->init(real_meta, proxima_index_params_);
  if (ailego_unlikely(builder_ret != 0 || streamer_ret != 0)) {
    LOG_ERROR(
        "Failed to init builder or streamer, builder_ret: %d, "
        "streamer_ret: %d",
        builder_ret, streamer_ret);
    return core::IndexError_Runtime;
  }

  return 0;
}

int DiskAnnIndex::open(const std::string &file_path,
                       StorageOptions storage_options) {
  ailego::Params storage_params;
  file_path_ = file_path;
  is_read_only_ = storage_options.read_only;
  switch (storage_options.type) {
    case StorageOptions::StorageType::kMMAP:
    case StorageOptions::StorageType::kBufferPool: {
      // NOTE: DiskAnn index is dumped via FileDumper (plain binary file), which
      // is not compatible with BufferStorage's IndexFormat layout. Fall back to
      // FileReadStorage for both MMAP and BufferPool storage types.
      storage_ = core::IndexFactory::CreateStorage("FileReadStorage");
      if (storage_ == nullptr) {
        LOG_ERROR("Failed to create FileReadStorage");
        return core::IndexError_Runtime;
      }
      int ret = storage_->init(storage_params);
      if (ret != 0) {
        LOG_ERROR("Failed to init FileReadStorage, path: %s, err: %s",
                  file_path_.c_str(), core::IndexError::What(ret));
        return ret;
      }
      break;
    }
    default: {
      LOG_ERROR("Unsupported storage type");
      return core::IndexError_Unsupported;
    }
  }

  if (!storage_options.create_new) {
    int ret = storage_->open(file_path_, false);
    if (ret != 0) {
      LOG_ERROR("Failed to open storage, path: %s, err: %s", file_path_.c_str(),
                core::IndexError::What(ret));
      return core::IndexError_Runtime;
    }
    if (streamer_ == nullptr || streamer_->open(storage_) != 0) {
      LOG_ERROR("Failed to open streamer, path: %s", file_path_.c_str());
      return core::IndexError_Runtime;
    }
    is_trained_ = true;
  }
  is_open_ = true;
  return 0;
}

int DiskAnnIndex::GenerateHolder() {
  return BuildMultiPassHolder(param_.data_type, param_.dimension, doc_cache_,
                              converter_, &holder_);
}

int DiskAnnIndex::CommitBuiltSnapshot() {
  if (builder_ == nullptr || file_path_.empty()) {
    LOG_ERROR("Cannot commit an uninitialized DiskAnn snapshot");
    return core::IndexError_NoReady;
  }

  auto dumper = core::IndexFactory::CreateDumper("FileDumper");
  if (dumper == nullptr) {
    LOG_ERROR("Failed to create FileDumper");
    return core::IndexError_Runtime;
  }

  const std::string temporary_path = MakeSnapshotTemporaryPath(file_path_);
  bool temporary_committed = false;
  AILEGO_DEFER([&]() {
    if (!temporary_committed &&
        ailego::FileHelper::IsExist(temporary_path.c_str())) {
      ailego::File::Delete(temporary_path);
    }
  });

  int ret = dumper->create(temporary_path);
  if (ret != 0) {
    LOG_ERROR("Failed to create dumper, path: %s, err: %s",
              temporary_path.c_str(), core::IndexError::What(ret));
    return core::IndexError_Runtime;
  }
  ret = builder_->dump(dumper);
  if (ret != 0) {
    LOG_ERROR("Failed to dump index, path: %s, err: %s", temporary_path.c_str(),
              core::IndexError::What(ret));
    dumper->close();
    return core::IndexError_Runtime;
  }
  ret = dumper->close();
  if (ret != 0) {
    LOG_ERROR("Failed to close dumper, path: %s, err: %s",
              temporary_path.c_str(), core::IndexError::What(ret));
    return core::IndexError_Runtime;
  }

  const core::IndexMeta &build_meta =
      converter_ != nullptr ? converter_->meta() : proxima_index_meta_;
  auto replacement_storage =
      core::IndexFactory::CreateStorage("FileReadStorage");
  auto replacement_streamer =
      core::IndexFactory::CreateStreamer("DiskAnnStreamer");
  if (replacement_storage == nullptr || replacement_streamer == nullptr) {
    LOG_ERROR("Failed to create replacement DiskAnn reader");
    return core::IndexError_Runtime;
  }

  ailego::Params storage_params;
  ret = replacement_storage->init(storage_params);
  if (ret != 0) {
    LOG_ERROR("Failed to initialize replacement storage, err: %s",
              core::IndexError::What(ret));
    return core::IndexError_Runtime;
  }
  ret = replacement_streamer->init(build_meta, proxima_index_params_);
  if (ret != 0) {
    LOG_ERROR("Failed to initialize replacement streamer, err: %s",
              core::IndexError::What(ret));
    return core::IndexError_Runtime;
  }
  ret = replacement_storage->open(temporary_path, false);
  if (ret != 0) {
    LOG_ERROR("Failed to open replacement storage, path: %s, err: %s",
              temporary_path.c_str(), core::IndexError::What(ret));
    return core::IndexError_Runtime;
  }
  ret = replacement_streamer->open(replacement_storage);
  if (ret != 0) {
    LOG_ERROR("Failed to validate replacement streamer, path: %s, err: %s",
              temporary_path.c_str(), core::IndexError::What(ret));
    return core::IndexError_Runtime;
  }

  if (!ReplaceFileAtomically(temporary_path, file_path_)) {
    LOG_ERROR("Failed to atomically replace DiskAnn index, path: %s, err: %s",
              file_path_.c_str(),
              ailego::FileHelper::GetLastErrorString().c_str());
    return core::IndexError_Runtime;
  }
  temporary_committed = true;

  auto previous_streamer = std::move(streamer_);
  auto previous_storage = std::move(storage_);
  streamer_ = std::move(replacement_streamer);
  storage_ = std::move(replacement_storage);
  if (previous_streamer != nullptr && previous_streamer->unload() != 0) {
    LOG_WARN("Failed to unload previous DiskAnn snapshot after replacement");
  }
  if (previous_storage != nullptr && previous_storage->close() != 0) {
    LOG_WARN("Failed to close previous DiskAnn storage after replacement");
  }

  return 0;
}

int DiskAnnIndex::add(const VectorData &vector, uint32_t doc_id) {
  if (is_trained_) {
    LOG_ERROR("this diskann index is trained");
    return core::IndexError_Runtime;
  }
  if (!std::holds_alternative<DenseVector>(vector.vector)) {
    LOG_ERROR("Invalid vector data");
    return core::IndexError_Runtime;
  }
  const DenseVector &dense_vector = std::get<DenseVector>(vector.vector);
  std::string out_vector_buffer = std::string(
      static_cast<const char *>(dense_vector.data),
      input_vector_meta_.dimension() * input_vector_meta_.unit_size());

  std::lock_guard<std::mutex> lock(mutex_);
  if (doc_cache_.size() <= doc_id) {
    std::string fake_data(
        input_vector_meta_.dimension() * input_vector_meta_.unit_size(), 0);
    doc_cache_.resize(doc_id + 1, std::make_pair(kInvalidKey, fake_data));
  }
  doc_cache_[doc_id] = std::make_pair(doc_id, out_vector_buffer);
  return 0;
}

int DiskAnnIndex::train() {
  if (!is_open_) {
    LOG_ERROR("Open DiskAnn index before training");
    return core::IndexError_NoReady;
  }
  if (is_read_only_) {
    LOG_ERROR("Cannot train a read-only DiskAnn index");
    return core::IndexError_Runtime;
  }

  int ret = GenerateHolder();
  if (ret != 0) {
    LOG_ERROR("Failed to generate holder, err: %s",
              core::IndexError::What(ret));
    return ret;
  }
  ret = builder_->train(holder_);
  if (ret != 0) {
    LOG_ERROR("Failed to train builder, err: %s", core::IndexError::What(ret));
    return ret;
  }
  ret = builder_->build(holder_);
  if (ret != 0) {
    LOG_ERROR("Failed to build index, err: %s", core::IndexError::What(ret));
    return ret;
  }
  ret = CommitBuiltSnapshot();
  if (ret != 0) {
    return ret;
  }
  is_trained_ = true;
  return 0;
}

int DiskAnnIndex::_dense_fetch(const uint32_t doc_id,
                               VectorDataBuffer *vector_data_buffer) {
  if (is_trained_) {
    return Index::_dense_fetch(doc_id, vector_data_buffer);
  } else {
    std::lock_guard<std::mutex> lock(mutex_);
    if (doc_id >= doc_cache_.size() ||
        doc_cache_[doc_id].first == kInvalidKey) {
      LOG_ERROR("Vector id does not exist: %u", doc_id);
      return core::IndexError_NoExist;
    }
    DenseVectorBuffer dense_vector_buffer;
    std::string &out_vector_buffer = dense_vector_buffer.data;
    out_vector_buffer = doc_cache_[doc_id].second;
    vector_data_buffer->vector_buffer = std::move(dense_vector_buffer);
    return 0;
  }
}

int DiskAnnIndex::_prepare_for_search(
    const VectorData & /*query*/,
    const BaseIndexQueryParam::Pointer &search_param,
    core::IndexContext::Pointer &context) {
  const auto &diskann_search_param =
      std::dynamic_pointer_cast<DiskAnnQueryParam>(search_param);
  if (diskann_search_param == nullptr) {
    LOG_ERROR("Invalid search param: expected DiskAnnQueryParam");
    return core::IndexError_Runtime;
  }

  if (search_param->group_by_param && search_param->group_by_param->group_by) {
    LOG_ERROR("group_by search is not supported for DiskAnn index");
    return core::IndexError_Unsupported;
  }

  context->set_topk(diskann_search_param->topk);
  context->set_fetch_vector(diskann_search_param->fetch_vector);
  if (diskann_search_param->filter) {
    context->set_filter(*diskann_search_param->filter);
  } else {
    context->reset_filter();
  }
  if (diskann_search_param->radius > 0.0f) {
    context->set_threshold(diskann_search_param->radius);
  } else {
    context->reset_threshold();
  }

  // Propagate the query-time beam-search list size into the context. Must be
  // at least topk to keep enough candidates for a correct result.
  ailego::Params params;
  params.set(
      core::PARAM_DISKANN_SEARCHER_LIST_SIZE,
      std::max(diskann_search_param->topk, diskann_search_param->list_size));
  const int ret = context->update(params);
  if (ret != 0) {
    LOG_ERROR("Failed to update DiskAnn search context: %s",
              core::IndexError::What(ret));
    return ret;
  }

  return 0;
}

int DiskAnnIndex::merge(const std::vector<Index::Pointer> &indexes,
                        const IndexFilter &filter,
                        const MergeOptions &options) {
  if (indexes.empty()) {
    return core::IndexError_Success;
  }
  if (!is_open_) {
    LOG_ERROR("Open DiskAnn index before merging");
    return core::IndexError_NoReady;
  }
  if (is_read_only_) {
    LOG_ERROR("Cannot merge into a read-only DiskAnn index");
    return core::IndexError_Runtime;
  }

  const bool was_trained = is_trained_;
  const core::IndexMeta &build_meta =
      converter_ != nullptr ? converter_->meta() : proxima_index_meta_;
  auto rollback_training_state = [&]() {
    is_trained_ = was_trained;
    if (!was_trained && builder_ != nullptr) {
      if (builder_->cleanup() != 0 ||
          builder_->init(build_meta, proxima_index_params_) != 0) {
        LOG_ERROR("Failed to reset DiskAnn builder after merge failure");
      }
    }
  };

  // A DiskAnn builder is single-use. Reinitialize it before rebuilding an
  // already trained target so repeated merge calls behave like other indexes.
  if (is_trained_) {
    if (builder_ == nullptr || builder_->cleanup() != 0) {
      LOG_ERROR("Failed to reset DiskAnn builder before merge");
      return core::IndexError_Runtime;
    }
    if (builder_->init(build_meta, proxima_index_params_) != 0) {
      LOG_ERROR("Failed to reinitialize DiskAnn builder before merge");
      return core::IndexError_Runtime;
    }
  }

  int pre_ret = Index::merge(indexes, filter, options);
  if (pre_ret != 0) {
    rollback_training_state();
    return pre_ret;
  }
  const int ret = CommitBuiltSnapshot();
  if (ret != 0) {
    rollback_training_state();
    return ret;
  }
  is_trained_ = true;
  return 0;
}

#endif  // DISKANN_SUPPORTED

}  // namespace zvec::core_interface
