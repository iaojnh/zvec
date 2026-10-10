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

#include <memory>
#include <mutex>
#include <string>
#include <zvec/core/interface/index.h>
#include "zvec/ailego/io/file.h"
#include "zvec/core/framework/index_error.h"
#include "buffered_input.h"
#include "holder_builder.h"

#if RABITQ_SUPPORTED
#include "algorithm/hnsw_rabitq/hnsw_rabitq_params.h"
#include "algorithm/hnsw_rabitq/rabitq_params.h"
#include "algorithm/ivf_rabitq/ivf_rabitq_params.h"
#include "algorithm/ivf_rabitq/ivf_rabitq_streamer.h"
#include "utility/utility_params.h"
#endif

namespace zvec::core_interface {

int IVFRabitqIndex::create_and_init_streamer(const BaseIndexParam &param) {
#if !RABITQ_SUPPORTED
  (void)param;
  LOG_ERROR("RaBitQ is not supported on this platform (Linux x86_64 only)");
  return core::IndexError_Unsupported;
#else
  param_ = dynamic_cast<const IVFRabitqIndexParam &>(param);

  if (is_sparse_) {
    LOG_ERROR("Sparse index is not supported for IVF RaBitQ");
    return core::IndexError_Runtime;
  }

  if (param_.nlist <= 0) {
    LOG_ERROR("nlist must be greater than 0, got %d", param_.nlist);
    return core::IndexError_InvalidArgument;
  }
  if (param_.sample_count < 0) {
    LOG_ERROR("sample_count must be greater than or equal to 0, got %d",
              param_.sample_count);
    return core::IndexError_InvalidArgument;
  }

  if (param.dimension < core::kMinRabitqDimSize ||
      param.dimension > core::kMaxRabitqDimSize) {
    LOG_ERROR("Unsupported dimension: %d, must be in [%d, %d]", param.dimension,
              core::kMinRabitqDimSize, core::kMaxRabitqDimSize);
    return core::IndexError_Unsupported;
  }

  proxima_index_params_.set(core::PARAM_IVF_RABITQ_NLIST, param_.nlist);
  proxima_index_params_.set(core::PARAM_RABITQ_TOTAL_BITS, param_.total_bits);
  proxima_index_params_.set(core::PARAM_RABITQ_SAMPLE_COUNT,
                            param_.sample_count);
  // Pass original dimension so builder can ignore extra dim from converter
  proxima_index_params_.set(core::PARAM_RABITQ_GENERAL_DIMENSION,
                            input_vector_meta_.dimension());

  // Create builder (for train/build/dump)
  builder_ = core::IndexFactory::CreateBuilder("IvfRabitqBuilder");
  if (ailego_unlikely(!builder_)) {
    LOG_ERROR("Failed to create IvfRabitqBuilder");
    return core::IndexError_Runtime;
  }
  if (ailego_unlikely(
          builder_->init(proxima_index_meta_, proxima_index_params_) != 0)) {
    LOG_ERROR("Failed to init IvfRabitqBuilder");
    return core::IndexError_Runtime;
  }

  // Create streamer (for search)
  auto streamer = std::make_shared<core::IvfRabitqStreamer>();
  streamer_ = streamer;
  if (ailego_unlikely(!streamer_)) {
    LOG_ERROR("Failed to create IvfRabitqStreamer");
    return core::IndexError_Runtime;
  }
  if (ailego_unlikely(
          streamer_->init(proxima_index_meta_, proxima_index_params_) != 0)) {
    LOG_ERROR("Failed to init IvfRabitqStreamer");
    return core::IndexError_Runtime;
  }
  return 0;
#endif  // RABITQ_SUPPORTED
}

int IVFRabitqIndex::open(const std::string &file_path,
                         StorageOptions storage_options) {
#if !RABITQ_SUPPORTED
  (void)file_path;
  (void)storage_options;
  LOG_ERROR("RaBitQ is not supported on this platform (Linux x86_64 only)");
  return core::IndexError_Unsupported;
#else
  ailego::Params storage_params;
  file_path_ = file_path;
  is_read_only_ = storage_options.read_only;

  const bool buffered =
      storage_options.type == StorageOptions::StorageType::kBufferPool;
  if (!buffered && storage_options.type != StorageOptions::StorageType::kMMAP)
    return core::IndexError_Unsupported;
  storage_params.set(core::BUFFER_READ_STORAGE_WARMUP_MODE,
                     core::BUFFER_READ_STORAGE_WARMUP_NONE);
  storage_ = core::IndexFactory::CreateStorage(
      buffered ? "BufferReadStorage" : "MMapFileReadStorage");
  if (storage_ == nullptr) {
    LOG_ERROR("Failed to create IVF RaBitQ storage");
    return core::IndexError_Runtime;
  }
  int ret = storage_->init(storage_params);
  if (ret != 0) {
    LOG_ERROR("Failed to init IVF RaBitQ storage, path: %s, err: %s",
              file_path_.c_str(), core::IndexError::What(ret));
    return ret;
  }

  if (is_read_only_ || !storage_options.create_new) {
    ret = storage_->open(file_path_, false);
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
  if (storage_options.create_new && !is_read_only_ &&
      storage_options.type == StorageOptions::StorageType::kBufferPool) {
    buffered_input_ = std::make_shared<BufferedInput>(input_vector_meta_,
                                                      file_path_ + ".input");
  }
  return 0;
#endif  // RABITQ_SUPPORTED
}

int IVFRabitqIndex::generate_holder() {
#if !RABITQ_SUPPORTED
  LOG_ERROR("RaBitQ is not supported on this platform (Linux x86_64 only)");
  return core::IndexError_Unsupported;
#else
  core::IndexHolder::Pointer input;
  if (buffered_input_) {
    int ret = buffered_input_->prepare();
    if (ret != 0) return ret;
    input = buffered_input_;
  } else {
    input = std::make_shared<DocCacheHolder>(input_vector_meta_, doc_cache_);
  }
  if (converter_) {
    int ret = core::IndexConverter::TrainAndTransform(converter_, input);
    if (ret != 0) return ret;
    input = converter_->result();
    if (!input) return core::IndexError_Runtime;
  }
  holder_ = std::move(input);
  return 0;
#endif  // RABITQ_SUPPORTED
}

int IVFRabitqIndex::add(const VectorData &vector, uint32_t doc_id) {
#if !RABITQ_SUPPORTED
  (void)vector;
  (void)doc_id;
  LOG_ERROR("RaBitQ is not supported on this platform (Linux x86_64 only)");
  return core::IndexError_Unsupported;
#else
  if (!is_open_ || is_read_only_) return core::IndexError_NoReady;
  if (is_trained_ || build_stage_ != BuildStage::kCollecting) {
    LOG_ERROR("this IVF RaBitQ index is trained");
    return core::IndexError_Runtime;
  }
  if (!std::holds_alternative<DenseVector>(vector.vector)) {
    LOG_ERROR("Invalid vector data");
    return core::IndexError_Runtime;
  }
  const DenseVector &dense_vector = std::get<DenseVector>(vector.vector);
  if (!dense_vector.data) return core::IndexError_InvalidArgument;
  std::lock_guard<std::mutex> lock(mutex_);
  if (buffered_input_) return buffered_input_->add(doc_id, dense_vector.data);
  std::string out_vector_buffer = std::string(
      static_cast<const char *>(dense_vector.data),
      input_vector_meta_.dimension() * input_vector_meta_.unit_size());

  if (doc_cache_->size() <= doc_id) {
    doc_cache_->resize(static_cast<size_t>(doc_id) + 1,
                       std::make_pair(kInvalidKey, std::string{}));
  }
  (*doc_cache_)[doc_id] = std::make_pair(doc_id, std::move(out_vector_buffer));
  return 0;
#endif  // RABITQ_SUPPORTED
}

int IVFRabitqIndex::train() {
#if !RABITQ_SUPPORTED
  LOG_ERROR("RaBitQ is not supported on this platform (Linux x86_64 only)");
  return core::IndexError_Unsupported;
#else
  if (is_trained_) return 0;
  if (!is_open_ || is_read_only_) return core::IndexError_NoReady;
  int ret = 0;
  if (build_stage_ == BuildStage::kCollecting) {
    ret = generate_holder();
    if (ret != 0) return ret;
    ret = builder_->train(holder_);
    if (ret != 0) return ret;
    build_stage_ = BuildStage::kTrained;
  }

  // Build index (assign vectors to centroids, quantize)
  if (build_stage_ == BuildStage::kTrained) {
    ret = builder_->build(holder_);
    if (ret != 0) return ret;
    build_stage_ = BuildStage::kBuilt;
    holder_.reset();
    buffered_input_.reset();
    doc_cache_.reset();
    converter_.reset();
  }

  // Dump to file
  if (build_stage_ == BuildStage::kBuilt) {
    auto dumper = core::IndexFactory::CreateDumper("FileDumper");
    if (!dumper) {
      LOG_ERROR("Failed to create FileDumper");
      return core::IndexError_Runtime;
    }
    ret = dumper->create(file_path_);
    if (ret != 0) {
      LOG_ERROR("Failed to create dumper at path: %s", file_path_.c_str());
      return core::IndexError_Runtime;
    }
    ret = builder_->dump(dumper);
    if (ret != 0) {
      LOG_ERROR("Failed to dump IVF RaBitQ index, ret=%d", ret);
      return core::IndexError_Runtime;
    }
    ret = dumper->close();
    if (ret != 0) return ret;
    build_stage_ = BuildStage::kDumped;
    // dump() succeeded; a failed reopen only needs the persisted output.
    builder_->cleanup();
  }

  // Reopen storage + streamer
  ret = storage_->open(file_path_, false);
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
  return 0;
#endif  // RABITQ_SUPPORTED
}

int IVFRabitqIndex::merge(const std::vector<Index::Pointer> &indexes,
                          const IndexFilter &filter,
                          const MergeOptions &options) {
#if !RABITQ_SUPPORTED
  (void)indexes;
  (void)filter;
  (void)options;
  LOG_ERROR("RaBitQ is not supported on this platform (Linux x86_64 only)");
  return core::IndexError_Unsupported;
#else
  int ret = Index::merge(indexes, filter, options);
  if (ret != 0) {
    return ret;
  }

  auto dumper = core::IndexFactory::CreateDumper("FileDumper");
  if (!dumper) {
    LOG_ERROR("Failed to create FileDumper");
    return core::IndexError_Runtime;
  }
  ret = dumper->create(file_path_);
  if (ret != 0) {
    LOG_ERROR("Failed to create dumper at path: %s", file_path_.c_str());
    return core::IndexError_Runtime;
  }
  ret = builder_->dump(dumper);
  if (ret != 0) {
    LOG_ERROR("Failed to dump IVF RaBitQ index, ret=%d", ret);
    return core::IndexError_Runtime;
  }
  ret = dumper->close();
  if (ret != 0) {
    LOG_ERROR("Failed to close dumper at path: %s", file_path_.c_str());
    return core::IndexError_Runtime;
  }

  // Only the persisted output is needed by the reader after a successful dump.
  builder_->cleanup();
  holder_.reset();
  buffered_input_.reset();
  doc_cache_.reset();
  build_stage_ = BuildStage::kDumped;

  ret = storage_->open(file_path_, false);
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
  return 0;
#endif  // RABITQ_SUPPORTED
}

int IVFRabitqIndex::_dense_fetch(const uint32_t doc_id,
                                 VectorDataBuffer *vector_data_buffer) {
  (void)doc_id;
  (void)vector_data_buffer;
  LOG_ERROR("Fetch is not supported for IVF RaBitQ index");
  return core::IndexError_Unsupported;
}

int IVFRabitqIndex::_prepare_for_search(
    const VectorData & /*vector_data*/,
    const BaseIndexQueryParam::Pointer &search_param,
    core::IndexContext::Pointer &context) {
#if !RABITQ_SUPPORTED
  (void)search_param;
  (void)context;
  LOG_ERROR("RaBitQ is not supported on this platform (Linux x86_64 only)");
  return core::IndexError_Unsupported;
#else
  const auto &ivf_rabitq_param =
      std::dynamic_pointer_cast<IVFRabitqQueryParam>(search_param);

  if (!ivf_rabitq_param) {
    LOG_ERROR("Invalid search param type, expected IVFRabitqQueryParam");
    return core::IndexError_InvalidArgument;
  }
  if (ivf_rabitq_param->nprobe == 0) {
    LOG_ERROR("nprobe must be greater than 0");
    return core::IndexError_InvalidArgument;
  }
  _set_group_by_on_context(search_param, context);
  context->set_topk(search_param->topk);
  context->set_fetch_vector(false);
  if (search_param->filter) {
    context->set_filter(std::move(*search_param->filter));
  }
  if (search_param->radius > 0.0f) {
    context->set_threshold(search_param->radius);
  }
  ailego::Params params;
  params.set(core::PARAM_IVF_RABITQ_NPROBE, ivf_rabitq_param->nprobe);
  return context->update(params);
#endif  // RABITQ_SUPPORTED
}

}  // namespace zvec::core_interface
