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
#include <cstddef>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <ailego/pattern/defer.h>
#include <zvec/ailego/io/file.h>
#if defined(_WIN32) || defined(_WIN64)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#include <cerrno>
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

enum class SnapshotReplaceResult {
  kNotReplaced,
  kReplaced,
  kReplacedNotDurable,
};

#if defined(_WIN32) || defined(_WIN64)
bool ReplaceOpenFileWithPosixSemantics(
    const std::filesystem::path &source_path,
    const std::filesystem::path &destination_path) {
  // FileRenameInfoEx is available at runtime on supported Windows Server
  // versions, but the project still builds against an older SDK view. Keep
  // the ABI declarations local so an open destination can be replaced while
  // existing readers retain the old file object.
  constexpr auto kFileRenameInfoEx = static_cast<FILE_INFO_BY_HANDLE_CLASS>(22);
  constexpr DWORD kReplaceIfExists = 0x00000001;
  constexpr DWORD kPosixSemantics = 0x00000002;
  struct ExtendedFileRenameInfo {
    DWORD flags;
    HANDLE root_directory;
    DWORD file_name_length;
    WCHAR file_name[1];
  };
  static_assert(offsetof(ExtendedFileRenameInfo, root_directory) ==
                offsetof(FILE_RENAME_INFO, RootDirectory));
  static_assert(offsetof(ExtendedFileRenameInfo, file_name_length) ==
                offsetof(FILE_RENAME_INFO, FileNameLength));
  static_assert(offsetof(ExtendedFileRenameInfo, file_name) ==
                offsetof(FILE_RENAME_INFO, FileName));

  std::error_code path_error;
  const std::filesystem::path absolute_destination =
      std::filesystem::absolute(destination_path, path_error);
  if (path_error) {
    ::SetLastError(ERROR_PATH_NOT_FOUND);
    return false;
  }
  const std::wstring &destination = absolute_destination.native();
  if (destination.size() >
      (std::numeric_limits<DWORD>::max)() / sizeof(WCHAR)) {
    ::SetLastError(ERROR_FILENAME_EXCED_RANGE);
    return false;
  }

  const size_t destination_bytes = destination.size() * sizeof(WCHAR);
  if (destination_bytes >
      (std::numeric_limits<DWORD>::max)() - sizeof(ExtendedFileRenameInfo)) {
    ::SetLastError(ERROR_FILENAME_EXCED_RANGE);
    return false;
  }
  const size_t rename_info_size =
      sizeof(ExtendedFileRenameInfo) + destination_bytes;

  std::unique_ptr<unsigned char[]> rename_buffer(
      new (std::nothrow) unsigned char[rename_info_size]);
  if (!rename_buffer) {
    ::SetLastError(ERROR_NOT_ENOUGH_MEMORY);
    return false;
  }
  std::memset(rename_buffer.get(), 0, rename_info_size);
  auto *rename_info =
      reinterpret_cast<ExtendedFileRenameInfo *>(rename_buffer.get());
  rename_info->flags = kReplaceIfExists | kPosixSemantics;
  rename_info->root_directory = nullptr;
  rename_info->file_name_length = static_cast<DWORD>(destination_bytes);
  std::memcpy(rename_info->file_name, destination.data(), destination_bytes);

  HANDLE source_handle =
      ::CreateFileW(source_path.c_str(), DELETE | SYNCHRONIZE,
                    FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                    nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (source_handle == INVALID_HANDLE_VALUE) {
    return false;
  }

  const BOOL renamed = ::SetFileInformationByHandle(
      source_handle, kFileRenameInfoEx, rename_info,
      static_cast<DWORD>(rename_info_size));
  const DWORD rename_error = renamed ? ERROR_SUCCESS : ::GetLastError();
  ::CloseHandle(source_handle);
  ::SetLastError(rename_error);
  return renamed != FALSE;
}
#endif

SnapshotReplaceResult ReplaceFileAtomically(const std::string &source,
                                            const std::string &destination) {
#if defined(_WIN32) || defined(_WIN64)
  const auto source_path = ailego::FileHelper::PathFromUtf8(source);
  const auto destination_path = ailego::FileHelper::PathFromUtf8(destination);
  // MoveFileExW cannot reliably replace an open destination. DiskAnn keeps
  // old snapshots readable until the final in-flight query releases them, so
  // use Windows POSIX rename semantics and retain MoveFileExW as a fallback
  // for older systems where FileRenameInfoEx is unavailable.
  return (ReplaceOpenFileWithPosixSemantics(source_path, destination_path) ||
          ::MoveFileExW(source_path.c_str(), destination_path.c_str(),
                        MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) !=
              0)
             ? SnapshotReplaceResult::kReplaced
             : SnapshotReplaceResult::kNotReplaced;
#else
  const size_t separator = destination.rfind('/');
  const std::string parent_directory =
      separator == std::string::npos
          ? "."
          : (separator == 0 ? "/" : destination.substr(0, separator));
  int flags = O_RDONLY;
#ifdef O_DIRECTORY
  flags |= O_DIRECTORY;
#endif
#ifdef O_CLOEXEC
  flags |= O_CLOEXEC;
#endif

  int directory_fd;
  do {
    directory_fd = ::open(parent_directory.c_str(), flags);
  } while (directory_fd < 0 && errno == EINTR);
  if (directory_fd < 0) {
    return SnapshotReplaceResult::kNotReplaced;
  }

  if (!ailego::File::Rename(source, destination)) {
    const int rename_error = errno;
    ::close(directory_fd);
    errno = rename_error;
    return SnapshotReplaceResult::kNotReplaced;
  }

  int sync_result;
  do {
    sync_result = ::fsync(directory_fd);
  } while (sync_result != 0 && errno == EINTR);
  const int sync_error = sync_result == 0 ? 0 : errno;
  ::close(directory_fd);
  if (sync_result != 0) {
    errno = sync_error;
    return SnapshotReplaceResult::kReplacedNotDurable;
  }
  return SnapshotReplaceResult::kReplaced;
#endif
}

}  // namespace
#endif

uint32_t DiskAnnIndex::get_doc_count() const {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!is_trained_) {
    return static_cast<uint32_t>(doc_cache_.size());
  }
  return Index::get_doc_count();
}

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

int DiskAnnIndex::CommitBuiltSnapshot(bool *snapshot_replaced) {
  if (snapshot_replaced != nullptr) {
    *snapshot_replaced = false;
  }
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

int DiskAnnIndex::CommitBuiltSnapshot(bool *snapshot_replaced) {
  if (snapshot_replaced != nullptr) {
    *snapshot_replaced = false;
  }
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

  const SnapshotReplaceResult replace_result =
      ReplaceFileAtomically(temporary_path, file_path_);
  if (replace_result == SnapshotReplaceResult::kNotReplaced) {
    LOG_ERROR("Failed to atomically replace DiskAnn index, path: %s, err: %s",
              file_path_.c_str(),
              ailego::FileHelper::GetLastErrorString().c_str());
    return core::IndexError_Runtime;
  }
  temporary_committed = true;
  if (snapshot_replaced != nullptr) {
    *snapshot_replaced = true;
  }

  core::IndexStreamer::Pointer previous_streamer;
  core::IndexStorage::Pointer previous_storage;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    previous_streamer = exchange_streamer(std::move(replacement_streamer));
    previous_storage = exchange_storage(std::move(replacement_storage));
  }
  // Do not explicitly unload the previous streamer. Searches, fetches and
  // providers atomically acquire their own shared_ptr snapshot; its file
  // reader must remain usable until the last in-flight operation releases it.
  // The old streamer and storage clean themselves up when these local owners
  // and all reader owners have gone away.
  (void)previous_streamer;
  (void)previous_storage;

  if (replace_result == SnapshotReplaceResult::kReplacedNotDurable) {
    LOG_ERROR(
        "DiskAnn snapshot was replaced but its directory entry could not be "
        "made durable, path: %s, err: %s",
        file_path_.c_str(), ailego::FileHelper::GetLastErrorString().c_str());
    return core::IndexError_WriteData;
  }

  return 0;
}

int DiskAnnIndex::add(const VectorData &vector, uint32_t doc_id) {
  if (!is_open_) {
    LOG_ERROR("Open DiskAnn index before adding vectors");
    return core::IndexError_NoReady;
  }
  if (is_read_only_) {
    LOG_ERROR("Cannot add to a read-only DiskAnn index");
    return core::IndexError_Runtime;
  }
  if (!std::holds_alternative<DenseVector>(vector.vector)) {
    LOG_ERROR("Invalid vector data");
    return core::IndexError_InvalidArgument;
  }
  const DenseVector &dense_vector = std::get<DenseVector>(vector.vector);
  if (dense_vector.data == nullptr) {
    LOG_ERROR("Invalid null vector data");
    return core::IndexError_InvalidArgument;
  }
  if (doc_id == (std::numeric_limits<uint32_t>::max)()) {
    LOG_ERROR("Invalid reserved document id: %u", doc_id);
    return core::IndexError_OutOfRange;
  }

  try {
    const size_t vector_size = input_vector_meta_.element_size();
    std::string out_vector_buffer(static_cast<const char *>(dense_vector.data),
                                  vector_size);

    std::lock_guard<std::mutex> lock(mutex_);
    if (is_trained_ || is_training_) {
      LOG_ERROR("Cannot add vectors while DiskAnn is trained or training");
      return core::IndexError_NoReady;
    }
    doc_cache_.insert_or_assign(doc_id, std::move(out_vector_buffer));
  } catch (const std::bad_alloc &) {
    LOG_ERROR("Not enough memory to cache vector for document id: %u", doc_id);
    return core::IndexError_NoMemory;
  } catch (const std::length_error &) {
    LOG_ERROR("Document id exceeds cache capacity: %u", doc_id);
    return core::IndexError_OutOfRange;
  }
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
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (is_trained_ || is_training_) {
      LOG_ERROR("DiskAnn index is already trained or training");
      return core::IndexError_NoReady;
    }
    is_training_ = true;
  }
  AILEGO_DEFER([&]() {
    std::lock_guard<std::mutex> lock(mutex_);
    is_training_ = false;
  });

  const core::IndexMeta &build_meta =
      converter_ != nullptr ? converter_->meta() : proxima_index_meta_;
  auto reset_builder_after_failure = [&]() -> bool {
    holder_.reset();
    bool reset_succeeded = true;
    if (converter_ != nullptr && converter_->cleanup() != 0) {
      LOG_ERROR(
          "Failed to release DiskAnn converter result after training "
          "failure");
      reset_succeeded = false;
    }
    if (builder_ == nullptr || builder_->cleanup() != 0 ||
        builder_->init(build_meta, proxima_index_params_) != 0) {
      LOG_ERROR("Failed to reset DiskAnn builder after training failure");
      reset_succeeded = false;
    }
    return reset_succeeded;
  };
  auto return_training_failure = [&](int failure) -> int {
    return reset_builder_after_failure() ? failure : core::IndexError_Runtime;
  };

  int ret = GenerateHolder();
  if (ret != 0) {
    LOG_ERROR("Failed to generate holder, err: %s",
              core::IndexError::What(ret));
    return return_training_failure(ret);
  }
  ret = builder_->train(holder_);
  if (ret != 0) {
    LOG_ERROR("Failed to train builder, err: %s", core::IndexError::What(ret));
    return return_training_failure(ret);
  }
  ret = builder_->build(holder_);
  if (ret != 0) {
    LOG_ERROR("Failed to build index, err: %s", core::IndexError::What(ret));
    return return_training_failure(ret);
  }
  bool snapshot_replaced = false;
  ret = CommitBuiltSnapshot(&snapshot_replaced);
  if (ret != 0 && !snapshot_replaced) {
    return return_training_failure(ret);
  }
  // The committed streamer owns the searchable snapshot. Drop all build-time
  // copies immediately so a trained mobile index does not retain the input
  // cache, holder, and builder entity for the rest of its lifetime.
  holder_.reset();
  {
    std::lock_guard<std::mutex> lock(mutex_);
    is_trained_ = true;
    decltype(doc_cache_) empty_cache;
    doc_cache_.swap(empty_cache);
  }
  if (builder_->cleanup() != 0) {
    LOG_WARN("Failed to release DiskAnn builder memory after training");
  }
  if (converter_ != nullptr && converter_->cleanup() != 0) {
    LOG_WARN("Failed to release DiskAnn converter memory after training");
  }
  return ret;
}

int DiskAnnIndex::_dense_fetch(const uint32_t doc_id,
                               VectorDataBuffer *vector_data_buffer) {
  if (is_trained_) {
    const auto streamer = streamer_snapshot();
    if (streamer == nullptr) {
      return core::IndexError_NoReady;
    }
    auto &context = acquire_context();
    if (context == nullptr) {
      LOG_ERROR("Failed to acquire DiskAnn fetch context");
      return core::IndexError_Runtime;
    }

    std::string stored_vector;
    const int ret = streamer->get_vector(doc_id, context, stored_vector);
    context->reset();
    if (ret != 0) {
      return ret;
    }
    const size_t expected_vector_size = streamer_vector_meta_.element_size();
    if (stored_vector.size() != expected_vector_size) {
      LOG_ERROR("Invalid fetched vector size: %zu, expected: %zu",
                stored_vector.size(), expected_vector_size);
      return core::IndexError_InvalidFormat;
    }

    DenseVectorBuffer dense_vector_buffer;
    if (reformer_ != nullptr) {
      dense_vector_buffer.data.resize(input_vector_meta_.element_size());
      if (reformer_->revert(stored_vector.data(), streamer_vector_meta_,
                            &dense_vector_buffer.data) != 0) {
        LOG_ERROR("Failed to revert fetched DiskAnn vector");
        return core::IndexError_Runtime;
      }
    } else {
      dense_vector_buffer.data = std::move(stored_vector);
    }
    vector_data_buffer->vector_buffer = std::move(dense_vector_buffer);
    return 0;
  } else {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto iter = doc_cache_.find(doc_id);
    if (iter == doc_cache_.end()) {
      LOG_ERROR("Vector id does not exist: %u", doc_id);
      return core::IndexError_NoExist;
    }
    DenseVectorBuffer dense_vector_buffer;
    std::string &out_vector_buffer = dense_vector_buffer.data;
    out_vector_buffer = iter->second;
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

  bool was_trained = false;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (is_training_) {
      LOG_ERROR("DiskAnn index is already training or merging");
      return core::IndexError_NoReady;
    }
    is_training_ = true;
    was_trained = is_trained_;
  }
  AILEGO_DEFER([&]() {
    std::lock_guard<std::mutex> lock(mutex_);
    is_training_ = false;
  });

  const core::IndexMeta &build_meta =
      converter_ != nullptr ? converter_->meta() : proxima_index_meta_;
  auto rollback_training_state = [&]() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      is_trained_ = was_trained;
    }
    holder_.reset();
    if (converter_ != nullptr && converter_->cleanup() != 0) {
      LOG_ERROR(
          "Failed to release DiskAnn converter result after merge "
          "failure");
    }
    if (builder_ == nullptr || builder_->cleanup() != 0 ||
        builder_->init(build_meta, proxima_index_params_) != 0) {
      LOG_ERROR("Failed to reset DiskAnn builder after merge failure");
    }
  };

  // A DiskAnn builder is single-use. Reinitialize it before rebuilding an
  // already trained target so repeated merge calls behave like other indexes.
  if (was_trained) {
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
  bool snapshot_replaced = false;
  const int ret = CommitBuiltSnapshot(&snapshot_replaced);
  if (ret != 0 && !snapshot_replaced) {
    rollback_training_state();
    return ret;
  }
  holder_.reset();
  {
    std::lock_guard<std::mutex> lock(mutex_);
    is_trained_ = true;
    decltype(doc_cache_) empty_cache;
    doc_cache_.swap(empty_cache);
  }
  if (builder_->cleanup() != 0) {
    LOG_WARN("Failed to release DiskAnn builder memory after merge");
  }
  if (converter_ != nullptr && converter_->cleanup() != 0) {
    LOG_WARN("Failed to release DiskAnn converter memory after merge");
  }
  return ret;
}

#endif  // DISKANN_SUPPORTED

}  // namespace zvec::core_interface
