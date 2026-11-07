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

#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <zvec/ailego/utility/file_helper.h>
#include <zvec/core/framework/index_error.h>
#include <zvec/core/framework/index_storage.h>

namespace zvec::core {

// Build-only, fixed-size scratch storage. All resident data pages share the
// ordinary Buffer Pool; reads copy into caller-owned, bounded working buffers.
// There are no escaping pins or persisted references to this temporary file.
class IndexStorage::Temporary {
 public:
  using Pointer = std::shared_ptr<Temporary>;

  Temporary(const Temporary &) = delete;
  Temporary &operator=(const Temporary &) = delete;

  static int DefaultPrefix(const std::string &name, std::string *out) {
    if (!out) return IndexError_InvalidArgument;
    std::string directory_name;
    if (!ailego::FileHelper::GetTemporaryDirectory(&directory_name))
      return IndexError_OpenFile;
    const auto directory = ailego::FileHelper::PathFromUtf8(directory_name);
    *out = ailego::FileHelper::PathToUtf8(directory / name);
    return 0;
  }

  static int Create(const std::string &prefix, size_t bytes, Pointer *out) {
    if (!out || bytes == 0 ||
        bytes > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
      return IndexError_InvalidArgument;
    }
    try {
      Pointer candidate(new Temporary());
      int ret = candidate->init(prefix, bytes);
      if (ret != 0) return ret;
      *out = std::move(candidate);
      return 0;
    } catch (const std::bad_alloc &) {
      return IndexError_NoMemory;
    } catch (const std::exception &) {
      return IndexError_Runtime;
    }
  }

  // An exclusively owned writable file for callers that append segments as
  // input arrives. It uses the same cleanup and pool as fixed-size scratch.
  static int CreateEmpty(const std::string &prefix, Pointer *out) {
    if (!out) return IndexError_InvalidArgument;
    try {
      Pointer candidate(new Temporary());
      int ret = candidate->init(prefix, 0);
      if (ret != 0) return ret;
      *out = std::move(candidate);
      return 0;
    } catch (const std::bad_alloc &) {
      return IndexError_NoMemory;
    } catch (const std::exception &) {
      return IndexError_Runtime;
    }
  }

  // Allocate independent streams in one temporary page table. The returned
  // segment must not outlive this owner; Buffer retains it automatically.
  int append(size_t bytes, IndexStorage::Segment::Pointer *out) try {
    if (!out || !bytes ||
        bytes > static_cast<size_t>(std::numeric_limits<int64_t>::max()))
      return IndexError_InvalidArgument;
    if (!storage_) return IndexError_NoReady;
    std::lock_guard<std::mutex> lock(mutex_);
    const std::string name = "buffer" + std::to_string(next_segment_++);
    int ret = storage_->append(name, bytes);
    if (ret != 0) return ret;
    auto segment = storage_->get(name);
    if (!segment) return IndexError_ReadData;
    *out = std::move(segment);
    return 0;
  } catch (const std::bad_alloc &) {
    return IndexError_NoMemory;
  } catch (const std::exception &) {
    return IndexError_WriteData;
  }

  // Native libraries may require their own file format. Keep their files
  // under the same exclusive owner instead of duplicating scratch cleanup.
  // Native file adapters require an existing parent directory.
  static int CreateDirectory(const std::string &prefix, Pointer *out) try {
    if (!out) return IndexError_InvalidArgument;
    Pointer candidate(new Temporary());
    int ret = candidate->init(prefix, 0, false);
    if (ret == 0) *out = std::move(candidate);
    return ret;
  } catch (const std::bad_alloc &) {
    return IndexError_NoMemory;
  } catch (const std::exception &) {
    return IndexError_OpenFile;
  }

  // Reserve a file name inside our exclusive directory. The native backend
  // creates its own header; a precreated empty file is not a valid index.
  int reserve_file(const std::string &name, std::string *out) try {
    if (!out || name.empty() || name == "." || name == ".." ||
        name.find_first_of("/\\:") != std::string::npos ||
        name.find('\0') != std::string::npos || (storage_ && name == "data"))
      return IndexError_InvalidArgument;
    std::lock_guard<std::mutex> lock(mutex_);
    std::string path = directory_ + "/" + name;
    for (const auto &file : files_)
      if (file.compare(path.c_str()) == 0) return IndexError_Duplicate;
    files_.emplace_back(path.data(), path.size());
    *out = std::move(path);
    return 0;
  } catch (const std::bad_alloc &) {
    return IndexError_NoMemory;
  } catch (const std::exception &) {
    return IndexError_OpenFile;
  }

  const IndexStorage::Pointer &storage() const {
    return storage_;
  }

  ~Temporary() {
    segment_.reset();
    if (storage_) storage_->close();
    storage_.reset();
    // Remove our backing file, registered native files and empty owned
    // directory. Preserve caller-owned parents and unregistered files.
    std::error_code error;
    if (!path_.empty()) {
      std::filesystem::remove(ailego::FileHelper::PathFromUtf8(path_), error);
    }
    for (const auto &file : files_) {
      std::filesystem::remove(std::filesystem::u8path(file.begin(), file.end()),
                              error);
    }
    if (!directory_.empty()) {
      std::filesystem::remove(ailego::FileHelper::PathFromUtf8(directory_),
                              error);
    }
  }

  int read(size_t offset, void *out, size_t bytes) const {
    if (!in_range(offset, bytes) || (!out && bytes != 0)) {
      return IndexError_InvalidArgument;
    }
    try {
      return bytes == 0 || segment_->fetch(offset, out, bytes) == bytes
                 ? 0
                 : segment_->read_error();
    } catch (const std::bad_alloc &) {
      return IndexError_NoMemory;
    } catch (const std::exception &) {
      return IndexError_ReadData;
    }
  }

  int write(size_t offset, const void *data, size_t bytes) {
    if (!in_range(offset, bytes) || (!data && bytes != 0)) {
      return IndexError_InvalidArgument;
    }
    try {
      return bytes == 0 || segment_->write(offset, data, bytes) == bytes
                 ? 0
                 : segment_->write_error();
    } catch (const std::bad_alloc &) {
      return IndexError_NoMemory;
    } catch (const std::exception &) {
      return IndexError_WriteData;
    }
  }

  int flush() {
    if (!storage_) return IndexError_NoReady;
    try {
      return storage_->flush();
    } catch (const std::bad_alloc &) {
      return IndexError_NoMemory;
    } catch (const std::exception &) {
      return IndexError_WriteData;
    }
  }

  const std::string &path() const {
    return path_;
  }

 private:
  Temporary() = default;

  bool in_range(size_t offset, size_t bytes) const {
    return segment_ && offset <= size_ && bytes <= size_ - offset;
  }

  static IndexStorage::Pointer CreateBackend();

  int init(const std::string &requested_prefix, size_t bytes,
           bool open_storage = true) {
    std::string prefix = requested_prefix;
    if (prefix.empty()) {
      int ret = DefaultPrefix("zvec-scratch", &prefix);
      if (ret != 0) return ret;
    }
    // FileDumper normally creates the output's parent directories. Scratch is
    // needed before dump, so preserve that behavior here, including UTF-8
    // paths on Windows. These caller-owned parents are never removed by us.
    const auto parent = ailego::FileHelper::PathFromUtf8(prefix).parent_path();
    if (!parent.empty() && open_storage) {
      std::error_code error;
      std::filesystem::create_directories(parent, error);
      if (error) return IndexError_OpenFile;
    }
    static std::atomic<uint64_t> sequence{0};
    const auto stamp =
        std::chrono::steady_clock::now().time_since_epoch().count();
    for (size_t attempt = 0; attempt < 64; ++attempt) {
      std::string directory =
          prefix + ".buffer-build-" + std::to_string(stamp) + "-" +
          std::to_string(sequence.fetch_add(1, std::memory_order_relaxed));
      std::string path = directory + "/data";
      std::error_code error;
      if (std::filesystem::create_directory(
              ailego::FileHelper::PathFromUtf8(directory), error)) {
        directory_ = std::move(directory);
        path_ = std::move(path);
        break;
      }
      if (error && error != std::errc::file_exists) return IndexError_OpenFile;
    }
    if (directory_.empty()) return IndexError_OpenFile;

    if (!open_storage) {
      path_.clear();
      return 0;
    }
    storage_ = CreateBackend();
    if (!storage_) return IndexError_NoExist;
    int ret = storage_->init(ailego::Params{});
    if (ret != 0) return ret;
    ret = storage_->open(path_, true);
    if (ret != 0) return ret;
    if (bytes == 0) return 0;
    ret = storage_->append("data", bytes);
    if (ret != 0) return ret;
    segment_ = storage_->get("data");
    if (!segment_) return IndexError_Runtime;
    size_ = bytes;
    return 0;
  }

  std::mutex mutex_;
  uint64_t next_segment_{0};
  ailego::BudgetedVector<ailego::BudgetedString> files_;
  std::string directory_;
  std::string path_;
  IndexStorage::Pointer storage_;
  IndexStorage::Segment::Pointer segment_;
  size_t size_{0};
};

}  // namespace zvec::core
