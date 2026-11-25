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

#include "concurrent_roaring_bitmap.h"
#include <atomic>
#include <chrono>
#include <filesystem>
#include <ailego/pattern/scope_guard.h>
#include <zvec/ailego/buffer/budgeted_allocator.h>
#include <zvec/ailego/hash/crc32c.h>


namespace zvec {


Status ConcurrentRoaringBitmap32::serialize(std::string *out) {
  std::unique_lock<std::shared_mutex> lock(mutex_);
  size_t bitmap_size = roaring_bitmap_portable_size_in_bytes(bitmap_);
  out->resize(bitmap_size);
  size_t written_size = roaring_bitmap_portable_serialize(bitmap_, out->data());
  if (written_size == bitmap_size) {
    return Status::OK();
  } else {
    LOG_ERROR("Failed to serialize bitmap");
    return Status::InternalError();
  }
}


Status ConcurrentRoaringBitmap32::deserialize(const std::string &in) {
  std::unique_lock<std::shared_mutex> lock(mutex_);
  auto *loaded = roaring_bitmap_portable_deserialize_safe(in.data(), in.size());
  if (loaded) {
    roaring_bitmap_free(bitmap_);
    bitmap_ = loaded;
    return Status::OK();
  } else {
    LOG_ERROR("Failed to deserialize bitmap");
    return Status::InternalError();
  }
}


Status ConcurrentRoaringBitmap64::serialize(const std::string &file_path,
                                            bool overwrite) try {
  std::unique_lock<std::shared_mutex> lock(mutex_);
  FILE file;
  const std::string file_str = "[" + file_path + "]";

  if (FILE::IsExist(file_path)) {
    if (!FILE::IsRegular(file_path)) {
      auto msg = debug_str(file_str, " is not a regular file");
      LOG_ERROR("%s", msg.c_str());
      return Status::InvalidArgument(msg);
    }
    if (!overwrite) {
      auto msg = debug_str(file_str, " already exists");
      LOG_ERROR("%s", msg.c_str());
      return Status::AlreadyExists(msg);
    }
  }

  // Serialize bitmap to buffer
  BitmapMetaHeader header{};
  size_t bitmap_size;
  std::vector<char, ailego::BudgetedAllocator<char>> bitmap_buffer;
  if (is_32bit_) {
    bitmap_size = bitmap32_->getSizeInBytes();
    bitmap_buffer.resize(bitmap_size);
    if (bitmap32_->write(bitmap_buffer.data()) != bitmap_size) {
      auto msg = debug_str("failed to serialize bitmap to buffer");
      LOG_ERROR("%s", msg.c_str());
      return Status::InternalError(msg);
    }
    header.is_32bit = 1;
  } else {
    bitmap_size = bitmap64_->getSizeInBytes();
    bitmap_buffer.resize(bitmap_size);
    if (bitmap64_->write(bitmap_buffer.data()) != bitmap_size) {
      auto msg = debug_str("failed to serialize bitmap to buffer");
      LOG_ERROR("%s", msg.c_str());
      return Status::InternalError(msg);
    }
    header.is_32bit = 0;
  }
  header.magic = roaring_magic_number;
  header.checksum = ailego::Crc32c::Hash(bitmap_buffer.data(), bitmap_size);
  header.timestamp = time(nullptr);

  // Create an owned sibling directory exclusively; never truncate an existing
  // bitmap (or a stale temporary file) before a successful replacement.
  static std::atomic<uint64_t> sequence{0};
  std::filesystem::path temporary_directory;
  bool created = false;
  for (int attempt = 0; attempt < 16 && !created; ++attempt) {
    temporary_directory = std::filesystem::u8path(
        file_path + ".tmp." +
        std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()) +
        "." + std::to_string(sequence.fetch_add(1)));
    std::error_code error;
    created = std::filesystem::create_directory(temporary_directory, error);
    if (error)
      return Status::InternalError("Cannot create bitmap temporary directory: ",
                                   error.message());
  }
  if (!created)
    return Status::InternalError("Cannot create bitmap temporary directory");
  const auto temporary_file = temporary_directory / "bitmap";
  auto cleanup = ailego::ScopeGuard::Make([&] {
    file.close();
    std::error_code ignored;
    std::filesystem::remove(temporary_file, ignored);
    std::filesystem::remove(temporary_directory, ignored);
  });
  if (!file.create(temporary_file.u8string(), 0))
    return Status::InternalError("Cannot create bitmap temporary file");

  // Write meta header to file
  if (file.write(&header, sizeof(header)) != sizeof(header)) {
    auto msg = debug_str("failed to serialize header to ", file_str);
    LOG_ERROR("%s", msg.c_str());
    return Status::InternalError(msg);
  }

  // Write serialized bitmap to file
  if (file.write(bitmap_buffer.data(), bitmap_size) != bitmap_size) {
    auto msg = debug_str("failed to write bitmap data to ", file_str);
    LOG_ERROR("%s", msg.c_str());
    return Status::InternalError(msg);
  };

  if (!file.flush())
    return Status::InternalError("Cannot flush bitmap temporary file");
  file.close();
  if (!FILE::Rename(temporary_file.u8string(), file_path))
    return Status::InternalError("Cannot replace bitmap file");

  LOG_DEBUG("%s: serialized bitmap to file[%s], checksum[%u], timestamp[%zu]",
            identifier_.c_str(), file_path.c_str(), header.checksum,
            (size_t)header.timestamp);
  return Status::OK();
} catch (const std::bad_alloc &) {
  return Status(StatusCode::RESOURCE_EXHAUSTED,
                "Insufficient bitmap serialization budget");
}


Status ConcurrentRoaringBitmap64::deserialize(
    const std::string &file_path) try {
  std::unique_lock<std::shared_mutex> lock(mutex_);
  FILE file;
  const std::string file_str = "[" + file_path + "]";

  if (!FILE::IsExist(file_path)) {
    auto msg = debug_str(file_str, " does not exist");
    LOG_ERROR("%s", msg.c_str());
    return Status::NotFound(msg);
  }
  if (!FILE::IsRegular(file_path)) {
    auto msg = debug_str(file_str, " is not a regular file");
    LOG_ERROR("%s", msg.c_str());
    return Status::InvalidArgument(msg);
  }
  if (!file.open(file_path.c_str(), true, false)) {
    auto msg = debug_str("failed to open ", file_str);
    LOG_ERROR("%s", msg.c_str());
    return Status::InternalError(msg);
  }

  // Deserialize and verify the meta header
  BitmapMetaHeader header{};
  if (file.size() < sizeof(BitmapMetaHeader)) {
    auto msg =
        debug_str(file_str, " is too small to to contain a valid bitmap");
    LOG_ERROR("%s", msg.c_str());
    return Status::InternalError(msg);
  }
  if (file.read(&header, sizeof(header)) != sizeof(header)) {
    auto msg = debug_str("failed to read meta header from ", file_str);
    LOG_ERROR("%s", msg.c_str());
    return Status::InternalError(msg);
  }
  if (header.magic != roaring_magic_number) {
    auto msg = debug_str("magic number mismatch, ", file_str);
    LOG_ERROR("%s", msg.c_str());
    return Status::InternalError(msg);
  }
  if (header.is_32bit != 0 && header.is_32bit != 1) {
    auto msg = debug_str("bitmap type mismatch, ", file_str);
    LOG_ERROR("%s", msg.c_str());
    return Status::InternalError(msg);
  }

  // Read from file to buffer
  size_t bitmap_size = file.size() - sizeof(BitmapMetaHeader);
  std::vector<char, ailego::BudgetedAllocator<char>> bitmap_buffer(bitmap_size);
  if (file.read(bitmap_buffer.data(), bitmap_size) != bitmap_size) {
    auto msg = debug_str("failed to read bitmap data from ", file_str);
    LOG_ERROR("%s", msg.c_str());
    return Status::InternalError(msg);
  }
  if (header.checksum !=
      ailego::Crc32c::Hash(bitmap_buffer.data(), bitmap_size)) {
    auto msg = debug_str("checksum mismatch, ", file_str);
    LOG_ERROR("%s", msg.c_str());
    return Status::InternalError(msg);
  }

  // Check container counts before calling readSafe, which may allocate its
  // container table before detecting a truncated payload.
  size_t offset = 0;
  uint64_t count = 1;
  if (!header.is_32bit) {
    if (bitmap_size < sizeof(count))
      return Status::InternalError("Truncated bitmap");
    std::memcpy(&count, bitmap_buffer.data(), sizeof(count));
    offset = sizeof(count);
    if (count > (bitmap_size - offset) / 12)
      return Status::InternalError("Invalid bitmap map size");
  }
  for (uint64_t i = 0; i < count; ++i) {
    if (!header.is_32bit) offset += sizeof(uint32_t);
    if (offset >= bitmap_size) return Status::InternalError("Truncated bitmap");
    const size_t bytes = roaring_bitmap_portable_deserialize_size(
        bitmap_buffer.data() + offset, bitmap_size - offset);
    if (!bytes) return Status::InternalError("Invalid bitmap payload");
    offset += bytes;
  }
  if (offset != bitmap_size)
    return Status::InternalError("Invalid bitmap length");
  ailego::MemoryReservation memory;
  memory.add(memory_allowance(bitmap_size));
  try {
    if (header.is_32bit) {
      auto loaded = std::make_unique<roaring::Roaring>(
          roaring::Roaring::readSafe(bitmap_buffer.data(), bitmap_size));
      bitmap32_ = std::move(loaded);
      bitmap64_.reset();
    } else {
      auto loaded = std::make_unique<roaring::Roaring64Map>(
          roaring::Roaring64Map::readSafe(bitmap_buffer.data(), bitmap_size));
      bitmap64_ = std::move(loaded);
      bitmap32_.reset();
    }
  } catch (const std::bad_alloc &) {
    throw;
  } catch (...) {
    return Status::InternalError("Failed to deserialize bitmap from ",
                                 file_path);
  }
  is_32bit_ = header.is_32bit != 0;
  if (is_32bit_ ? bitmap32_->isEmpty() : bitmap64_->isEmpty())
    memory.retain_at_most(0);
  memory_ = std::move(memory);

  LOG_DEBUG(
      "%s: deserialized bitmap from file[%s], checksum[%u], timestamp[%zu]",
      identifier_.c_str(), file_path.c_str(), header.checksum,
      (size_t)header.timestamp);
  return Status::OK();
} catch (const std::bad_alloc &) {
  return Status(StatusCode::RESOURCE_EXHAUSTED,
                "Insufficient bitmap load budget");
}


}  // namespace zvec
