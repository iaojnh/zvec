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
#include <algorithm>
#include <array>
#include <atomic>
#include <mutex>
#include <shared_mutex>
#include <ailego/pattern/scope_guard.h>
#include <zvec/core/framework/index_error.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/core/framework/index_mapping.h>
#include <zvec/core/framework/index_version.h>
#include "utility_params.h"

namespace zvec {
namespace core {

/*! MMap File Storage
 */
class MMapFileStorage : public IndexStorage {
  struct OpenFile {
    IndexMapping mapping;
    // Directory operations exclude writes. Independent segments can write in
    // parallel.
    std::shared_mutex directory_mutex;
    std::array<std::mutex, 32> segment_mutexes;
    std::atomic<bool> dirty{false};
    bool retired{false};
    bool persist_on_close{true};

    bool is_dirty() const {
      return dirty.load(std::memory_order_relaxed) || mapping.is_header_dirty();
    }

    int flush() {
      if (dirty.exchange(false, std::memory_order_relaxed)) mapping.refresh(0);
      return mapping.flush();
    }
  };

 public:
  /*! Index Storage Segment
   */
  class Segment : public IndexStorage::Segment,
                  public std::enable_shared_from_this<Segment> {
   public:
    //! Index Storage Pointer
    typedef std::shared_ptr<Segment> Pointer;

    //! Constructor
    Segment(std::shared_ptr<OpenFile> owner, IndexMapping::Segment *segment,
            std::mutex &mutex)
        : segment_(segment),
          owner_(std::move(owner)),
          mutex_(mutex),
          capacity_(static_cast<size_t>(segment->meta()->data_size +
                                        segment->meta()->padding_size)) {}

    //! Destructor
    ~Segment() override = default;

    //! Retrieve size of data
    size_t data_size() const override {
      std::lock_guard lock(mutex_);
      return static_cast<size_t>(segment_->meta()->data_size);
    }

    //! Retrieve crc of data
    uint32_t data_crc() const override {
      std::lock_guard lock(mutex_);
      return segment_->meta()->data_crc;
    }

    //! Retrieve size of padding
    size_t padding_size() const override {
      std::lock_guard lock(mutex_);
      return static_cast<size_t>(segment_->meta()->padding_size);
    }

    //! Retrieve capacity of segment
    size_t capacity() const override {
      return capacity_;
    }

    //! Fetch data from segment (with own buffer)
    size_t fetch(size_t offset, void *buf, size_t len) const override {
      std::lock_guard lock(mutex_);
      const size_t size = segment_->meta()->data_size;
      if (offset >= size || !len) return 0;
      len = std::min(len, size - offset);
      memmove(buf, (const uint8_t *)segment_->data() + offset, len);
      return len;
    }

    //! Read data from segment
    size_t read(size_t offset, MemoryBlock &data, size_t len) override {
      const size_t size = data_size();
      offset = std::min(offset, size);
      len = std::min(len, size - offset);
      data = MemoryBlock::MakeMappedView((uint8_t *)segment_->data() + offset,
                                         owner_);
      return len;
    }

    //! Write data into the storage with offset
    size_t write(size_t offset, const void *data, size_t len) override {
      if (offset > capacity_ || len > capacity_ - offset || (len && !data))
        return 0;
      std::shared_lock directory(owner_->directory_mutex);
      if (owner_->retired) return 0;
      std::lock_guard lock(mutex_);
      const size_t data_tail = offset + len;
      auto meta = segment_->meta();
      if (data_tail > meta->data_size) {
        meta->data_size = data_tail;
        meta->padding_size = capacity_ - data_tail;
      }
      owner_->dirty.store(true, std::memory_order_relaxed);
      if (len) memmove((uint8_t *)segment_->data() + offset, data, len);
      segment_->set_dirty();
      return len;
    }

    //! Resize size of data
    size_t resize(size_t size) override {
      std::shared_lock directory(owner_->directory_mutex);
      std::lock_guard lock(mutex_);
      if (owner_->retired) return segment_->meta()->data_size;
      auto meta = segment_->meta();
      if (meta->data_size != size) {
        if (size > capacity_) {
          size = capacity_;
        }
        meta->data_size = size;
        meta->padding_size = capacity_ - size;
        owner_->dirty.store(true, std::memory_order_relaxed);
      }
      return size;
    }

    //! Update crc of data
    void update_data_crc(uint32_t crc) override {
      std::shared_lock directory(owner_->directory_mutex);
      if (owner_->retired) return;
      std::lock_guard lock(mutex_);
      segment_->meta()->data_crc = crc;
      owner_->dirty.store(true, std::memory_order_relaxed);
    }

    //! Clone the segment
    IndexStorage::Segment::Pointer clone() override {
      return shared_from_this();
    }

    //! Stable base data pointer — valid for the lifetime of the mmap.
    const uint8_t *base_data() const override {
      return (const uint8_t *)segment_->data();
    }

   private:
    IndexMapping::Segment *segment_{};
    std::shared_ptr<OpenFile> owner_;
    std::mutex &mutex_;
    size_t capacity_{};
  };

  //! Destructor
  ~MMapFileStorage() override {
    close_index(/*disposing=*/true);
  }

  //! Initialize storage
  int init(const ailego::Params &params) override {
    std::lock_guard lock(lifecycle_mutex_);
    uint32_t val = params.get_as_uint32(MMAPFILE_STORAGE_SEGMENT_META_CAPACITY);
    if (val != 0) {
      segment_meta_capacity_ = val;
    }
    params.get(MMAPFILE_STORAGE_COPY_ON_WRITE, &copy_on_write_);
    params.get(MMAPFILE_STORAGE_FORCE_FLUSH, &force_flush_);
    params.get(MMAPFILE_STORAGE_MEMORY_LOCKED, &memory_locked_);
    params.get(MMAPFILE_STORAGE_MEMORY_WARMUP, &memory_warmup_);
    return 0;
  }

  //! Cleanup storage
  int cleanup() override {
    return close_index();
  }

  //! Open storage
  int open(const std::string &path, bool create_if_missing) override try {
    std::lock_guard lock(lifecycle_mutex_);
    auto previous = file_;
    std::unique_lock<std::shared_mutex> writers;
    if (previous) {
      writers = std::unique_lock(previous->directory_mutex);
      if (previous->persist_on_close && previous->is_dirty()) {
        const int ret = previous->flush();
        if (ret != 0) return ret;
      }
    }
    if (!ailego::File::IsExist(path) && create_if_missing) {
      size_t last_slash = path.rfind('/');
      if (last_slash != std::string::npos) {
        ailego::File::MakePath(path.substr(0, last_slash));
      }

      int error_code = this->init_index(path);
      if (error_code != 0) {
        return error_code;
      }
    }
    auto next =
        std::allocate_shared<OpenFile>(ailego::BudgetedAllocator<OpenFile>{});
    const int ret = next->mapping.open(path, copy_on_write_, force_flush_);
    if (ret != 0) return ret;
    next->persist_on_close = !copy_on_write_ || force_flush_;
    if (previous) previous->retired = true;
    file_ = std::move(next);
    return 0;
  } catch (const std::bad_alloc &) {
    return IndexError_NoMemory;
  }

  //! Flush storage
  int flush() override {
    std::lock_guard lock(lifecycle_mutex_);
    if (!file_) return 0;
    std::unique_lock directory(file_->directory_mutex);
    return file_->flush();
  }

  //! Close storage
  int close() override {
    return close_index();
  }

  //! Append a segment into storage
  int append(const std::string &id, size_t size) override {
    std::lock_guard lock(lifecycle_mutex_);
    if (!file_) return IndexError_Runtime;
    std::unique_lock directory(file_->directory_mutex);
    return file_->mapping.append(id, size);
  }

  //! Refresh meta information (checksum, update time, etc.)
  void refresh(uint64_t chkp) override {
    std::lock_guard lock(lifecycle_mutex_);
    if (!file_) return;
    std::unique_lock directory(file_->directory_mutex);
    file_->mapping.refresh(chkp);
    file_->dirty.store(false, std::memory_order_relaxed);
  }

  //! Retrieve check point of storage
  uint64_t check_point() const override {
    std::lock_guard lock(lifecycle_mutex_);
    return file_ ? file_->mapping.check_point() : 0;
  }

  //! Retrieve a segment by id
  IndexStorage::Segment::Pointer get(const std::string &id, int) override {
    std::lock_guard lock(lifecycle_mutex_);
    if (!file_) return {};
    std::unique_lock directory(file_->directory_mutex);
    auto segment = file_->mapping.map(id, memory_warmup_, memory_locked_);
    if (!segment) return {};
    auto &mutex = file_->segment_mutexes[std::hash<std::string>{}(id) %
                                         file_->segment_mutexes.size()];
    return std::make_shared<Segment>(file_, segment, mutex);
  }

  //! Test if it a segment exists
  bool has(const std::string &id) const override {
    std::lock_guard lock(lifecycle_mutex_);
    return file_ && file_->mapping.has(id);
  }

  //! Retrieve magic number of index
  uint32_t magic() const override {
    std::lock_guard lock(lifecycle_mutex_);
    return file_ ? file_->mapping.magic() : 0;
  }

 protected:
  //! Initialize index version segment
  int init_version_segment(IndexMapping &mapping) {
    size_t data_size = std::strlen(IndexVersion::Details());
    int error_code = mapping.append(INDEX_VERSION_SEGMENT_NAME, data_size);
    if (error_code != 0) {
      return error_code;
    }

    auto *segment = mapping.map(INDEX_VERSION_SEGMENT_NAME, false, false);
    if (!segment) {
      return IndexError_MMapFile;
    }
    auto meta = segment->meta();
    size_t capacity = static_cast<size_t>(meta->padding_size + meta->data_size);
    memcpy(segment->data(), IndexVersion::Details(), data_size);
    segment->set_dirty();
    meta->data_crc = ailego::Crc32c::Hash(segment->data(), data_size, 0);
    meta->data_size = data_size;
    meta->padding_size = capacity - data_size;
    return 0;
  }

  //! Initialize index file
  int init_index(const std::string &path) {
    IndexMapping mapping;
    int ret = mapping.create(path, segment_meta_capacity_, /*exclusive=*/true);
    if (ret != 0) return ret;

    bool ready = false;
    const auto cleanup = ailego::ScopeGuard::Make([&] {
      mapping.close();
      if (!ready && !ailego::File::Delete(path))
        LOG_ERROR("Failed to remove incomplete index %s", path.c_str());
    });
    ret = init_version_segment(mapping);
    if (ret != 0) return ret;
    mapping.refresh(0);
    ret = mapping.flush();
    ready = ret == 0;
    return ret;
  }

  bool is_huge_page() const override {
    std::lock_guard lock(lifecycle_mutex_);
    return file_ && file_->mapping.huge_page();
  }

  bool is_dirty() const override {
    std::lock_guard lock(lifecycle_mutex_);
    return file_ && file_->is_dirty();
  }

  int close_index(bool disposing = false) {
    std::lock_guard lock(lifecycle_mutex_);
    auto previous = file_;
    if (!previous) return 0;
    std::unique_lock directory(previous->directory_mutex);
    int ret = 0;
    if (previous->persist_on_close && previous->is_dirty()) {
      ret = previous->flush();
      if (ret != 0 && !disposing) return ret;
    }
    previous->retired = true;
    file_.reset();
    return ret;
  }

 private:
  uint32_t segment_meta_capacity_{1024 * 1024};
  bool copy_on_write_{false};
  bool force_flush_{false};
  bool memory_locked_{false};
  bool memory_warmup_{false};
  std::shared_ptr<OpenFile> file_;
  mutable std::mutex lifecycle_mutex_;
};

INDEX_FACTORY_REGISTER_STORAGE(MMapFileStorage);

}  // namespace core
}  // namespace zvec
