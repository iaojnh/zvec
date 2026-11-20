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

#include <ailego/pattern/scope_guard.h>
#include <zvec/ailego/io/mmap_file.h>
#include <zvec/ailego/logger/logger.h>
#include <zvec/ailego/utility/file_helper.h>
#include <zvec/core/framework/index_error.h>
#include <zvec/core/framework/index_mapping.h>
#include "ailego/utility/memory_helper.h"

#if defined(_WIN32) || defined(_WIN64)
#include <Windows.h>
#endif

#ifdef __linux__
#include <sys/statfs.h>
#include <sys/vfs.h>
#ifndef HUGETLBFS_MAGIC
#define HUGETLBFS_MAGIC 0x958458f6
#endif
#endif

namespace zvec {
namespace core {

static inline size_t CalcPageAlignedSize(size_t size, bool huge_size) {
  size_t page_size = ailego::MemoryHelper::PageSize();
  if (huge_size) {
    page_size = ailego::MemoryHelper::HugePageSize();
  }
  if (size > (std::numeric_limits<size_t>::max)() - (page_size - 1)) return 0;
  return (size + page_size - 1) / page_size * page_size;
}

static bool WritePadding(ailego::File &file, size_t size) {
  constexpr char padding[4096]{};
  while (size) {
    const size_t bytes = std::min(size, sizeof(padding));
    if (file.write(padding, bytes) != bytes) return false;
    size -= bytes;
  }
  return true;
}

static int UnpackMappingSize(ailego::File &file, uint64_t offset, size_t *len) {
  IndexFormat::MetaHeader header;
  if (offset > static_cast<uint64_t>((std::numeric_limits<ssize_t>::max)()) ||
      file.read(static_cast<ssize_t>(offset), &header, sizeof(header)) !=
          sizeof(header)) {
    LOG_ERROR("Failed to read file, %s",
              ailego::FileHelper::GetLastErrorString().c_str());
    return IndexError_ReadData;
  }

  if (header.meta_header_size != sizeof(IndexFormat::MetaHeader) ||
      header.meta_footer_size != sizeof(IndexFormat::MetaFooter)) {
    return IndexError_InvalidValue;
  }

  if (ailego::Crc32c::Hash(&header, sizeof(header), header.header_crc) !=
      header.header_crc) {
    return IndexError_InvalidChecksum;
  }

  if ((int32_t)header.meta_footer_offset < 0) {
    return IndexError_Unsupported;
  }

  *len = header.meta_footer_offset + header.meta_footer_size;
  const size_t file_size = file.size();
  if (offset > file_size || *len > file_size - offset) {
    return IndexError_InvalidLength;
  }
  return 0;
}

int IndexMapping::open(const std::string &path, bool cow, bool full_mode) {
  IndexMapping candidate;
  const int ret = candidate.open_mapping(path, cow, full_mode);
  if (ret == 0) swap(candidate);
  return ret;
}

int IndexMapping::open_mapping(const std::string &path, bool cow,
                               bool full_mode) try {
  path_ = path;
  full_mode_ = full_mode;
  copy_on_write_ = cow;
  huge_page_ = ishugetlbfs(path);

  bool read_only = copy_on_write_ && !full_mode_;
  if (!file_.open(path.c_str(), read_only, false)) {
    LOG_ERROR("Failed to open file %s, %s", path.c_str(),
              ailego::FileHelper::GetLastErrorString().c_str());
    return IndexError_OpenFile;
  }

  for (;;) {
    size_t mapping_size = 0;
    const int size_ret =
        UnpackMappingSize(file_, current_header_start_offset_, &mapping_size);
    if (size_ret != 0) return size_ret;
    const int ret = init_index_mapping(mapping_size);
    if (ret != 0) return ret;
    const auto next = footer_->next_meta_header_offset;
    if (next == 0) return 0;
    if (next < index_size_ || next >= file_.size())
      return IndexError_InvalidValue;
    current_header_start_offset_ = next;
  }
} catch (const std::bad_alloc &) {
  return IndexError_NoMemory;
}

int IndexMapping::create(const std::string &path, size_t seg_meta_capacity) {
  return create(path, seg_meta_capacity, false);
}

int IndexMapping::create(const std::string &path, size_t seg_meta_capacity,
                         bool exclusive) try {
  close();
  path_ = path;
  seg_meta_capacity_ = seg_meta_capacity;
  huge_page_ = ishugetlbfs(path);
  bool ready = false;
  const auto rollback = ailego::ScopeGuard::Make([&] {
    if (ready) return;
    const bool remove = exclusive && file_.is_valid();
    close();
    if (remove && !ailego::File::Delete(path))
      LOG_ERROR("Failed to remove incomplete index %s", path.c_str());
  });
  const int ret = init_meta_section(exclusive);
  ready = ret == 0;
  return ret;
} catch (const std::bad_alloc &) {
  return IndexError_NoMemory;
}

size_t IndexMapping::meta_mapping_length() const {
  constexpr size_t overhead =
      sizeof(IndexFormat::MetaHeader) + sizeof(IndexFormat::MetaFooter);
  // A nonnegative signed footer offset is required by IndexMapping::open().
  if (seg_meta_capacity_ > static_cast<size_t>(INT32_MAX) - overhead) return 0;
  const size_t len =
      CalcPageAlignedSize(seg_meta_capacity_ + overhead, huge_page_);
  return len <= static_cast<size_t>(INT32_MAX) ? len : 0;
}

int IndexMapping::init_meta_section() {
  return init_meta_section(false);
}

int IndexMapping::init_meta_section(bool exclusive) {
  const size_t page_size = huge_page_ ? ailego::MemoryHelper::HugePageSize()
                                      : ailego::MemoryHelper::PageSize();
  if (current_header_start_offset_ % page_size != 0)
    return IndexError_InvalidValue;
  const size_t len = meta_mapping_length();
  if (!len ||
      current_header_start_offset_ >
          static_cast<size_t>((std::numeric_limits<ssize_t>::max)()) - len)
    return IndexError_InvalidLength;

  auto &pool = ailego::MemoryLimitPool::get_instance();
  const size_t charge = pool.capacity() ? len : 0;
  using Pending = ailego::MemoryLimitPool::PendingAllocation;
  Pending mapping_memory(pool, Pending::Kind::kWorkspace);
  if (!mapping_memory.reserve(charge)) return IndexError_NoMemory;
  header_addr_map_.initialize();
  HeaderMap pending;
  pending.emplace(current_header_start_offset_,
                  MetaMapping{nullptr, len, charge});
  auto pending_node = pending.extract(pending.begin());

  if (!file_.is_valid() && !(exclusive ? file_.create_exclusive(path_.c_str())
                                       : file_.create(path_.c_str(), 0))) {
    LOG_ERROR("Failed to create file %s, %s", path_.c_str(),
              ailego::FileHelper::GetLastErrorString().c_str());
    return IndexError_CreateFile;
  }

  IndexFormat::MetaHeader meta_header;
  IndexFormat::MetaFooter meta_footer;
  const auto segments_meta_size =
      static_cast<uint32_t>(len - sizeof(meta_header) - sizeof(meta_footer));
  IndexFormat::SetupMetaHeader(&meta_header, len - sizeof(meta_footer), len);
  IndexFormat::SetupMetaFooter(&meta_footer);
  meta_footer.segments_meta_size = segments_meta_size;
  meta_footer.total_size = len;
  IndexFormat::UpdateMetaFooter(&meta_footer, check_point_);

  if (huge_page_) {
    const int opts = ailego::File::MMAP_SHARED | ailego::File::MMAP_HUGE_PAGE;
    auto *addr = static_cast<char *>(ailego::File::MemoryMap(
        file_.native_handle(), current_header_start_offset_, len, opts));
    if (!addr) return IndexError_MMapFile;
    std::memcpy(addr, &meta_header, sizeof(meta_header));
    std::memset(addr + sizeof(meta_header), 0, segments_meta_size);
    std::memcpy(addr + meta_header.meta_footer_offset, &meta_footer,
                sizeof(meta_footer));
    if (!ailego::File::TryMemoryUnmap(addr, len)) {
      mapping_memory.commit();
      LOG_ERROR("Failed to release temporary huge-page index metadata");
      return IndexError_MMapFile;
    }
  } else {
    if (!file_.seek(current_header_start_offset_, ailego::File::Origin::Begin))
      return IndexError_SeekFile;
    if (file_.write(&meta_header, sizeof(meta_header)) != sizeof(meta_header) ||
        !WritePadding(file_, segments_meta_size) ||
        file_.write(&meta_footer, sizeof(meta_footer)) != sizeof(meta_footer))
      return IndexError_WriteData;
  }

  const int ret =
      init_index_mapping_admitted(len, std::move(pending_node), mapping_memory);
  if (ret == 0) index_size_ = current_header_start_offset_ + len;
  return ret;
}

void IndexMapping::swap(IndexMapping &rhs) noexcept {
  using std::swap;
  swap(segment_ids_offset_, rhs.segment_ids_offset_);
  swap(segment_start_, rhs.segment_start_);
  swap(header_, rhs.header_);
  swap(header_addr_map_, rhs.header_addr_map_);
  swap(footer_, rhs.footer_);
  swap(segments_, rhs.segments_);
  swap(index_size_, rhs.index_size_);
  swap(check_point_, rhs.check_point_);
  swap(file_, rhs.file_);
  swap(path_, rhs.path_);
  swap(copy_on_write_, rhs.copy_on_write_);
  swap(full_mode_, rhs.full_mode_);
  swap(header_dirty_, rhs.header_dirty_);
  swap(huge_page_, rhs.huge_page_);
  swap(seg_meta_capacity_, rhs.seg_meta_capacity_);
  swap(current_header_start_offset_, rhs.current_header_start_offset_);
}

void IndexMapping::release_meta_mapping(const MetaMapping &mapping) {
  if (!ailego::File::TryMemoryUnmap(mapping.header, mapping.length)) {
    LOG_ERROR("Failed to unmap index metadata; retaining its memory charge");
    return;
  }
  if (mapping.charge) {
    ailego::MemoryLimitPool::get_instance().release_unmapped_external(
        mapping.charge, mapping.projection);
  }
}

void IndexMapping::close() {
  this->unmap_all();
  for (const auto &item : header_addr_map_) {
    release_meta_mapping(item.second);
  }
  // Reset members
  segment_ids_offset_ = 0;
  segment_start_ = nullptr;
  header_ = nullptr;
  header_addr_map_.clear();
  footer_ = nullptr;
  index_size_ = 0u;
  check_point_ = 0;
  segments_.clear();
  current_header_start_offset_ = 0;
  seg_meta_capacity_ = 0;
  file_.close();
  copy_on_write_ = false;
  full_mode_ = false;
  header_dirty_ = false;
  huge_page_ = false;
}

void IndexMapping::refresh(uint64_t check_point) {
  if (!footer_) return;
  check_point_ = std::max(check_point_, check_point);
  for (auto item : header_addr_map_) {
    auto header = item.second.header;
    auto footer = reinterpret_cast<IndexFormat::MetaFooter *>(
        reinterpret_cast<uint8_t *>(header) + header->meta_footer_offset);
    auto segment_start = reinterpret_cast<IndexFormat::SegmentMeta *>(
        reinterpret_cast<uint8_t *>(header) +
        (header->meta_footer_offset - footer->segments_meta_size));
    footer->segments_meta_crc =
        ailego::Crc32c::Hash(segment_start, footer->segments_meta_size, 0);
    IndexFormat::UpdateMetaFooter(footer, check_point_);
  }
  header_dirty_ = true;
}

int IndexMapping::append(const std::string &id, size_t size) try {
  if (!footer_) return IndexError_NoReady;
  if (id.find('\0') != std::string::npos) return IndexError_InvalidArgument;
  size = CalcPageAlignedSize(size, huge_page_);
  if (size == 0) return IndexError_InvalidArgument;
  if (segments_.find(id) != segments_.end()) return IndexError_Duplicate;

  const size_t id_size = id.size() + 1;
  const size_t need_size = sizeof(IndexFormat::SegmentMeta) + id_size;
  if (need_size > footer_->segments_meta_size) return IndexError_InvalidLength;
  const bool split =
      sizeof(IndexFormat::SegmentMeta) * footer_->segment_count + need_size >
      segment_ids_offset_;
  const size_t next_header_len = split ? meta_mapping_length() : 0;
  const auto max_file_size =
      static_cast<size_t>((std::numeric_limits<ssize_t>::max)());
  if ((split && !next_header_len) ||
      index_size_ > max_file_size - next_header_len ||
      size > max_file_size - next_header_len - index_size_)
    return IndexError_InvalidLength;

  // Prepare the entire directory on first use. Node insertion into either
  // directory now needs no allocation, including on libraries with sentinels.
  ailego::LazyMap<SegmentMap> pending;
  pending.emplace(std::string_view(id), SegmentInfo{});
  auto pending_node = pending.extract(pending.begin());

  auto *old_header = header_;
  auto *old_footer = footer_;
  auto *old_segment_start = segment_start_;
  const auto old_ids_offset = segment_ids_offset_;
  const auto old_header_offset = current_header_start_offset_;
  const auto old_capacity = seg_meta_capacity_;
  const auto old_size = index_size_;
  size_t old_file_size = 0;
  bool rollback_split = false;
  const auto rollback = ailego::ScopeGuard::Make([&] {
    if (!rollback_split) return;
    auto it = header_addr_map_.find(old_size);
    if (it != header_addr_map_.end()) {
      release_meta_mapping(it->second);
      header_addr_map_.erase(it);
    }
    header_ = old_header;
    footer_ = old_footer;
    segment_start_ = old_segment_start;
    segment_ids_offset_ = old_ids_offset;
    current_header_start_offset_ = old_header_offset;
    seg_meta_capacity_ = old_capacity;
    index_size_ = old_size;
    if (file_.size() > old_file_size && !file_.truncate(old_file_size)) {
      LOG_ERROR("Failed to roll back unlinked index extent in %s",
                path_.c_str());
    }
  });
  if (split) {
    // Persist earlier copy-on-write changes before writing a new file extent.
    if (copy_on_write_) {
      const int ret = flush();
      if (ret != 0) return ret;
    }
    old_file_size = file_.size();
    rollback_split = true;
    current_header_start_offset_ = old_size;
    const int ret = init_meta_section();
    if (ret != 0) return ret;
  }

  if (!copy_on_write_ && !file_.truncate(index_size_ + size)) {
    LOG_ERROR("Failed to truncate file, %s",
              ailego::FileHelper::GetLastErrorString().c_str());
    return IndexError_TruncateFile;
  }

  // Update segment table
  segment_ids_offset_ -= static_cast<uint32_t>(id_size);
  IndexFormat::SegmentMeta *segment = segment_start_ + footer_->segment_count;
  segment->segment_id_offset = segment_ids_offset_;
  segment->data_index =
      index_size_ - header_->content_offset - current_header_start_offset_;
  segment->data_size = 0;
  segment->data_crc = 0;
  segment->padding_size = size;
  memcpy((uint8_t *)segment_start_ + segment_ids_offset_, id.c_str(), id_size);
  index_size_ += size;

  // Update index footer
  footer_->segments_meta_crc =
      ailego::Crc32c::Hash(segment_start_, footer_->segments_meta_size, 0);
  footer_->segment_count += 1;
  footer_->content_size += size;
  footer_->total_size += size;
  IndexFormat::UpdateMetaFooter(footer_, 0);
  pending_node.key() = std::string_view(
      reinterpret_cast<const char *>(segment_start_) + segment_ids_offset_,
      id_size - 1);
  pending_node.mapped() =
      SegmentInfo{Segment{segment}, current_header_start_offset_, header_};
  if (segments_.empty()) {
    pending.insert(std::move(pending_node));
    segments_.swap(pending);
  } else {
    segments_.insert(std::move(pending_node));
  }
  if (split) {
    old_footer->next_meta_header_offset = current_header_start_offset_;
    IndexFormat::UpdateMetaFooter(old_footer, old_footer->check_point);
    rollback_split = false;
  }
  header_dirty_ = true;
  return 0;
} catch (const std::bad_alloc &) {
  return IndexError_NoMemory;
}

IndexMapping::Segment *IndexMapping::map(const std::string &id, bool warmup,
                                         bool locked) {
  auto iter = segments_.find(id);
  if (iter == segments_.end()) {
    return nullptr;
  }
  SegmentInfo &segment_info = iter->second;
  Segment *item = &segment_info.segment;
  if (!item->data()) {
    auto meta = item->meta();
    size_t mapping_size = meta->data_size + meta->padding_size;
    size_t offset = segment_info.segment_header_start_offset +
                    segment_info.segment_header->content_offset +
                    meta->data_index;

    void *addr = nullptr;
    if (!copy_on_write_) {
      int opts = ailego::File::MMAP_SHARED;
      if (huge_page_) {
        opts |= ailego::File::MMAP_HUGE_PAGE;
      }
      addr = ailego::File::MemoryMap(file_.native_handle(), offset,
                                     mapping_size, opts);
    } else {
      size_t file_size = file_.size();
      int opts = ailego::File::MMAP_POPULATE;
      if (huge_page_) {
        opts |= ailego::File::MMAP_HUGE_PAGE;
      }
      if (offset < file_size && mapping_size <= file_size - offset) {
        addr = ailego::File::MemoryMap(file_.native_handle(), offset,
                                       mapping_size, opts);
      } else {
        addr = ailego::File::MemoryMap(mapping_size, opts);
      }
    }

    if (!addr) {
      LOG_ERROR("Map segment failed, segment id %s", id.c_str());
      return nullptr;
    }
    item->set_data(addr, mapping_size);

    // Lock memory
    if (locked) {
      ailego::File::MemoryLock(item->data(), mapping_size);
    }
    // Warmup memory
    if (warmup && meta->data_size) {
      ailego::File::MemoryWarmup(item->data(), meta->data_size);
    }
  }
  return item;
}

void IndexMapping::unmap(const std::string &id) {
  auto iter = segments_.find(id);
  if (iter != segments_.end()) {
    SegmentInfo &segment_info = iter->second;
    Segment *item = &segment_info.segment;

    if (item->data()) {
      ailego::File::MemoryUnmap(item->data(), item->mapping_size_);
      item->set_data(nullptr, 0);
    }
  }
}

void IndexMapping::unmap_all() {
  for (auto iter = segments_.begin(); iter != segments_.end(); ++iter) {
    SegmentInfo &segment_info = iter->second;
    Segment *item = &segment_info.segment;

    if (item->data()) {
      ailego::File::MemoryUnmap(item->data(), item->mapping_size_);
      item->set_data(nullptr, 0);
    }
  }
}

int IndexMapping::flush() {
  if ((file_.size() < index_size_) && !file_.truncate(index_size_)) {
    LOG_ERROR("Failed to truncate file size %zu, %s", index_size_,
              ailego::FileHelper::GetLastErrorString().c_str());
    return IndexError_TruncateFile;
  }

  for (auto iter = segments_.begin(); iter != segments_.end(); ++iter) {
    SegmentInfo &segment_info = iter->second;
    Segment *item = &segment_info.segment;
    if (!item->data() || !item->dirty()) {
      continue;
    }

    const size_t segment_size = item->mapping_size_;
    if (full_mode_ && copy_on_write_) {
      size_t off = segment_info.segment_header_start_offset +
                   segment_info.segment_header->content_offset +
                   item->meta()->data_index;
      if (file_.write(off, item->data(), segment_size) != segment_size) {
        LOG_ERROR("Failed to write segment, size %zu, %s", segment_size,
                  ailego::FileHelper::GetLastErrorString().c_str());
        return IndexError_WriteData;
      }
    } else {
      if (!ailego::File::MemoryFlush(item->data(), segment_size))
        return IndexError_WriteData;
    }
    item->reset_dirty();
  }

  if (!header_dirty_) {
    return 0;
  }

  if (full_mode_ && copy_on_write_) {
    for (auto item : header_addr_map_) {
      auto header_start_offset = item.first;
      auto header = item.second.header;
      if (file_.write(header_start_offset, header, item.second.length) !=
          item.second.length) {
        LOG_ERROR("Failed to write segment, size %zu, %s", item.second.length,
                  ailego::FileHelper::GetLastErrorString().c_str());
        return IndexError_WriteData;
      }
    }
  } else {
    for (auto item : header_addr_map_) {
      auto header = item.second.header;
      if (!ailego::File::MemoryFlush(header, item.second.length))
        return IndexError_WriteData;
    }
  }
  header_dirty_ = false;
  return 0;
}

int IndexMapping::init_index_mapping(size_t len) {
  auto &pool = ailego::MemoryLimitPool::get_instance();
  const size_t charge = pool.capacity() ? len : 0;
  using Pending = ailego::MemoryLimitPool::PendingAllocation;
  Pending mapping_memory(pool, Pending::Kind::kWorkspace);
  if (!mapping_memory.reserve(charge)) return IndexError_NoMemory;
  HeaderMap pending;
  pending.emplace(current_header_start_offset_,
                  MetaMapping{nullptr, len, charge});
  auto pending_node = pending.extract(pending.begin());
  return init_index_mapping_admitted(len, std::move(pending_node),
                                     mapping_memory);
}

int IndexMapping::init_index_mapping_admitted(
    size_t len, HeaderMap::node_type pending_node,
    ailego::MemoryLimitPool::PendingAllocation &mapping_memory) {
  if (header_addr_map_.find(current_header_start_offset_) !=
      header_addr_map_.end())
    return IndexError_InvalidValue;
  // Both open and create enter here. A lazy directory may allocate its map
  // object/sentinel even when the node has already been admitted.
  header_addr_map_.initialize();
  // The managed header already has a pending reservation for the whole view.
  // MAP_POPULATE would run the separate optional prefault admission again.
  int opts = ailego::File::MMAP_SHARED;
  if (copy_on_write_) {
    opts = pending_node.mapped().charge ? 0 : ailego::File::MMAP_POPULATE;
  }
  if (huge_page_) {
    opts |= ailego::File::MMAP_HUGE_PAGE;
  }
  uint8_t *start = reinterpret_cast<uint8_t *>(ailego::File::MemoryMap(
      file_.native_handle(), current_header_start_offset_, len, opts));
  if (!start) {
    LOG_ERROR("Failed to map file, %s",
              ailego::FileHelper::GetLastErrorString().c_str());
    return IndexError_MMapFile;
  }

  // Keep the mapping independently owned until its directory entry exists.
  // A failed insertion must release the view as well as its committed charge.
  MetaMapping mapping = pending_node.mapped();
  mapping.header = reinterpret_cast<IndexFormat::MetaHeader *>(start);
  mapping.projection = mapping_memory.commit_mapping();
  bool registered = false;
  const auto rollback = ailego::ScopeGuard::Make([&] {
    if (!registered) release_meta_mapping(mapping);
  });
  pending_node.mapped() = mapping;
  if (!header_addr_map_.insert(std::move(pending_node)).inserted)
    return IndexError_InvalidValue;
  registered = true;
  header_ = mapping.header;

  // Unpack header
  if (header_->meta_header_size != sizeof(IndexFormat::MetaHeader)) {
    return IndexError_InvalidLength;
  }
  if (ailego::Crc32c::Hash(header_, sizeof(*header_), header_->header_crc) !=
      header_->header_crc) {
    return IndexError_InvalidChecksum;
  }

  switch (header_->version) {
    case IndexFormat::FORMAT_VERSION:
      break;
    default:
      LOG_ERROR("Unsupported index version: %u", header_->version);
      return IndexError_Unsupported;
  }

  // Unpack footer
  if (header_->meta_footer_size != sizeof(IndexFormat::MetaFooter)) {
    return IndexError_InvalidLength;
  }
  if ((int32_t)header_->meta_footer_offset < 0) {
    return IndexError_Unsupported;
  }
  size_t footer_offset = header_->meta_footer_offset;
  if (footer_offset + header_->meta_footer_size > len) {
    return IndexError_InvalidLength;
  }

  footer_ = reinterpret_cast<IndexFormat::MetaFooter *>(start + footer_offset);
  if (footer_offset < sizeof(*header_) ||
      footer_->segments_meta_size > footer_offset - sizeof(*header_) ||
      header_->content_offset < footer_offset + sizeof(*footer_))
    return IndexError_InvalidLength;

  seg_meta_capacity_ = footer_->segments_meta_size;
  const size_t file_size = file_.size();
  if (current_header_start_offset_ > file_size ||
      footer_->total_size > file_size - current_header_start_offset_ ||
      header_->content_offset > footer_->total_size ||
      footer_->content_size > footer_->total_size - header_->content_offset ||
      footer_->content_padding_size >
          footer_->total_size - header_->content_offset - footer_->content_size)
    return IndexError_InvalidLength;
  index_size_ = current_header_start_offset_ + footer_->total_size;
  if (ailego::Crc32c::Hash(footer_, sizeof(*footer_), footer_->footer_crc) !=
      footer_->footer_crc) {
    return IndexError_InvalidChecksum;
  }

  // Unpack segment table
  if (sizeof(IndexFormat::SegmentMeta) * footer_->segment_count >
      footer_->segments_meta_size) {
    return IndexError_InvalidLength;
  }

  segment_start_ = reinterpret_cast<IndexFormat::SegmentMeta *>(
      start + (footer_offset - footer_->segments_meta_size));
  if (ailego::Crc32c::Hash(segment_start_, footer_->segments_meta_size, 0u) !=
      footer_->segments_meta_crc) {
    LOG_ERROR("Index segments meta checksum is invalid.");
    return IndexError_InvalidChecksum;
  }

  segment_ids_offset_ = footer_->segments_meta_size;
  for (IndexFormat::SegmentMeta *iter = segment_start_,
                                *end = segment_start_ + footer_->segment_count;
       iter != end; ++iter) {
    if (iter->segment_id_offset >= footer_->segments_meta_size) {
      return IndexError_InvalidValue;
    }
    if (iter->data_index > footer_->content_size) {
      return IndexError_InvalidValue;
    }
    const size_t extent = footer_->total_size - header_->content_offset;
    if (iter->data_size > footer_->content_size - iter->data_index ||
        iter->padding_size > extent - iter->data_index - iter->data_size)
      return IndexError_InvalidLength;

    if (iter->segment_id_offset < segment_ids_offset_) {
      segment_ids_offset_ = iter->segment_id_offset;
    }
    const char *name = reinterpret_cast<const char *>(segment_start_) +
                       iter->segment_id_offset;
    const size_t max_length =
        footer_->segments_meta_size - iter->segment_id_offset;
    const size_t length = ::strnlen(name, max_length);
    if (length == max_length) return IndexError_InvalidValue;
    segments_.emplace(
        std::string_view(name, length),
        SegmentInfo{Segment{iter}, current_header_start_offset_, header_});
  }
  if (sizeof(IndexFormat::SegmentMeta) * footer_->segment_count >
      segment_ids_offset_) {
    return IndexError_InvalidLength;
  }

  check_point_ = std::max(check_point_, uint64_t{footer_->check_point});

  // if (header_->version == IndexFormat::COMPATIBLE_FORMAT_VERSION_0X0002) {
  //   header_->version = IndexFormat::CURRENT_FORMAT_VERSION;
  //   LOG_INFO("Index file format upgraded");
  //   IndexFormat::UpdateMetaHeader(header_);
  //   footer_->segments_meta_crc =
  //       ailego::Crc32c::Hash(segment_start_, footer_->segments_meta_size, 0);
  //   IndexFormat::UpdateMetaFooter(footer_, 0);
  //   header_dirty_ = true;
  // }

  return 0;
}

bool IndexMapping::ishugetlbfs(const std::string &path) const {
#ifdef __linux__
  struct statfs buf;
  if (statfs(path.c_str(), &buf) != 0) {
    const auto slash = path.find_last_of('/');
    const std::string parent =
        slash == std::string::npos ? "." : path.substr(0, slash ? slash : 1);
    if (statfs(parent.c_str(), &buf) != 0) return false;
  }
  return static_cast<unsigned long>(buf.f_type) == HUGETLBFS_MAGIC;
#else
  static_cast<void>(path);
  return false;
#endif
}

}  // namespace core
}  // namespace zvec
