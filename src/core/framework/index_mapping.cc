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

#include <zvec/ailego/io/mmap_file.h>
#include <zvec/ailego/logger/logger.h>
#include <zvec/ailego/utility/file_helper.h>
#include <zvec/core/framework/index_error.h>
#include <zvec/core/framework/index_mapping.h>
#include <zvec/core/framework/index_metadata.h>
#include "ailego/pattern/scope_guard.h"
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
  return (size + page_size - 1) / page_size * page_size;
}

static inline bool WritePadding(ailego::File &file, size_t size) {
  std::string padding(ailego::MemoryHelper::PageSize(), 0);
  for (size_t i = 0, count = size / padding.size(); i < count; ++i) {
    if (file.write(padding.data(), padding.size()) != padding.size()) {
      return false;
    }
  }
  padding.resize(size % padding.size());
  if (padding.size()) {
    if (file.write(padding.data(), padding.size()) != padding.size()) {
      return false;
    }
  }
  return true;
}

int IndexMapping::open(const std::string &path, bool cow, bool full_mode) try {
  IndexMapping candidate;
  candidate.path_ = path;
  candidate.full_mode_ = full_mode;
  candidate.copy_on_write_ = cow;
  candidate.huge_page_ = ishugetlbfs(path);

  const bool read_only = cow && !full_mode;
  if (!candidate.file_.open(path.c_str(), read_only, false)) {
    LOG_ERROR("Failed to open file %s, %s", path.c_str(),
              ailego::FileHelper::GetLastErrorString().c_str());
    return IndexError_OpenFile;
  }

  const size_t total = candidate.file_.size();
  size_t start = 0;
  do {
    IndexFormat::MetaHeader header;
    if (sizeof(header) > total - start) {
      return IndexError_InvalidLength;
    }
    if (candidate.file_.read(start, &header, sizeof(header)) !=
        sizeof(header)) {
      return IndexError_ReadData;
    }
    size_t footer_offset;
    int ret =
        index_metadata::FooterOffset(header, total - start, &footer_offset);
    if (ret != 0) {
      return ret;
    }
    if (static_cast<int32_t>(header.meta_footer_offset) < 0 ||
        header.version != IndexFormat::FORMAT_VERSION) {
      return IndexError_Unsupported;
    }
    candidate.current_header_start_offset_ = start;
    ret = candidate.init_index_mapping(footer_offset +
                                       sizeof(IndexFormat::MetaFooter));
    if (ret != 0) {
      return ret;
    }
    ret = index_metadata::NextSection(*candidate.footer_, start, total, &start);
    if (ret != 0) {
      return ret;
    }
  } while (start != 0);
  swap(candidate);
  return 0;
} catch (const std::bad_alloc &) {
  return IndexError_NoMemory;
}

void IndexMapping::swap(IndexMapping &rhs) noexcept {
  using std::swap;
  swap(segment_ids_offset_, rhs.segment_ids_offset_);
  swap(segment_start_, rhs.segment_start_);
  swap(header_, rhs.header_);
  header_addr_map_.swap(rhs.header_addr_map_);
  swap(footer_, rhs.footer_);
  segments_.swap(rhs.segments_);
  swap(index_size_, rhs.index_size_);
  swap(file_, rhs.file_);
  path_.swap(rhs.path_);
  swap(copy_on_write_, rhs.copy_on_write_);
  swap(full_mode_, rhs.full_mode_);
  swap(header_dirty_, rhs.header_dirty_);
  swap(huge_page_, rhs.huge_page_);
  swap(seg_meta_capacity_, rhs.seg_meta_capacity_);
  swap(current_header_start_offset_, rhs.current_header_start_offset_);
}

int IndexMapping::create(const std::string &path, size_t seg_meta_capacity) {
  path_ = path;
  seg_meta_capacity_ = seg_meta_capacity;
  current_header_start_offset_ = 0;

  // write() & copying to mmap() will auto extend the file size
  if (!file_.create(path.c_str(), 0)) {
    LOG_ERROR("Failed to create file %s, %s", path.c_str(),
              ailego::FileHelper::GetLastErrorString().c_str());
    return IndexError_CreateFile;
  }
  huge_page_ = ishugetlbfs(path);
  if (huge_page_) {
    return init_hugepage_meta_section();
  }
  return init_meta_section();
}

int IndexMapping::init_meta_section() {
  if (current_header_start_offset_ % ailego::MemoryHelper::PageSize() != 0) {
    LOG_ERROR("File offset %zu is not a multiple of the page size: %zu",
              (size_t)current_header_start_offset_,
              ailego::MemoryHelper::PageSize());
    return IndexError_InvalidValue;
  }

  auto &path = path_;
  size_t len =
      CalcPageAlignedSize(seg_meta_capacity_ + sizeof(IndexFormat::MetaHeader) +
                              sizeof(IndexFormat::MetaFooter),
                          false);

  IndexFormat::MetaHeader meta_header;
  IndexFormat::MetaFooter meta_footer;

  // Write index header
  IndexFormat::SetupMetaHeader(&meta_header, len - sizeof(meta_footer), len);
  if (!file_.seek(current_header_start_offset_, ailego::File::Origin::Begin)) {
    LOG_ERROR("Failed to seek file %s, %s", path.c_str(),
              ailego::FileHelper::GetLastErrorString().c_str());
    return IndexError_SeekFile;
  }
  if (file_.write(&meta_header, sizeof(meta_header)) != sizeof(meta_header)) {
    LOG_ERROR("Failed to write file: %s, %s", path.c_str(),
              ailego::FileHelper::GetLastErrorString().c_str());
    return IndexError_WriteData;
  }

  // Write padding data
  uint32_t segments_meta_size =
      static_cast<uint32_t>(len - (sizeof(meta_header) + sizeof(meta_footer)));
  if (!WritePadding(file_, segments_meta_size)) {
    LOG_ERROR("Failed to write file: %s, %s", path.c_str(),
              ailego::FileHelper::GetLastErrorString().c_str());
    return IndexError_WriteData;
  }

  // Write index footer
  IndexFormat::SetupMetaFooter(&meta_footer);
  meta_footer.segments_meta_size = segments_meta_size;
  meta_footer.total_size = len;
  IndexFormat::UpdateMetaFooter(&meta_footer, 0);
  if (file_.write(&meta_footer, sizeof(meta_footer)) != sizeof(meta_footer)) {
    LOG_ERROR("Failed to write file: %s, %s", path.c_str(),
              ailego::FileHelper::GetLastErrorString().c_str());
    return IndexError_WriteData;
  }
  return this->init_index_mapping(len);
}

int IndexMapping::init_hugepage_meta_section() {
  ssize_t file_offset = (ssize_t)current_header_start_offset_;
  if (file_offset % ailego::MemoryHelper::HugePageSize() != 0) {
    LOG_ERROR("File offset %zu is not a multiple of the page size: %zu",
              file_offset, ailego::MemoryHelper::HugePageSize());
    return IndexError_InvalidValue;
  }

  size_t len =
      CalcPageAlignedSize(seg_meta_capacity_ + sizeof(IndexFormat::MetaHeader) +
                              sizeof(IndexFormat::MetaFooter),
                          true);
  int opts = ailego::File::MMAP_SHARED | ailego::File::MMAP_HUGE_PAGE;
  void *addr =
      ailego::File::MemoryMap(file_.native_handle(), file_offset, len, opts);

  IndexFormat::MetaHeader meta_header;
  IndexFormat::MetaFooter meta_footer;

  // Write index header
  IndexFormat::SetupMetaHeader(&meta_header, len - sizeof(meta_footer), len);
  memcpy((char *)addr + file_offset, &meta_header, sizeof(meta_header));
  file_offset += sizeof(meta_header);

  // Write padding data
  uint32_t segments_meta_size =
      static_cast<uint32_t>(len - (sizeof(meta_header) + sizeof(meta_footer)));
  std::string padding(ailego::MemoryHelper::HugePageSize(), 0);
  for (size_t i = 0, count = segments_meta_size / padding.size(); i < count;
       ++i) {
    memcpy((char *)addr + file_offset, padding.data(), padding.size());
    file_offset += padding.size();
  }
  padding.resize(segments_meta_size % padding.size());
  if (padding.size()) {
    memcpy((char *)addr + file_offset, padding.data(), padding.size());
    file_offset += padding.size();
  }

  // Write index footer
  IndexFormat::SetupMetaFooter(&meta_footer);
  meta_footer.segments_meta_size = segments_meta_size;
  meta_footer.total_size = len;
  IndexFormat::UpdateMetaFooter(&meta_footer, 0);
  memcpy((char *)addr + file_offset, &meta_footer, sizeof(meta_footer));
  file_offset += sizeof(meta_footer);

  return this->init_index_mapping(len);
}

void IndexMapping::close() {
  // Unmap all memory
  this->unmap_all();
  for (auto item : header_addr_map_) {
    ailego::File::MemoryUnmap(item.second.header, item.second.length);
  }
  // Reset members
  segment_ids_offset_ = 0;
  segment_start_ = nullptr;
  header_ = nullptr;
  header_addr_map_.clear();
  footer_ = nullptr;
  index_size_ = 0u;
  segments_.clear();
  file_.close();
  copy_on_write_ = false;
  full_mode_ = false;
  header_dirty_ = false;
  huge_page_ = false;
  current_header_start_offset_ = 0;
}

void IndexMapping::refresh(uint64_t check_point) {
  // support add_with_id
  for (auto item : header_addr_map_) {
    auto header = item.second.header;
    auto footer = reinterpret_cast<IndexFormat::MetaFooter *>(
        reinterpret_cast<uint8_t *>(header) + header->meta_footer_offset);
    auto segment_start = reinterpret_cast<IndexFormat::SegmentMeta *>(
        reinterpret_cast<uint8_t *>(header) +
        (header->meta_footer_offset - footer->segments_meta_size));
    footer->segments_meta_crc =
        ailego::Crc32c::Hash(segment_start, footer->segments_meta_size, 0);
    IndexFormat::UpdateMetaFooter(footer, check_point);
  }
  header_dirty_ = true;
}

int IndexMapping::append(const std::string &id, size_t size) {
  size = CalcPageAlignedSize(size, huge_page_);
  if (size == 0) {
    return IndexError_InvalidArgument;
  }

  if (segments_.find(id) != segments_.end()) {
    return IndexError_Duplicate;
  }

  size_t id_size = std::strlen(id.c_str()) + 1;
  size_t need_size = sizeof(IndexFormat::SegmentMeta) + id_size;
  if (sizeof(IndexFormat::SegmentMeta) * footer_->segment_count + need_size >
      segment_ids_offset_) {
    LOG_DEBUG("segment meta section expanded: %s", path_.c_str());
    footer_->next_meta_header_offset = index_size_;
    refresh(0);
    flush();
    // mmap file storage write() will update segment's meta
    // ailego::File::MemoryUnmap(header_, header_->content_offset);
    header_ = nullptr;
    footer_ = nullptr;

    current_header_start_offset_ = index_size_;
    const int ret =
        huge_page_ ? init_hugepage_meta_section() : init_meta_section();
    if (ret != 0) {
      return ret;
    }
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
  segments_.emplace(
      id, SegmentInfo{Segment{segment}, current_header_start_offset_, header_});
  header_dirty_ = true;
  return 0;
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
      if (offset < file_size) {
        ailego_assert(offset + mapping_size <= file_size);
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
    item->set_data(addr);

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
      ailego::File::MemoryUnmap(
          item->data(), item->meta()->data_size + item->meta()->padding_size);
      item->set_data(nullptr);
    }
  }
}

void IndexMapping::unmap_all() {
  for (auto iter = segments_.begin(); iter != segments_.end(); ++iter) {
    SegmentInfo &segment_info = iter->second;
    Segment *item = &segment_info.segment;

    if (item->data()) {
      ailego::File::MemoryUnmap(
          item->data(), item->meta()->data_size + item->meta()->padding_size);
      item->set_data(nullptr);
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

    size_t segment_size = item->meta()->data_size + item->meta()->padding_size;
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
      ailego::File::MemoryFlush(item->data(), segment_size);
    }
    item->reset_dirty();
  }

  if (!header_dirty_) {
    return 0;
  }

  header_dirty_ = false;
  if (full_mode_ && copy_on_write_) {
    for (auto item : header_addr_map_) {
      auto header_start_offset = item.first;
      auto header = item.second.header;
      const size_t length = item.second.length;
      if (file_.write(header_start_offset, header, length) != length) {
        LOG_ERROR("Failed to write segment, size %zu, %s", length,
                  ailego::FileHelper::GetLastErrorString().c_str());
        return IndexError_WriteData;
      }
    }
  } else {
    for (auto item : header_addr_map_) {
      ailego::File::MemoryFlush(item.second.header, item.second.length);
    }
  }
  return 0;
}

int IndexMapping::init_index_mapping(size_t len) {
  int opts =
      copy_on_write_ ? ailego::File::MMAP_POPULATE : ailego::File::MMAP_SHARED;
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

  bool owns_mapping = true;
  const auto unmap = ailego::ScopeGuard::Make([&] {
    if (owns_mapping) {
      ailego::File::MemoryUnmap(start, len);
    }
  });
  auto *header = reinterpret_cast<IndexFormat::MetaHeader *>(start);
  index_size_ = file_.size();
  if (current_header_start_offset_ > index_size_ ||
      len > index_size_ - current_header_start_offset_) {
    return IndexError_InvalidLength;
  }
  const size_t remaining = index_size_ - current_header_start_offset_;
  size_t footer_offset;
  int ret = index_metadata::FooterOffset(*header, remaining, &footer_offset);
  if (ret != 0) {
    return ret;
  }
  if (header->version != IndexFormat::FORMAT_VERSION ||
      static_cast<int32_t>(header->meta_footer_offset) < 0) {
    return IndexError_Unsupported;
  }
  if (footer_offset % alignof(IndexFormat::MetaFooter) != 0 ||
      footer_offset + sizeof(IndexFormat::MetaFooter) != len ||
      header->content_offset != len) {
    return IndexError_InvalidLength;
  }
  auto *footer =
      reinterpret_cast<IndexFormat::MetaFooter *>(start + footer_offset);
  ret = index_metadata::CheckFooter(*header, *footer, footer_offset, remaining);
  if (ret != 0) {
    return ret;
  }
  const size_t table_offset = footer_offset - footer->segments_meta_size;
  if (table_offset % alignof(IndexFormat::SegmentMeta) != 0) {
    return IndexError_InvalidLength;
  }

  auto *table =
      reinterpret_cast<IndexFormat::SegmentMeta *>(start + table_offset);
  ret = index_metadata::CheckTable(*footer, table);
  if (ret != 0) {
    return ret;
  }

  if (!header_addr_map_
           .try_emplace(current_header_start_offset_, MetaMapping{header, len})
           .second) {
    return IndexError_Duplicate;
  }
  owns_mapping = false;
  uint32_t ids_offset = footer->segments_meta_size;
  for (size_t i = 0; i < footer->segment_count; ++i) {
    std::string_view id;
    ret = index_metadata::SegmentId(*footer, table, table[i], &id);
    if (ret != 0) {
      return ret;
    }
    ids_offset = std::min(ids_offset, table[i].segment_id_offset);
    if (!segments_
             .try_emplace(std::string(id),
                          SegmentInfo{Segment{table + i},
                                      current_header_start_offset_, header})
             .second) {
      return IndexError_Duplicate;
    }
  }
  header_ = header;
  footer_ = footer;
  segment_start_ = table;
  segment_ids_offset_ = ids_offset;
  seg_meta_capacity_ = footer->segments_meta_size;
  return 0;
}

bool IndexMapping::ishugetlbfs(const std::string &path) const {
#ifdef __linux__
  struct statfs buf;
  if (statfs(path.c_str(), &buf) != 0) {
    perror("statfs");
    return false;
  }
  return static_cast<unsigned long>(buf.f_type) == HUGETLBFS_MAGIC;
#else
  static_cast<void>(path);
  return false;
#endif
}

}  // namespace core
}  // namespace zvec
