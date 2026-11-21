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
#include <cstring>
#include <optional>
#include <ailego/pattern/defer.h>
#include <zvec/ailego/pattern/expected.hpp>
#include "inverted_codec.h"
#include "inverted_column_indexer.h"


namespace zvec {


void InvertedColumnIndexer::initialize_range_cache() const {
  std::call_once(range_cache_once_, [&]() {
    // Bound retained bitmap payload plus per-container and range metadata.
    // Large indexes or unusual range layouts retain the existing search path.
    constexpr size_t kMaxBytes = 32 * 1024 * 1024;
    constexpr size_t kStride = 64;
    constexpr size_t kMaxRanges = 4096;
    if ((uint64_t(max_id_) / 65536 + 1) * (8192 + 64) > kMaxBytes) {
      return;
    }
    auto cache = std::make_shared<RangeCache>();
    std::unique_ptr<rocksdb::Iterator> iter(
        ctx_.db_->NewIterator(ctx_.read_opts_, cf_ranges_));
    std::unique_ptr<roaring_bitmap_t, decltype(&roaring_bitmap_free)> prefix(
        roaring_bitmap_create(), roaring_bitmap_free);
    if (!prefix) {
      return;
    }
    size_t bytes = 0;
    uint64_t expected_count = 0;
    auto save_checkpoint = [&]() {
      roaring_bitmap_repair_after_lazy(prefix.get());
      // Subtracting complete ranges is valid only when their document sets
      // are disjoint. Fall back if a malformed or multiply indexed scalar
      // column contains the same document in different ranges.
      if (roaring_bitmap_get_cardinality(prefix.get()) != expected_count) {
        return false;
      }
      roaring_statistics_t stats;
      roaring_bitmap_statistics(prefix.get(), &stats);
      bytes += roaring_bitmap_size_in_bytes(prefix.get()) +
               size_t(stats.n_containers) * 64 + sizeof(RangeCache::Checkpoint);
      if (bytes > kMaxBytes) {
        return false;
      }
      auto *copy = roaring_bitmap_copy(prefix.get());
      if (!copy) {
        return false;
      }
      cache->checkpoints.push_back(
          {cache->ranges.size(), std::shared_ptr<const roaring_bitmap_t>(
                                     copy, [](const roaring_bitmap_t *p) {
                                       roaring_bitmap_free(
                                           const_cast<roaring_bitmap_t *>(p));
                                     })});
      return true;
    };
    if (!save_checkpoint()) {
      return;
    }
    for (iter->SeekToFirst(); iter->Valid(); iter->Next()) {
      const auto key = iter->key();
      if (cache->ranges.size() == kMaxRanges || key.size() < 9) {
        return;
      }
      uint64_t begin_size;
      std::memcpy(&begin_size, key.data() + key.size() - 8, 8);
      if (begin_size > key.size() - 9) {
        return;
      }
      RangeCache::Range range{key.ToString(),
                              std::string(key.data(), begin_size),
                              std::string(key.data() + begin_size + 1,
                                          key.size() - begin_size - 9)};
      if (range.begin > range.end ||
          (!cache->ranges.empty() && cache->ranges.back().end >= range.begin)) {
        return;
      }
      bytes += 2 * (range.key.size() + range.begin.size() + range.end.size() +
                    sizeof(RangeCache::Range));
      if (bytes > kMaxBytes) {
        return;
      }
      roaring_bitmap_t *part = nullptr;
      if (!InvertedIndexCodec::Deserialize(iter->value().data(),
                                           iter->value().size(), &part)
               .ok()) {
        return;
      }
      expected_count += roaring_bitmap_get_cardinality(part);
      roaring_bitmap_lazy_or_inplace(prefix.get(), part, true);
      roaring_bitmap_free(part);
      cache->ranges.emplace_back(std::move(range));
      if (cache->ranges.size() % kStride == 0 && !save_checkpoint()) {
        return;
      }
    }
    if (!iter->status().ok() || cache->ranges.empty()) {
      return;
    }
    if (cache->checkpoints.back().count != cache->ranges.size() &&
        !save_checkpoint()) {
      return;
    }
    range_cache_ = std::move(cache);
  });
}


Result<roaring_bitmap_t *> InvertedColumnIndexer::get_bitmap_lt_cached(
    const std::string &term, bool include_eq) const {
  const auto &cache = *range_cache_;
  // Only complete ranges are covered by a checkpoint. The boundary range is
  // still evaluated using its original term postings, including < versus <=.
  auto boundary = std::partition_point(
      cache.ranges.begin(), cache.ranges.end(),
      [&](const RangeCache::Range &r) {
        return cmp_lt(r.end.data(), r.end.size(), term.data(), term.size(),
                      include_eq);
      });
  const size_t count = boundary - cache.ranges.begin();
  auto checkpoint = std::lower_bound(
      cache.checkpoints.begin(), cache.checkpoints.end(), count,
      [](const RangeCache::Checkpoint &p, size_t n) { return p.count < n; });
  if (checkpoint != cache.checkpoints.begin() &&
      (checkpoint == cache.checkpoints.end() ||
       count - (checkpoint - 1)->count <= checkpoint->count - count)) {
    --checkpoint;
  }
  std::unique_ptr<roaring_bitmap_t, decltype(&roaring_bitmap_free)> bitmap(
      roaring_bitmap_copy(checkpoint->bitmap.get()), roaring_bitmap_free);
  if (!bitmap) {
    return tl::make_unexpected(Status::InternalError());
  }
  std::unique_ptr<rocksdb::Iterator> iter(
      ctx_.db_->NewIterator(ctx_.read_opts_, cf_ranges_));
  const size_t first = std::min(count, checkpoint->count);
  const size_t last = std::max(count, checkpoint->count);
  if (first < last) {
    iter->Seek(cache.ranges[first].key);
  }
  for (size_t i = first; i < last; ++i, iter->Next()) {
    if (!iter->Valid() || iter->key() != cache.ranges[i].key) {
      return tl::make_unexpected(Status::InternalError());
    }
    if (checkpoint->count <= count) {
      auto s = InvertedIndexCodec::merge_or(
          iter->value().data(), iter->value().size(), true, bitmap.get());
      if (!s.ok()) {
        return tl::make_unexpected(s);
      }
    } else {
      // Scalar ranges are disjoint. Removing complete ranges from the next
      // checkpoint gives the same prefix as adding them to the previous one.
      roaring_bitmap_t *remove = nullptr;
      auto s = InvertedIndexCodec::Deserialize(iter->value().data(),
                                               iter->value().size(), &remove);
      if (!s.ok()) {
        return tl::make_unexpected(s);
      }
      roaring_bitmap_andnot_inplace(bitmap.get(), remove);
      roaring_bitmap_free(remove);
    }
  }
  if (!iter->status().ok()) {
    return tl::make_unexpected(Status::InternalError());
  }
  if (boundary != cache.ranges.end()) {
    iter.reset(ctx_.db_->NewIterator(ctx_.read_opts_, cf_terms_));
    for (iter->Seek(boundary->begin); iter->Valid(); iter->Next()) {
      if (!cmp_lt(iter->key().data(), iter->key().size(), term.data(),
                  term.size(), include_eq)) {
        break;
      }
      auto s = InvertedIndexCodec::merge_or(
          iter->value().data(), iter->value().size(), true, bitmap.get());
      if (!s.ok()) {
        return tl::make_unexpected(s);
      }
    }
    if (!iter->status().ok()) {
      return tl::make_unexpected(Status::InternalError());
    }
  }
  roaring_bitmap_repair_after_lazy(bitmap.get());
  return bitmap.release();
}


Result<roaring_bitmap_t *> InvertedColumnIndexer::get_bitmap_eq(
    const std::string &term) const {
  PinnableSlice bitmap_slice;
  auto s = ctx_.db_->Get(ctx_.read_opts_, cf_terms_, term, &bitmap_slice);
  if (!s.ok()) {
    if (s.code() == rocksdb::Status::kNotFound) {
      return nullptr;
    }
    LOG_ERROR(
        "Failed to retrieve data for term[%s] from %s, code[%d], reason[%s]",
        term.c_str(), id().c_str(), s.code(), s.ToString().c_str());
    return tl::make_unexpected(Status::InternalError());
  }

  roaring_bitmap_t *bitmap{nullptr};
  Status status = InvertedIndexCodec::Deserialize(bitmap_slice.data(),
                                                  bitmap_slice.size(), &bitmap);
  if (status.ok()) {
    return bitmap;
  } else {
    LOG_ERROR(
        "Failed to deserialize bitmap for term[%s] from %s, bitmap size[%zu]",
        term.c_str(), id().c_str(), bitmap_slice.size());
    return tl::make_unexpected(Status::InternalError());
  }
}


Result<roaring_bitmap_t *> InvertedColumnIndexer::get_bitmap_contain(
    const std::vector<std::string> &terms, bool is_any) const {
  if (terms.empty()) {
    LOG_ERROR("Terms is empty");
    return tl::make_unexpected(Status::InvalidArgument());
  }

  // Shall we sort the terms here? Does it make any difference in performance?
  std::vector<Slice> slice_terms(terms.begin(), terms.end());
  std::vector<PinnableSlice> bitmap_slices;
  bitmap_slices.resize(terms.size());
  std::vector<rocksdb::Status> statuses;
  statuses.resize(terms.size());
  ctx_.db_->MultiGet(ctx_.read_opts_, cf_terms_, slice_terms.size(),
                     slice_terms.data(), bitmap_slices.data(), statuses.data(),
                     false);

  roaring_bitmap_t *bitmap{nullptr};
  Status s = Status::OK();
  AILEGO_DEFER([&]() {
    if (!s.ok() && bitmap) {
      roaring_bitmap_free(bitmap);
    }
  });

  auto init_or_merge_at_i = [&](size_t i) {
    if (bitmap == nullptr) {
      s = InvertedIndexCodec::Deserialize(bitmap_slices[i].data(),
                                          bitmap_slices[i].size(), &bitmap);
      if (!s.ok()) {
        LOG_ERROR("Failed to deserialize bitmap for term[%s] from %s",
                  terms[i].c_str(), id().c_str());
      }
      return;
    }

    if (is_any) {
      s = InvertedIndexCodec::merge_or(bitmap_slices[i].data(),
                                       bitmap_slices[i].size(), true, bitmap);
    } else {
      s = InvertedIndexCodec::merge_and(bitmap_slices[i].data(),
                                        bitmap_slices[i].size(), bitmap);
    }
    if (!s.ok()) {
      LOG_ERROR("Failed to merge bitmap for term[%s] from %s", terms[i].c_str(),
                id().c_str());
    }
  };

  for (size_t i = 0; i < terms.size(); i++) {
    if (statuses[i].ok()) {
      init_or_merge_at_i(i);
      if (!s.ok()) {
        return tl::make_unexpected(s);
      }
    } else if (statuses[i].code() == rocksdb::Status::kNotFound) {
      if (!is_any) {  // For contain_all, if any term is not found, return empty
        s = Status::NotFound();
        return nullptr;
      }
    } else {
      LOG_ERROR(
          "Failed to retrieve data for term[%s] from %s, code[%d], reason[%s]",
          terms[i].c_str(), id().c_str(), statuses[i].code(),
          statuses[i].ToString().c_str());
      s = Status::InternalError();
      return tl::make_unexpected(s);
    }
  }

  if (is_any && bitmap) {
    roaring_bitmap_repair_after_lazy(bitmap);
  }
  return bitmap;
}


Result<roaring_bitmap_t *> InvertedColumnIndexer::get_bitmap_ne(
    const std::string &term) const {
  if (sealed_) {
    auto ret = get_bitmap_eq(term);
    if (ret) {
      ret = flip_bitmap(ret.value());
    } else {
      LOG_ERROR("Failed to retrieve bitmap for term[%s] from %s", term.c_str(),
                id().c_str());
    }
    return ret;
  } else {
    roaring_bitmap_t *bitmap = roaring_bitmap_create();
    if (!bitmap) {
      LOG_ERROR("Failed to create bitmap");
      return tl::make_unexpected(Status::InternalError());
    }
    auto iter = ctx_.db_->NewIterator(ctx_.read_opts_, cf_terms_);
    AILEGO_DEFER([&]() { delete iter; });
    Status s;
    iter->SeekToFirst();
    while (iter->Valid()) {
      if (iter->key() == term) {
        iter->Next();
        continue;
      }
      s = InvertedIndexCodec::merge_or(iter->value().data(),
                                       iter->value().size(), true, bitmap);
      if (s.ok()) {
        iter->Next();
      } else {
        roaring_bitmap_free(bitmap);
        LOG_ERROR("Failed to merge bitmap from %s", id().c_str());
        return tl::make_unexpected(s);
      }
    }
    roaring_bitmap_repair_after_lazy(bitmap);
    return bitmap;
  }
}


Result<roaring_bitmap_t *> InvertedColumnIndexer::get_bitmap_not_contain(
    const std::vector<std::string> &terms, bool is_any) const {
  if (terms.empty()) {
    LOG_ERROR("Terms is empty");
    return tl::make_unexpected(Status::InvalidArgument());
  }

  auto non_null_result = get_bitmap_non_null();
  if (!non_null_result) {
    return non_null_result;
  }
  auto *non_null_bitmap = non_null_result.value();
  AILEGO_DEFER([&]() {
    if (non_null_bitmap) {
      roaring_bitmap_free(non_null_bitmap);
    }
  });

  auto ret = get_bitmap_contain(terms, is_any);
  if (ret) {
    if (ret.value() == nullptr) {
      ret = roaring_bitmap_create();
    }
    roaring_bitmap_flip_inplace(ret.value(), 0, max_id_ + 1);
  } else {
    LOG_ERROR("Failed to retrieve bitmap[%s] from %s, term size[%zu]",
              is_any ? "contain_any" : "contain_all", id().c_str(),
              terms.size());
    return ret;
  }

  roaring_bitmap_and_inplace(ret.value(), non_null_bitmap);
  return ret;
}


Result<roaring_bitmap_t *> InvertedColumnIndexer::get_bitmap_lt(
    const std::string &term, bool include_eq) const {
  if (field_.element_data_type() == DataType::BOOL) {
    LOG_ERROR("Bool type is not supported for range query");
    return tl::make_unexpected(Status::InternalError());
  }

  // For range queries that match most values, it's more efficient to compute
  // the result by getting the complement and flipping bits.
  if (range_covers_most_values(term, CompareOp::LT)) {
    auto ret = get_bitmap_gt(term, !include_eq);
    if (ret) {
      ret = flip_bitmap(ret.value());
    } else {
      LOG_ERROR("Failed to retrieve range bitmap for term[%s] from %s",
                term.c_str(), id().c_str());
    }
    return ret;
  }

  if (sealed_ && cf_ranges_) {
    initialize_range_cache();
    if (range_cache_) {
      return get_bitmap_lt_cached(term, include_eq);
    }
  }

  Status s = Status::OK();
  roaring_bitmap_t *bitmap = roaring_bitmap_create();
  if (!bitmap) {
    LOG_ERROR("Failed to create bitmap");
    return tl::make_unexpected(Status::InternalError());
  }
  AILEGO_DEFER([&]() {
    if (!s.ok()) {
      roaring_bitmap_free(bitmap);
    }
  });

  rocksdb::Iterator *iter_point, *iter_range;
  iter_point = ctx_.db_->NewIterator(ctx_.read_opts_, cf_terms_);
  iter_point->SeekToFirst();
  AILEGO_DEFER([&]() { delete iter_point; });
  if (sealed_ && cf_ranges_) {
    iter_range = ctx_.db_->NewIterator(ctx_.read_opts_, cf_ranges_);
    iter_range->SeekToFirst();
  } else {
    iter_range = nullptr;
  }
  AILEGO_DEFER([&]() {
    if (iter_range) {
      delete iter_range;
    }
  });

  bool lt;  // True if the current range or term is "less than" the search term

  // Process pre-aggregated range entries and merge matching bitmaps
  if (iter_range && iter_range->Valid()) {
    std::optional<std::string> point_seek_start_pos;
    // 1. Merge ranges where the end boundary is less than the search term
    while (iter_range->Valid()) {
      char *range_begin, *range_end;
      size_t range_begin_key_size, range_end_key_size;
      InvertedIndexCodec::decode_range_key(
          iter_range->key().data(), iter_range->key().size(), &range_begin,
          &range_begin_key_size, &range_end, &range_end_key_size);
      lt = cmp_lt(range_end, range_end_key_size, term.data(), term.length(),
                  include_eq);
      if (!lt) {
        point_seek_start_pos.emplace(range_begin, range_begin_key_size);
        break;
      }
      s = InvertedIndexCodec::merge_or(
          iter_range->value().data(), iter_range->value().size(), true, bitmap);
      if (!s.ok()) {
        LOG_ERROR("Failed to merge range bitmap from %s", id().c_str());
        return tl::make_unexpected(s);
      }
      iter_range->Next();
    }
    // 2. Change the start position of the point iterator
    if (point_seek_start_pos) {
      iter_point->Seek(*point_seek_start_pos);
      if (iter_point->Valid() && iter_point->key() != *point_seek_start_pos) {
        LOG_ERROR(
            "Failed to initialize the point iterator, seek_pos[%s], "
            "first_key_found[%s], term[%s]",
            (*point_seek_start_pos).c_str(),
            iter_point->key().ToStringView().data(), term.c_str());
        s = Status::InternalError();
        return tl::make_unexpected(s);
      }
    } else {
      iter_point->SeekToLast();
    }
  }

  // Process individual point entries
  while (iter_point->Valid()) {
    lt = cmp_lt(iter_point->key().data(), iter_point->key().size(), term.data(),
                term.size(), include_eq);
    if (!lt) {
      break;
    }
    s = InvertedIndexCodec::merge_or(iter_point->value().data(),
                                     iter_point->value().size(), true, bitmap);
    if (!s.ok()) {
      LOG_ERROR("Failed to merge range bitmap from %s", id().c_str());
      return tl::make_unexpected(s);
    }
    iter_point->Next();
  }

  roaring_bitmap_repair_after_lazy(bitmap);
  return bitmap;
}


Result<roaring_bitmap_t *> InvertedColumnIndexer::get_bitmap_gt(
    const std::string &term, bool include_eq) const {
  if (field_.element_data_type() == DataType::BOOL) {
    LOG_ERROR("Bool type is not supported for range query");
    return tl::make_unexpected(Status::InternalError());
  }

  // For range queries that match most values, it's more efficient to compute
  // the result by getting the complement and flipping bits.
  if (range_covers_most_values(term, CompareOp::GT)) {
    auto ret = get_bitmap_lt(term, !include_eq);
    if (ret) {
      ret = flip_bitmap(ret.value());
    } else {
      LOG_ERROR("Failed to retrieve range bitmap for term[%s] from %s",
                term.c_str(), id().c_str());
    }
    return ret;
  }

  Status s = Status::OK();
  roaring_bitmap_t *bitmap = roaring_bitmap_create();
  if (!bitmap) {
    LOG_ERROR("Failed to create bitmap");
    return tl::make_unexpected(Status::InternalError());
  }
  AILEGO_DEFER([&]() {
    if (!s.ok()) {
      roaring_bitmap_free(bitmap);
    }
  });

  rocksdb::Iterator *iter_point, *iter_range;
  iter_point = ctx_.db_->NewIterator(ctx_.read_opts_, cf_terms_);
  AILEGO_DEFER([&]() { delete iter_point; });
  if (sealed_ && cf_ranges_) {
    iter_range = ctx_.db_->NewIterator(ctx_.read_opts_, cf_ranges_);
  } else {
    iter_range = nullptr;
  }
  AILEGO_DEFER([&]() {
    if (iter_range) {
      delete iter_range;
    }
  });

  std::optional<std::string> point_seek_end_pos;

  // Process pre-aggregated range entries and merge matching bitmaps
  if (iter_range) {
    // 1. Seek to the first range entry that is greater than the search term
    iter_range->Seek(term);
    if (iter_range->Valid()) {
      char *range_begin, *range_end;
      size_t range_begin_key_size, range_end_key_size;
      InvertedIndexCodec::decode_range_key(
          iter_range->key().data(), iter_range->key().size(), &range_begin,
          &range_begin_key_size, &range_end, &range_end_key_size);
      int ret =
          cmp(range_begin, range_begin_key_size, term.data(), term.size());
      if (ret == 0 && !include_eq) {
        iter_range->Next();
        if (iter_range->Valid()) {
          InvertedIndexCodec::decode_range_key(
              iter_range->key().data(), iter_range->key().size(), &range_begin,
              &range_begin_key_size, &range_end, &range_end_key_size);
          point_seek_end_pos.emplace(range_begin, range_begin_key_size);
        }
      } else {
        point_seek_end_pos.emplace(range_begin, range_begin_key_size);
      }
    }
    // 2. Merge ranges where the begin boundary is greater than the search term
    while (iter_range->Valid()) {
      s = InvertedIndexCodec::merge_or(
          iter_range->value().data(), iter_range->value().size(), true, bitmap);
      if (!s.ok()) {
        LOG_ERROR("Failed to merge range bitmap from %s", id().c_str());
        return tl::make_unexpected(s);
      }
      iter_range->Next();
    }
  }

  // Process individual point entries
  iter_point->Seek(term);
  if (!include_eq) {
    if (iter_point->Valid() && iter_point->key() == term) {
      iter_point->Next();
    }
  }
  while (iter_point->Valid()) {
    if (point_seek_end_pos &&
        cmp(iter_point->key().data(), iter_point->key().size(),
            (*point_seek_end_pos).data(), (*point_seek_end_pos).size()) >= 0) {
      break;
    }
    s = InvertedIndexCodec::merge_or(iter_point->value().data(),
                                     iter_point->value().size(), true, bitmap);
    if (!s.ok()) {
      LOG_ERROR("Failed to merge range bitmap from %s", id().c_str());
      return tl::make_unexpected(s);
    }
    iter_point->Next();
  }

  roaring_bitmap_repair_after_lazy(bitmap);
  return bitmap;
}


Result<roaring_bitmap_t *> InvertedColumnIndexer::get_bitmap_array_len_eq(
    uint32_t len) const {
  std::string encoded_len = InvertedIndexCodec::Encode(
      std::string((char *)&len, sizeof(uint32_t)), DataType::UINT32);

  PinnableSlice bitmap_slice;
  auto rs =
      ctx_.db_->Get(ctx_.read_opts_, cf_array_len_, encoded_len, &bitmap_slice);
  if (!rs.ok()) {
    if (rs.code() == rocksdb::Status::kNotFound) {
      return nullptr;
    }
    LOG_ERROR(
        "Failed to retrieve data for len[%u] from %s, code[%d], reason[%s]",
        len, id().c_str(), rs.code(), rs.ToString().c_str());
    return tl::make_unexpected(Status::InternalError());
  }

  roaring_bitmap_t *bitmap{nullptr};
  Status status = InvertedIndexCodec::Deserialize(bitmap_slice.data(),
                                                  bitmap_slice.size(), &bitmap);
  if (status.ok()) {
    return bitmap;
  } else {
    LOG_ERROR(
        "Failed to deserialize bitmap for len[%u] from %s, bitmap size[%zu]",
        len, id().c_str(), bitmap_slice.size());
    return tl::make_unexpected(Status::InternalError());
  }
}


Result<roaring_bitmap_t *> InvertedColumnIndexer::get_bitmap_array_len_ne(
    uint32_t len) const {
  std::string encoded_len = InvertedIndexCodec::Encode(
      std::string((char *)&len, sizeof(uint32_t)), DataType::UINT32);

  roaring_bitmap_t *bitmap = roaring_bitmap_create();
  if (!bitmap) {
    LOG_ERROR("Failed to create bitmap");
    return tl::make_unexpected(Status::InternalError());
  }
  auto iter = ctx_.db_->NewIterator(ctx_.read_opts_, cf_array_len_);
  AILEGO_DEFER([&]() { delete iter; });
  Status s;
  iter->SeekToFirst();
  while (iter->Valid()) {
    if (iter->key() == encoded_len) {
      iter->Next();
      continue;
    }
    s = InvertedIndexCodec::merge_or(iter->value().data(), iter->value().size(),
                                     true, bitmap);
    if (s.ok()) {
      iter->Next();
    } else {
      roaring_bitmap_free(bitmap);
      LOG_ERROR("Failed to merge bitmap from %s", id().c_str());
      return tl::make_unexpected(s);
    }
  }
  roaring_bitmap_repair_after_lazy(bitmap);
  return bitmap;
}


Result<roaring_bitmap_t *> InvertedColumnIndexer::get_bitmap_array_len_lt(
    uint32_t len, bool include_eq) const {
  std::string encoded_len = InvertedIndexCodec::Encode(
      std::string((char *)&len, sizeof(uint32_t)), DataType::UINT32);

  roaring_bitmap_t *bitmap = roaring_bitmap_create();
  if (!bitmap) {
    LOG_ERROR("Failed to create bitmap");
    return tl::make_unexpected(Status::InternalError());
  }
  auto iter = ctx_.db_->NewIterator(ctx_.read_opts_, cf_array_len_);
  AILEGO_DEFER([&]() { delete iter; });
  Status s;
  iter->SeekToFirst();
  while (iter->Valid()) {
    bool lt = cmp_lt(iter->key().data(), iter->key().size(), encoded_len.data(),
                     encoded_len.size(), include_eq);
    if (!lt) {
      break;
    }
    s = InvertedIndexCodec::merge_or(iter->value().data(), iter->value().size(),
                                     true, bitmap);
    if (s.ok()) {
      iter->Next();
    } else {
      roaring_bitmap_free(bitmap);
      LOG_ERROR("Failed to merge bitmap from %s", id().c_str());
      return tl::make_unexpected(s);
    }
  }
  roaring_bitmap_repair_after_lazy(bitmap);
  return bitmap;
}


Result<roaring_bitmap_t *> InvertedColumnIndexer::get_bitmap_array_len_gt(
    uint32_t len, bool include_eq) const {
  std::string encoded_len = InvertedIndexCodec::Encode(
      std::string((char *)&len, sizeof(uint32_t)), DataType::UINT32);

  roaring_bitmap_t *bitmap = roaring_bitmap_create();
  if (!bitmap) {
    LOG_ERROR("Failed to create bitmap");
    return tl::make_unexpected(Status::InternalError());
  }
  auto iter = ctx_.db_->NewIterator(ctx_.read_opts_, cf_array_len_);
  AILEGO_DEFER([&]() { delete iter; });
  Status s;
  iter->Seek(encoded_len);
  if (!include_eq) {
    if (iter->Valid() && iter->key() == encoded_len) {
      iter->Next();
    }
  }
  while (iter->Valid()) {
    s = InvertedIndexCodec::merge_or(iter->value().data(), iter->value().size(),
                                     true, bitmap);
    if (s.ok()) {
      iter->Next();
    } else {
      roaring_bitmap_free(bitmap);
      LOG_ERROR("Failed to merge bitmap from %s", id().c_str());
      return tl::make_unexpected(s);
    }
  }
  roaring_bitmap_repair_after_lazy(bitmap);
  return bitmap;
}


Result<roaring_bitmap_t *> InvertedColumnIndexer::get_bitmap_like(
    std::string term) const {
  // convert to `is not null` if `%` is the only character
  if (term == "%") {
    return get_bitmap_non_null();
  }
  size_t percent_loc = std::string::npos;
  size_t size = 0;
  int percent_count = 0;
  // unescape \% and \_, detect % location
  for (size_t i = 0; i < term.size(); i++) {
    if (term[i] == '\\') {
      i++;
      if (i < term.size()) {
        term[size++] = term[i];
      }
      continue;
    }
    if (term[i] == '%') {
      percent_loc = size;
      percent_count += 1;
    }
    term[size++] = term[i];
  }
  term.resize(size);
  // convert to `=` filter if no percent
  if (percent_count == 0) {
    return get_bitmap_eq(term);
  } else if (percent_count != 1) {
    return tl::make_unexpected(Status::InvalidArgument(
        "like should have exactly one percent, unescaped:", term));
  }
  if (percent_loc == 0) {
    return get_bitmap_suffix(term.substr(1));
  } else if (percent_loc == size - 1) {
    return get_bitmap_prefix(term.substr(0, percent_loc));
  } else {
    std::string prefix = term.substr(0, percent_loc);
    std::string suffix = term.substr(percent_loc + 1, size - percent_loc - 1);
    return get_bitmap_prefix(prefix, suffix);
  }
}


Result<roaring_bitmap_t *> InvertedColumnIndexer::get_bitmap_prefix(
    const std::string &term, const std::string &suffix) const {
  auto iter = ctx_.db_->NewIterator(ctx_.read_opts_, cf_terms_);
  AILEGO_DEFER([&]() { delete iter; });

  roaring_bitmap_t *bitmap = roaring_bitmap_create();
  if (!bitmap) {
    LOG_ERROR("Failed to create bitmap");
    return tl::make_unexpected(Status::InternalError());
  }

  Status s;
  iter->Seek(term);
  while (iter->Valid()) {
    if (!has_prefix(iter->key().data(), iter->key().size(), term.data(),
                    term.size())) {
      break;
    }
    // Validate both literals on the same term; LIKE cannot overlap them.
    if (!suffix.empty() &&
        (iter->key().size() < term.size() + suffix.size() ||
         memcmp(iter->key().data() + iter->key().size() - suffix.size(),
                suffix.data(), suffix.size()) != 0)) {
      iter->Next();
      continue;
    }
    s = InvertedIndexCodec::merge_or(iter->value().data(), iter->value().size(),
                                     true, bitmap);
    if (!s.ok()) {
      roaring_bitmap_free(bitmap);
      LOG_ERROR("Failed to merge range bitmap from %s", id().c_str());
      return tl::make_unexpected(s);
    }
    iter->Next();
  }

  roaring_bitmap_repair_after_lazy(bitmap);
  return bitmap;
}


Result<roaring_bitmap_t *> InvertedColumnIndexer::get_bitmap_suffix(
    const std::string &term) const {
  if (!cf_reversed_terms_) {
    LOG_ERROR("%s doesn't support suffix matching", id().c_str());
    return tl::make_unexpected(Status::PermissionDenied());
  }

  auto iter = ctx_.db_->NewIterator(ctx_.read_opts_, cf_reversed_terms_);
  AILEGO_DEFER([&]() { delete iter; });

  roaring_bitmap_t *bitmap = roaring_bitmap_create();
  if (!bitmap) {
    LOG_ERROR("Failed to create bitmap");
    return tl::make_unexpected(Status::InternalError());
  }

  Status s;
  auto reversed_term = encode_reversed(term);
  iter->Seek(reversed_term);
  while (iter->Valid()) {
    if (!has_prefix(iter->key().data(), iter->key().size(),
                    reversed_term.data(), reversed_term.size())) {
      break;
    }
    s = InvertedIndexCodec::merge_or(iter->value().data(), iter->value().size(),
                                     true, bitmap);
    if (!s.ok()) {
      roaring_bitmap_free(bitmap);
      LOG_ERROR("Failed to merge range bitmap from %s", id().c_str());
      return tl::make_unexpected(s);
    }
    iter->Next();
  }

  roaring_bitmap_repair_after_lazy(bitmap);
  return bitmap;
}


Result<roaring_bitmap_t *> InvertedColumnIndexer::get_bitmap_null() const {
  return null_bitmap_.copy();
}


Result<roaring_bitmap_t *> InvertedColumnIndexer::get_bitmap_non_null() const {
  if (sealed_) {
    roaring_bitmap_t *bitmap = null_bitmap_.copy();
    roaring_bitmap_flip_inplace(bitmap, 0, max_id_ + 1);
    return bitmap;
  } else {
    Status s = Status::OK();
    // Empty arrays have a length entry but no term entries.
    auto *cf = field_.is_array_type() ? cf_array_len_ : cf_terms_;
    auto iter = ctx_.db_->NewIterator(ctx_.read_opts_, cf);
    AILEGO_DEFER([&]() { delete iter; });
    roaring_bitmap_t *bitmap = roaring_bitmap_create();
    if (!bitmap) {
      LOG_ERROR("Failed to create bitmap");
      return tl::make_unexpected(Status::InternalError());
    }
    iter->SeekToFirst();
    while (iter->Valid()) {
      s = InvertedIndexCodec::merge_or(iter->value().data(),
                                       iter->value().size(), true, bitmap);
      if (s.ok()) {
        iter->Next();
      } else {
        roaring_bitmap_free(bitmap);
        LOG_ERROR("Failed to merge bitmap from %s", id().c_str());
        return tl::make_unexpected(s);
      }
    }
    roaring_bitmap_repair_after_lazy(bitmap);
    return bitmap;
  }
}


InvertedSearchResult::Ptr InvertedColumnIndexer::search(
    const std::string &value, CompareOp op) const {
  if (field_.is_array_type()) {
    LOG_ERROR("%s: array type doesn't support single value search",
              id().c_str());
    return nullptr;
  }

  std::string encoded_value = encode(value);
  auto search_res = std::make_shared<InvertedSearchResult>();
  Result<roaring_bitmap_t *> bitmap_res;

  switch (op) {
    case CompareOp::EQ: {
      bitmap_res = get_bitmap_eq(encoded_value);
      break;
    }
    case CompareOp::NE: {
      bitmap_res = get_bitmap_ne(encoded_value);
      break;
    }
    case CompareOp::LT: {
      bitmap_res = get_bitmap_lt(encoded_value, false);
      break;
    }
    case CompareOp::LE: {
      bitmap_res = get_bitmap_lt(encoded_value, true);
      break;
    }
    case CompareOp::GT: {
      bitmap_res = get_bitmap_gt(encoded_value, false);
      break;
    }
    case CompareOp::GE: {
      bitmap_res = get_bitmap_gt(encoded_value, true);
      break;
    }
    case CompareOp::LIKE: {
      bitmap_res = get_bitmap_like(std::move(encoded_value));
      break;
    }
    case CompareOp::HAS_PREFIX: {
      bitmap_res = get_bitmap_prefix(std::move(encoded_value));
      break;
    }
    case CompareOp::HAS_SUFFIX: {
      bitmap_res = get_bitmap_suffix(std::move(encoded_value));
      break;
    }
    default:
      LOG_ERROR("%s: unsupported operator[%u]", id().c_str(),
                static_cast<uint32_t>(op));
      return nullptr;
  }

  if (bitmap_res) {
    search_res->set_and_own_bitmap(bitmap_res.value());
    return search_res;
  } else {
    LOG_ERROR("%s: failed to search, code[%d]", id().c_str(),
              static_cast<int>(bitmap_res.error().code()));
    return nullptr;
  }
}


InvertedSearchResult::Ptr InvertedColumnIndexer::multi_search(
    const std::vector<std::string> &values, CompareOp op) const {
  auto encoded_values = encode(values);
  auto search_res = std::make_shared<InvertedSearchResult>();
  Result<roaring_bitmap_t *> bitmap_res;

  switch (op) {
    case CompareOp::CONTAIN_ANY: {
      bitmap_res = get_bitmap_contain(encoded_values, true);
      break;
    }
    case CompareOp::CONTAIN_ALL: {
      bitmap_res = get_bitmap_contain(encoded_values, false);
      break;
    }
    case CompareOp::NOT_CONTAIN_ANY: {
      bitmap_res = get_bitmap_not_contain(encoded_values, true);
      break;
    }
    case CompareOp::NOT_CONTAIN_ALL: {
      bitmap_res = get_bitmap_not_contain(encoded_values, false);
      break;
    }
    default:
      LOG_ERROR("%s: unsupported operator[%u]", id().c_str(),
                static_cast<uint32_t>(op));
      return nullptr;
  }

  if (bitmap_res) {
    search_res->set_and_own_bitmap(bitmap_res.value());
    return search_res;
  } else {
    LOG_ERROR("%s: failed to search, code[%d]", id().c_str(),
              static_cast<int>(bitmap_res.error().code()));
    return nullptr;
  }
}


InvertedSearchResult::Ptr InvertedColumnIndexer::search_array_len(
    uint32_t len, CompareOp op) const {
  if (!field_.is_array_type()) {
    LOG_ERROR("%s: non-array type doesn't array length search", id().c_str());
    return nullptr;
  }

  auto search_res = std::make_shared<InvertedSearchResult>();
  Result<roaring_bitmap_t *> bitmap_res;

  switch (op) {
    case CompareOp::EQ: {
      bitmap_res = get_bitmap_array_len_eq(len);
      break;
    }
    case CompareOp::NE: {
      bitmap_res = get_bitmap_array_len_ne(len);
      break;
    }
    case CompareOp::LT: {
      bitmap_res = get_bitmap_array_len_lt(len, false);
      break;
    }
    case CompareOp::LE: {
      bitmap_res = get_bitmap_array_len_lt(len, true);
      break;
    }
    case CompareOp::GT: {
      bitmap_res = get_bitmap_array_len_gt(len, false);
      break;
    }
    case CompareOp::GE: {
      bitmap_res = get_bitmap_array_len_gt(len, true);
      break;
    }
    default:
      LOG_ERROR("%s: unsupported operator[%u]", id().c_str(),
                static_cast<uint32_t>(op));
      return nullptr;
  }

  if (bitmap_res) {
    search_res->set_and_own_bitmap(bitmap_res.value());
    return search_res;
  } else {
    LOG_ERROR("%s: failed to search, code[%d]", id().c_str(),
              static_cast<int>(bitmap_res.error().code()));
    return nullptr;
  }
}


InvertedSearchResult::Ptr InvertedColumnIndexer::search_null() const {
  auto search_res = std::make_shared<InvertedSearchResult>();
  auto bitmap_res = get_bitmap_null();
  if (bitmap_res) {
    search_res->set_and_own_bitmap(bitmap_res.value());
    return search_res;
  } else {
    LOG_ERROR("%s: failed to search, code[%d]", id().c_str(),
              static_cast<int>(bitmap_res.error().code()));
    return nullptr;
  }
}


InvertedSearchResult::Ptr InvertedColumnIndexer::search_non_null() const {
  auto search_res = std::make_shared<InvertedSearchResult>();
  auto bitmap_res = get_bitmap_non_null();
  if (bitmap_res) {
    search_res->set_and_own_bitmap(bitmap_res.value());
    return search_res;
  } else {
    LOG_ERROR("%s: failed to search, code[%d]", id().c_str(),
              static_cast<int>(bitmap_res.error().code()));
    return nullptr;
  }
}


Status InvertedColumnIndexer::evaluate_ratio(const std::string &value,
                                             CompareOp op, uint64_t *total_size,
                                             uint64_t *range_size) const {
  if (field_.is_array_type()) {
    LOG_ERROR("%s: array type doesn't support ratio evaluation", id().c_str());
    return Status::PermissionDenied();
  }

  if (sealed_ && doc_range_stat_) {
    std::string encoded_value = encode(value);
    doc_range_stat_->evaluate_ratio(encoded_value, op, total_size, range_size);
  } else {
    *range_size = 0;
    *total_size = 1;
  }
  return Status::OK();
}


inline Status InvertedColumnIndexer::estimate_range_ratio(
    const std::string &term, CompareOp op, uint64_t *total_count,
    uint64_t *matching_count) const {
  if (field_.is_array_type() || field_.element_data_type() == DataType::BOOL) {
    LOG_ERROR("%s: type[%d] doesn't support range ratio estimation",
              id().c_str(), (int)field_.data_type());
    return Status::PermissionDenied();
  }

  if (sealed_ && doc_range_stat_) {
    doc_range_stat_->evaluate_ratio(term, op, total_count, matching_count);
  } else {
    *matching_count = 0;
    *total_count = 1;
  }
  return Status::OK();
}


inline bool InvertedColumnIndexer::range_covers_most_values(
    const std::string &term, CompareOp op) const {
  constexpr float HIGH_SELECTIVITY_THRESHOLD = 0.7;

  // Estimation is only available for sealed indexes as they have the cumulative
  // distribution index.
  if (!sealed_) {
    return false;
  }

  uint64_t total_cnt{0}, matching_cnt{0};
  if (auto s = estimate_range_ratio(term, op, &total_cnt, &matching_cnt);
      s.ok()) {
    return (total_cnt != 0) &&
           ((1.0f * matching_cnt / total_cnt) > HIGH_SELECTIVITY_THRESHOLD);

  } else {
    return false;
  }
}


inline roaring_bitmap_t *InvertedColumnIndexer::flip_bitmap(
    roaring_bitmap_t *bitmap) const {
  roaring_bitmap_t *ret;
  if (ret = bitmap; ret == nullptr) {
    ret = null_bitmap_.copy();
  } else {
    roaring_bitmap_or_inplace(ret, null_bitmap_.bitmap());
  }
  roaring_bitmap_flip_inplace(ret, 0, max_id_ + 1);
  return ret;
}


}  // namespace zvec
