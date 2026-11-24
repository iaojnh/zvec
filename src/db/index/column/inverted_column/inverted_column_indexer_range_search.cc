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
#include <memory>
#include "inverted_column_indexer.h"

namespace zvec {
RangeQueryPlan InvertedColumnIndexer::plan_range(const std::string &term,
                                                 bool query_left,
                                                 bool include_eq) const {
  const auto *index = range_prefix_.get();
  RangeQueryPlan plan;
  // Every candidate builds either L or its complement R. GT is the complement
  // of LE, and GE is the complement of LT, including at a repeated value.
  plan.left_inclusive = query_left ? include_eq : !include_eq;
  if (!index) {
    const bool complement = range_covers_most_values(
        term, query_left ? CompareOp::LT : CompareOp::GT);
    const bool build_left = query_left != complement;
    plan.path =
        build_left ? RangeQueryPath::ScanLeft : RangeQueryPath::ScanRight;
    plan.flip = complement;
    return plan;
  }

  const auto &buckets = index->buckets;
  const auto matches_left = [&](const std::string &value) {
    return cmp_lt(value.data(), value.size(), term.data(), term.size(),
                  plan.left_inclusive);
  };
  // [0,left_end) consists of complete left buckets; [right_begin,n) of
  // complete right buckets. At most one bucket needs exact term processing.
  const size_t left_end =
      std::partition_point(buckets.begin(), buckets.end(),
                           [&](const InvertedRangePrefix::Bucket &bucket) {
                             return matches_left(bucket.end);
                           }) -
      buckets.begin();
  const size_t right_begin =
      std::partition_point(buckets.begin(), buckets.end(),
                           [&](const InvertedRangePrefix::Bucket &bucket) {
                             return matches_left(bucket.begin);
                           }) -
      buckets.begin();
  const size_t boundary = left_end < right_begin ? left_end : SIZE_MAX;
  // A range bitmap does not tell us how many distinct point postings lie on
  // either side of a boundary. Even a small change in the number of complete
  // buckets can otherwise select thousands more point seeks/merges. Preserve
  // the existing physical boundary side until finer term-cost statistics are
  // available; index use and the final complement are still chosen below.
  const bool legacy_build_left =
      query_left != range_covers_most_values(
                        term, query_left ? CompareOp::LT : CompareOp::GT);
  const auto boundary_side_allowed = [&](const RangeQueryPlan &candidate) {
    const bool builds_left_boundary =
        candidate.path == RangeQueryPath::ScanLeft ||
        candidate.path == RangeQueryPath::PrefixAdd;
    return boundary == SIZE_MAX || builds_left_boundary == legacy_build_left;
  };
  const auto sum_bytes = [&](size_t begin, size_t end) {
    return (end ? buckets[end - 1].cumulative_bytes : 0) -
           (begin ? buckets[begin - 1].cumulative_bytes : 0);
  };
  // This is a deterministic byte-work proxy, not a calibrated latency model.
  // A boundary is conservatively charged as its complete source bucket; using
  // zero would spuriously favor scanning a bucket containing a large term.
  const uint64_t boundary_bytes =
      boundary != SIZE_MAX ? buckets[boundary].bytes : 0;
  const uint64_t flip_bytes =
      ((static_cast<uint64_t>(max_id_.load()) + 64) / 64) * sizeof(uint64_t);
  auto make_plan = [&](RangeQueryPath path, size_t begin, size_t end,
                       size_t prefix_index, bool flip) {
    RangeQueryPlan candidate;
    candidate.path = path;
    candidate.prefix_index = prefix_index;
    candidate.range_begin = begin;
    candidate.range_end = end;
    candidate.boundary_bucket = boundary;
    candidate.left_inclusive = plan.left_inclusive;
    candidate.flip = flip;
    candidate.estimated_work = sum_bytes(begin, end) + boundary_bytes;
    if (prefix_index != SIZE_MAX) {
      candidate.estimated_work +=
          index->prefixes[prefix_index].serialized_bytes;
    }
    if (flip) {
      candidate.estimated_work += flip_bytes;
    }
    return candidate;
  };

  // Start from direct execution. Equal-cost ties prefer no final flip and no
  // subtraction; the proxy intentionally does not assign fitted weights.
  const bool initial_build_left =
      boundary == SIZE_MAX ? query_left : legacy_build_left;
  plan = initial_build_left ? make_plan(RangeQueryPath::ScanLeft, 0, left_end,
                                        SIZE_MAX, !query_left)
                            : make_plan(RangeQueryPath::ScanRight, right_begin,
                                        buckets.size(), SIZE_MAX, query_left);
  auto consider = [&](RangeQueryPlan candidate) {
    if (!boundary_side_allowed(candidate)) {
      return;
    }
    const auto penalty = [](const RangeQueryPlan &p) {
      return static_cast<unsigned>(p.flip) +
             static_cast<unsigned>(p.path == RangeQueryPath::PrefixSubtract);
    };
    if (candidate.estimated_work < plan.estimated_work ||
        (candidate.estimated_work == plan.estimated_work &&
         penalty(candidate) < penalty(plan))) {
      plan = candidate;
    }
  };
  consider(
      make_plan(RangeQueryPath::ScanLeft, 0, left_end, SIZE_MAX, !query_left));
  consider(make_plan(RangeQueryPath::ScanRight, right_begin, buckets.size(),
                     SIZE_MAX, query_left));

  // The prefix set is bounded. Considering all eligible prefixes also
  // accounts for a farther but substantially smaller run-compressed prefix.
  for (size_t i = 0; i < index->prefixes.size(); ++i) {
    const size_t end = index->prefixes[i].end_bucket;
    if (end <= left_end) {
      consider(
          make_plan(RangeQueryPath::PrefixAdd, end, left_end, i, !query_left));
    }
  }
  // The two halves of the boundary bucket can contain very different numbers
  // of point postings. Without reading them, do not switch to subtraction for
  // an apparent saving of only one boundary bucket's serialized bytes. This is
  // a conservative near-tie guard, not a bound on actual term-iteration time.
  const uint64_t best_non_subtract_work = plan.estimated_work;
  for (size_t i = 0; i < index->prefixes.size(); ++i) {
    const size_t end = index->prefixes[i].end_bucket;
    if (end >= right_begin) {
      auto candidate = make_plan(RangeQueryPath::PrefixSubtract, right_begin,
                                 end, i, !query_left);
      if (candidate.estimated_work < best_non_subtract_work &&
          best_non_subtract_work - candidate.estimated_work > boundary_bytes) {
        consider(candidate);
      }
    }
  }
  return plan;
}

Result<roaring_bitmap_t *> InvertedColumnIndexer::execute_range_plan(
    const std::string &term, const RangeQueryPlan &plan) const {
  const auto *index = range_prefix_.get();
  if (!index) {
    auto result = plan.path == RangeQueryPath::ScanLeft
                      ? get_bitmap_lt_raw(term, plan.left_inclusive)
                      : get_bitmap_gt_raw(term, !plan.left_inclusive);
    if (result && plan.flip) {
      result = flip_bitmap(result.value());
    }
    return result;
  }

  const bool subtract = plan.path == RangeQueryPath::PrefixSubtract;
  InvertedRangePrefix::BitmapPtr bitmap;
  if (plan.prefix_index == SIZE_MAX) {
    bitmap.reset(roaring_bitmap_create());
  } else {
    const auto &prefix = index->prefixes[plan.prefix_index];
    std::string value;
    const auto status =
        ctx_.db_->Get(ctx_.read_opts_, cf_prefixes_,
                      index->buckets[prefix.end_bucket - 1].key, &value);
    roaring_bitmap_t *raw = nullptr;
    if (status.IsNotFound()) {
      // A derived prefix may be absent in an older/partially persisted index.
      // The source postings remain authoritative.
      auto result = get_bitmap_lt_raw(term, plan.left_inclusive);
      if (result && plan.flip) result = flip_bitmap(result.value());
      return result;
    }
    if (!status.ok()) return tl::make_unexpected(Status::InternalError());
    auto decoded =
        InvertedIndexCodec::Deserialize(value.data(), value.size(), &raw);
    bitmap.reset(raw);
    if (!decoded.ok() || !bitmap ||
        roaring_bitmap_get_cardinality(bitmap.get()) !=
            index->buckets[prefix.end_bucket - 1].cumulative_cardinality ||
        roaring_bitmap_maximum(bitmap.get()) > max_id_.load() ||
        (null_bitmap_.bitmap() &&
         roaring_bitmap_intersect(bitmap.get(), null_bitmap_.bitmap()))) {
      return tl::make_unexpected(Status::InternalError());
    }
  }
  if (!bitmap) {
    return tl::make_unexpected(Status::InternalError());
  }
  bool needs_repair = false;
  auto merge = [&](const rocksdb::Slice &value) {
    roaring_bitmap_t *raw = nullptr;
    auto status =
        InvertedIndexCodec::Deserialize(value.data(), value.size(), &raw);
    InvertedRangePrefix::BitmapPtr posting(raw);
    if (!status.ok() || !posting) return Status::InternalError();
    if (subtract) {
      roaring_bitmap_andnot_inplace(bitmap.get(), posting.get());
    } else {
      roaring_bitmap_lazy_or_inplace(bitmap.get(), posting.get(), true);
      needs_repair = true;
    }
    return Status::OK();
  };

  if (plan.range_begin < plan.range_end) {
    std::unique_ptr<rocksdb::Iterator> ranges(
        ctx_.db_->NewIterator(ctx_.read_opts_, cf_ranges_));
    ranges->Seek(index->buckets[plan.range_begin].key);
    for (size_t i = plan.range_begin; i < plan.range_end; ++i) {
      if (!ranges->Valid() || ranges->key() != index->buckets[i].key) {
        return tl::make_unexpected(Status::InternalError());
      }
      auto status = merge(ranges->value());
      if (!status.ok()) {
        return tl::make_unexpected(status);
      }
      ranges->Next();
    }
    if (!ranges->status().ok()) {
      return tl::make_unexpected(Status::InternalError());
    }
  }

  if (plan.boundary_bucket != SIZE_MAX) {
    const auto &bucket = index->buckets[plan.boundary_bucket];
    const bool boundary_left = plan.path == RangeQueryPath::ScanLeft ||
                               plan.path == RangeQueryPath::PrefixAdd;
    std::unique_ptr<rocksdb::Iterator> terms(
        ctx_.db_->NewIterator(ctx_.read_opts_, cf_terms_));
    if (boundary_left) {
      terms->Seek(bucket.begin);
    } else {
      terms->Seek(term);
      if (plan.left_inclusive && terms->Valid() && terms->key() == term) {
        terms->Next();
      }
    }
    while (terms->Valid()) {
      const auto key = terms->key();
      if (cmp(key.data(), key.size(), bucket.end.data(), bucket.end.size()) >
              0 ||
          (boundary_left && !cmp_lt(key.data(), key.size(), term.data(),
                                    term.size(), plan.left_inclusive))) {
        break;
      }
      auto status = merge(terms->value());
      if (!status.ok()) {
        return tl::make_unexpected(status);
      }
      terms->Next();
    }
    if (!terms->status().ok()) {
      return tl::make_unexpected(Status::InternalError());
    }
  }

  // An exact persisted boundary needs no residual merge. Repair cardinality
  // only when lazy OR was used.
  if (needs_repair) {
    roaring_bitmap_repair_after_lazy(bitmap.get());
  }
  if (plan.flip) {
    flip_bitmap(bitmap.get());
  }
  return bitmap.release();
}

Result<roaring_bitmap_t *> InvertedColumnIndexer::get_bitmap_lt(
    const std::string &term, bool include_eq) const {
  if (field_.element_data_type() == DataType::BOOL) {
    return tl::make_unexpected(Status::InternalError());
  }
  const auto plan = plan_range(term, true, include_eq);
  return execute_range_plan(term, plan);
}

Result<roaring_bitmap_t *> InvertedColumnIndexer::get_bitmap_gt(
    const std::string &term, bool include_eq) const {
  if (field_.element_data_type() == DataType::BOOL) {
    return tl::make_unexpected(Status::InternalError());
  }
  const auto plan = plan_range(term, false, include_eq);
  return execute_range_plan(term, plan);
}

}  // namespace zvec
