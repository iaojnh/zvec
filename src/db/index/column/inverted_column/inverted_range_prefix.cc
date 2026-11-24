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

#include "inverted_range_prefix.h"
#include <cstring>
#include <zvec/ailego/encoding/json/mod_json_plus.h>
#include "inverted_codec.h"

namespace zvec {
namespace {

bool decode_bucket(const std::string &key,
                   InvertedRangePrefix::Bucket &bucket) {
  if (key.size() < 9) return false;
  uint64_t size = 0;
  std::memcpy(&size, key.data() + key.size() - 8, 8);
  if (size > key.size() - 9 || key[size] != '\0') return false;
  bucket.key = key;
  bucket.begin.assign(key.data(), size);
  bucket.end.assign(key.data() + size + 1, key.size() - size - 9);
  return InvertedIndexCodec::CMP(bucket.begin.data(), bucket.begin.size(),
                                 bucket.end.data(), bucket.end.size()) <= 0;
}

bool follows(const InvertedRangePrefix &index,
             const InvertedRangePrefix::Bucket &bucket) {
  return index.buckets.empty() ||
         InvertedIndexCodec::CMP(index.buckets.back().end.data(),
                                 index.buckets.back().end.size(),
                                 bucket.begin.data(), bucket.begin.size()) < 0;
}

bool get_uint(const ailego::JsonObject &object, const char *name,
              uint64_t &value) {
  ailego::JsonValue number;
  if (!object.get(name, &number) || !number.is_integer() ||
      number.as_integer() < 0)
    return false;
  value = number.as_integer();
  return true;
}

}  // namespace

Status InvertedRangePrefix::Build(RocksdbContext &context,
                                  rocksdb::ColumnFamilyHandle *ranges,
                                  rocksdb::ColumnFamilyHandle *prefixes,
                                  uint32_t max_id,
                                  const roaring_bitmap_t *null_bitmap) {
  InvertedRangePrefix index;
  BitmapPtr cumulative(roaring_bitmap_create());
  if (!cumulative) return Status::InternalError();
  rocksdb::ReadOptions options = context.read_opts_;
  options.fill_cache = false;
  std::unique_ptr<rocksdb::Iterator> iter(
      context.db_->NewIterator(options, ranges));
  uint64_t source_bytes = 0;
  for (iter->SeekToFirst(); iter->Valid(); iter->Next()) {
    Bucket bucket;
    if (!decode_bucket(iter->key().ToString(), bucket) ||
        !follows(index, bucket)) {
      // The existing range-key encoding need not preserve bucket order for
      // binary strings containing NULs. Keep the optional index disabled,
      // as #881 does, rather than make a previously valid seal fail.
      return Status::OK();
    }
    roaring_bitmap_t *raw = nullptr;
    auto status = InvertedIndexCodec::Deserialize(iter->value().data(),
                                                  iter->value().size(), &raw);
    BitmapPtr bitmap(raw);
    if (!status.ok() || !bitmap) return Status::InternalError();
    bucket.cardinality = roaring_bitmap_get_cardinality(bitmap.get());
    bucket.bytes = iter->value().size();
    index.total_cardinality += bucket.cardinality;
    source_bytes += bucket.bytes;
    bucket.cumulative_cardinality = index.total_cardinality;
    bucket.cumulative_bytes = source_bytes;
    index.buckets.push_back(std::move(bucket));
    roaring_bitmap_or_inplace(cumulative.get(), bitmap.get());
  }
  if (!iter->status().ok()) return Status::InternalError();
  const uint64_t null_count =
      null_bitmap ? roaring_bitmap_get_cardinality(null_bitmap) : 0;
  // Prefix planning may choose complements. Prove full, disjoint DocID domain
  // coverage first, so it cannot create matches for gaps or NULLs.
  if (index.buckets.size() < 2 || !index.total_cardinality ||
      index.total_cardinality + null_count != uint64_t(max_id) + 1 ||
      roaring_bitmap_get_cardinality(cumulative.get()) !=
          index.total_cardinality ||
      roaring_bitmap_maximum(cumulative.get()) > max_id ||
      (null_count &&
       (roaring_bitmap_maximum(null_bitmap) > max_id ||
        roaring_bitmap_intersect(cumulative.get(), null_bitmap)))) {
    return Status::OK();
  }

  roaring_bitmap_clear(cumulative.get());
  size_t bucket_index = 0;
  unsigned quantile = 1;
  for (iter->SeekToFirst(); iter->Valid(); iter->Next(), ++bucket_index) {
    if (bucket_index >= index.buckets.size() ||
        iter->key() != index.buckets[bucket_index].key)
      return Status::InternalError();
    roaring_bitmap_t *raw = nullptr;
    auto status = InvertedIndexCodec::Deserialize(iter->value().data(),
                                                  iter->value().size(), &raw);
    BitmapPtr bitmap(raw);
    if (!status.ok() || !bitmap) return Status::InternalError();
    roaring_bitmap_or_inplace(cumulative.get(), bitmap.get());
    const uint64_t cardinality =
        index.buckets[bucket_index].cumulative_cardinality;
    if (quantile > 15 ||
        cardinality < (index.total_cardinality * quantile + 15) / 16)
      continue;
    do {
      ++quantile;
    } while (quantile <= 15 &&
             cardinality >= (index.total_cardinality * quantile + 15) / 16);
    // A heavy term may cross several quantiles: store its boundary only once.
    if (bucket_index + 1 == index.buckets.size()) continue;
    std::string value;
    status = InvertedIndexCodec::Serialize(cumulative.get(), &value);
    if (!status.ok()) return status;
    const auto stored = context.db_->Put(
        context.write_opts_, prefixes, index.buckets[bucket_index].key, value);
    if (!stored.ok()) return Status::InternalError();
    index.prefixes.push_back({bucket_index + 1, value.size()});
  }
  if (!iter->status().ok() || bucket_index != index.buckets.size())
    return Status::InternalError();
  if (index.prefixes.empty()) return Status::OK();

  ailego::JsonArray buckets_json, prefixes_json;
  for (const auto &bucket : index.buckets) {
    ailego::JsonObject item;
    item.set("key", ailego::JsonString(bucket.key).encode());
    item.set("cardinality", ailego::JsonValue(bucket.cardinality));
    item.set("bytes", ailego::JsonValue(bucket.bytes));
    buckets_json.push(item);
  }
  for (const auto &prefix : index.prefixes) {
    ailego::JsonObject item;
    item.set("end_bucket", ailego::JsonValue(uint64_t(prefix.end_bucket)));
    item.set("bytes", ailego::JsonValue(prefix.serialized_bytes));
    prefixes_json.push(item);
  }
  ailego::JsonObject metadata;
  metadata.set("version", ailego::JsonValue(int64_t{1}));
  metadata.set("max_id", ailego::JsonValue(uint64_t(max_id)));
  metadata.set("null_count", ailego::JsonValue(null_count));
  metadata.set("buckets", buckets_json);
  metadata.set("prefixes", prefixes_json);
  // The empty key cannot be a range key. Publish metadata last; readers only
  // use it after the indexer's sealed marker is visible.
  const auto stored = context.db_->Put(
      context.write_opts_, prefixes, "",
      ailego::JsonValue(std::move(metadata)).as_json_string().as_stl_string());
  return stored.ok() ? Status::OK() : Status::InternalError();
}

std::unique_ptr<InvertedRangePrefix> InvertedRangePrefix::Load(
    RocksdbContext &context, rocksdb::ColumnFamilyHandle *family,
    uint32_t max_id, const roaring_bitmap_t *null_bitmap) {
  if (!family || (null_bitmap && !roaring_bitmap_is_empty(null_bitmap) &&
                  roaring_bitmap_maximum(null_bitmap) > max_id))
    return nullptr;
  std::string value;
  if (!context.db_->Get(context.read_opts_, family, "", &value).ok())
    return nullptr;
  ailego::JsonValue json;
  ailego::JsonParser parser;
  if (!parser.parse(value.c_str(), &json) || !json.is_object()) return nullptr;
  const auto object = json.as_object();
  uint64_t version, stored_max, stored_null;
  if (!get_uint(object, "version", version) || version != 1 ||
      !get_uint(object, "max_id", stored_max) || stored_max != max_id ||
      !get_uint(object, "null_count", stored_null) ||
      stored_null !=
          (null_bitmap ? roaring_bitmap_get_cardinality(null_bitmap) : 0))
    return nullptr;
  ailego::JsonArray buckets_json, prefixes_json;
  if (!object.get("buckets", &buckets_json) ||
      !object.get("prefixes", &prefixes_json) || buckets_json.empty() ||
      prefixes_json.empty() || prefixes_json.size() > 15)
    return nullptr;
  auto index = std::make_unique<InvertedRangePrefix>();
  uint64_t source_bytes = 0;
  for (const auto &entry : buckets_json) {
    Bucket bucket;
    ailego::JsonString key;
    if (!entry.is_object() || !entry.as_object().get("key", &key) ||
        !decode_bucket(key.decode().as_stl_string(), bucket) ||
        !follows(*index, bucket) ||
        !get_uint(entry.as_object(), "cardinality", bucket.cardinality) ||
        !get_uint(entry.as_object(), "bytes", bucket.bytes) || !bucket.bytes ||
        bucket.cardinality > uint64_t(max_id) + 1 - index->total_cardinality ||
        bucket.bytes > UINT64_MAX - source_bytes)
      return nullptr;
    index->total_cardinality += bucket.cardinality;
    source_bytes += bucket.bytes;
    bucket.cumulative_cardinality = index->total_cardinality;
    bucket.cumulative_bytes = source_bytes;
    index->buckets.push_back(std::move(bucket));
  }
  if (index->total_cardinality + stored_null != uint64_t(max_id) + 1)
    return nullptr;
  size_t previous = 0;
  for (const auto &entry : prefixes_json) {
    uint64_t end, bytes;
    if (!entry.is_object() || !get_uint(entry.as_object(), "end_bucket", end) ||
        !get_uint(entry.as_object(), "bytes", bytes) || !bytes ||
        end <= previous || end >= index->buckets.size())
      return nullptr;
    index->prefixes.push_back({size_t(end), bytes});
    previous = end;
  }
  return index;
}

}  // namespace zvec
