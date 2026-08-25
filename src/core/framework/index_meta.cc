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

#include <cstring>
#include <limits>
#include <zvec/ailego/encoding/json.h>
#include <zvec/core/framework/index_meta.h>

namespace zvec {
namespace core {

/*! Index Meta Buffer Format
 */
struct IndexMetaFormatHeader {
  uint32_t header_size;
  uint32_t meta_type;
  uint32_t major_order;
  uint32_t data_type;
  uint32_t dimension;
  uint32_t unit_size;
  uint32_t space_id;
  uint32_t attachment_offset;
  uint32_t attachment_size;
  uint32_t extra_meta_size;
  uint8_t reserved_[4088];
};

static_assert(sizeof(IndexMetaFormatHeader) % 32 == 0,
              "IndexMetaBufferFormat must be aligned with 32 bytes");

namespace {

bool ComputeElementSize(uint32_t data_type, uint32_t unit_size,
                        uint32_t dimension, uint32_t extra_meta_size,
                        uint32_t *element_size) {
  if (data_type > static_cast<uint32_t>(IndexMeta::DataType::DT_BINARY64)) {
    return false;
  }

  const auto type = static_cast<IndexMeta::DataType>(data_type);
  const uint32_t expected_unit_size = IndexMeta::UnitSizeof(type);
  if (unit_size != expected_unit_size) {
    return false;
  }

  uint64_t base_size = 0;
  switch (type) {
    case IndexMeta::DataType::DT_UNDEFINED:
      break;
    case IndexMeta::DataType::DT_FP16:
    case IndexMeta::DataType::DT_FP32:
    case IndexMeta::DataType::DT_FP64:
    case IndexMeta::DataType::DT_INT8:
    case IndexMeta::DataType::DT_INT16:
      base_size = static_cast<uint64_t>(dimension) * unit_size;
      break;
    case IndexMeta::DataType::DT_INT4: {
      const uint64_t values_per_unit = static_cast<uint64_t>(unit_size) * 2;
      base_size = (static_cast<uint64_t>(dimension) + values_per_unit - 1) /
                  values_per_unit * unit_size;
      break;
    }
    case IndexMeta::DataType::DT_BINARY32:
    case IndexMeta::DataType::DT_BINARY64: {
      const uint64_t values_per_unit = static_cast<uint64_t>(unit_size) * 8;
      base_size = (static_cast<uint64_t>(dimension) + values_per_unit - 1) /
                  values_per_unit * unit_size;
      break;
    }
  }

  const uint64_t total_size = base_size + extra_meta_size;
  if (total_size > std::numeric_limits<uint32_t>::max()) {
    return false;
  }
  *element_size = static_cast<uint32_t>(total_size);
  return true;
}

}  // namespace

void IndexMeta::serialize(std::string *out) const {
  ailego::Params attachment;
  IndexMetaFormatHeader format;
  memset(&format, 0, sizeof(format));
  format.header_size = sizeof(format);
  format.meta_type = static_cast<uint32_t>(meta_type_);
  format.major_order = static_cast<uint32_t>(major_order_);
  format.data_type = static_cast<uint32_t>(data_type_);
  format.dimension = dimension_;
  format.unit_size = unit_size_;
  format.space_id = space_id_;
  format.extra_meta_size = extra_meta_size_;

  if (!metric_name_.empty()) {
    ailego::Params item;
    item.set("name", metric_name_);
    item.set("revision", metric_revision_);
    item.set("params", metric_params_);
    attachment.set("metric", std::move(item));
  }

  if (!converter_name_.empty()) {
    ailego::Params item;
    item.set("name", converter_name_);
    item.set("revision", converter_revision_);
    item.set("params", converter_params_);
    attachment.set("converter", std::move(item));
  }
  if (!reformer_name_.empty()) {
    ailego::Params item;
    item.set("name", reformer_name_);
    item.set("revision", reformer_revision_);
    item.set("params", reformer_params_);
    attachment.set("reformer", std::move(item));
  }
  if (!quantizer_name_.empty()) {
    ailego::Params item;
    item.set("name", quantizer_name_);
    item.set("revision", quantizer_revision_);
    item.set("params", quantizer_params_);
    attachment.set("quantizer", std::move(item));
  }
  if (!trainer_name_.empty()) {
    ailego::Params item;
    item.set("name", trainer_name_);
    item.set("revision", trainer_revision_);
    item.set("params", trainer_params_);
    attachment.set("trainer", std::move(item));
  }
  if (!builder_name_.empty()) {
    ailego::Params item;
    item.set("name", builder_name_);
    item.set("revision", builder_revision_);
    item.set("params", builder_params_);
    attachment.set("builder", std::move(item));
  }
  if (!reducer_name_.empty()) {
    ailego::Params item;
    item.set("name", reducer_name_);
    item.set("revision", reducer_revision_);
    item.set("params", reducer_params_);
    attachment.set("reducer", std::move(item));
  }
  if (!searcher_name_.empty()) {
    ailego::Params item;
    item.set("name", searcher_name_);
    item.set("revision", searcher_revision_);
    item.set("params", searcher_params_);
    attachment.set("searcher", std::move(item));
  }
  if (!streamer_name_.empty()) {
    ailego::Params item;
    item.set("name", streamer_name_);
    item.set("revision", streamer_revision_);
    item.set("params", streamer_params_);
    attachment.set("streamer", std::move(item));
  }

  if (!attributes_.empty()) {
    attachment.set("attributes", attributes_);
  }

  std::string attachment_buffer;
  if (!attachment.empty()) {
    ailego::Params::SerializeToBuffer(attachment, &attachment_buffer);
    if (attachment_buffer.size() > std::numeric_limits<uint32_t>::max()) {
      out->clear();
      return;
    }
    format.attachment_offset = sizeof(format);
    format.attachment_size = static_cast<uint32_t>(attachment_buffer.size());
  }
  out->assign(reinterpret_cast<const char *>(&format), sizeof(format));
  out->append(attachment_buffer);
}

bool IndexMeta::deserialize(const void *data, size_t len) {
  this->clear();
  if (data == nullptr || sizeof(IndexMetaFormatHeader) > len) {
    return false;
  }

  IndexMetaFormatHeader format;
  std::memcpy(&format, data, sizeof(format));
  if (format.header_size < sizeof(IndexMetaFormatHeader) ||
      format.header_size > len) {
    return false;
  }

  if (format.meta_type >
          static_cast<uint32_t>(IndexMeta::MetaType::MT_SPARSE) ||
      format.major_order >
          static_cast<uint32_t>(IndexMeta::MajorOrder::MO_COLUMN)) {
    return false;
  }

  uint32_t element_size = 0;
  if (!ComputeElementSize(format.data_type, format.unit_size, format.dimension,
                          format.extra_meta_size, &element_size)) {
    return false;
  }

  // Read attachment
  ailego::Params attachment;
  if (format.attachment_size != 0) {
    if (format.attachment_offset < format.header_size ||
        format.attachment_offset > len ||
        format.attachment_size > len - format.attachment_offset) {
      return false;
    }
    std::string str(
        reinterpret_cast<const char *>(data) + format.attachment_offset,
        format.attachment_size);
    if (!ailego::Params::ParseFromBuffer(str, &attachment)) {
      return false;
    }
  }

  meta_type_ = static_cast<IndexMeta::MetaType>(format.meta_type);
  major_order_ = static_cast<IndexMeta::MajorOrder>(format.major_order);
  data_type_ = static_cast<IndexMeta::DataType>(format.data_type);
  dimension_ = format.dimension;
  unit_size_ = format.unit_size;
  extra_meta_size_ = format.extra_meta_size;
  element_size_ = element_size;
  space_id_ = format.space_id;

  ailego::Params item;
  if (attachment.get("metric", &item)) {
    item.get("name", &metric_name_);
    item.get("revision", &metric_revision_);
    item.get("params", &metric_params_);
  }
  if (attachment.get("converter", &item)) {
    item.get("name", &converter_name_);
    item.get("revision", &converter_revision_);
    item.get("params", &converter_params_);
  }
  if (attachment.get("reformer", &item)) {
    item.get("name", &reformer_name_);
    item.get("revision", &reformer_revision_);
    item.get("params", &reformer_params_);
  }
  if (attachment.get("quantizer", &item)) {
    item.get("name", &quantizer_name_);
    item.get("revision", &quantizer_revision_);
    item.get("params", &quantizer_params_);
  }
  if (attachment.get("trainer", &item)) {
    item.get("name", &trainer_name_);
    item.get("revision", &trainer_revision_);
    item.get("params", &trainer_params_);
  }
  if (attachment.get("builder", &item)) {
    item.get("name", &builder_name_);
    item.get("revision", &builder_revision_);
    item.get("params", &builder_params_);
  }
  if (attachment.get("reducer", &item)) {
    item.get("name", &reducer_name_);
    item.get("revision", &reducer_revision_);
    item.get("params", &reducer_params_);
  }
  if (attachment.get("searcher", &item)) {
    item.get("name", &searcher_name_);
    item.get("revision", &searcher_revision_);
    item.get("params", &searcher_params_);
  }
  if (attachment.get("streamer", &item)) {
    item.get("name", &streamer_name_);
    item.get("revision", &streamer_revision_);
    item.get("params", &streamer_params_);
  }
  attachment.get("attributes", &attributes_);

  return true;
}

}  // namespace core
}  // namespace zvec
