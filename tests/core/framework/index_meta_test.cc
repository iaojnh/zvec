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

#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <gtest/gtest.h>
#include <zvec/core/framework/index_meta.h>

namespace zvec::core {
namespace {

void WriteHeaderWord(std::string *buffer, size_t word, uint32_t value) {
  ASSERT_NE(nullptr, buffer);
  ASSERT_GE(buffer->size(), (word + 1) * sizeof(value));
  std::memcpy(buffer->data() + word * sizeof(value), &value, sizeof(value));
}

TEST(IndexMeta, RejectsMalformedHeaderAndAttachmentBounds) {
  IndexMeta source(IndexMeta::DataType::DT_FP32, 8);
  source.set_trainer("test_trainer", 1, ailego::Params());
  std::string serialized;
  source.serialize(&serialized);
  ASSERT_FALSE(serialized.empty());

  IndexMeta parsed;
  EXPECT_FALSE(parsed.deserialize(nullptr, serialized.size()));
  EXPECT_FALSE(parsed.deserialize(serialized.data(), sizeof(uint32_t)));

  std::string oversized_header = serialized;
  WriteHeaderWord(&oversized_header, 0,
                  static_cast<uint32_t>(serialized.size() + 1));
  EXPECT_FALSE(
      parsed.deserialize(oversized_header.data(), oversized_header.size()));

  std::string wrapped_attachment = serialized;
  WriteHeaderWord(&wrapped_attachment, 7,
                  std::numeric_limits<uint32_t>::max() - 3);
  WriteHeaderWord(&wrapped_attachment, 8, 8);
  EXPECT_FALSE(
      parsed.deserialize(wrapped_attachment.data(), wrapped_attachment.size()));

  std::string attachment_in_header = serialized;
  WriteHeaderWord(&attachment_in_header, 7, sizeof(uint32_t));
  EXPECT_FALSE(parsed.deserialize(attachment_in_header.data(),
                                  attachment_in_header.size()));
}

TEST(IndexMeta, RejectsInvalidTypesUnitsAndElementSizeOverflow) {
  IndexMeta source(IndexMeta::DataType::DT_FP32, 8);
  std::string serialized;
  source.serialize(&serialized);

  IndexMeta parsed;
  std::string invalid_type = serialized;
  WriteHeaderWord(&invalid_type, 3,
                  static_cast<uint32_t>(IndexMeta::DataType::DT_UINT8) + 1);
  EXPECT_FALSE(parsed.deserialize(invalid_type.data(), invalid_type.size()));

  std::string invalid_unit = serialized;
  WriteHeaderWord(&invalid_unit, 5, sizeof(double));
  EXPECT_FALSE(parsed.deserialize(invalid_unit.data(), invalid_unit.size()));

  std::string overflowing_size = serialized;
  WriteHeaderWord(&overflowing_size, 3,
                  static_cast<uint32_t>(IndexMeta::DataType::DT_FP64));
  WriteHeaderWord(&overflowing_size, 4, std::numeric_limits<uint32_t>::max());
  WriteHeaderWord(&overflowing_size, 5, sizeof(double));
  EXPECT_FALSE(
      parsed.deserialize(overflowing_size.data(), overflowing_size.size()));
}

TEST(IndexMeta, DeserializesFromUnalignedStorage) {
  IndexMeta source(IndexMeta::DataType::DT_FP16, 7);
  source.set_metric("SquaredEuclidean", 3, ailego::Params());
  std::string serialized;
  source.serialize(&serialized);

  std::string unaligned(1, '\0');
  unaligned.append(serialized);

  IndexMeta parsed;
  ASSERT_TRUE(parsed.deserialize(unaligned.data() + 1, serialized.size()));
  EXPECT_EQ(source.data_type(), parsed.data_type());
  EXPECT_EQ(source.dimension(), parsed.dimension());
  EXPECT_EQ(source.element_size(), parsed.element_size());
  EXPECT_EQ(source.metric_name(), parsed.metric_name());
}

}  // namespace
}  // namespace zvec::core
