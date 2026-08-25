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

#include <array>
#include <cstring>
#include <memory>
#include <string>
#include <gtest/gtest.h>
#include <zvec/core/framework/index_factory.h>

namespace zvec {
namespace core {
namespace {

class TestHolder : public IndexHolder {
 public:
  explicit TestHolder(bool fail_reads) : fail_reads_(fail_reads) {}

  class Iterator : public IndexHolder::Iterator {
   public:
    Iterator(bool fail_reads, const std::array<float, 4> *values)
        : fail_reads_(fail_reads), values_(values) {}

    const void *data() const override {
      return fail_reads_ ? nullptr : values_->data();
    }

    bool is_valid() const override {
      return valid_;
    }

    uint64_t key() const override {
      return 0;
    }

    void next() override {
      valid_ = false;
    }

   private:
    bool fail_reads_{false};
    bool valid_{true};
    const std::array<float, 4> *values_{nullptr};
  };

  size_t count() const override {
    return 1;
  }

  size_t dimension() const override {
    return values_.size();
  }

  IndexMeta::DataType data_type() const override {
    return IndexMeta::DataType::DT_FP32;
  }

  size_t element_size() const override {
    return values_.size() * sizeof(float);
  }

  bool multipass() const override {
    return true;
  }

  Iterator::Pointer create_iterator() override {
    return Iterator::Pointer(new Iterator(fail_reads_, &values_));
  }

 private:
  bool fail_reads_{false};
  std::array<float, 4> values_{1.0F, 2.0F, 3.0F, 4.0F};
};

IndexMeta MakeMeta(const std::string &converter_name) {
  IndexMeta meta(IndexMeta::DataType::DT_FP32, 4);
  meta.set_metric(
      converter_name == "BinaryConverter" ? "InnerProduct" : "SquaredEuclidean",
      0, ailego::Params());
  return meta;
}

TEST(ConverterReadFailure, TransformWrappersPropagateNullData) {
  for (const char *name :
       {"HalfFloatConverter", "CosineNormalizeConverter",
        "Int8QuantizerConverter", "Int8StreamingConverter", "MipsConverter",
        "UniformUint7Converter", "UniformUint8Converter", "BinaryConverter"}) {
    SCOPED_TRACE(name);
    auto converter = IndexFactory::CreateConverter(name);
    ASSERT_NE(nullptr, converter);
    ASSERT_EQ(0, converter->init(MakeMeta(name), ailego::Params()));
    if (std::string(name) == "Int8QuantizerConverter" ||
        std::string(name) == "MipsConverter" ||
        std::string(name) == "UniformUint7Converter" ||
        std::string(name) == "UniformUint8Converter") {
      ASSERT_EQ(0, converter->train(std::make_shared<TestHolder>(false)));
    }
    ASSERT_EQ(0, converter->transform(std::make_shared<TestHolder>(true)));
    auto result = converter->result();
    ASSERT_NE(nullptr, result);
    auto iterator = result->create_iterator();
    ASSERT_NE(nullptr, iterator);
    ASSERT_TRUE(iterator->is_valid());
    EXPECT_EQ(nullptr, iterator->data());

    ASSERT_EQ(0, converter->transform(std::make_shared<TestHolder>(false)));
    iterator = converter->result()->create_iterator();
    ASSERT_NE(nullptr, iterator);
    ASSERT_TRUE(iterator->is_valid());
    EXPECT_NE(nullptr, iterator->data());
  }
}

TEST(ConverterReadFailure, TrainingReturnsReadData) {
  for (const char *name : {"Int8QuantizerConverter", "MipsConverter",
                           "UniformUint7Converter", "UniformUint8Converter"}) {
    SCOPED_TRACE(name);
    auto converter = IndexFactory::CreateConverter(name);
    ASSERT_NE(nullptr, converter);
    ASSERT_EQ(0, converter->init(MakeMeta(name), ailego::Params()));
    EXPECT_EQ(IndexError_ReadData,
              converter->train(std::make_shared<TestHolder>(true)));
  }
}

TEST(ConverterReadFailure, BinaryConverterInitializesQuantizer) {
  auto converter = IndexFactory::CreateConverter("BinaryConverter");
  ASSERT_NE(nullptr, converter);
  ASSERT_EQ(0, converter->init(MakeMeta("BinaryConverter"), ailego::Params()));
  ASSERT_EQ(0, converter->transform(std::make_shared<TestHolder>(false)));
  auto iterator = converter->result()->create_iterator();
  ASSERT_NE(nullptr, iterator);
  ASSERT_TRUE(iterator->is_valid());
  const void *data = iterator->data();
  ASSERT_NE(nullptr, data);
  uint32_t encoded = 0;
  std::memcpy(&encoded, data, sizeof(encoded));
  EXPECT_EQ(0xFU, encoded);
}

TEST(ConverterReadFailure, BinaryReformerPreservesBatchBoundaries) {
  auto reformer = IndexFactory::CreateReformer("BinaryReformer");
  ASSERT_NE(nullptr, reformer);
  ASSERT_EQ(0, reformer->init(ailego::Params()));

  const std::array<float, 8> queries{-1.0F, -2.0F, -3.0F, -4.0F,
                                     1.0F,  2.0F,  3.0F,  4.0F};
  const IndexQueryMeta query_meta(IndexMeta::DataType::DT_FP32, 4);
  IndexQueryMeta output_meta;
  std::string output;
  ASSERT_EQ(0, reformer->transform(queries.data(), query_meta, 2, &output,
                                   &output_meta));
  ASSERT_EQ(2 * sizeof(uint32_t), output.size());
  EXPECT_EQ(IndexMeta::DataType::DT_BINARY32, output_meta.data_type());
  EXPECT_EQ(32U, output_meta.dimension());

  std::array<uint32_t, 2> encoded{};
  std::memcpy(encoded.data(), output.data(), output.size());
  EXPECT_EQ(0U, encoded[0]);
  EXPECT_EQ(0xFU, encoded[1]);
}

}  // namespace
}  // namespace core
}  // namespace zvec
