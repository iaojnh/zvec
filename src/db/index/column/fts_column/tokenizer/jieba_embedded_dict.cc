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

#include "jieba_embedded_dict.h"
#include <lz4.h>
#include <cstddef>
#include <istream>
#include <limits>
#include <streambuf>
#include <string>
#include <vector>
#include <zvec/ailego/buffer/memory_reservation.h>

namespace zvec::fts {
namespace {

#include "jieba_dict_lz4.inc"
#include "jieba_hmm_model_lz4.inc"

// The cache outlives tokenizer instances. Destroy the parsed payload before
// refunding its allowance, including during static teardown and failed loads.
template <typename T>
struct CachedResource {
  ailego::MemoryReservation memory;
  std::shared_ptr<const T> value;
};

// Reference the decompressed text without istringstream's C++17 string copy.
// The caller keeps the text alive until parsing has finished.
class MemoryInputBuffer : public std::streambuf {
 public:
  explicit MemoryInputBuffer(std::string &text) {
    char *begin = text.data();
    setg(begin, begin, begin + text.size());
  }
};

template <std::size_t N>
Result<std::string> decompress(const unsigned char (&data)[N],
                               std::size_t original_size) {
  static_assert(N <= std::numeric_limits<int>::max());
  if (original_size >
      static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    return tl::make_unexpected(Status::InternalError(
        "Embedded Jieba resource exceeds LZ4 size limit"));
  }
  std::string text(original_size, '\0');
  int size =
      LZ4_decompress_safe(reinterpret_cast<const char *>(data), text.data(),
                          static_cast<int>(N), static_cast<int>(original_size));
  if (size < 0 || static_cast<std::size_t>(size) != original_size) {
    return tl::make_unexpected(
        Status::InternalError("Failed to decompress embedded Jieba resource"));
  }
  return text;
}

}  // namespace

size_t embedded_jieba_dict_bytes() {
  return kJiebaDictLz4OriginalSize;
}

size_t embedded_jieba_hmm_model_bytes() {
  return kJiebaHmmModelLz4OriginalSize;
}

Result<EmbeddedJiebaDict> get_embedded_jieba_dict() {
  static const Result<EmbeddedJiebaDict> dict =
      []() -> Result<EmbeddedJiebaDict> {
    auto owner =
        std::make_shared<CachedResource<std::vector<cppjieba::DictUnit>>>();
    owner->memory.add_array(embedded_jieba_dict_bytes(),
                            kEmbeddedJiebaDictionaryMultiplier);
    auto text = decompress(kJiebaDictLz4, kJiebaDictLz4OriginalSize);
    if (!text) {
      return tl::make_unexpected(text.error());
    }
    MemoryInputBuffer buffer(*text);
    std::istream stream(&buffer);
    auto entries = cppjieba::DictTrie::LoadDict(stream);
    owner->value = std::move(entries.node_infos);
    // DictTrie retains node_infos, not the DictCacheEntry. Let those aliases
    // retain admission too, even if a tokenizer outlives static cache teardown.
    entries.node_infos =
        decltype(entries.node_infos)(owner, owner->value.get());
    return std::make_shared<const cppjieba::DictTrie::DictCacheEntry>(
        std::move(entries));
  }();
  return dict;
}

Result<EmbeddedJiebaHmmModel> get_embedded_jieba_hmm_model() {
  static const Result<EmbeddedJiebaHmmModel> model =
      []() -> Result<EmbeddedJiebaHmmModel> {
    auto owner = std::make_shared<CachedResource<cppjieba::HMMModel>>();
    owner->memory.add_array(embedded_jieba_hmm_model_bytes(),
                            kEmbeddedJiebaModelMultiplier);
    auto text = decompress(kJiebaHmmModelLz4, kJiebaHmmModelLz4OriginalSize);
    if (!text) {
      return tl::make_unexpected(text.error());
    }
    MemoryInputBuffer buffer(*text);
    std::istream stream(&buffer);
    owner->value = std::make_shared<const cppjieba::HMMModel>(stream);
    return EmbeddedJiebaHmmModel(owner, owner->value.get());
  }();
  return model;
}

}  // namespace zvec::fts
