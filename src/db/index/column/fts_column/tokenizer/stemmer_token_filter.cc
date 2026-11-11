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

#include "stemmer_token_filter.h"
#include <limits>
#include <unordered_map>
#include <zvec/ailego/buffer/memory_reservation.h>
#include <zvec/ailego/logger/logger.h>
#include "db/common/binding_memory.h"

extern "C" {
#include <libstemmer.h>
}

namespace zvec::fts {

namespace {
struct StemmerDeleter {
  void operator()(sb_stemmer *stemmer) const {
    sb_stemmer_delete(stemmer);
  }
};

class CachedStemmer {
 public:
  explicit CachedStemmer(const std::string &language) {
    // Include the native context and cache node/key before creating them.
    memory_.add(kContextBytes);
    stemmer_.reset(sb_stemmer_new(language.c_str(), nullptr));
    if (!stemmer_) throw std::bad_alloc();
  }

  void stem(std::string &text) {
    if (text.size() > static_cast<size_t>(std::numeric_limits<int>::max()))
      throw std::bad_alloc();
    const size_t length = std::max(max_word_bytes_, text.size());
    // Snowball retains its growing word buffer and algorithm scratch until
    // the context dies. Each language/thread needs its own peak allowance;
    // the operation's temporary FTS allowance cannot cover their sum.
    const size_t target = detail::BindingMemoryBytes(length, 8, kContextBytes);
    if (target > memory_.bytes()) memory_.add(target - memory_.bytes());
    max_word_bytes_ = length;
    const auto *result = sb_stemmer_stem(
        stemmer_.get(), reinterpret_cast<const unsigned char *>(text.data()),
        static_cast<int>(text.size()));
    if (!result) throw std::bad_alloc();
    text.assign(reinterpret_cast<const char *>(result),
                sb_stemmer_length(stemmer_.get()));
  }

 private:
  static constexpr size_t kContextBytes = 4096;
  // Destroy the stemmer's retained buffers before refunding admission.
  ailego::MemoryReservation memory_;
  size_t max_word_bytes_{0};
  std::unique_ptr<sb_stemmer, StemmerDeleter> stemmer_;
};

struct ThreadLocalStemmerCache {
  std::unordered_map<std::string, std::unique_ptr<CachedStemmer>> stemmers;

  CachedStemmer &get(const std::string &lang) {
    auto it = stemmers.find(lang);
    if (it == stemmers.end()) {
      // Ownership also cleans up if map insertion fails after native creation.
      auto stemmer = std::make_unique<CachedStemmer>(lang);
      it = stemmers.emplace(lang, std::move(stemmer)).first;
    }
    return *it->second;
  }
};
}  // namespace

bool StemmerTokenFilter::init(const ailego::JsonObject &config) {
  std::string lang;
  if (config.get("stemmer_lang", &lang) && !lang.empty()) {
    language_ = lang;
  }
  auto *test_stemmer = sb_stemmer_new(language_.c_str(), nullptr);
  if (!test_stemmer) {
    LOG_ERROR("[StemmerTokenFilter] failed to create stemmer for language: %s",
              language_.c_str());
    return false;
  }
  sb_stemmer_delete(test_stemmer);
  return true;
}

std::vector<Token> StemmerTokenFilter::filter(std::vector<Token> tokens) const {
  static thread_local ThreadLocalStemmerCache tls_cache;
  auto &stemmer = tls_cache.get(language_);
  for (auto &token : tokens) stemmer.stem(token.text);
  return tokens;
}

}  // namespace zvec::fts
