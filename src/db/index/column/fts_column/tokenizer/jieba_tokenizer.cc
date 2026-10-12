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

#include "jieba_tokenizer.h"
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <zvec/ailego/logger/logger.h>
#include <zvec/db/config.h>

namespace zvec::fts {

static std::string get_string_or_default(const ailego::JsonObject &config,
                                         const char *key,
                                         const std::string &default_value) {
  auto val = config[key];
  if (val.is_string()) {
    std::string result = val.as_string().c_str();
    if (!result.empty()) {
      return result;
    }
  }
  return default_value;
}

// Priority: per-field config > ZVEC_JIEBA_DICT_DIR > GlobalConfig.
static std::string resolve_jieba_dict_dir(const ailego::JsonObject &config) {
  std::string dir = get_string_or_default(config, "jieba_dict_dir", "");
  if (!dir.empty()) {
    return dir;
  }
  if (const char *env = std::getenv("ZVEC_JIEBA_DICT_DIR"); env && *env) {
    return env;
  }
  return GlobalConfig::Instance().jieba_dict_dir();
}

Status JiebaTokenizer::init(const ailego::JsonObject &config) {
  std::string user_dict_path =
      get_string_or_default(config, "user_dict_path", "");

  std::string mode_str = get_string_or_default(config, "cut_mode", "search");
  if (mode_str == "search") {
    cut_mode_ = CutMode::kSearch;
  } else if (mode_str == "mix") {
    cut_mode_ = CutMode::kMix;
  } else if (mode_str == "full") {
    cut_mode_ = CutMode::kFull;
  } else if (mode_str == "hmm") {
    cut_mode_ = CutMode::kHmm;
  } else {
    return Status::InvalidArgument("JiebaTokenizer: unknown cut_mode '",
                                   mode_str, "'");
  }

  bool needs_dict = cut_mode_ != CutMode::kHmm;
  bool needs_model = cut_mode_ != CutMode::kFull;

  std::string dict_dir = resolve_jieba_dict_dir(config);
  if ((needs_dict || needs_model) && dict_dir.empty()) {
    return Status::InvalidArgument(
        "JiebaTokenizer: jieba_dict_dir not configured. Set via "
        "extra_params.jieba_dict_dir, ZVEC_JIEBA_DICT_DIR env var, "
        "or zvec.set_default_jieba_dict_dir() / "
        "zvec.init(jieba_dict_dir=...).");
  }

  std::string dict_path = needs_dict ? dict_dir + "/jieba.dict.utf8" : "";
  std::string model_path = needs_model ? dict_dir + "/hmm_model.utf8" : "";

  reset();

  try {
    // Dictionaries scale with configuration, not collection row count. Keep
    // a planning allowance for expanded trie/hash nodes and loader scratch;
    // do not pretend that third-party STL allocations are exact pool bytes.
    dictionary_memory_.add(8 * 1024 * 1024);
    for (const auto &path :
         {dict_path, model_path, needs_dict ? user_dict_path : std::string{}}) {
      if (path.empty()) continue;
      std::error_code error;
      const auto bytes =
          std::filesystem::file_size(std::filesystem::u8path(path), error);
      if (error || bytes > std::numeric_limits<size_t>::max() / 16) {
        reset();
        return Status::InvalidArgument("Cannot size Jieba dictionary: ", path);
      }
      dictionary_memory_.add_array(static_cast<size_t>(bytes), 16);
    }
    if (needs_dict) {
      dict_trie_ =
          std::make_unique<cppjieba::DictTrie>(dict_path, user_dict_path);
    }
    if (needs_model) {
      hmm_model_ = std::make_unique<cppjieba::HMMModel>(model_path);
    }
    switch (cut_mode_) {
      case CutMode::kSearch:
        query_seg_ = std::make_unique<cppjieba::QuerySegment>(dict_trie_.get(),
                                                              hmm_model_.get());
        break;
      case CutMode::kMix:
        mix_seg_ = std::make_unique<cppjieba::MixSegment>(dict_trie_.get(),
                                                          hmm_model_.get());
        break;
      case CutMode::kFull:
        full_seg_ = std::make_unique<cppjieba::FullSegment>(dict_trie_.get());
        break;
      case CutMode::kHmm:
        hmm_seg_ = std::make_unique<cppjieba::HMMSegment>(hmm_model_.get());
        break;
    }
  } catch (const std::bad_alloc &) {
    reset();
    return Status(StatusCode::RESOURCE_EXHAUSTED,
                  "Insufficient Jieba dictionary budget");
  } catch (const std::exception &e) {
    LOG_ERROR("JiebaTokenizer init failed: %s", e.what());
    reset();
    return Status::InvalidArgument("JiebaTokenizer init failed: ", e.what());
  }

  initialized_ = true;
  LOG_INFO("JiebaTokenizer init success. dict_dir[%s] cut_mode[%s]",
           dict_dir.c_str(), mode_str.c_str());
  return Status::OK();
}

JiebaTokenizer::~JiebaTokenizer() = default;

JiebaTokenizer::JiebaTokenizer(JiebaTokenizer &&other) noexcept {
  *this = std::move(other);
}

JiebaTokenizer &JiebaTokenizer::operator=(JiebaTokenizer &&other) noexcept {
  if (this == &other) return *this;
  reset();  // Free old dictionaries before releasing their allowance.
  dictionary_memory_ = std::move(other.dictionary_memory_);
  dict_trie_ = std::move(other.dict_trie_);
  hmm_model_ = std::move(other.hmm_model_);
  query_seg_ = std::move(other.query_seg_);
  mix_seg_ = std::move(other.mix_seg_);
  full_seg_ = std::move(other.full_seg_);
  hmm_seg_ = std::move(other.hmm_seg_);
  cut_mode_ = other.cut_mode_;
  initialized_ = std::exchange(other.initialized_, false);
  return *this;
}

void JiebaTokenizer::reset() {
  query_seg_.reset();
  mix_seg_.reset();
  full_seg_.reset();
  hmm_seg_.reset();
  dict_trie_.reset();
  hmm_model_.reset();
  dictionary_memory_.retain_at_most(0);
  initialized_ = false;
}

std::vector<Token> JiebaTokenizer::tokenize(const std::string &text) const {
  std::vector<Token> tokens;
  if (!initialized_ || text.empty()) {
    return tokens;
  }

  std::vector<cppjieba::Word> words;
  switch (cut_mode_) {
    case CutMode::kSearch:
      query_seg_->Cut(text, words, true);
      break;
    case CutMode::kMix:
      mix_seg_->Cut(text, words, true);
      break;
    case CutMode::kFull:
      full_seg_->Cut(text, words);
      break;
    case CutMode::kHmm:
      hmm_seg_->Cut(text, words);
      break;
  }

  tokens.reserve(words.size());
  // Position = output sequence index, not cppjieba's unicode_offset:
  // overlapping sub-words emitted after long parents share unicode_offset,
  // which breaks PhraseDocIterator's strict anchor+1 adjacency check.
  uint32_t seq = 0;
  for (const auto &word : words) {
    if (word.word.empty()) {
      continue;
    }
    Token token;
    token.text = word.word;
    token.offset = word.offset;
    token.position = seq++;
    tokens.push_back(std::move(token));
  }

  return tokens;
}

}  // namespace zvec::fts
