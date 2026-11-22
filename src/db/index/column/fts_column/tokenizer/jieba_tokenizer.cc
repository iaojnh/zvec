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
#include <fstream>
#include <limits>
#include <zvec/ailego/logger/logger.h>
#include "jieba_embedded_dict.h"

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

// Priority: per-field config > ZVEC_JIEBA_DICT_DIR > embedded defaults.
static std::string resolve_jieba_dict_dir(const ailego::JsonObject &config) {
  std::string dir = get_string_or_default(config, "jieba_dict_dir", "");
  if (!dir.empty()) {
    return dir;
  }
  if (const char *env = std::getenv("ZVEC_JIEBA_DICT_DIR"); env && *env) {
    return env;
  }
  return {};
}

// cppjieba aborts when a file cannot be opened. Validate configured paths
// before constructing its dictionary or model, preserving UTF-8 path support.
static Status check_jieba_file(const std::string &path, bool allow_empty) {
  std::ifstream stream;
  cppjieba::OpenInputFile(stream, path);
  if (!stream.is_open()) {
    return Status::InvalidArgument("JiebaTokenizer: cannot read file '", path,
                                   "'");
  }
  auto first = stream.peek();
  if (stream.bad() ||
      (!allow_empty && first == std::ifstream::traits_type::eof())) {
    return Status::InvalidArgument(
        "JiebaTokenizer: cannot read nonempty file '", path, "'");
  }
  return Status::OK();
}

Result<JiebaTokenizer::Config> JiebaTokenizer::parse_config(
    const ailego::JsonObject &config) {
  Config result;
  result.user_dict_path = get_string_or_default(config, "user_dict_path", "");
  result.mode_name = get_string_or_default(config, "cut_mode", "search");
  if (result.mode_name == "search") {
    result.mode = CutMode::kSearch;
  } else if (result.mode_name == "mix") {
    result.mode = CutMode::kMix;
  } else if (result.mode_name == "full") {
    result.mode = CutMode::kFull;
  } else if (result.mode_name == "hmm") {
    result.mode = CutMode::kHmm;
  } else {
    return tl::make_unexpected(Status::InvalidArgument(
        "JiebaTokenizer: unknown cut_mode '", result.mode_name, "'"));
  }

  const bool needs_dict = result.mode != CutMode::kHmm;
  const bool needs_model = result.mode != CutMode::kFull;
  result.dict_dir = resolve_jieba_dict_dir(config);
  if (needs_dict) result.dict_path = result.dict_dir + "/jieba.dict.utf8";
  if (needs_model) result.model_path = result.dict_dir + "/hmm_model.utf8";

  if (!result.dict_dir.empty()) {
    for (const auto &path : {result.dict_path, result.model_path}) {
      if (path.empty()) continue;
      auto status = check_jieba_file(path, false);
      if (!status.ok()) return tl::make_unexpected(std::move(status));
    }
  }
  if (needs_dict && !result.user_dict_path.empty()) {
    for (const auto &path : limonp::Split(result.user_dict_path, "|;")) {
      auto status = check_jieba_file(path, true);
      if (!status.ok()) return tl::make_unexpected(std::move(status));
    }
  }
  return result;
}

Status JiebaTokenizer::validate_config(const ailego::JsonObject &config) {
  auto parsed = parse_config(config);
  return parsed ? Status::OK() : std::move(parsed.error());
}

Status JiebaTokenizer::init(const ailego::JsonObject &config) {
  reset();
  auto parsed = parse_config(config);
  if (!parsed) return std::move(parsed.error());
  const auto &options = *parsed;
  cut_mode_ = options.mode;
  const bool needs_dict = cut_mode_ != CutMode::kHmm;
  const bool needs_model = cut_mode_ != CutMode::kFull;
  const auto &dict_dir = options.dict_dir;
  const auto &dict_path = options.dict_path;
  const auto &model_path = options.model_path;
  const auto &user_dict_path = options.user_dict_path;
  const auto &mode_str = options.mode_name;

  try {
    // Dictionaries scale with configuration, not collection row count. Keep
    // a planning allowance for expanded trie/hash nodes and loader scratch;
    // do not pretend that third-party STL allocations are exact pool bytes.
    dictionary_memory_.add(8 * 1024 * 1024);
    auto reserve_file = [this](const std::string &path) {
      std::error_code error;
      const auto bytes =
          std::filesystem::file_size(std::filesystem::u8path(path), error);
      if (error || bytes > std::numeric_limits<size_t>::max() / 16) {
        return Status::InvalidArgument("Cannot size Jieba dictionary: ", path);
      }
      dictionary_memory_.add_array(static_cast<size_t>(bytes), 16);
      return Status::OK();
    };
    if (dict_dir.empty()) {
      // Shared defaults retain their own allowance after this tokenizer dies.
      // Only trie construction/backing is per instance; the HMM is shared.
      if (needs_dict)
        dictionary_memory_.add_array(embedded_jieba_dict_bytes(),
                                     kEmbeddedJiebaDictionaryMultiplier);
    } else {
      for (const auto &path : {dict_path, model_path}) {
        if (path.empty()) continue;
        const auto status = reserve_file(path);
        if (!status.ok()) {
          reset();
          return status;
        }
      }
    }
    if (needs_dict && !user_dict_path.empty()) {
      for (const auto &path : limonp::Split(user_dict_path, "|;")) {
        const auto status = reserve_file(path);
        if (!status.ok()) {
          reset();
          return status;
        }
      }
    }
    if (needs_dict) {
      if (dict_dir.empty()) {
        auto dict = get_embedded_jieba_dict();
        if (!dict) {
          reset();
          return dict.error();
        }
        dict_trie_ =
            std::make_unique<cppjieba::DictTrie>(**dict, user_dict_path);
      } else {
        // The path constructor retains every dictionary in a process-wide
        // cache. Load through the stream API instead so the final tokenizer
        // releases external entries together with their instance allowance.
        std::ifstream stream;
        cppjieba::OpenInputFile(stream, dict_path);
        if (!stream.is_open()) {
          reset();
          return Status::InvalidArgument("Cannot read Jieba dictionary: ",
                                         dict_path);
        }
        auto entries = cppjieba::DictTrie::LoadDict(stream);
        dict_trie_ =
            std::make_unique<cppjieba::DictTrie>(entries, user_dict_path);
      }
    }
    if (needs_model) {
      if (dict_dir.empty()) {
        auto model = get_embedded_jieba_hmm_model();
        if (!model) {
          reset();
          return model.error();
        }
        hmm_model_ = *model;
      } else {
        hmm_model_ = std::make_shared<const cppjieba::HMMModel>(model_path);
      }
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
