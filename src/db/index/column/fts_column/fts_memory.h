// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <cctype>
#include <zvec/ailego/encoding/json/mod_json_plus.h>
#include <zvec/db/index_params.h>
#include "db/common/query_result_memory.h"

namespace zvec::detail {

// Conservative pre-admission for APIs whose token/map/WriteBatch types use
// ordinary allocators. Account for analyzer expansion, not just input bytes.
// This is a planning allowance, not an allocator or an RSS measurement.
inline size_t EstimateFtsWorkspace(const FtsIndexParams &params,
                                   const std::string &text) {
  if (text.empty()) return 0;
  ailego::JsonValue config;
  if (!params.extra_params().empty() &&
      !config.parse(params.extra_params().c_str()))
    throw std::invalid_argument("Invalid FTS analyzer configuration");
  const auto number = [&](const char *key, size_t fallback) {
    if (!config.is_object()) return fallback;
    const auto value = config.as_object()[key];
    return value.is_integer() && value.as_integer() > 0
               ? static_cast<size_t>(value.as_integer())
               : fallback;
  };
  const auto multiply = [](size_t a, size_t b) {
    QueryResultEstimate size;
    size.add(a, b);
    return size.bytes();
  };
  size_t tokens = 0;
  size_t characters = 0;
  bool ascii = true;
  bool in_word = false;
  for (unsigned char byte : text) {
    if ((byte & 0xc0) != 0x80) ++characters;
    ascii = ascii && byte < 128;
    const bool word = params.tokenizer_name() == "whitespace"
                          ? !std::isspace(byte)
                          : std::isalnum(byte) != 0;
    if (word && !in_word) ++tokens;
    in_word = word;
  }
  size_t payload = text.size();
  const auto &name = params.tokenizer_name();
  QueryResultEstimate estimate;
  estimate.add(4096);
  if (name == "ngram") {
    const size_t min = number("ngram_min", 2);
    const size_t max = number("ngram_max", 2);
    if (max < min || max - min > 1) throw std::bad_alloc();
    const size_t variants = max - min + 1;
    tokens = multiply(characters, variants);
    // Each byte can occur in at most max windows of each gram length.
    payload =
        multiply(text.size(), multiply(std::min(max, characters), variants));
    estimate.add(64, characters);  // Unicode spans and vector growth.
  } else if (name == "jieba") {
    // cppjieba's DAG has at most MAX_WORD_LENGTH (512) outgoing edges
    // per codepoint. Full mode may return all of them; search/mix/hmm
    // emit far fewer, but still need the DAG during tokenization.
    constexpr size_t kMaxWordLength = 512;
    bool full = false;
    bool hmm = false;
    if (config.is_object()) {
      const auto mode = config.as_object()["cut_mode"];
      full = mode.is_string() && mode.as_stl_string() == "full";
      hmm = mode.is_string() && mode.as_stl_string() == "hmm";
    }
    const size_t width = std::min(kMaxWordLength, characters);
    tokens = multiply(characters, full ? width : 3);
    payload = multiply(text.size(), full ? multiply(width, width + 1) / 2 : 4);
    estimate.add(hmm ? 128 : multiply(width, 32), characters);
  } else if (name == "standard" || name.empty()) {
    if (ascii) {
      // Punctuation may join runs; splitting at max_token_length can add
      // no more than one token per complete input-length interval.
      tokens += text.size() / number("max_token_length", 255);
    } else {
      tokens = characters;
      estimate.add(64, characters);  // Unicode word-break scratch.
    }
    tokens = std::max(tokens, std::min(text.size() / 4 + 1, size_t{4096}));
  } else if (name != "whitespace") {
    // New analyzers need an explicit expansion policy before budgeted use.
    throw std::bad_alloc();
  }
  if (!ascii) {
    for (size_t i = 0; i < params.filters().size(); ++i)
      payload = multiply(payload, 4);
  }
  estimate.add(256, tokens);  // Token vectors, map nodes, positions, ASTs.
  estimate.add(12, payload);  // Filter copies and growing cross-CF batch.
  estimate.add(96, multiply(tokens, params.filters().size()));
  return estimate.bytes();
}

// Share one backing high water across indexing and query materialization.
// Separate analyzer allowances could cumulatively consume the whole pool.
class FtsWriteAdmission {
 public:
  explicit FtsWriteAdmission(bool enabled = true)
      : memory_(enabled ? &SharedResultMemoryAllowance() : nullptr) {}

  void add(const FtsIndexParams &params, const std::string &text) {
    if (memory_.enabled()) memory_.add(EstimateFtsWorkspace(params, text));
  }

 private:
  ResultMemoryAllowance::Lease memory_;
};

}  // namespace zvec::detail
