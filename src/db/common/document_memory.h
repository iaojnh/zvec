// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <zvec/db/doc.h>
#include "binding_memory.h"

namespace zvec::detail {

struct DocumentMemoryAccess {
  static const Doc::Value *find_field(const Doc &doc, const std::string &name) {
    const auto it = doc.fields_.find(name);
    return it == doc.fields_.end() ? nullptr : &it->second;
  }
  static void retain(Doc &doc, std::shared_ptr<void> memory) {
    doc.memory_owner_ = std::move(memory);
  }
  static bool budgeted(const Doc &doc) {
    return bool(doc.memory_owner_);
  }
  static size_t field_names_bytes(const Doc &doc) {
    size_t bytes =
        BindingMemoryBytes(doc.fields_.size(), sizeof(std::string) + 64);
    for (const auto &[name, value] : doc.fields_)
      bytes = BindingMemoryBytes(1, name.size(), bytes);
    return bytes;
  }
  // Estimate a deep copy without first allocating field_names or serialized
  // data. Include nodes, buckets, string objects and allocator rounding.
  static size_t copy_bytes(const Doc &doc) {
    size_t bytes = BindingMemoryBytes(1, doc.pk_.size(), sizeof(Doc) + 256);
    auto add = [&bytes](size_t count, size_t width = 1) {
      bytes = BindingMemoryBytes(count, width, bytes);
    };
    for (const auto &[name, value] : doc.fields_) {
      add(BindingMemoryBytes(1, name.size(), 256));
      std::visit(
          [&add](const auto &v) {
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<T, std::monostate> ||
                          std::is_arithmetic_v<T>) {
              return;
            } else if constexpr (std::is_same_v<T, std::string>) {
              add(v.size());
            } else if constexpr (std::is_same_v<T, std::vector<std::string>>) {
              add(v.size(), sizeof(std::string) + 32);
              for (const auto &text : v) add(text.size());
            } else if constexpr (std::is_same_v<
                                     T, std::pair<std::vector<uint32_t>,
                                                  std::vector<float>>> ||
                                 std::is_same_v<
                                     T, std::pair<std::vector<uint32_t>,
                                                  std::vector<float16_t>>>) {
              add(v.first.size(), sizeof(uint32_t));
              add(v.second.size(), sizeof(typename T::second_type::value_type));
            } else {
              add(v.size(), sizeof(typename T::value_type));
            }
          },
          value);
    }
    return bytes;
  }
};

}  // namespace zvec::detail
