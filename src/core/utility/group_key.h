// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <string>
#include <string_view>
#include <zvec/ailego/buffer/memory_reservation.h>

namespace zvec::core {
// Immutable owned group name. Keep std::string callback compatibility while
// charging its allocation before copying. Transparent lookups allocate nothing.
class GroupKey {
 public:
  GroupKey(const std::string &value) : GroupKey(std::string_view(value)) {}
  GroupKey(const char *value) : GroupKey(std::string_view(value)) {}
  GroupKey(std::string_view value) {
    memory_.add_array(value.size() + 1, 2);
    value_.assign(value.data(), value.size());
    memory_.retain_at_most(value_.capacity() + 1);
  }
  GroupKey(const GroupKey &other) : GroupKey(other.value_) {}
  GroupKey(GroupKey &&) noexcept = default;
  operator const std::string &() const {
    return value_;
  }
  const char *data() const {
    return value_.data();
  }
  size_t size() const {
    return value_.size();
  }
  friend bool operator<(const GroupKey &a, const GroupKey &b) {
    return a.value_ < b.value_;
  }
  friend bool operator<(const GroupKey &a, const std::string &b) {
    return a.value_ < b;
  }
  friend bool operator<(const std::string &a, const GroupKey &b) {
    return a < b.value_;
  }

 private:
  ailego::MemoryReservation memory_;
  std::string value_;
};
}  // namespace zvec::core
