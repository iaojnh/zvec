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
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

namespace zvec {
namespace core {

// File allocation is independent of both OS mapping pages and buffer pages.
constexpr uint64_t kDiskAlignment = 4096;

inline bool AlignDiskSize(uint64_t size, uint64_t *aligned) {
  if (size > std::numeric_limits<uint64_t>::max() - (kDiskAlignment - 1)) {
    return false;
  }
  *aligned = (size + kDiskAlignment - 1) & ~(kDiskAlignment - 1);
  return true;
}

// A fixed-width wire field. Its object representation is always little endian;
// no native integer or compiler bitfield representation is written to disk.
// Natural alignment permits atomic access to mutable directory counters.
template <typename T>
class DiskUInt {
  static_assert(std::is_unsigned<T>::value, "Unsigned wire integer required");

 public:
  DiskUInt() = default;
  DiskUInt(T value) : encoded_(Encode(value)) {}

  operator T() const {
    return Decode(encoded_);
  }

  DiskUInt &operator=(T value) {
    encoded_ = Encode(value);
    return *this;
  }

  DiskUInt &operator+=(T value) {
    return *this = static_cast<T>(*this) + value;
  }

  DiskUInt &operator-=(T value) {
    return *this = static_cast<T>(*this) - value;
  }

  DiskUInt &operator++() {
    return *this += 1;
  }

  T operator++(int) {
    T old = *this;
    *this += 1;
    return old;
  }

  // Only atomic storage access should bypass the value conversion.
  const T *encoded_storage() const {
    return &encoded_;
  }

  T *encoded_storage() {
    return &encoded_;
  }

  static T Encode(T value) {
    T encoded;
    auto *bytes = reinterpret_cast<unsigned char *>(&encoded);
    for (size_t i = 0; i < sizeof(T); ++i) {
      bytes[i] = static_cast<unsigned char>(value >> (i * 8));
    }
    return encoded;
  }

  static T Decode(T encoded) {
    const auto *bytes = reinterpret_cast<const unsigned char *>(&encoded);
    T value = 0;
    for (size_t i = 0; i < sizeof(T); ++i) {
      value |= static_cast<T>(bytes[i]) << (i * 8);
    }
    return value;
  }

 private:
  T encoded_;
};

using DiskUInt16 = DiskUInt<uint16_t>;
using DiskUInt32 = DiskUInt<uint32_t>;
using DiskUInt64 = DiskUInt<uint64_t>;
static_assert(sizeof(DiskUInt16) == 2 && sizeof(DiskUInt32) == 4 &&
                  sizeof(DiskUInt64) == 8,
              "Wire integers must not contain padding");
static_assert(std::is_trivially_copyable<DiskUInt64>::value,
              "Wire integers must be byte-copyable");

// Vector and neighbor arrays are read directly by SIMD kernels. The format
// specifies IEEE binary16/binary32 and little-endian integer arrays. Reject a
// host needing payload conversion instead of silently interpreting its bytes.
inline bool NativeDiskPayloadSupported() {
  static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
                "Vector files require IEEE 754 binary32");
  const uint32_t value = 1;
  return *reinterpret_cast<const unsigned char *>(&value) == 1;
}

}  // namespace core
}  // namespace zvec
