// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "db/common/result_memory_allowance.h"

namespace zvec::detail {

inline size_t FetchMemoryBytes(size_t count, size_t width, size_t extra = 0) {
  if (width && count > (std::numeric_limits<size_t>::max() - extra) / width)
    throw std::bad_alloc();
  return count * width + extra;
}

// Admit native input copies before pybind's automatic vector/string caster
// would allocate them. The lease precedes its payloads so destruction returns
// active capacity only after all copies have been destroyed.
class PythonFetchInputs {
 public:
  PythonFetchInputs(pybind11::handle ids, pybind11::handle fields,
                    ResultMemoryAllowance *owner)
      // UTF-8 temporaries can acquire a CPython small-object arena even when
      // every ID misses. Match the 64-bit Python result backing quantum.
      : memory_(owner, false, 1024 * 1024) {
    if (!owner) {
      try {
        pks = pybind11::cast<std::vector<std::string>>(ids);
        output_fields =
            pybind11::cast<std::optional<std::vector<std::string>>>(fields);
      } catch (const pybind11::cast_error &) {
        throw pybind11::type_error(
            "Fetch expects sequences of strings for IDs and output fields");
      }
      return;
    }
    read_strings(ids, &pks);
    if (!fields.is_none()) {
      output_fields.emplace();
      read_strings(fields, &*output_fields);
    }
  }

 public:
  static size_t string_bytes(pybind11::handle value) {
    size_t bytes;
    if (PyUnicode_Check(value.ptr())) {
      const auto length = PyUnicode_GetLength(value.ptr());
      if (length < 0) throw pybind11::error_already_set();
      const size_t width =
          PyUnicode_IS_ASCII(value.ptr())                       ? 1
          : PyUnicode_KIND(value.ptr()) == PyUnicode_1BYTE_KIND ? 2
          : PyUnicode_KIND(value.ptr()) == PyUnicode_2BYTE_KIND ? 3
                                                                : 4;
      bytes = FetchMemoryBytes(static_cast<size_t>(length), width);
    } else if (PyBytes_Check(value.ptr())) {
      bytes = static_cast<size_t>(PyBytes_GET_SIZE(value.ptr()));
    } else if (PyByteArray_Check(value.ptr())) {
      bytes = static_cast<size_t>(PyByteArray_GET_SIZE(value.ptr()));
    } else {
      throw pybind11::type_error("Fetch IDs and output fields must be strings");
    }
    // Native string plus transient UTF-8 bytes, terminators and allocator
    // rounding. Do not populate a persistent UTF-8 cache on caller strings.
    return FetchMemoryBytes(bytes, 2, 64);
  }

  static std::string copy_string(pybind11::handle value) {
    if (PyUnicode_Check(value.ptr())) {
      auto encoded = pybind11::reinterpret_steal<pybind11::object>(
          PyUnicode_AsUTF8String(value.ptr()));
      if (!encoded) throw pybind11::error_already_set();
      return {PyBytes_AS_STRING(encoded.ptr()),
              static_cast<size_t>(PyBytes_GET_SIZE(encoded.ptr()))};
    }
    if (PyBytes_Check(value.ptr()))
      return {PyBytes_AS_STRING(value.ptr()),
              static_cast<size_t>(PyBytes_GET_SIZE(value.ptr()))};
    return {PyByteArray_AS_STRING(value.ptr()),
            static_cast<size_t>(PyByteArray_GET_SIZE(value.ptr()))};
  }

 private:
  void admit(size_t bytes) {
    pybind11::gil_scoped_release release;
    memory_.add(bytes);
  }

  void read_strings(pybind11::handle source, std::vector<std::string> *out) {
    if (PyUnicode_Check(source.ptr()) || PyBytes_Check(source.ptr()) ||
        PyByteArray_Check(source.ptr()))
      throw pybind11::type_error("Fetch expects a sequence of strings");
    constexpr size_t kSlotBytes = 3 * sizeof(std::string);
    if (PyList_CheckExact(source.ptr()) || PyTuple_CheckExact(source.ptr())) {
      const bool list = PyList_CheckExact(source.ptr());
      const auto count = [&] {
        return list ? PyList_GET_SIZE(source.ptr())
                    : PyTuple_GET_SIZE(source.ptr());
      };
      const auto item = [&](Py_ssize_t i) -> pybind11::handle {
        return list ? PyList_GET_ITEM(source.ptr(), i)
                    : PyTuple_GET_ITEM(source.ptr(), i);
      };
      size_t admitted = 0;
      while (true) {
        const auto size = count();
        size_t needed = FetchMemoryBytes(static_cast<size_t>(size), kSlotBytes);
        for (Py_ssize_t i = 0; i < size; ++i)
          needed = FetchMemoryBytes(1, string_bytes(item(i)), needed);
        if (needed > admitted) {
          admit(needed - admitted);
          admitted = needed;
          // Admission releases the GIL. Recheck mutable lists/bytearrays
          // before copying, then hold the GIL through the entire conversion.
          continue;
        }
        out->reserve(static_cast<size_t>(size));
        for (Py_ssize_t i = 0; i < size; ++i)
          out->push_back(copy_string(item(i)));
        return;
      }
    }
    // Unknown-length iterables must not first become an unbudgeted tuple.
    // Consume incrementally; rejection may leave an iterator partially read.
    for (auto value :
         pybind11::reinterpret_borrow<pybind11::iterable>(source)) {
      auto held = pybind11::reinterpret_borrow<pybind11::object>(value);
      size_t admitted = 0;
      while (true) {
        const size_t needed =
            FetchMemoryBytes(1, string_bytes(held), kSlotBytes);
        if (needed <= admitted) break;
        admit(needed - admitted);
        admitted = needed;
      }
      out->push_back(copy_string(held));
    }
  }

  ResultMemoryAllowance::Lease memory_;

 public:
  std::vector<std::string> pks;
  std::optional<std::vector<std::string>> output_fields;
};

// Used only while the GIL is released. Charge actual node AND rebound bucket
// allocations before allocating. Replaced buckets remain in this call's
// conservative allowance until it ends; shared backing high water survives
// free(), which need not return ordinary-heap pages to the OS.
template <typename T>
class FetchWorkspaceAllocator {
 public:
  using value_type = T;
  explicit FetchWorkspaceAllocator(
      ResultMemoryAllowance::Lease *memory = nullptr)
      : memory_(memory) {}
  template <typename U>
  FetchWorkspaceAllocator(const FetchWorkspaceAllocator<U> &other)
      : memory_(other.memory_) {}
  T *allocate(size_t n) {
    const auto bytes = FetchMemoryBytes(n, sizeof(T), 32);
    if (memory_) memory_->add(bytes);
    return std::allocator<T>{}.allocate(n);
  }
  void deallocate(T *ptr, size_t n) noexcept {
    std::allocator<T>{}.deallocate(ptr, n);
  }
  template <typename U>
  bool operator==(const FetchWorkspaceAllocator<U> &other) const {
    return memory_ == other.memory_;
  }
  template <typename U>
  bool operator!=(const FetchWorkspaceAllocator<U> &other) const {
    return !(*this == other);
  }

 private:
  template <typename>
  friend class FetchWorkspaceAllocator;
  ResultMemoryAllowance::Lease *memory_;
};

}  // namespace zvec::detail
