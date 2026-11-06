// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <pybind11/numpy.h>
#include "db/common/binding_memory.h"
#include "db/common/constants.h"
#include "python_fetch_memory.h"

namespace zvec::detail {

// Own admission across the public Python query-preparation pipeline, before
// NumPy conversion or pybind's automatic string/vector setters can allocate.
// Inputs follow the query contract: callers must not mutate them during a call.
class PythonQueryMemory {
 public:
  PythonQueryMemory(ResultMemoryAllowance *owner, pybind11::handle queries,
                    pybind11::handle filter, pybind11::handle fields,
                    pybind11::handle group_field)
      : memory_(owner, false, 1024 * 1024),
        queries_(sequence(queries, false)),
        fields_(sequence(fields, true)) {
    if (!memory_.enabled()) return;
    size_t admitted = 0;
    while (true) {
      const auto routes = std::max(size_t{1}, pybind11::len(queries_));
      // Each route copies common fields; MultiQuery adds another set and
      // copies targets through SubQuery and its vector caster.
      size_t bytes = BindingMemoryBytes(routes, 1024);
      bytes = BindingMemoryBytes(routes + 2, text_bytes(filter), bytes);
      bytes = BindingMemoryBytes(4, text_bytes(group_field), bytes);
      if (!fields_.is_none()) {
        const auto count = pybind11::len(fields_);
        if (count > kMaxOutputFieldSize) too_many_fields();
        size_t field_bytes = BindingMemoryBytes(count, 3 * sizeof(std::string));
        for (auto field : fields_)
          field_bytes = BindingMemoryBytes(1, text_bytes(field), field_bytes);
        bytes = BindingMemoryBytes(routes + 2, field_bytes, bytes);
      }
      for (auto item : queries_) {
        auto query = pybind11::reinterpret_borrow<pybind11::object>(item);
        const auto name = query.attr("field_name");
        bytes = BindingMemoryBytes(4, text_bytes(name), bytes);
        const auto id = query.attr("id");
        bytes = BindingMemoryBytes(4, text_bytes(id), bytes);
        const auto vector = query.attr("vector");
        if (!vector.is_none()) {
          bytes = BindingMemoryBytes(1, vector_bytes(vector), bytes);
        }
        const auto fts = query.attr("fts");
        if (!fts.is_none()) {
          bytes = BindingMemoryBytes(4, text_bytes(fts.attr("query_string")),
                                     bytes);
          bytes = BindingMemoryBytes(4, text_bytes(fts.attr("match_string")),
                                     bytes);
        }
      }
      if (bytes <= admitted) break;
      admit(bytes - admitted);
      admitted = bytes;
      // Admission releases the GIL; recheck containers before publishing.
    }
  }

  const pybind11::object &queries() const {
    return queries_;
  }
  const pybind11::object &output_fields() const {
    return fields_;
  }

  // ID-based queries discover their vector only after the budgeted Fetch.
  // Retain its conversion allowance with the rest of this query preparation.
  void reserve_vector(pybind11::handle value) {
    if (!memory_.enabled()) return;
    size_t admitted = 0;
    while (true) {
      const auto bytes = vector_bytes(value);
      if (bytes <= admitted) return;
      admit(bytes - admitted);
      admitted = bytes;
    }
  }

 private:
  static void too_many_fields() {
    throw pybind11::value_error(
        "Invalid query: too many output fields, the maximum allowed is " +
        std::to_string(kMaxOutputFieldSize));
  }
  static size_t text_bytes(pybind11::handle value) {
    if (value.is_none()) return 0;
    if (PyUnicode_Check(value.ptr()) || PyBytes_Check(value.ptr()) ||
        PyByteArray_Check(value.ptr()))
      return BindingMemoryBytes(2, PythonFetchInputs::string_bytes(value));
    // Preserve public validation of invalid field/FTS parameter types.
    return 64;
  }
  static size_t dense_elements(pybind11::handle value, size_t depth = 0) {
    if (depth > 64)
      throw pybind11::value_error("Query vector nesting is too deep");
    if (pybind11::isinstance<pybind11::array>(value))
      return static_cast<size_t>(
          pybind11::reinterpret_borrow<pybind11::array>(value).size());
    if (!PyList_Check(value.ptr()) && !PyTuple_Check(value.ptr())) return 1;
    const auto count = pybind11::len(value);
    size_t elements = 0;
    for (size_t i = 0; i < count; ++i) {
      auto item = PyList_Check(value.ptr()) ? PyList_GET_ITEM(value.ptr(), i)
                                            : PyTuple_GET_ITEM(value.ptr(), i);
      elements =
          BindingMemoryBytes(1, dense_elements(item, depth + 1), elements);
    }
    return elements;
  }
  static size_t vector_bytes(pybind11::handle value) {
    if (PyDict_Check(value.ptr()))
      return BindingMemoryBytes(pybind11::len(value), 128, 256);
    // dtype conversion/flattening plus owned SubQuery/MultiQuery vector data.
    return BindingMemoryBytes(dense_elements(value), 32, 256);
  }
  void admit(size_t bytes) {
    pybind11::gil_scoped_release release;
    memory_.add(bytes);
  }
  pybind11::object sequence(pybind11::handle value, bool fields) {
    if (value.is_none())
      return fields ? pybind11::object(pybind11::none())
                    : pybind11::object(pybind11::tuple());
    if (PyList_CheckExact(value.ptr()) || PyTuple_CheckExact(value.ptr())) {
      if (fields && pybind11::len(value) > kMaxOutputFieldSize)
        too_many_fields();
      return pybind11::reinterpret_borrow<pybind11::object>(value);
    }
    if (PyUnicode_Check(value.ptr()) || PyBytes_Check(value.ptr()) ||
        PyByteArray_Check(value.ptr()))
      throw pybind11::type_error("Query inputs expect a sequence");
    pybind11::list out;
    for (auto item : pybind11::reinterpret_borrow<pybind11::iterable>(value)) {
      auto held = pybind11::reinterpret_borrow<pybind11::object>(item);
      if (fields && out.size() == kMaxOutputFieldSize) too_many_fields();
      admit(64);
      out.append(held);
    }
    return out;
  }
  ResultMemoryAllowance::Lease memory_;
  pybind11::object queries_;
  pybind11::object fields_;
};

}  // namespace zvec::detail
