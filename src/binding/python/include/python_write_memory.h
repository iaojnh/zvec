// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "db/common/constants.h"
#include "db/common/document_memory.h"
#include "python_doc.h"
#include "python_fetch_memory.h"

namespace zvec::detail {

// The public wrapper creates this owner BEFORE convert_to_cpp_doc. Every
// intermediate Doc retains it, including when a Python traceback survives a
// failed conversion. No materialization mutex is held across Python calls.
class PythonWriteMemory {
 public:
  explicit PythonWriteMemory(ResultMemoryAllowance *owner, size_t count)
      : memory_(std::make_shared<ResultMemoryAllowance::Lease>(owner, false,
                                                               1024 * 1024)) {
    if (count > kMaxWriteBatchSize)
      throw pybind11::value_error(
          "Too many docs: exceeds max write batch size of " +
          std::to_string(kMaxWriteBatchSize));
    admit(BindingMemoryBytes(count, sizeof(void *) + 64));
  }

  Doc::Ptr new_doc() {
    if (created_ == kMaxWriteBatchSize)
      throw pybind11::value_error(
          "Too many docs: exceeds max write batch size of " +
          std::to_string(kMaxWriteBatchSize));
    ResultMemoryAllowance::PendingCopy pending;
    reserve(sizeof(Doc) + 256, pending);
    pending.start();
    auto doc = std::make_shared<Doc>();
    if (memory_->enabled()) DocumentMemoryAccess::retain(*doc, memory_);
    ++created_;
    return doc;
  }

  void set_pk(Doc &doc, pybind11::handle value) {
    ResultMemoryAllowance::PendingCopy pending;
    reserve_value(pending, [&] { return PythonFetchInputs::string_bytes(value); });
    pending.start();
    doc.set_pk(PythonFetchInputs::copy_string(value));
  }

  pybind11::bytes field_name(pybind11::handle value) {
    // Schema lookup itself crosses a string caster before set_any is reached.
    // Admit that copy too, and avoid a UTF-8 cache on the caller's Unicode key.
    ResultMemoryAllowance::PendingCopy pending;
    reserve_value(pending, [&] { return PythonFetchInputs::string_bytes(value); });
    pending.start();
    return pybind11::bytes(PythonFetchInputs::copy_string(value));
  }

  bool set_any(Doc &doc, const std::string &field, const FieldSchema &schema,
               pybind11::object value) {
    const auto type = schema.data_type();
    const bool strings = type == DataType::ARRAY_STRING;
    const bool sparse = type == DataType::SPARSE_VECTOR_FP16 ||
                        type == DataType::SPARSE_VECTOR_FP32;
    const bool sequence = schema.is_array_type() || schema.is_vector_field();
    // Match the STL caster: a scalar string/bytes or dict is not a dense
    // vector/array. Iterating it here would silently accept chars or keys.
    if (sequence && !sparse &&
        (PyUnicode_Check(value.ptr()) || PyBytes_Check(value.ptr()) ||
         PyByteArray_Check(value.ptr()) || PyDict_Check(value.ptr())))
      throw pybind11::type_error("Field '" + field +
                                 "': expected an array or vector sequence");
    if (type == DataType::STRING && !value.is_none() &&
        !PyUnicode_Check(value.ptr()) && !PyBytes_Check(value.ptr()) &&
        !PyByteArray_Check(value.ptr())) {
      throw pybind11::type_error(
          "Field '" + field + "': expected STRING, got " +
          std::string(pybind11::str(pybind11::type::of(value))));
    }
    // The STL caster otherwise materializes unknown-length iterables before
    // we can account for them. Subclasses may also override iteration/length;
    // normalize them once, incrementally, under the same lease.
    if (!value.is_none() && sequence && !sparse &&
        !PyList_CheckExact(value.ptr()) && !PyTuple_CheckExact(value.ptr())) {
      pybind11::list staged;
      for (auto item :
           pybind11::reinterpret_borrow<pybind11::iterable>(value)) {
        auto held = pybind11::reinterpret_borrow<pybind11::object>(item);
        ResultMemoryAllowance::PendingCopy pending;
        reserve_value(pending, [&] {
          return strings ? BindingMemoryBytes(
                               1, PythonFetchInputs::string_bytes(held), 128)
                         : size_t{128};
        });
        pending.start();
        staged.append(held);
      }
      value = std::move(staged);
    }
    ResultMemoryAllowance::PendingCopy pending;
    reserve_value(pending, [&] {
      size_t bytes = BindingMemoryBytes(1, field.size(), 256);
      if (value.is_none()) return bytes;
      if (type == DataType::STRING)
        return BindingMemoryBytes(1, PythonFetchInputs::string_bytes(value),
                                  bytes);
      if (sequence) {
        // The sparse caster traverses the native dict, ignoring an overridden
        // __len__. Charge the same entries it will actually copy.
        const auto count = PyDict_Check(value.ptr())
                               ? static_cast<size_t>(PyDict_Size(value.ptr()))
                               : pybind11::len(value);
        bytes = BindingMemoryBytes(count,
                                   sparse    ? 32
                                   : strings ? 64
                                             : 8,
                                   bytes);
        if (strings) {
          // Inspect borrowed items only; do not first copy a tuple or populate
          // UTF-8 caches. Recheck after every admission that releases the GIL.
          for (size_t i = 0; i < count; ++i) {
            auto item = PyList_Check(value.ptr())
                            ? PyList_GET_ITEM(value.ptr(), i)
                            : PyTuple_GET_ITEM(value.ptr(), i);
            bytes = BindingMemoryBytes(1, PythonFetchInputs::string_bytes(item),
                                       bytes);
          }
        }
      }
      return bytes;
    });
    pending.start();
    // Avoid pybind's persistent UTF-8 cache on caller-owned Unicode objects:
    // such a cache could outlive the staging lease after every write.
    if (type == DataType::STRING && !value.is_none())
      return doc.set(field, PythonFetchInputs::copy_string(value));
    if (strings && !value.is_none()) {
      std::vector<std::string> copied;
      copied.reserve(pybind11::len(value));
      for (auto item : value)
        copied.push_back(PythonFetchInputs::copy_string(item));
      return doc.set(field, std::move(copied));
    }
    return ZVecPyDoc::set_field(doc, field, schema, value);
  }

 private:
  void admit(size_t bytes) {
    pybind11::gil_scoped_release release;
    memory_->add(bytes);
  }
  void reserve(size_t bytes, ResultMemoryAllowance::PendingCopy &pending) {
    pybind11::gil_scoped_release release;
    memory_->add_pending(bytes, pending);
  }
  template <typename Estimate>
  void reserve_value(ResultMemoryAllowance::PendingCopy &pending,
                     Estimate estimate) {
    size_t admitted = 0;
    while (true) {
      const size_t needed = estimate();
      if (needed <= admitted) return;
      reserve(needed - admitted, pending);
      admitted = needed;
    }
  }
  std::shared_ptr<ResultMemoryAllowance::Lease> memory_;
  size_t created_{0};
};

// The private binding also accepts independently constructed _Docs. Copy them
// explicitly so pybind cannot allocate an entire vector before entering us.
class PythonWriteInputs {
 public:
  PythonWriteInputs(pybind11::handle source, ResultMemoryAllowance *owner)
      : memory_(owner, false, 1024 * 1024) {
    for (auto item : pybind11::reinterpret_borrow<pybind11::iterable>(source)) {
      auto held = pybind11::reinterpret_borrow<pybind11::object>(item);
      if (docs.size() == kMaxWriteBatchSize)
        throw pybind11::value_error(
            "Too many docs: exceeds max write batch size of " +
            std::to_string(kMaxWriteBatchSize));
      const auto &doc = pybind11::cast<const Doc &>(held);
      ResultMemoryAllowance::PendingCopy pending;
      size_t admitted = 0;
      while (true) {
        const auto needed = BindingMemoryBytes(
            1, DocumentMemoryAccess::copy_bytes(doc), 3 * sizeof(Doc) + 512);
        if (needed <= admitted) break;
        {
          pybind11::gil_scoped_release release;
          memory_.add_pending(needed - admitted, pending);
        }
        admitted = needed;
      }
      pending.start();
      docs.push_back(doc);
    }
  }

 private:
  ResultMemoryAllowance::Lease memory_;

 public:
  std::vector<Doc> docs;
};

}  // namespace zvec::detail
