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

#include "python_collection.h"
#include <algorithm>
#include <limits>
#include <string_view>
#include <unordered_set>
#include <arrow/api.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <zvec/db/collection.h>
#include <zvec/db/doc_iterator.h>
#include "db/collection_fetch_internal.h"
#include "db/collection_query_internal.h"
#include "db/common/result_memory_allowance.h"
#include "python_doc.h"

namespace zvec {

namespace {

detail::ResultMemoryAllowance &forward_result_allowance();
void reserve_query_forward_results(const DocPtrList &docs,
                                   const CollectionSchema &schema,
                                   detail::ResultMemoryAllowance::Lease &lease);

// Batch-materialize a DocPtrList into a list of (id, score, fields, vectors)
// tuples in a single GIL-held section, avoiding per-doc _Doc wrappers and
// per-doc Python->C++ crossings on the hot query path. The forward/vector field
// lists are resolved once per batch rather than once per doc.
py::list docs_to_tuples(const DocPtrList &docs,
                        const CollectionSchema &schema) {
  const auto forward_fields = schema.forward_fields();
  const auto vector_fields = schema.vector_fields();
  py::list out(docs.size());
  for (size_t i = 0; i < docs.size(); ++i) {
    if (docs[i]) {
      out[i] = ZVecPyDoc::doc_to_tuple_with_fields(*docs[i], forward_fields,
                                                   vector_fields);
    } else {
      out[i] = py::none();
    }
  }
  return out;
}

}  // namespace

inline void throw_if_error(const Status &status) {
  switch (status.code()) {
    case StatusCode::OK:
      return;
    case StatusCode::NOT_FOUND:
      throw py::key_error(status.message());
    case StatusCode::INVALID_ARGUMENT:
      throw py::value_error(status.message());
    case StatusCode::RESOURCE_EXHAUSTED:
      PyErr_SetString(PyExc_MemoryError, status.message().c_str());
      throw py::error_already_set();
    case StatusCode::INTERNAL_ERROR:
    case StatusCode::ALREADY_EXISTS:
    case StatusCode::NOT_SUPPORTED:
    case StatusCode::PERMISSION_DENIED:
    case StatusCode::FAILED_PRECONDITION:
    case StatusCode::UNKNOWN:
    default:
      throw std::runtime_error(status.message());
  }
}


template <typename T>
T unwrap_expected(const tl::expected<T, Status> &exp) {
  if (exp.has_value()) {
    return exp.value();
  }
  throw_if_error(exp.error());
  return T{};
}

template <typename T>
T unwrap_expected(tl::expected<T, Status> &&exp) {
  if (exp.has_value()) {
    return std::move(exp).value();
  }
  throw_if_error(exp.error());
  return T{};
}

// Run a query with the GIL released, capturing docs + schema atomically under a
// single schema read lock (internal::query_result_snapshot), then materialize
// the batch into tuples after the GIL is reacquired and the read lock released.
template <typename Query>
py::list execute_for_python(const Collection &collection, const Query &query) {
  const bool budgeted = !unwrap_expected(collection.options()).enable_mmap_ &&
                        ailego::MemoryLimitPool::get_instance().capacity();
  detail::ResultMemoryAllowance::Lease memory(
      budgeted ? &forward_result_allowance() : nullptr);
  Result<internal::QueryResultSnapshot> result;
  {
    py::gil_scoped_release release;
    result = internal::query_result_snapshot(collection, query);
  }
  // GIL restored, schema read lock already released.
  auto snapshot = unwrap_expected(std::move(result));
  reserve_query_forward_results(snapshot.docs, *snapshot.schema, memory);
  return docs_to_tuples(snapshot.docs, *snapshot.schema);
}

// Transfer ownership of the native buffer without a per-result Python loop.
template <typename T>
py::array_t<T> owned_array(std::vector<T> values) {
  auto buffer = std::make_unique<std::vector<T>>(std::move(values));
  const auto size = static_cast<py::ssize_t>(buffer->size());
  auto *ptr = buffer.get();
  py::capsule owner(ptr,
                    [](void *p) { delete static_cast<std::vector<T> *>(p); });
  buffer.release();
  return py::array_t<T>({size}, {static_cast<py::ssize_t>(sizeof(T))},
                        ptr->data(), owner);
}

namespace {

detail::ResultMemoryAllowance &forward_result_allowance() {
  // Construct the pool first so this reservation is released before it at
  // process shutdown. Collection close must not refund retained heap backing.
  return detail::SharedResultMemoryAllowance();
}

class PythonResultEstimate {
 public:
  void add(size_t bytes, size_t count = 1) {
    if (count && bytes > (std::numeric_limits<size_t>::max() - bytes_) / count)
      throw std::bad_alloc();
    bytes_ += bytes * count;
  }
  void string(std::string_view value) {
    add(128);  // Unicode object/header and allocator size-class rounding.
    // A single non-BMP character can make the whole Python string UCS-4.
    // ASCII remains one byte per character; never assume UTF-8 byte length
    // also bounds the in-memory representation of a mixed Unicode string.
    const bool ascii = std::all_of(value.begin(), value.end(),
                                   [](unsigned char c) { return c < 128; });
    add(value.size(), ascii ? 1 : 4);
  }
  size_t bytes() const {
    return bytes_;
  }

 private:
  size_t bytes_{0};
};

template <typename T>
void estimate_query_array(const Doc &doc, const std::string &name,
                          PythonResultEstimate &estimate) {
  const auto &values = doc.get_ref<std::vector<T>>(name);
  estimate.add(128);
  estimate.add(sizeof(void *), values.size());
  if constexpr (std::is_same_v<T, std::string>) {
    for (const auto &value : values) estimate.string(value);
  } else {
    estimate.add(64, values.size());
  }
}

void reserve_query_forward_results(
    const DocPtrList &docs, const CollectionSchema &schema,
    detail::ResultMemoryAllowance::Lease &lease) {
  if (!lease.enabled()) return;
  try {
    py::gil_scoped_release release;
    PythonResultEstimate estimate;
    const auto fields = schema.forward_fields();
    for (const auto &doc : docs) {
      estimate.add(512);
      if (!doc) continue;
      estimate.string(doc->pk_ref());
      for (const auto &field : fields) {
        const auto &name = field->name();
        if (!doc->has_value(name)) continue;
        estimate.string(name);
        estimate.add(64);
        switch (field->data_type()) {
          case DataType::STRING:
          case DataType::BINARY:
            estimate.string(doc->get_ref<std::string>(name));
            break;
          case DataType::ARRAY_STRING:
          case DataType::ARRAY_BINARY:
            estimate_query_array<std::string>(*doc, name, estimate);
            break;
          case DataType::ARRAY_BOOL:
            estimate_query_array<bool>(*doc, name, estimate);
            break;
          case DataType::ARRAY_INT32:
            estimate_query_array<int32_t>(*doc, name, estimate);
            break;
          case DataType::ARRAY_INT64:
            estimate_query_array<int64_t>(*doc, name, estimate);
            break;
          case DataType::ARRAY_UINT32:
            estimate_query_array<uint32_t>(*doc, name, estimate);
            break;
          case DataType::ARRAY_UINT64:
            estimate_query_array<uint64_t>(*doc, name, estimate);
            break;
          case DataType::ARRAY_FLOAT:
            estimate_query_array<float>(*doc, name, estimate);
            break;
          case DataType::ARRAY_DOUBLE:
            estimate_query_array<double>(*doc, name, estimate);
            break;
          default:
            estimate.add(64);
            break;
        }
      }
    }
    lease.add(estimate.bytes());
  } catch (const std::bad_alloc &) {
    PyErr_SetString(PyExc_MemoryError,
                    "Insufficient shared budget for Python query results; "
                    "reduce topk or output fields");
    throw py::error_already_set();
  }
}

void estimate_forward_value(const arrow::Array &array, int64_t row,
                            PythonResultEstimate &estimate) {
  if (array.IsNull(row)) return;
  switch (array.type_id()) {
    case arrow::Type::STRING: {
      const auto value =
          static_cast<const arrow::StringArray &>(array).GetView(row);
      estimate.string({value.data(), value.size()});
      break;
    }
    case arrow::Type::LIST: {
      const auto &list = static_cast<const arrow::ListArray &>(array);
      if (list.values()->type_id() == arrow::Type::BINARY) return;
      estimate.add(128);
      const auto offset = list.value_offset(row);
      const auto length = list.value_length(row);
      estimate.add(sizeof(void *), static_cast<size_t>(length));
      for (int64_t i = 0; i < length; ++i)
        estimate_forward_value(*list.values(), offset + i, estimate);
      break;
    }
    default:
      estimate.add(64);  // Numeric objects; bool/None are conservative here.
      break;
  }
}

void reserve_forward_results(const internal::ForwardFetchResult &forward,
                             detail::ResultMemoryAllowance::Lease &lease) {
  if (!lease.enabled()) return;
  try {
    py::gil_scoped_release release;
    PythonResultEstimate estimate;
    for (const auto &table : forward.tables) {
      // Doc/tuple/outer-dict entries, then field-dict slots. Include the
      // wrapper objects created after FetchTuples returns, not only payloads.
      estimate.add(512, static_cast<size_t>(table->num_rows()));
      for (int c = 1; c < table->num_columns(); ++c) {
        if (c >= 2) {
          estimate.string(table->field(c)->name());
          estimate.add(64, static_cast<size_t>(table->num_rows()));
        }
        for (const auto &chunk : table->column(c)->chunks()) {
          for (int64_t row = 0; row < chunk->length(); ++row)
            estimate_forward_value(*chunk, row, estimate);
        }
      }
    }
    // Reserve the entire native chunk before creating any Python values.
    // The lease accumulates all chunks of this API call, including split ones.
    lease.add(estimate.bytes());
  } catch (const std::bad_alloc &) {
    PyErr_SetString(PyExc_MemoryError,
                    "Insufficient shared budget for Python fetch results; "
                    "reduce fetch batch bytes or concurrent requests");
    throw py::error_already_set();
  }
}

py::object forward_value_to_python(const arrow::Array &array, int64_t row) {
  if (array.IsNull(row)) return py::none();
  switch (array.type_id()) {
    case arrow::Type::STRING: {
      const auto value =
          static_cast<const arrow::StringArray &>(array).GetView(row);
      return py::str(value.data(), value.size());
    }
    case arrow::Type::BOOL:
      return py::bool_(
          static_cast<const arrow::BooleanArray &>(array).Value(row));
    case arrow::Type::INT32:
      return py::int_(static_cast<const arrow::Int32Array &>(array).Value(row));
    case arrow::Type::INT64:
      return py::int_(static_cast<const arrow::Int64Array &>(array).Value(row));
    case arrow::Type::UINT32:
      return py::int_(
          static_cast<const arrow::UInt32Array &>(array).Value(row));
    case arrow::Type::UINT64:
      return py::int_(
          static_cast<const arrow::UInt64Array &>(array).Value(row));
    case arrow::Type::FLOAT:
      return py::float_(
          static_cast<const arrow::FloatArray &>(array).Value(row));
    case arrow::Type::DOUBLE:
      return py::float_(
          static_cast<const arrow::DoubleArray &>(array).Value(row));
    case arrow::Type::LIST: {
      const auto &list = static_cast<const arrow::ListArray &>(array);
      if (list.values()->type_id() == arrow::Type::BINARY) return py::none();
      const auto offset = list.value_offset(row);
      const auto length = list.value_length(row);
      py::list result(length);
      for (int64_t i = 0; i < length; ++i) {
        result[i] = forward_value_to_python(*list.values(), offset + i);
      }
      return result;
    }
    default:
      // Binary fields are not exposed by the Python DataType API; retain the
      // legacy Doc converter's None result for unsupported field types.
      return py::none();
  }
}

py::object forward_value_to_python(const arrow::ChunkedArray &column,
                                   int64_t row, bool *is_null = nullptr) {
  for (const auto &chunk : column.chunks()) {
    if (row < chunk->length()) {
      if (is_null) *is_null = chunk->IsNull(row);
      return forward_value_to_python(*chunk, row);
    }
    row -= chunk->length();
  }
  throw std::runtime_error("Invalid forward fetch row");
}

void forward_to_tuples(internal::ForwardFetchResult &forward, py::dict &out) {
  for (auto &table : forward.tables) {
    std::vector<py::str> names;
    for (int c = 2; c < table->num_columns(); ++c) {
      names.emplace_back(table->field(c)->name());
    }
    for (int64_t row = 0; row < table->num_rows(); ++row) {
      auto key = forward_value_to_python(*table->column(1), row);
      if (key.is_none()) throw std::runtime_error("Missing forward fetch key");
      py::dict fields;
      for (int c = 2; c < table->num_columns(); ++c) {
        bool is_null;
        auto value = forward_value_to_python(*table->column(c), row, &is_null);
        // Null fields are absent from Doc too. Unsupported non-null fields
        // keep the legacy converter's None behavior.
        if (!is_null) fields[names[c - 2]] = std::move(value);
      }
      out[key] = py::make_tuple(key, 0.0, std::move(fields), py::none());
    }
    table.reset();
  }
}

internal::FetchResultSnapshot fetch_snapshot_for_python(
    const Collection &collection, const std::vector<std::string> &pks,
    const std::optional<std::vector<std::string>> &output_fields,
    bool include_vector, const CollectionSchema *schema) {
  Result<internal::FetchResultSnapshot> result;
  {
    py::gil_scoped_release release;
    result = internal::fetch_result_snapshot(collection, pks, output_fields,
                                             include_vector, schema, true);
  }
  return unwrap_expected(std::move(result));
}

void fetch_chunk_for_python(
    const Collection &collection, const std::vector<std::string> &pks,
    const std::optional<std::vector<std::string>> &output_fields,
    bool include_vector, const CollectionSchema &schema, py::dict &out,
    detail::ResultMemoryAllowance::Lease &allowance) {
  auto snapshot = fetch_snapshot_for_python(collection, pks, output_fields,
                                            include_vector, &schema);
  if (snapshot.forward) {
    if (snapshot.forward->batch_exceeded) {
      snapshot.forward.reset();
      if (pks.size() <= 1) throw std::runtime_error("Fetch allocation failed");
      const auto middle = pks.begin() + pks.size() / 2;
      fetch_chunk_for_python(collection, {pks.begin(), middle}, output_fields,
                             include_vector, schema, out, allowance);
      fetch_chunk_for_python(collection, {middle, pks.end()}, output_fields,
                             include_vector, schema, out, allowance);
      return;
    }
    reserve_forward_results(*snapshot.forward, allowance);
    forward_to_tuples(*snapshot.forward, out);
    return;
  }
  const auto forward_fields = schema.forward_fields();
  const auto vector_fields = schema.vector_fields();
  while (!snapshot.docs.empty()) {
    auto it = snapshot.docs.begin();
    if (it->second) {
      out[py::str(it->first)] = ZVecPyDoc::doc_to_tuple_with_fields(
          *it->second, forward_fields, vector_fields);
    }
    snapshot.docs.erase(it);
  }
}

}  // namespace

// Only one bounded native result batch is alive at a time. The final Python
// values belong to the caller; a high-water allowance covers materialization,
// not arbitrary caller retention. Release all database locks before taking the
// GIL, including on error/DDL paths. Native reads may run concurrently, but
// only one budgeted call accumulates Python results at a time. Waiting calls
// retain at most their bounded native chunk, not growing Python dictionaries.
py::dict fetch_for_python(
    const Collection &collection, const std::vector<std::string> &pks,
    const std::optional<std::vector<std::string>> &output_fields,
    bool include_vector) {
  auto initial = fetch_snapshot_for_python(collection, {}, output_fields,
                                           include_vector, nullptr);
  const auto schema = std::move(initial.schema);
  const bool budgeted = initial.forward &&
                        !unwrap_expected(collection.options()).enable_mmap_ &&
                        ailego::MemoryLimitPool::get_instance().capacity();
  detail::ResultMemoryAllowance::Lease allowance(
      budgeted ? &forward_result_allowance() : nullptr,
      /*serialize_materialization=*/true);
  py::dict out;
  // Columnar results remain charged; don't shrink them to Doc-sized chunks.
  const size_t batch_rows = initial.forward ? 256 : 32;
  std::unordered_set<std::string_view> seen;
  size_t begin = 0;
  do {
    std::vector<std::string> batch;
    batch.reserve(std::min(batch_rows, pks.size() - begin));
    while (begin < pks.size() && batch.size() < batch_rows) {
      const auto &pk = pks[begin++];
      if (seen.emplace(pk).second) {
        batch.push_back(pk);
      }
    }
    fetch_chunk_for_python(collection, batch, output_fields, include_vector,
                           *schema, out, allowance);
  } while (begin < pks.size());
  return out;
}

void ZVecPyCollection::Initialize(pybind11::module_ &m) {
  py::class_<GroupResult>(m, "_GroupResult")
      .def_readonly("group_by_value", &GroupResult::group_by_value_)
      .def_readonly("docs", &GroupResult::docs_);

  bind_iterator(m);

  py::class_<Collection, Collection::Ptr> collection(m, "_Collection");
  bind_db_methods(collection);
  bind_ddl_methods(collection);
  bind_dml_methods(collection);
  bind_dql_methods(collection);
  collection.def(py::pickle(
      [](const Collection &c) {
        return py::make_tuple(c.path(), c.schema(), c.options());
      },
      [](py::tuple t) {
        if (t.size() != 3) {
          throw std::runtime_error("Invalid tuple size for Collection pickle");
        }
        std::string path = t[0].cast<std::string>();
        auto schema = t[1].cast<CollectionSchema>();
        CollectionOptions options = t[2].cast<CollectionOptions>();
        auto result = Collection::Open(path, options);
        // auto result = Collection::CreateAndOpen(path, schema, options);
        return unwrap_expected(result);
      }));
}

void ZVecPyCollection::bind_db_methods(
    py::class_<Collection, Collection::Ptr> &col) {
  col.def_static("CreateAndOpen",
                 [](const std::string &path, const CollectionSchema &schema,
                    const CollectionOptions &options) {
                   Result<Collection::Ptr> result;
                   {
                     py::gil_scoped_release release;
                     result = Collection::CreateAndOpen(path, schema, options);
                   }
                   return unwrap_expected(result);
                 })
      .def_static("Open", [](const std::string &path,
                             const CollectionOptions &options) {
        Result<Collection::Ptr> result;
        {
          py::gil_scoped_release release;
          result = Collection::Open(path, options);
        }
        return unwrap_expected(result);
      });
}


void ZVecPyCollection::bind_ddl_methods(
    py::class_<Collection, Collection::Ptr> &col) {
  // bind collection properties
  col.def("Path",
          [](const Collection &self) {
            auto ret = self.path();
            return unwrap_expected(ret);
          })
      .def("Options",
           [](const Collection &self) {
             auto ret = self.options();
             return unwrap_expected(ret);
           })
      .def("Schema",
           [](const Collection &self) {
             auto ret = self.schema();
             return unwrap_expected(ret);
           })
      .def("Stats", [](const Collection &self) {
        auto ret = self.stats();
        return unwrap_expected(ret);
      });

  // bind collection ddl methods
  col.def("Close",
          [](Collection &self) {
            Status status;
            {
              py::gil_scoped_release release;
              status = self.close();
            }
            throw_if_error(status);
          })
      .def("Destroy",
           [](Collection &self) {
             Status status;
             {
               py::gil_scoped_release release;
               status = self.destroy();
             }
             throw_if_error(status);
           })
      .def("Flush", [](Collection &self) {
        Status status;
        {
          py::gil_scoped_release release;
          status = self.flush();
        }
        throw_if_error(status);
      });

  // binding index ddl methods
  col.def("CreateIndex",
          [](Collection &self, const std::string &column_name,
             const IndexParams::Ptr &index_options,
             const CreateIndexOptions &options) {
            Status status;
            {
              py::gil_scoped_release release;
              status = self.create_index(column_name, index_options, options);
            }
            throw_if_error(status);
          })
      .def("DropIndex",
           [](Collection &self, const std::string &column_name) {
             Status status;
             {
               py::gil_scoped_release release;
               status = self.drop_index(column_name);
             }
             throw_if_error(status);
           })
      .def("Optimize", [](Collection &self, const OptimizeOptions &options) {
        Status status;
        {
          py::gil_scoped_release release;
          status = self.optimize(options);
        }
        throw_if_error(status);
      });

  // binding column ddl methods
  col.def("AddColumn",
          [](Collection &self, const FieldSchema::Ptr &column_schema,
             const std::string &expression, const AddColumnOptions &options) {
            Status status;
            {
              py::gil_scoped_release release;
              status = self.add_column(column_schema, expression, options);
            }
            throw_if_error(status);
          })
      .def("DropColumn",
           [](Collection &self, std::string &column_name) {
             Status status;
             {
               py::gil_scoped_release release;
               status = self.drop_column(column_name);
             }
             throw_if_error(status);
           })
      .def("AlterColumn", [](Collection &self, std::string &column_name,
                             const std::string &rename,
                             const FieldSchema::Ptr &new_column_schema,
                             const AlterColumnOptions &options) {
        Status status;
        {
          py::gil_scoped_release release;
          status = self.alter_column(column_name, rename, new_column_schema,
                                     options);
        }
        throw_if_error(status);
      });
}

void ZVecPyCollection::bind_dml_methods(
    py::class_<Collection, Collection::Ptr> &col) {
  // bind collection upsert/insert/update/delete methods
  col.def("Insert",
          [](Collection &self, std::vector<Doc> &docs) {
            Result<WriteResults> result;
            {
              py::gil_scoped_release release;
              result = self.insert(docs);
            }
            return unwrap_expected(result);
          })
      .def("Update",
           [](Collection &self, std::vector<Doc> &docs) {
             Result<WriteResults> result;
             {
               py::gil_scoped_release release;
               result = self.update(docs);
             }
             return unwrap_expected(result);
           })
      .def("Upsert",
           [](Collection &self, std::vector<Doc> &docs) {
             Result<WriteResults> result;
             {
               py::gil_scoped_release release;
               result = self.upsert(docs);
             }
             return unwrap_expected(result);
           })
      .def("Delete",
           [](Collection &self, const std::vector<std::string> &pks) {
             Result<WriteResults> result;
             {
               py::gil_scoped_release release;
               result = self.delete_(pks);
             }
             return unwrap_expected(result);
           })
      .def("DeleteByFilter", [](Collection &self, const std::string &filter) {
        Status status;
        {
          py::gil_scoped_release release;
          status = self.delete_by_filter(filter);
        }
        throw_if_error(status);
      });
}

void ZVecPyCollection::bind_dql_methods(
    py::class_<Collection, Collection::Ptr> &col) {
  // Query with the GIL released, then materialize all hits into
  // (id, score, fields, vectors) tuples in one crossing (see docs_to_tuples).
  // execute_for_python captures the docs and the schema snapshot atomically
  // under one read lock, so concurrent DDL cannot desynchronize them, while
  // the binding signature stays unchanged from the legacy per-doc binding.
  col.def(
         "Query",
         [](const Collection &self, const SearchQuery &query) {
           return execute_for_python(self, query);
         },
         py::arg("query"),
         "Execute a query and return results as a list of "
         "(id, score, fields, vectors) tuples materialized in one batch.")
      // MultiQuery: multi query with reranker
      .def(
          "Query",
          [](const Collection &self, const MultiQuery &query) {
            return execute_for_python(self, query);
          },
          py::arg("query"),
          "Execute a multi query with re-ranking and return results as a "
          "list of (id, score, fields, vectors) tuples materialized in one "
          "batch.")
      .def(
          "QueryInternalIds",
          [](const Collection &self, const SearchQuery &query,
             bool return_scores) -> py::object {
            Result<InternalIdsQueryResult> result;
            {
              py::gil_scoped_release release;
              result = self.query_internal_ids(query, return_scores);
            }
            auto output = unwrap_expected(std::move(result));
            auto ids = owned_array(std::move(output.ids));
            if (!return_scores) return ids;
            return py::make_tuple(std::move(ids),
                                  owned_array(std::move(output.scores)));
          },
          py::arg("query"), py::arg("return_scores") = false,
          R"doc(Execute a single query and return internal numeric IDs.

Accepts the same single-target SearchQuery object as Query. Returns an owning
int64 array, or (IDs, float32 scores) with return_scores=True. Scores include
refinement. Eligible dense queries use the specialized low-latency path;
other query shapes use regular Query semantics. Use ResolveInternalIds to
convert IDs to user primary keys.
)doc")
      .def(
          "ResolveInternalIds",
          [](const Collection &self, const std::vector<int64_t> &ids) {
            Result<std::vector<std::optional<std::string>>> result;
            {
              py::gil_scoped_release release;
              result = self.resolve_internal_ids(ids);
            }
            return unwrap_expected(std::move(result));
          },
          py::arg("ids"),
          R"doc(Convert internal numeric IDs to user primary keys.

Positions are preserved. Padding ID -1 and IDs that are deleted or unknown
are returned as None.
)doc")
      .def("GroupByQuery",
           [](const Collection &self, const GroupByVectorQuery &query) {
             Result<GroupResults> result;
             {
               py::gil_scoped_release release;
               result = self.group_by_query(query);
             }
             return unwrap_expected(std::move(result));
           })
      .def(
          "Fetch",
          [](const Collection &self, const std::vector<std::string> &pks,
             const std::optional<std::vector<std::string>> &output_fields,
             bool include_vector) {
            Result<DocPtrMap> result;
            {
              py::gil_scoped_release release;
              result = self.fetch(pks, output_fields, include_vector);
            }
            // return DocPtrMap
            return unwrap_expected(std::move(result));
          },
          py::arg("pks"), py::arg("output_fields") = py::none(),
          py::arg("include_vector") = true)
      .def("FetchTuples", &fetch_for_python, py::arg("pks"),
           py::arg("output_fields") = py::none(),
           py::arg("include_vector") = true)
      .def(
          "CreateIterator",
          [](Collection &self,
             const std::optional<std::vector<std::string>> &output_fields,
             bool include_vector) {
            IteratorOptions options;
            options.output_fields_ = output_fields;
            options.include_vector_ = include_vector;
            Result<DocIterator::Ptr> result;
            {
              py::gil_scoped_release release;
              result = self.create_iterator(options);
            }
            return unwrap_expected(result);
          },
          py::arg("output_fields") = py::none(),
          py::arg("include_vector") = true,
          // The collection must outlive the returned iterator (documented
          // contract); keep it
          // alive automatically in the binding too.
          py::keep_alive<0, 1>(),
          "Create a document iterator to traverse all documents.")
      .def(
          "_debug_hnsw_storage_mode",
          [](const Collection &self, const std::string &column_name) {
            const auto result = self.debug_get_hnsw_storage_mode(column_name);
            return unwrap_expected(result);
          },
          py::arg("column_name"),
          "Debug-only: returns the storage mode of the HNSW entity on the "
          "given vector column. One of 'mmap', 'buffer_pool', 'contiguous'. "
          "Raises KeyError if no HNSW index exists on the column, or "
          "ValueError if the column's index is not an HNSW index. Intended "
          "for introspection and testing only; not part of the stable API.");
}

void ZVecPyCollection::bind_iterator(py::module_ &m) {
  // Document iterator: Python iterator protocol (__iter__ / __next__).
  // Constructed only via Collection.create_iterator (no py::init).
  py::class_<DocIterator, DocIterator::Ptr>(m, "_DocIterator")
      .def("__iter__", [](py::object self) { return self; })
      .def("__next__",
           [](DocIterator &self) {
             Result<Doc::Ptr> result;
             {
               py::gil_scoped_release release;
               result = self.next();
             }
             // !has_value() -> error (raises); value()==nullptr -> EOF
             auto doc = unwrap_expected(result);
             if (doc == nullptr) {
               throw py::stop_iteration();
             }
             return doc;
           })
      .def("close", [](DocIterator &self) {
        py::gil_scoped_release release;
        self.close();
      });
}

}  // namespace zvec
