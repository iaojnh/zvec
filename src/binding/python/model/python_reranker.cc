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

#include "python_reranker.h"
#include <algorithm>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <zvec/db/collection.h>
#include <zvec/db/reranker.h>
#include "db/common/binding_memory.h"
#include "python_fetch_memory.h"

namespace zvec {
namespace {

using RerankLease = detail::ResultMemoryAllowance::Lease;
using PendingRerankCopy = detail::ResultMemoryAllowance::PendingCopy;
constexpr size_t kPythonArena = 1024 * 1024;

std::shared_ptr<RerankLease> new_query_lease() {
  return std::make_shared<RerankLease>(&detail::SharedResultMemoryAllowance(),
                                       false, kPythonArena);
}

void admit(RerankLease &memory, size_t bytes) {
  if (!bytes || !ailego::MemoryLimitPool::get_instance().capacity()) return;
  py::gil_scoped_release release;
  memory.add(bytes);
}

template <typename Estimate>
void admit_current(RerankLease &memory, PendingRerankCopy &pending,
                    Estimate estimate) {
  size_t admitted = 0;
  while (true) {
    const size_t bytes = estimate();
    if (bytes <= admitted) {
      pending.start();
      return;
    }
    if (ailego::MemoryLimitPool::get_instance().capacity()) {
      py::gil_scoped_release release;
      memory.add_pending(bytes - admitted, pending);
    }
    admitted = bytes;
  }
}

template <typename T>
std::vector<T> stage_values(py::iterable source, RerankLease &memory) {
  std::vector<T> values;
  for (auto item : source) {
    auto held = py::reinterpret_borrow<py::object>(item);
    admit(memory, 3 * sizeof(T));
    values.push_back(py::cast<T>(held));
  }
  return values;
}

struct PythonWeightedParams {
  explicit PythonWeightedParams(py::iterable source) {
    set_weights(source);
  }

  void set_weights(py::iterable source) {
    auto next = new_query_lease();
    auto values = stage_values<double>(source, *next);
    next->retain_at_most(
        detail::BindingMemoryBytes(values.capacity(), sizeof(double)));
    weights = std::move(values);
    memory = std::move(next);
  }

  std::shared_ptr<RerankLease> memory;
  std::vector<double> weights;
};

struct RouteStorage {
  std::shared_ptr<RerankLease> memory;
  std::vector<SubQuery> retired;
};

// The native query is destroyed before any lease for its input fields.
struct MultiQueryMemory {
  std::shared_ptr<RouteStorage> routes;
  std::shared_ptr<RerankLease> filter_memory;
  std::shared_ptr<RerankLease> fields;
  std::shared_ptr<RerankLease> weights;
};

struct PythonMultiQuery : MultiQueryMemory, MultiQuery {
  ~PythonMultiQuery() {
    retire_routes();
  }

  void retire_routes() {
    // Moving the vector preserves references already exported to Python.
    if (routes && routes.use_count() > 1)
      routes->retired = std::move(queries);
  }
};

size_t target_copy_bytes(const QueryTarget &target) {
  const auto add_string = [](size_t bytes, std::string_view text) {
    return detail::BindingMemoryBytes(1, text.size(), bytes);
  };
  size_t bytes = add_string(128, target.field_name_);
  if (auto vector = target.get_vector_view()) {
    bytes = add_string(bytes, vector->query_vector_);
    bytes = add_string(bytes, vector->sparse_indices_);
    return add_string(bytes, vector->sparse_values_);
  }
  const auto &fts = *target.get_fts_clause();
  return add_string(add_string(bytes, fts.query_string_), fts.match_string_);
}

size_t python_string_bytes(std::string_view value) {
  const bool ascii = std::all_of(value.begin(), value.end(),
                                 [](unsigned char c) { return c < 128; });
  // Account for both the plain Python string and its owner-backed copy.
  return detail::BindingMemoryBytes(value.size(), ascii ? 2 : 8, 256);
}

std::shared_ptr<SubQuery> make_subquery(
    const QueryTarget &target, int count,
    const std::shared_ptr<RerankLease> &memory) {
  struct SubQueryCopy {
    std::shared_ptr<RerankLease> memory;
    SubQuery value;
  };
  auto copy = std::make_shared<SubQueryCopy>();
  copy->memory = memory;
  copy->value = SubQuery{target, count};
  // Keep copied subqueries independent of the original Python buffer.
  if (const auto *view = target.get_vector_view_clause()) {
    copy->value.target_.clause_ =
        VectorClause{std::string(view->query_vector_),
                     std::string(view->sparse_indices_),
                     std::string(view->sparse_values_)};
  }
  return std::shared_ptr<SubQuery>(copy, &copy->value);
}

template <typename Estimate, typename Convert>
py::object copy_query_value(Estimate estimate, Convert convert) {
  auto memory = new_query_lease();
  PendingRerankCopy pending;
  admit_current(*memory, pending, estimate);
  auto value = convert();
  if (!memory->bytes()) return value;
  return py::module_::import("zvec.model._budgeted_result")
      .attr("_retain_field")(std::move(value), detail::result_memory_owner(memory));
}

void set_multi_routes(PythonMultiQuery &query, py::iterable source) {
  auto next = std::make_shared<RouteStorage>();
  next->memory = new_query_lease();
  std::vector<SubQuery> values;
  for (auto item : source) {
    auto held = py::reinterpret_borrow<py::object>(item);
    const auto &route = held.cast<const SubQuery &>();
    PendingRerankCopy pending;
    admit_current(*next->memory, pending, [&] {
      return detail::BindingMemoryBytes(
          1, target_copy_bytes(route.target_), 3 * sizeof(SubQuery));
    });
    values.push_back(route);
  }
  query.retire_routes();
  query.queries = std::move(values);
  query.routes = std::move(next);
}

py::object multi_route_handles(PythonMultiQuery &query) {
  auto memory = new_query_lease();
  PendingRerankCopy pending;
  admit_current(*memory, pending, [&] {
    return detail::BindingMemoryBytes(query.queries.size(), 256);
  });
  struct RouteExport {
    std::shared_ptr<RerankLease> memory;
    std::shared_ptr<RouteStorage> storage;
  };
  auto owner = std::make_shared<RouteExport>(
      RouteExport{memory, query.routes});
  py::list out;
  for (auto &route : query.queries)
    out.append(std::shared_ptr<SubQuery>(owner, &route));
  if (!memory->bytes()) return out;
  return py::module_::import("zvec.model._budgeted_result")
      .attr("_ResultList")(out, detail::result_memory_owner(memory));
}

void set_multi_filter(PythonMultiQuery &query, py::object source) {
  auto next = new_query_lease();
  PendingRerankCopy pending;
  admit_current(*next, pending, [&] {
    return detail::PythonFetchInputs::utf8_size_bound(source)
               ? detail::PythonFetchInputs::string_bytes(source)
               : size_t{0};
  });
  auto value = detail::PythonFetchInputs::copy_string(source);
  next->retain_at_most(detail::BindingMemoryBytes(1, value.capacity(), 32));
  query.filter = std::move(value);
  query.filter_memory = std::move(next);
}

void set_multi_fields(PythonMultiQuery &query, py::object source) {
  if (source.is_none()) {
    query.output_fields.reset();
    query.fields.reset();
    return;
  }
  auto next = new_query_lease();
  std::vector<std::string> values;
  detail::PythonFetchInputs::read_strings(source, &values, *next);
  query.output_fields = std::move(values);
  query.fields = std::move(next);
}

template <typename StageDoc>
std::vector<DocPtrList> stage_routes(py::iterable source, RerankLease &memory,
                                      StageDoc stage_doc) {
  std::vector<DocPtrList> routes;
  for (auto route : source) {
    auto held_route = py::reinterpret_borrow<py::iterable>(route);
    admit(memory, 3 * sizeof(DocPtrList));
    auto &docs = routes.emplace_back();
    for (auto item : held_route) {
      docs.push_back(stage_doc(py::reinterpret_borrow<py::object>(item)));
    }
  }
  return routes;
}

reranker::RerankParams stage_strategy(py::handle params, RerankLease &memory) {
  if (py::isinstance<reranker::RrfParams>(params))
    return params.cast<reranker::RrfParams>();
  if (py::isinstance<PythonWeightedParams>(params)) {
    const auto &weights = params.cast<const PythonWeightedParams &>().weights;
    PendingRerankCopy pending;
    admit_current(memory, pending, [&] {
      return detail::BindingMemoryBytes(weights.size(), sizeof(double));
    });
    return reranker::WeightedParams{weights};
  }
  if (py::isinstance<reranker::CallbackParams>(params))
    return params.cast<reranker::CallbackParams>();
  throw py::type_error(
      "params must be _RrfParams, _WeightedParams, or _CallbackParams");
}

size_t max_ranked_docs(const std::vector<DocPtrList> &routes, int topn) {
  if (topn <= 0) return 0;
  size_t count = 0;
  for (const auto &route : routes)
    count += std::min(static_cast<size_t>(topn) - count, route.size());
  return count;
}

DocPtrList unwrap_rerank(Result<DocPtrList> result) {
  if (result.has_value()) return std::move(result).value();
  const auto &status = result.error();
  if (status.code() == StatusCode::RESOURCE_EXHAUSTED) {
    PyErr_SetString(PyExc_MemoryError, status.message().c_str());
    throw py::error_already_set();
  }
  throw std::runtime_error(status.message());
}

py::object rerank_native_docs(py::object params, py::iterable source,
                              py::iterable fields, int topn) {
  auto &allowance = detail::SharedResultMemoryAllowance();
  RerankLease workspace(&allowance, false, kPythonArena);
  auto strategy = stage_strategy(params, workspace);
  auto schema = stage_values<FieldSchema::Ptr>(fields, workspace);
  auto routes = stage_routes(source, workspace, [&](py::object doc) {
    admit(workspace, 3 * sizeof(Doc::Ptr));
    auto held = doc.cast<Doc::Ptr>();
    if (!held) throw py::type_error("Reranking expects documents");
    return held;
  });
  auto memory = std::make_shared<RerankLease>(&allowance, false, kPythonArena);
  const bool callback = std::holds_alternative<reranker::CallbackParams>(strategy);
  if (!callback) {
    // The native score-based algorithms update their input scores in place.
    // Admit both the vector and its Python wrappers before those mutations.
    const size_t count = max_ranked_docs(routes, topn);
    admit(workspace, detail::BindingMemoryBytes(count, sizeof(Doc::Ptr)));
    admit(*memory, detail::BindingMemoryBytes(count, 128));
  }
  auto ranked = unwrap_rerank(reranker::rerank(strategy, routes, schema, topn));
  const size_t result_bytes = detail::BindingMemoryBytes(ranked.size(), 128);
  if (callback)
    admit(*memory, result_bytes);
  else
    memory->retain_at_most(result_bytes);
  auto result = py::cast(std::move(ranked));
  if (!memory->bytes()) return result;
  return py::module_::import("zvec.model._budgeted_result")
      .attr("_ResultList")(std::move(result), detail::result_memory_owner(memory));
}

size_t mapping_size(py::handle value) {
  if (value.is_none()) return 0;
  return PyDict_Check(value.ptr())
             ? static_cast<size_t>(PyDict_Size(value.ptr()))
             : py::len(value);
}

py::list rerank_python_docs(py::object params, py::iterable source,
                            py::iterable fields, int topn) {
  auto &allowance = detail::SharedResultMemoryAllowance();
  RerankLease workspace(&allowance, false, kPythonArena);
  reranker::RerankParams strategy =
      py::isinstance<py::list>(params)
          ? reranker::RerankParams{reranker::WeightedParams{
                stage_values<double>(params, workspace)}}
          : stage_strategy(params, workspace);
  if (std::holds_alternative<reranker::CallbackParams>(strategy))
    throw py::type_error("Standalone score fusion requires RRF or weights");
  auto schema = stage_values<FieldSchema::Ptr>(fields, workspace);
  std::vector<py::object> originals;
  auto routes = stage_routes(source, workspace, [&](py::object original) {
    py::object key = original.attr("id");
    py::object score = original.attr("score");
    PendingRerankCopy pending;
    admit_current(workspace, pending, [&] {
      return detail::BindingMemoryBytes(
          1, detail::PythonFetchInputs::string_bytes(key),
          sizeof(Doc) + 3 * (sizeof(Doc::Ptr) + sizeof(py::object)) + 128);
    });
    auto doc = std::make_shared<Doc>();
    doc->set_pk(detail::PythonFetchInputs::copy_string(key));
    doc->set_score(score.is_none() ? 0.0f : score.cast<float>());
    doc->set_doc_id(originals.size());
    originals.push_back(std::move(original));
    return doc;
  });
  admit(workspace, detail::BindingMemoryBytes(max_ranked_docs(routes, topn),
                                             sizeof(Doc::Ptr)));
  auto native = [&] {
    py::gil_scoped_release release;
    return reranker::rerank(strategy, routes, schema, topn);
  }();
  auto ranked = unwrap_rerank(std::move(native));

  auto memory = std::make_shared<RerankLease>(&allowance, false, kPythonArena);
  py::object owner;
  py::object dict_type;
  py::list out;
  for (const auto &doc : ranked) {
    const auto &original = originals.at(doc->doc_id());
    PendingRerankCopy pending;
    admit_current(*memory, pending, [&] {
      // _replace stages shallow copies before normalization/wrapping.
      const size_t field_count = mapping_size(original.attr("fields"));
      const size_t vectors = mapping_size(original.attr("vectors"));
      return detail::BindingMemoryBytes(
          vectors, 256, detail::BindingMemoryBytes(field_count, 256, 512));
    });
    auto copy = original.attr("_replace")(py::arg("score") = doc->score());
    if (memory->bytes()) {
      if (!owner) {
        owner = detail::result_memory_owner(memory);
        dict_type = py::module_::import("zvec.model._budgeted_result")
                        .attr("_ResultDict");
      }
      copy.attr("fields") = dict_type(copy.attr("fields"), owner);
      copy.attr("vectors") = dict_type(copy.attr("vectors"), owner);
    }
    out.append(copy);
  }
  return out;
}

}  // namespace

std::shared_ptr<SubQuery> python_subquery_from_search(const SearchQuery &query) {
  auto memory = new_query_lease();
  PendingRerankCopy pending;
  admit_current(*memory, pending, [&] {
    return detail::BindingMemoryBytes(
        1, target_copy_bytes(query.target_), sizeof(SubQuery));
  });
  return make_subquery(query.target_, query.topk_, memory);
}

void ZVecPyReranker::Initialize(py::module_ &m) {
  // Bind RrfParams
  py::class_<reranker::RrfParams>(m, "_RrfParams")
      .def(py::init<int>(), py::arg("rank_constant") = 60)
      .def_readwrite("rank_constant", &reranker::RrfParams::rank_constant);

  // Bind WeightedParams
  py::class_<PythonWeightedParams>(m, "_WeightedParams")
      .def(py::init<py::iterable>(), py::arg("weights"))
      .def_property(
          "weights",
          [](const PythonWeightedParams &params) {
            return copy_query_value(
                [&] { return detail::BindingMemoryBytes(params.weights.size(), 80); },
                [&] { return py::cast(params.weights); });
          },
          &PythonWeightedParams::set_weights);

  // Bind CallbackParams
  py::class_<reranker::CallbackParams>(m, "_CallbackParams")
      .def(py::init<reranker::CallbackParams::Callback>(), py::arg("callback"));

  // Standalone rerank execution function
  m.def("_reranker_rerank", &rerank_native_docs, py::arg("params"),
        py::arg("results"), py::arg("fields"), py::arg("topn"));
  m.def("_rerank_python_docs", &rerank_python_docs, py::arg("params"),
        py::arg("results"), py::arg("fields"), py::arg("topn"));

  // Bind MultiQuery struct
  py::class_<MultiQuery>(m, "_MultiQueryValue");
  py::class_<PythonMultiQuery, MultiQuery>(m, "_MultiQuery",
                                         py::multiple_inheritance())
      .def(py::init<>())
      .def_property("queries", &multi_route_handles, &set_multi_routes)
      .def_readwrite("topk", &MultiQuery::topk)
      .def_property(
          "filter",
          [](const PythonMultiQuery &query) {
            return copy_query_value(
                [&] { return python_string_bytes(query.filter); },
                [&] { return py::str(query.filter); });
          },
          &set_multi_filter)
      .def_readwrite("include_vector", &MultiQuery::include_vector)
      .def_property(
          "output_fields",
          [](const PythonMultiQuery &query) {
            return copy_query_value(
                [&] {
                  if (!query.output_fields) return size_t{0};
                  size_t bytes = detail::BindingMemoryBytes(
                      query.output_fields->size(), 2 * sizeof(void *), 256);
                  for (const auto &field : *query.output_fields)
                    bytes = detail::BindingMemoryBytes(
                        1, python_string_bytes(field), bytes);
                  return bytes;
                },
                [&] { return py::cast(query.output_fields); });
          },
          &set_multi_fields)
      .def(
          "set_rerank_rrf",
          [](PythonMultiQuery &q, int rank_constant) {
            q.rerank = reranker::RrfParams{rank_constant};
            q.weights.reset();
          },
          py::arg("rank_constant") = 60)
      .def(
          "set_rerank_weighted",
          [](PythonMultiQuery &q, py::iterable weights) {
            PythonWeightedParams staged(weights);
            q.rerank = reranker::WeightedParams{std::move(staged.weights)};
            q.weights = std::move(staged.memory);
          },
          py::arg("weights"))
      .def(
          "set_rerank_callback",
          [](PythonMultiQuery &q, reranker::CallbackParams::Callback callback) {
            q.rerank = reranker::CallbackParams{std::move(callback)};
            q.weights.reset();
          },
          py::arg("callback"));
}

}  // namespace zvec
