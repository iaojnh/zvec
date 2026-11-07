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
constexpr size_t kPythonArena = 1024 * 1024;

void admit(RerankLease &memory, size_t bytes) {
  if (!bytes || !ailego::MemoryLimitPool::get_instance().capacity()) return;
  py::gil_scoped_release release;
  memory.add(bytes);
}

template <typename Estimate>
void admit_current(RerankLease &memory, Estimate estimate) {
  size_t admitted = 0;
  while (true) {
    const size_t bytes = estimate();
    if (bytes <= admitted) return;
    admit(memory, bytes - admitted);
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
  if (py::isinstance<reranker::WeightedParams>(params)) {
    const auto &weights = params.cast<const reranker::WeightedParams &>();
    admit_current(memory, [&] {
      return detail::BindingMemoryBytes(weights.weights.size(), sizeof(double));
    });
    return weights;
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
    admit_current(workspace, [&] {
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
    admit_current(*memory, [&] {
      // _replace stages shallow copies before normalization/wrapping.
      const size_t fields = mapping_size(original.attr("fields"));
      const size_t vectors = mapping_size(original.attr("vectors"));
      return detail::BindingMemoryBytes(
          vectors, 256, detail::BindingMemoryBytes(fields, 256, 512));
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

void ZVecPyReranker::Initialize(py::module_ &m) {
  // Bind RrfParams
  py::class_<reranker::RrfParams>(m, "_RrfParams")
      .def(py::init<int>(), py::arg("rank_constant") = 60)
      .def_readwrite("rank_constant", &reranker::RrfParams::rank_constant);

  // Bind WeightedParams
  py::class_<reranker::WeightedParams>(m, "_WeightedParams")
      .def(py::init<std::vector<double>>(), py::arg("weights"))
      .def_readwrite("weights", &reranker::WeightedParams::weights);

  // Bind CallbackParams
  py::class_<reranker::CallbackParams>(m, "_CallbackParams")
      .def(py::init<reranker::CallbackParams::Callback>(), py::arg("callback"));

  // Standalone rerank execution function
  m.def("_reranker_rerank", &rerank_native_docs, py::arg("params"),
        py::arg("results"), py::arg("fields"), py::arg("topn"));
  m.def("_rerank_python_docs", &rerank_python_docs, py::arg("params"),
        py::arg("results"), py::arg("fields"), py::arg("topn"));

  // Bind MultiQuery struct
  py::class_<MultiQuery>(m, "_MultiQuery")
      .def(py::init<>())
      .def_readwrite("queries", &MultiQuery::queries)
      .def_readwrite("topk", &MultiQuery::topk)
      .def_readwrite("filter", &MultiQuery::filter)
      .def_readwrite("include_vector", &MultiQuery::include_vector)
      .def_readwrite("output_fields", &MultiQuery::output_fields)
      .def(
          "set_rerank_rrf",
          [](MultiQuery &q, int rank_constant) {
            q.rerank = reranker::RrfParams{rank_constant};
          },
          py::arg("rank_constant") = 60)
      .def(
          "set_rerank_weighted",
          [](MultiQuery &q, std::vector<double> weights) {
            q.rerank = reranker::WeightedParams{std::move(weights)};
          },
          py::arg("weights"))
      .def(
          "set_rerank_callback",
          [](MultiQuery &q, reranker::CallbackParams::Callback callback) {
            q.rerank = reranker::CallbackParams{std::move(callback)};
          },
          py::arg("callback"));
}

}  // namespace zvec
