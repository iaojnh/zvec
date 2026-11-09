include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)
include(${PROJECT_ROOT_DIR}/cmake/option.cmake)

option(ZVEC_BUILD_ACCEPTANCE_TESTS
  "Build extended memory, storage and training acceptance tests" OFF)

# Daily CI keeps admission/accounting, I/O failure and public API regressions.
# These additional suites exercise the same contracts across complete builds,
# storage layouts, training implementations and resource-pressure scenarios.
set(ZVEC_ACCEPTANCE_TEST_TARGETS
  collection_batch_fetch_test
  collection_write_budget_test
  db_memory_storage_test
  multi_segment_topk_test
  rocksdb_cache_allocator_test
  rocksdb_context_test
  rocksdb_memtable_factory_test
  combined_result_memory_test
  ddl_memory_test
  segment_stream_test
  buffered_input_test
  diskann_buffered_build_test
  diskann_paged_metadata_test
  ivf_memory_lifecycle_test
  ivf_rabitq_buffered_input_test
  flat_memory_test
  flat_workspace_budget_test
  ivf_dumper_storage_test
  hnsw_sparse_buffered_builder_test
  kmeans_paged_mean_test
  kmeans_storage_test
  centroid_serialization_memory_test
  graph_segment_writer_test
  paged_build_buffer_test
  paged_record_sort_test
  paged_graph_index_hash_test
  temporary_buffer_storage_test
  turbo_quantizer_scratch_test
  integer_quantizer_memory_test
  rabitq_training_memory_test
  hnsw_rabitq_dump_memory_test
  worker_memory_test
  native_worker_memory_test
  vector_build_source_test)

if(ZVEC_BUILD_ACCEPTANCE_TESTS)
  add_custom_target(acceptance.build)
endif()

# TODO(windows): this is for test_util.h, should be removed
include_directories(${PROJECT_ROOT_DIR})

cc_directories(ailego)
cc_directories(db)
cc_directories(core)
cc_directories(turbo)
if(BUILD_ZVEC_SHARED AND NOT IOS AND NOT ANDROID)
    cc_directories(shared)
endif()
if(BUILD_C_BINDINGS)
    cc_directories(c)
endif()

# Reuse the unit-test sources so the two suites share fixtures in one place.
# Opt in, build acceptance.build, then run ctest -L acceptance.
if(ZVEC_BUILD_ACCEPTANCE_TESTS)
  foreach(CC_SOURCE
      core/interface/pq_buffered_training_test.cc
      core/algorithm/cluster/stratified_cluster_storage_test.cc
      core/utility/appendable_build_buffer_test.cc)
    get_filename_component(CC_TARGET ${CC_SOURCE} NAME_WE)
    string(REGEX REPLACE "_test$" "_acceptance_test" CC_TARGET "${CC_TARGET}")
    cc_gtest(
      NAME ${CC_TARGET}
      STRICT
      SRCS ${CC_SOURCE}
      DEFS ZVEC_ACCEPTANCE_TESTS
      LIBS zvec_ailego core_framework core_metric core_interface core_knn_flat
           core_utility core_quantizer sparsehash core_knn_hnsw core_mix_reducer
           core_knn_flat_sparse core_knn_hnsw_sparse core_knn_ivf
           core_knn_hnsw_rabitq core_knn_ivf_rabitq core_knn_vamana
           core_knn_diskann core_knn_cluster zvec_turbo
      INCS . ${PROJECT_ROOT_DIR}/src/core ${PROJECT_ROOT_DIR}/src/core/algorithm
           ${PROJECT_ROOT_DIR}/src/turbo
    )
  endforeach()
endif()
