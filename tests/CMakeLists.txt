include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)
include(${PROJECT_ROOT_DIR}/cmake/option.cmake)

option(ZVEC_BUILD_ACCEPTANCE_TESTS
  "Build resource-intensive buffer and training acceptance tests" OFF)

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
  add_custom_target(acceptance.build)
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
    set_tests_properties(${CC_TARGET} PROPERTIES
      LABELS "acceptance;buffer" RUN_SERIAL TRUE TIMEOUT 900)
    add_dependencies(acceptance.build ${CC_TARGET})
  endforeach()
endif()
