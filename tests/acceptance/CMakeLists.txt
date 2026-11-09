include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)

# Opt in with -DZVEC_BUILD_ACCEPTANCE_TESTS=ON, build acceptance.build,
# then run ctest --output-on-failure -L acceptance in the build directory.
# Default unit tests keep admission, rollback, retry and ownership coverage.
add_custom_target(acceptance.build)

foreach(CC_TARGET
    appendable_build_buffer_acceptance_test
    pq_buffered_training_acceptance_test
    stratified_cluster_storage_acceptance_test)
  cc_gtest(
    NAME ${CC_TARGET}
    STRICT
    SRCS ${CC_TARGET}.cc
    LIBS zvec_ailego core_framework core_metric core_interface core_knn_flat
         core_utility core_quantizer sparsehash core_knn_hnsw core_mix_reducer
         core_knn_flat_sparse core_knn_hnsw_sparse core_knn_ivf
         core_knn_hnsw_rabitq core_knn_ivf_rabitq core_knn_vamana
         core_knn_diskann core_knn_cluster zvec_turbo
    INCS . ${PROJECT_ROOT_DIR}/src/core ${PROJECT_ROOT_DIR}/src/core/algorithm
         ${PROJECT_ROOT_DIR}/src/turbo
  )
  set_tests_properties(${CC_TARGET} PROPERTIES
    LABELS "acceptance;buffer"
    RUN_SERIAL TRUE
    TIMEOUT 900
  )
  add_dependencies(acceptance.build ${CC_TARGET})
endforeach()
