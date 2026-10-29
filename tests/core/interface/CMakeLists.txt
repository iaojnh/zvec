include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)

file(GLOB_RECURSE ALL_TEST_SRCS *_test.cc)

foreach(CC_SRCS ${ALL_TEST_SRCS})
  get_filename_component(CC_TARGET ${CC_SRCS} NAME_WE)
  set(INTERFACE_TEST_LIBS zvec_ailego core_framework core_metric core_interface core_knn_flat core_utility core_quantizer sparsehash core_knn_hnsw core_mix_reducer
      core_knn_flat_sparse core_knn_hnsw_sparse core_knn_ivf core_knn_hnsw_rabitq core_knn_ivf_rabitq core_knn_vamana core_knn_diskann zvec_turbo)
  if((ANDROID OR IOS) AND CC_TARGET STREQUAL "hnsw_platform_test")
    # A device test must be self-contained when pushed/installed. Exercise the
    # same static SDK linkage as mobile consumers instead of component DSOs.
    set(INTERFACE_TEST_LIBS zvec::core_static)
  endif()
  cc_gtest(
      NAME ${CC_TARGET}
      STRICT
      LIBS ${INTERFACE_TEST_LIBS}
      SRCS ${CC_SRCS}
      INCS . ${PROJECT_ROOT_DIR}/src/core ${PROJECT_ROOT_DIR}/src/core/algorithm
    )
endforeach()

# Keep a machine-readable record of the graph/storage matrix and the actual
# FP16 architectures exercised by each desktop runner.
set_tests_properties(hnsw_platform_test PROPERTIES ENVIRONMENT
    "GTEST_OUTPUT=xml:${CMAKE_BINARY_DIR}/test_tmp/hnsw_platform_test/results.xml")

# Simulator CI launches the default com.zvec.<target> bundle identifier.
# Only physical devices need a custom identifier for provisioning profiles.
if(IOS AND NOT CMAKE_OSX_SYSROOT MATCHES "[Ss]imulator")
  set(ZVEC_HNSW_TEST_BUNDLE_ID "com.zvec.hnsw-platform-test" CACHE STRING
      "Bundle identifier used when signing the HNSW device test app")
  set_target_properties(hnsw_platform_test PROPERTIES
      MACOSX_BUNDLE_GUI_IDENTIFIER "${ZVEC_HNSW_TEST_BUNDLE_ID}"
      MACOSX_BUNDLE_INFO_PLIST "${CMAKE_CURRENT_SOURCE_DIR}/hnsw_platform_test.plist.in")
endif()
