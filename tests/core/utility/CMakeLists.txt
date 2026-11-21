
include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)
include(${PROJECT_ROOT_DIR}/cmake/option.cmake)

file(GLOB_RECURSE ALL_TEST_SRCS *_test.cc)

foreach(CC_SRCS ${ALL_TEST_SRCS})
  get_filename_component(CC_TARGET ${CC_SRCS} NAME_WE)
  # Native direct AIO prefetch is only implemented on non-Android Linux.
  # Do not produce an empty GTest executable on the other platforms.
  if(CC_TARGET STREQUAL "chunk_read_prefetch_test" AND
     (NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR ANDROID))
    continue()
  endif()
  cc_gtest(
    NAME ${CC_TARGET} STRICT
    LIBS zvec_ailego core_framework core_utility
    SRCS ${CC_SRCS}
    INCS . ${PROJECT_ROOT_DIR}/src/core/
  )
  cc_test_suite(zvec_ailego ${CC_TARGET})
endforeach()
