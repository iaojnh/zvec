
include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)
include(${PROJECT_ROOT_DIR}/cmake/option.cmake)

file(GLOB_RECURSE ALL_TEST_SRCS *_test.cc)

# Exercise the exact failure mode seen when a host and a component DSO both
# embed zvec_ailego's static archive. DiskANN is Linux-only, but the singleton
# contract itself is also verified on macOS where Mach-O has different weak
# symbol coalescing rules.
if(NOT WIN32 AND NOT IOS AND NOT ANDROID)
  cc_library(
    NAME cross_dso_singleton_probe SHARED EXCLUDE STRICT
    SRCS buffer/cross_dso_singleton_probe.cc
    LIBS zvec_ailego
  )
endif()

foreach(CC_SRCS ${ALL_TEST_SRCS})
  get_filename_component(CC_TARGET ${CC_SRCS} NAME_WE)
  set(CC_TEST_LIBS zvec_ailego)
  if(CC_TARGET STREQUAL "cross_dso_singleton_test")
    if(WIN32 OR IOS OR ANDROID)
      continue()
    endif()
    list(APPEND CC_TEST_LIBS cross_dso_singleton_probe)
  endif()
  cc_gtest(
    NAME ${CC_TARGET} STRICT
    LIBS ${CC_TEST_LIBS}
    SRCS ${CC_SRCS}
  )
  cc_test_suite(zvec_ailego ${CC_TARGET})
endforeach()
