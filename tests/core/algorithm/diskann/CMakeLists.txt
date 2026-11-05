include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)

file(GLOB_RECURSE ALL_TEST_SRCS *_test.cc)

# The full DiskAnn suite repeatedly builds 10k-vector indexes and is intended
# for desktop CI. Mobile CI runs a focused compatibility test that covers the
# portable I/O path, failure recovery, concurrency, and an end-to-end
# build/dump/load/search cycle.
if(ANDROID)
  list(FILTER ALL_TEST_SRCS INCLUDE REGEX
       "diskann_(mobile_compat|android_io)_test\\.cc$")
elseif(IOS)
  # Mobile CI runs the focused portable-I/O and end-to-end compatibility test.
  list(FILTER ALL_TEST_SRCS INCLUDE REGEX "diskann_(mobile_compat|apple_io)_test\\.cc$")
else()
  if(NOT APPLE)
    list(FILTER ALL_TEST_SRCS EXCLUDE REGEX "/diskann_apple_io_test\\.cc$")
  endif()
  list(FILTER ALL_TEST_SRCS EXCLUDE REGEX "/diskann_android_io_test\\.cc$")
  if(WIN32)
    list(FILTER ALL_TEST_SRCS EXCLUDE REGEX
         "/diskann_file_reader_test\\.cc$|/diskann_mobile_compat_test\\.cc$")
  else()
    list(FILTER ALL_TEST_SRCS EXCLUDE REGEX
         "/diskann_file_reader_windows_test\\.cc$")
  endif()

  if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    list(FILTER ALL_TEST_SRCS EXCLUDE REGEX
         "/diskann_file_reader_aio_test\\.cc$|/diskann_libaio_lifetime_test\\.cc$")
  endif()
endif()

foreach(CC_SRCS ${ALL_TEST_SRCS})
  get_filename_component(CC_TARGET ${CC_SRCS} NAME_WE)
  set(EXTRA_TEST_LIBS)
  if(CC_TARGET STREQUAL "diskann_mmap_view_test")
    set(EXTRA_TEST_LIBS core_knn_flat core_mix_reducer)
  endif()
  cc_gtest(
      NAME ${CC_TARGET}
      STRICT
      LIBS zvec_ailego core_framework core_utility core_metric core_quantizer core_knn_cluster core_knn_diskann ${EXTRA_TEST_LIBS}
      SRCS ${CC_SRCS}
      INCS . ${PROJECT_ROOT_DIR}/src/core ${PROJECT_ROOT_DIR}/src/core/algorithm/diskann ${PROJECT_ROOT_DIR}/src/turbo
    )
  if(ANDROID AND CC_TARGET STREQUAL "diskann_android_io_test")
    set_property(TARGET ${CC_TARGET} APPEND_STRING PROPERTY LINK_FLAGS
      " -Wl,--wrap=open -Wl,--wrap=open64 -Wl,--wrap=__open_2 -Wl,--wrap=__open64_2 -Wl,--wrap=pread -Wl,--wrap=pread64 -Wl,--wrap=__pread_chk -Wl,--wrap=__pread64_chk")
  endif()
  if(APPLE AND NOT IOS AND CC_TARGET STREQUAL "diskann_apple_io_test")
    target_link_libraries(${CC_TARGET} proc)
  endif()
endforeach()
