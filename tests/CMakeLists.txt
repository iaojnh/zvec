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

if(ZVEC_BUILD_ACCEPTANCE_TESTS)
    cc_directories(acceptance)
endif()
