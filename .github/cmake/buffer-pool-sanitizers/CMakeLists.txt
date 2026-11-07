cmake_minimum_required(VERSION 3.16)
project(zvec_buffer_pool_sanitizers LANGUAGES C CXX)

if(NOT ZVEC_SOURCE_DIR OR NOT ZVEC_SANITIZER MATCHES "^(address,undefined|thread)$")
  message(FATAL_ERROR "Set ZVEC_SOURCE_DIR and ZVEC_SANITIZER=address,undefined or thread")
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-O1 -g -fno-omit-frame-pointer "-fsanitize=${ZVEC_SANITIZER}")
add_link_options("-fsanitize=${ZVEC_SANITIZER}")
find_package(Threads REQUIRED)
if(ZVEC_GTEST_SOURCE_DIR)
  add_subdirectory("${ZVEC_GTEST_SOURCE_DIR}" googletest EXCLUDE_FROM_ALL)
  set(ZVEC_GTEST_MAIN gtest_main)
else()
  find_package(GTest REQUIRED)
  set(ZVEC_GTEST_MAIN GTest::gtest_main)
endif()

# Compile the real first-party pool/I/O code without rebuilding DB dependencies.
# System GTest is a test dependency; no thirdparty source files are patched.
file(GLOB_RECURSE AILEGO_SOURCES CONFIGURE_DEPENDS
  "${ZVEC_SOURCE_DIR}/src/ailego/*.cc"
  "${ZVEC_SOURCE_DIR}/src/ailego/*.c")
add_library(ailego_sanitized STATIC ${AILEGO_SOURCES})
target_include_directories(ailego_sanitized PUBLIC
  "${ZVEC_SOURCE_DIR}/src/include" "${ZVEC_SOURCE_DIR}/src")
target_link_libraries(ailego_sanitized PUBLIC Threads::Threads ${CMAKE_DL_LIBS})
if(UNIX AND NOT APPLE)
  target_link_libraries(ailego_sanitized PUBLIC rt)
endif()
enable_testing()
set(TEST_TARGETS
  process_rss_budget_test pool_admission_test retained_block_allocator_test
  budgeted_mapped_buffer_test budgeted_aligned_buffer_test
  eviction_queue_budget_test slab_backing_budget_test vector_page_table_test)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  list(APPEND TEST_TARGETS vector_page_table_bypass_aio_test)
  add_executable(iouring_loader_test
    "${ZVEC_SOURCE_DIR}/tests/ailego/io/iouring_loader_test.cc")
  target_link_libraries(iouring_loader_test PRIVATE ailego_sanitized ${ZVEC_GTEST_MAIN})
  file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/work/iouring_loader_test")
  add_test(NAME iouring_loader_test COMMAND iouring_loader_test)
  set_tests_properties(iouring_loader_test PROPERTIES
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/work/iouring_loader_test" TIMEOUT 300)
endif()
foreach(name IN LISTS TEST_TARGETS)
  add_executable(${name} "${ZVEC_SOURCE_DIR}/tests/ailego/buffer/${name}.cc")
  target_link_libraries(${name} PRIVATE ailego_sanitized ${ZVEC_GTEST_MAIN})
  file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/work/${name}")
  add_test(NAME ${name} COMMAND ${name})
  set_tests_properties(${name} PROPERTIES
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/work/${name}" TIMEOUT 300)
endforeach()
