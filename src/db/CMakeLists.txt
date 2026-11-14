include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)
include(${PROJECT_ROOT_DIR}/cmake/option.cmake)

cc_directory(common)
cc_directory(index)
cc_directory(sqlengine)

file(GLOB_RECURSE ALL_DB_SRCS *.cc *.c *.h)
get_target_property(ZVEC_ROCKSDB_SOURCE_DIR rocksdb SOURCE_DIR)
set_source_files_properties(
  ${CMAKE_CURRENT_SOURCE_DIR}/common/rocksdb_memtable_factory.cc
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/common
  PROPERTIES INCLUDE_DIRECTORIES ${ZVEC_ROCKSDB_SOURCE_DIR}
)

# Source properties are directory-scoped: configure both packed and component
# targets, including portable builds with global architecture detection disabled.
include(${PROJECT_ROOT_DIR}/cmake/fts_bitpacking.cmake)
zvec_configure_fts_bitpacking()

set(ZVEC_DB_LIBRARY_OPTIONS)
if(WIN32 AND BUILD_ZVEC_SHARED)
  list(APPEND ZVEC_DB_LIBRARY_OPTIONS
    OBJECTS
    EXPORT_DEF ZVEC_DB_BUILD_SHARED
  )
endif()

cc_library(
  NAME zvec STATIC STRICT SRCS_NO_GLOB PACKED ${ZVEC_DB_LIBRARY_OPTIONS}
  SRCS ${ALL_DB_SRCS}
  INCS . ${CMAKE_CURRENT_BINARY_DIR}
  PUBINCS ${PROJECT_ROOT_DIR}/src/include/
  LIBS
    zvec_ailego
    zvec_core
    glog
    roaring
    rocksdb
    antlr4
    FastPFOR
    cppjieba
    lz4
    snowball
    Arrow::arrow_static
    Arrow::parquet_static
    Arrow::arrow_compute
    Arrow::arrow_acero
    utf8proc
  VERSION "${PROXIMA_ZVEC_VERSION}"
)
