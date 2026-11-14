include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)
include(${PROJECT_ROOT_DIR}/cmake/option.cmake)

include(${PROJECT_ROOT_DIR}/cmake/fts_bitpacking.cmake)
zvec_configure_fts_bitpacking()

cc_library(
    NAME zvec_index STATIC STRICT
    SRCS *.cc segment/*.cc column/vector_column/*.cc column/inverted_column/*.cc column/fts_column/*.cc column/fts_column/tokenizer/*.cc column/fts_column/posting/*.cc column/fts_column/iterator/*.cc storage/*.cc storage/wal/*.cc common/*.cc common/manifest/*.cc
    LIBS zvec_common
         rocksdb
         core_interface
         Arrow::arrow_static
         Arrow::parquet_static
         Arrow::arrow_compute
         cppjieba
         lz4
         snowball
         FastPFOR
         utf8proc
    INCS .  ${PROJECT_ROOT_DIR}/src
    VERSION "${PROXIMA_ZVEC_VERSION}"
  )
