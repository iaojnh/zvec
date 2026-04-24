include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)
include(${PROJECT_ROOT_DIR}/cmake/option.cmake)

cc_library(
    NAME core_knn_cluster STATIC SHARED STRICT ALWAYS_LINK
    SRCS *.cc
    LIBS zvec_ailego core_framework 
    INCS . ${PROJECT_ROOT_DIR}/src/core ${PROJECT_ROOT_DIR}/src/core/cluster
    LDFLAGS "-Wl,--exclude-libs,libparquet.a:libarrow.a:libarrow_bundled_dependencies.a"
    VERSION "${PROXIMA_ZVEC_VERSION}"
  )
