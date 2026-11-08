# Runtime dispatch must advertise only kernels compiled by the target toolchain.
# Keep per-kernel flags independent of AUTO_DETECT_ARCH and the package baseline.
function(zvec_configure_fts_bitpacking)
  set(_dir "${PROJECT_ROOT_DIR}/src/db/index/column/fts_column/posting")
  set(_dispatch "${_dir}/bitpacked_simd_dispatch.cc")
  if(MSVC)
    set(_sse_flags "")
    set(_avx_flags "/arch:AVX2")
  else()
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
      set(_baseline "-march=x86-64")
    else()
      set(_baseline "-march=i686")
    endif()
    set(_sse_flags "${_baseline};-msse4.1")
    set(_avx_flags "${_baseline};-mavx2")
  endif()
  configure_simd_kernel("${_dir}/bitpacked_simd_sse41.cc" "${_dispatch}"
    "${_sse_flags}" "
    #if !defined(__SSE4_1__) && !(defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86)))
    #error Kernel ISA unavailable
    #endif
    int main() { return 0; }"
    ZVEC_FTS_HAVE_SSE41)
  configure_simd_kernel("${_dir}/bitpacked_simd_avx2.cc" "${_dispatch}"
    "${_avx_flags}" "
    #if !defined(__AVX2__) && !(defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86)))
    #error Kernel ISA unavailable
    #endif
    int main() { return 0; }"
    ZVEC_FTS_HAVE_AVX2)
endfunction()
