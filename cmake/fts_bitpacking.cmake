# Probe runtime-dispatched kernels independently of global architecture defaults.
# AUTO_DETECT_ARCH=OFF must not leave CPU dispatch pointing at empty stubs.
function(zvec_configure_fts_bitpacking)
  set(_dir "${PROJECT_ROOT_DIR}/src/db/index/column/fts_column/posting")
  if(MSVC)
    set(_sse41 "")
    set(_avx2 /arch:AVX2)
  else()
    set(_sse41 -msse4.1)
    set(_avx2 -mavx2)
  endif()
  foreach(_isa SSE41 AVX2)
    string(TOLOWER "${_isa}" _lower)
    if(_isa STREQUAL SSE41)
      set(_feature __SSE4_1__)
    else()
      set(_feature __AVX2__)
    endif()
    configure_simd_kernel("${_dir}/bitpacked_simd_${_lower}.cc"
      "${_dir}/bitpacked_simd_dispatch.cc" "${_${_lower}}" "
      #if !defined(${_feature}) && !(defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86)))
      #error Kernel ISA unavailable
      #endif
      #include <immintrin.h>
      int main() { return 0; }" ZVEC_FTS_HAVE_${_isa})
  endforeach()
endfunction()
