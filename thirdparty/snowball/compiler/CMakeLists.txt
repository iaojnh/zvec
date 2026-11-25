# Keep host-tool flags in their own directory so the generated runtime library
# and the SDK retain the caller's Release optimization settings.
if(CMAKE_C_COMPILER_ID STREQUAL "MSVC" AND
   (CMAKE_C_COMPILER_ARCHITECTURE_ID STREQUAL "ARM64" OR
    CMAKE_SYSTEM_PROCESSOR MATCHES "^(ARM64|arm64|aarch64)$"))
  # MSVC 19.52 ARM64's optimized Snowball compiler rejects valid backwardmode
  # rules even with C11 enabled. This executable only runs during the build.
  # Remove conflicting options here rather than adding /Od after /O2, which
  # would introduce D9025 warnings. No generated stemmer is compiled here.
  set(_snowball_configs DEBUG RELEASE RELWITHDEBINFO MINSIZEREL
                        ${CMAKE_CONFIGURATION_TYPES} ${CMAKE_BUILD_TYPE})
  list(TRANSFORM _snowball_configs TOUPPER)
  list(REMOVE_DUPLICATES _snowball_configs)
  set(_snowball_flag_vars CMAKE_C_FLAGS)
  foreach(_config IN LISTS _snowball_configs)
    list(APPEND _snowball_flag_vars "CMAKE_C_FLAGS_${_config}")
  endforeach()
  foreach(_var IN LISTS _snowball_flag_vars)
    set(_previous "")
    while(NOT "${_previous}" STREQUAL "${${_var}}")
      set(_previous "${${_var}}")
      string(REGEX REPLACE
        "(^|[ \t])[-/]O([12dx]|b[0-3]|i-?|s|t|y-?)([ \t]|$)" "\\1\\3"
        ${_var} "${${_var}}")
    endwhile()
  endforeach()
  add_compile_options(/Od)
endif()

add_executable(snowball_compiler EXCLUDE_FROM_ALL ${_snowball_compiler_srcs})
target_include_directories(snowball_compiler PRIVATE
                           "${SNOWBALL_SOURCE_DIR}/compiler")
if(MSVC)
  target_compile_definitions(snowball_compiler PRIVATE _CRT_SECURE_NO_WARNINGS)
  # C_STANDARD 99 supplies no /std option with MSVC. Use its C11 mode.
  set_target_properties(snowball_compiler PROPERTIES C_STANDARD 11)
else()
  set_target_properties(snowball_compiler PROPERTIES C_STANDARD 99)
endif()
set_target_properties(snowball_compiler PROPERTIES C_STANDARD_REQUIRED ON)
