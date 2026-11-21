cmake_minimum_required(VERSION 3.26)
project(snowball_codegen_smoke LANGUAGES C)

enable_testing()
# Build the real dependency in isolation, including its native code generator
# and every enabled UTF-8 algorithm, before starting the full SDK build.
add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/../../../thirdparty/snowball"
                 "${CMAKE_CURRENT_BINARY_DIR}/snowball")
add_executable(snowball_codegen_smoke smoke.c)
set_target_properties(snowball_codegen_smoke PROPERTIES
                      C_STANDARD 11 C_STANDARD_REQUIRED ON)
target_link_libraries(snowball_codegen_smoke PRIVATE snowball)
add_test(NAME snowball_codegen_smoke COMMAND snowball_codegen_smoke)
