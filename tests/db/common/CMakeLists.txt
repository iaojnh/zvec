include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)
include(${PROJECT_ROOT_DIR}/cmake/option.cmake)

if(APPLE)
    set(APPLE_FRAMEWORK_LIBS
        -framework CoreFoundation
        -framework CoreGraphics
        -framework CoreData
        -framework CoreText
        -framework Security
        -framework Foundation
        -Wl,-U,_MallocExtension_ReleaseFreeMemory
        -Wl,-U,_ProfilerStart
        -Wl,-U,_ProfilerStop
        -Wl,-U,_RegisterThriftProtocol
    )
endif()

if(APPLE AND NOT IOS)
    add_library(native_image_memory_macos_fixture SHARED
        native_image_memory_macos_fixture.cc)
    add_library(native_image_memory_macos_bundle MODULE
        native_image_memory_macos_fixture.cc)
endif()
if(WIN32)
    add_library(native_image_memory_windows_fixture SHARED
        native_image_memory_windows_fixture.cc)
endif()

file(GLOB ALL_TEST_SRCS *_test.cc)
foreach(CC_SRCS ${ALL_TEST_SRCS})
    get_filename_component(CC_TARGET ${CC_SRCS} NAME_WE)
    if(CC_TARGET STREQUAL "native_image_memory_macos_test" AND NOT APPLE)
        continue()
    endif()
    if(CC_TARGET STREQUAL "native_image_memory_windows_test" AND NOT WIN32)
        continue()
    endif()
    set(TEST_LIBRARY zvec_common)
    if(CC_TARGET STREQUAL "result_memory_allowance_test")
        # This regression now exercises Arrow results and Doc conversion,
        # which need the complete DB target and its transitive dependencies.
        set(TEST_LIBRARY zvec)
    elseif(CC_TARGET STREQUAL "native_image_memory_macos_test" OR
           CC_TARGET STREQUAL "native_image_memory_windows_test")
        # Exercise the real reservation pool without unrelated DB components.
        set(TEST_LIBRARY zvec_ailego)
    endif()
    cc_gmock(
        NAME ${CC_TARGET} STRICT
        LIBS ${TEST_LIBRARY}
        ${CMAKE_THREAD_LIBS_INIT}
        ${CMAKE_DL_LIBS}
        SRCS ${CC_SRCS}
        INCS .. ../../src
        LDFLAGS ${APPLE_FRAMEWORK_LIBS}
    )
    if(CC_TARGET STREQUAL "native_image_memory_macos_test" AND NOT IOS)
        add_dependencies(${CC_TARGET} native_image_memory_macos_fixture
            native_image_memory_macos_bundle)
        target_compile_definitions(${CC_TARGET} PRIVATE
            ZVEC_NATIVE_IMAGE_MACOS_DYLIB="$<TARGET_FILE:native_image_memory_macos_fixture>"
            ZVEC_NATIVE_IMAGE_MACOS_BUNDLE="$<TARGET_FILE:native_image_memory_macos_bundle>")
    endif()
    if(CC_TARGET STREQUAL "native_image_memory_windows_test")
        add_dependencies(${CC_TARGET} native_image_memory_windows_fixture)
        target_compile_definitions(${CC_TARGET} PRIVATE
            ZVEC_NATIVE_IMAGE_WINDOWS_DLL="$<TARGET_FILE:native_image_memory_windows_fixture>")
    endif()
    cc_test_suite(zvec_common ${CC_TARGET})
endforeach()
