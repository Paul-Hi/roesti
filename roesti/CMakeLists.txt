project(roesti)

message(STATUS "================================================")
message(STATUS "Adding roesti library!")

file(GLOB_RECURSE HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}include/*.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}include/*.cuh)
file(GLOB_RECURSE SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)

add_library(roesti
    SHARED
    ${HEADERS}
    ${SOURCES}
)

target_link_libraries(roesti
    PUBLIC
    roesti::torch
)

target_include_directories(roesti
    PUBLIC
    ${ROESTI_INCLUDE_DIR}
    ${TORCH_INCLUDE_DIRS}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

set_target_properties(roesti
    PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_SEPARABLE_COMPILATION ON
)

target_compile_definitions(roesti
    PUBLIC
    $<$<CONFIG:Debug>:ROESTI_DEBUG>
    $<$<CXX_COMPILER_ID:MSVC>: _CRT_SECURE_NO_WARNINGS>
    PRIVATE
    $<$<BOOL:${WIN32}>:WIN32>
    $<$<BOOL:${LINUX}>:LINUX>
)

target_compile_options(roesti
    PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>: /W4>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>: -Wall -Wextra -Wno-c++98-compat -Wno-c++98-compat-pedantic -Wno-gnu-zero-variadic-macro-arguments -Wno-global-constructors -Wno-documentation-unknown-command>
    $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>
)

add_library(roesti::roesti ALIAS roesti)

install(TARGETS roesti DESTINATION lib)
install(DIRECTORY roesti/include/ DESTINATION include FILES_MATCHING PATTERN "*.hpp;*.cuh")
message(STATUS "================================================")
