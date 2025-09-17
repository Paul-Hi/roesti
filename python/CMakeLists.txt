project(pyroesti)

set(APP_NAME pyroesti)

message(STATUS "================================================")
message(STATUS "Adding pyroesti!")

message(STATUS "Using pybind11 from ${pybind11_SOURCE_DIR}")

file(GLOB_RECURSE HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)
file(GLOB_RECURSE SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

pybind11_add_module(${APP_NAME}
    ${SOURCES}
    ${HEADERS}
)

set_target_properties(${APP_NAME}
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}$<$<CONFIG:Debug>:/debug>$<$<CONFIG:Release>:/release>/lib
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}$<$<CONFIG:Debug>:/debug>$<$<CONFIG:Release>:/release>/lib
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}$<$<CONFIG:Debug>:/debug>$<$<CONFIG:Release>:/release>/bin
)

# Try to find torch_python library from possible locations
find_library(TORCH_PYTHON_LIBRARY
    NAMES torch_python
    PATHS
        ${torch_SOURCE_DIR}/lib
    NO_DEFAULT_PATH
)

target_link_libraries(${APP_NAME}
    PUBLIC
    roesti::roesti
    roesti::torch
    ${TORCH_PYTHON_LIBRARY}
)

target_include_directories(${APP_NAME}
    PRIVATE
    ${ROESTI_INCLUDE_DIR}
)

install(TARGETS ${APP_NAME} DESTINATION lib)
message(STATUS "================================================")
