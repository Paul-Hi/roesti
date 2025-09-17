cmake_minimum_required(VERSION 3.26)

project(ROESTI
    VERSION 0.01
    DESCRIPTION "Small Cuda Rasterizer."
    LANGUAGES CXX C CUDA
)

set(CMAKE_COLOR_MAKEFILE ON)

if(UNIX AND NOT APPLE)
    set(LINUX TRUE)
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING
        "Choose the type of build, options are: Debug, Release" FORCE)
endif(NOT CMAKE_BUILD_TYPE)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${PROJECT_SOURCE_DIR}/install/roesti/" CACHE PATH "Install path prefix" FORCE)
endif(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

option(ROESTI_ENABLE_TESTBENCH "Enable the testbench build." ON)
option(ROESTI_ENABLE_PYTHON_BINDINGS "Enable the python bindings build." ON)

set(VERSION_MAJOR 0 CACHE STRING "Project major version number.")
set(VERSION_MINOR 0 CACHE STRING "Project minor version number.")
set(VERSION_PATCH 1 CACHE STRING "Project patch version number.")
mark_as_advanced(VERSION_MAJOR VERSION_MINOR VERSION_PATCH)

message(STATUS "================================================")
message(STATUS "        ____   ___  _____ ____ _____ ___        ")
message(STATUS "       |  _ \\ / _ \\| ____/ ___|_   _|_ _|       ")
message(STATUS "       | |_) | | | |  _| \\___ \\ | |  | |        ")
message(STATUS "       |  _ <| |_| | |___ ___) || |  | |        ")
message(STATUS "       |_| \\_\\\\___/|_____|____/ |_| |___|       ")
message(STATUS "================================================")
message(STATUS "Copyright (c) 2025-present, Paul Himmler")
message(STATUS "================================================")
message(STATUS "Version v" ${VERSION_MAJOR} "." ${VERSION_MINOR} ${VERSION_PATCH} " Licensed Apache License 2.0")
message(STATUS "================================================")
message(STATUS "Build Type: " ${CMAKE_BUILD_TYPE})

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    message(STATUS "Setting Project Standards.")
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
endif()

file(TO_CMAKE_PATH "${PROJECT_BINARY_DIR}/CMakeLists.txt" LOC_PATH)

if(EXISTS "${LOC_PATH}")
    message(FATAL_ERROR "You cannot build in a source directory (or any directory with a CMakeLists.txt file). Please make a build subdirectory. Feel free to remove CMakeCache.txt and CMakeFiles.")
endif()

message(STATUS "================================================")

message(STATUS "Fetching Dependencies.")

# dependencies
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake/")
include(FetchContent)
set(FETCHCONTENT_TRY_FIND_PACKAGE_MODE ALWAYS)
set(FETCHCONTENT_QUIET TRUE)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

# Torch
include(torch)

if(ROESTI_ENABLE_TESTBENCH)
    include(saf)
endif()

if(ROESTI_ENABLE_PYTHON_BINDINGS)
    include(pybind11)
endif()

message(STATUS "================================================")

set(ROESTI_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/roesti/include)

add_subdirectory(roesti)

if(ROESTI_ENABLE_TESTBENCH)
    add_subdirectory(testbench)
endif()

if(ROESTI_ENABLE_PYTHON_BINDINGS)
    add_subdirectory(python)
endif()

message(STATUS "================================================")