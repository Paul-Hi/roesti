if(TARGET roesti::torch)
    return()
endif()

# backup old value opf FETCHCONTENT_TRY_FIND_PACKAGE_MODE
set(_FETCHCONTENT_TRY_FIND_PACKAGE_MODE_OLD ${FETCHCONTENT_TRY_FIND_PACKAGE_MODE})
set(FETCHCONTENT_TRY_FIND_PACKAGE_MODE NEVER)

include(FetchContent)

FetchContent_Declare(
    Torch
    URL https://download.pytorch.org/libtorch/cu129/libtorch-shared-with-deps-2.8.0%2Bcu129.zip
)

FetchContent_GetProperties(Torch)

if(NOT torch_POPULATED)
    message(STATUS "Fetching libtorch")
    FetchContent_MakeAvailable(Torch)
endif()

set(CMAKE_PREFIX_PATH "${torch_SOURCE_DIR}" ${CMAKE_PREFIX_PATH})

message(STATUS "Creating Target 'roesti::torch'")

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")


add_library(roesti::torch INTERFACE IMPORTED)
set_target_properties(roesti::torch PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${TORCH_INCLUDE_DIRS}"
    INTERFACE_LINK_LIBRARIES "${TORCH_LIBRARIES}"
)

# message(STATUS "${TORCH_CXX_FLAGS} : ${TORCH_LIBRARIES}")
# add_library(roesti::torch)

set(FETCHCONTENT_TRY_FIND_PACKAGE_MODE ${_FETCHCONTENT_TRY_FIND_PACKAGE_MODE_OLD})
unset(_FETCHCONTENT_TRY_FIND_PACKAGE_MODE_OLD)

message(STATUS "================================================")
message(STATUS "================================================")