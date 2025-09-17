if(TARGET roesti::saf)
    return()
endif()

include(FetchContent)

FetchContent_Declare(
    saf
    GIT_REPOSITORY https://github.com/Paul-Hi/SAF.git
    GIT_TAG 3266cafe49f6a76c05325bb9133dd3d7ca226732
)

FetchContent_GetProperties(saf)

if(NOT saf_POPULATED)
    set(SAF_ENABLE_CUDA_INTEROP ON CACHE INTERNAL "Enable CUDA interop in SAF")
    FetchContent_MakeAvailable(saf)
endif()

message(STATUS "Creating Target 'roesti::saf'")

add_library(roesti::saf ALIAS saf)

message(STATUS "================================================")
message(STATUS "================================================")