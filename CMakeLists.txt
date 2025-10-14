cmake_minimum_required(VERSION 3.20)

project(CUDA_BEC LANGUAGES CXX CUDA)

# Standards
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Position-independent code like -fPIC
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Detect CUDA toolkit and libraries
find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED COMPONENTS CXX)

# Try to auto-detect GPU compute capability (first GPU) via nvidia-smi, fallback to 75
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    execute_process(
        COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
        OUTPUT_VARIABLE _GPU_COMPUTE_CAP
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(_GPU_COMPUTE_CAP)
        # Take the first line if multiple GPUs are listed
        string(REGEX MATCH "^[0-9]+\.?[0-9]*" _GPU_COMPUTE_CAP "${_GPU_COMPUTE_CAP}")
        string(REPLACE "." "" _GPU_ARCH "${_GPU_COMPUTE_CAP}")
        if(_GPU_ARCH)
            set(CMAKE_CUDA_ARCHITECTURES ${_GPU_ARCH} CACHE STRING "CUDA arch" FORCE)
        endif()
    endif()
    if(NOT CMAKE_CUDA_ARCHITECTURES)
        message(WARNING "Could not detect GPU compute capability, using 75 as default")
        set(CMAKE_CUDA_ARCHITECTURES 75 CACHE STRING "CUDA arch" FORCE)
    endif()
endif()

# Include directories (shared)
set(PROJ_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils
)

# Shared sources
set(UTILS_CUDA_SOURCES
    src/utils/diffint.cu
    src/utils/simpson3d_kernel.cu
    src/utils/simpson3d_integrator.cu
)

set(UTILS_CPP_SOURCES
    src/utils/multiarray.cpp
    src/utils/cfg.cpp
    src/utils/CudaArray.cpp
)

# Create a static library for shared utils (mix of CUDA and C++)
add_library(cuda_utils STATIC
    ${UTILS_CUDA_SOURCES}
    ${UTILS_CPP_SOURCES}
)

target_include_directories(cuda_utils
    PUBLIC
        ${PROJ_INCLUDE_DIRS}
)

target_link_libraries(cuda_utils
    PUBLIC
        CUDA::cudart
        CUDA::cufft
        OpenMP::OpenMP_CXX
)

# Optimize and match Makefile flags where appropriate
target_compile_options(cuda_utils
    PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3 --fmad=true>
        $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

# imag3d-cuda executable
add_executable(imag3d-cuda
    src/imag3d-cuda/imag3d-cuda.cu
)

target_include_directories(imag3d-cuda
    PRIVATE
        ${PROJ_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/src/imag3d-cuda
)

target_link_libraries(imag3d-cuda
    PRIVATE
        cuda_utils
)

target_compile_options(imag3d-cuda
    PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3 --fmad=true>
        $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

# real3d-cuda executable
add_executable(real3d-cuda
    src/real3d-cuda/real3d-cuda.cu
)

target_include_directories(real3d-cuda
    PRIVATE
        ${PROJ_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/src/real3d-cuda
)

target_link_libraries(real3d-cuda
    PRIVATE
        cuda_utils
)

target_compile_options(real3d-cuda
    PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3 --fmad=true>
        $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

# Nice output
add_custom_command(TARGET imag3d-cuda POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "Build complete: imag3d-cuda")

add_custom_command(TARGET real3d-cuda POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "Build complete: real3d-cuda")


