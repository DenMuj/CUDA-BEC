cmake_minimum_required(VERSION 3.18)

# Project configuration
project(CUDA-BEC 
    VERSION 1.0.0
    LANGUAGES CXX CUDA
    DESCRIPTION "CUDA Dipolar BEC"
)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Enable CUDA separable compilation
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Find required packages
find_package(CUDA REQUIRED)
find_package(OpenMP REQUIRED)

# Check if modern CUDA targets are available
set(USE_MODERN_CUDA_TARGETS FALSE)

# Try to find CUDA with modern targets
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    # Check if the modern targets exist
    if(TARGET CUDA::cudart AND TARGET CUDA::cufft)
        set(USE_MODERN_CUDA_TARGETS TRUE)
        message(STATUS "Using modern CUDA targets (CUDAToolkit)")
    endif()
endif()

# If modern targets not found, use traditional CUDA linking
if(NOT USE_MODERN_CUDA_TARGETS)
    message(STATUS "Using traditional CUDA linking")
    # Ensure CUDA libraries are found
    if(NOT CUDA_FOUND)
        message(FATAL_ERROR "CUDA not found")
    endif()
endif()

# Platform-specific settings
if(WIN32)
    # Windows-specific settings
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler /W3")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W3")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler /O2")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2")
elseif(UNIX)
    # Linux/Unix-specific settings
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fPIC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -O3")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
endif()

# CUDA compute capability detection
# Try to detect GPU compute capability automatically
find_program(NVIDIA_SMI nvidia-smi)
if(NVIDIA_SMI)
    execute_process(
        COMMAND ${NVIDIA_SMI} --query-gpu=compute_cap --format=csv,noheader
        OUTPUT_VARIABLE GPU_COMPUTE_CAPABILITY_RAW
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE NVIDIA_SMI_RESULT
    )
    
    if(NVIDIA_SMI_RESULT EQUAL 0 AND GPU_COMPUTE_CAPABILITY_RAW)
        # Extract first line and remove decimal point
        string(REGEX REPLACE "\n.*" "" GPU_COMPUTE_CAPABILITY_LINE "${GPU_COMPUTE_CAPABILITY_RAW}")
        string(REGEX REPLACE "\\." "" GPU_COMPUTE_CAPABILITY "${GPU_COMPUTE_CAPABILITY_LINE}")
        message(STATUS "Detected GPU compute capability: ${GPU_COMPUTE_CAPABILITY}")
    else()
        set(GPU_COMPUTE_CAPABILITY "75")  # Default fallback
        message(WARNING "Could not detect GPU compute capability, using sm_75 as default")
    endif()
else()
    set(GPU_COMPUTE_CAPABILITY "75")  # Default fallback
    message(WARNING "nvidia-smi not found, using sm_75 as default")
endif()

# Set CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES ${GPU_COMPUTE_CAPABILITY})

# CUDA compiler flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++20")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --fmad=true")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fopenmp")

# C++ compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

# Include directories
include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/src/utils)

# Source file definitions
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

set(IMAG3D_SOURCES
    src/imag3d-cuda/imag3d-cuda.cu
)

set(REAL3D_SOURCES
    src/real3d-cuda/real3d-cuda.cu
)

# Create shared library for common utilities
add_library(cuda_bec_utils STATIC
    ${UTILS_CUDA_SOURCES}
    ${UTILS_CPP_SOURCES}
)

# Set target properties for the utilities library
target_include_directories(cuda_bec_utils PUBLIC
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src/utils
)

if(USE_MODERN_CUDA_TARGETS)
    target_link_libraries(cuda_bec_utils PUBLIC
        CUDA::cudart
        CUDA::cufft
        OpenMP::OpenMP_CXX
    )
else()
    target_link_libraries(cuda_bec_utils PUBLIC
        ${CUDA_LIBRARIES}
        ${CUDA_CUFFT_LIBRARIES}
        OpenMP::OpenMP_CXX
    )
endif()

# Set CUDA properties for the utilities library
set_target_properties(cuda_bec_utils PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Create imag3d-cuda executable
add_executable(imag3d-cuda ${IMAG3D_SOURCES})

# Set target properties for imag3d-cuda
target_include_directories(imag3d-cuda PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src/utils
    ${CMAKE_SOURCE_DIR}/src/imag3d-cuda
)

if(USE_MODERN_CUDA_TARGETS)
    target_link_libraries(imag3d-cuda
        cuda_bec_utils
        CUDA::cudart
        CUDA::cufft
        OpenMP::OpenMP_CXX
    )
else()
    target_link_libraries(imag3d-cuda
        cuda_bec_utils
        ${CUDA_LIBRARIES}
        ${CUDA_CUFFT_LIBRARIES}
        OpenMP::OpenMP_CXX
    )
endif()

# Set CUDA properties for imag3d-cuda
set_target_properties(imag3d-cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Create real3d-cuda executable
add_executable(real3d-cuda ${REAL3D_SOURCES})

# Set target properties for real3d-cuda
target_include_directories(real3d-cuda PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src/utils
    ${CMAKE_SOURCE_DIR}/src/real3d-cuda
)

if(USE_MODERN_CUDA_TARGETS)
    target_link_libraries(real3d-cuda
        cuda_bec_utils
        CUDA::cudart
        CUDA::cufft
        OpenMP::OpenMP_CXX
    )
else()
    target_link_libraries(real3d-cuda
        cuda_bec_utils
        ${CUDA_LIBRARIES}
        ${CUDA_CUFFT_LIBRARIES}
        OpenMP::OpenMP_CXX
    )
endif()

# Set CUDA properties for real3d-cuda
set_target_properties(real3d-cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Platform-specific post-build steps
if(WIN32)
    # Windows-specific post-build steps (if needed)
    message(STATUS "Windows build configuration complete")
elseif(UNIX)
    # Linux-specific post-build steps (if needed)
    message(STATUS "Linux build configuration complete")
endif()

# Print build information
message(STATUS "CUDA-BEC CMake Configuration:")
message(STATUS "  Project: ${PROJECT_NAME} v${PROJECT_VERSION}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  CUDA Standard: ${CMAKE_CUDA_STANDARD}")
message(STATUS "  CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Install Prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "  Use Modern CUDA Targets: ${USE_MODERN_CUDA_TARGETS}")
message(STATUS "  CUDA Found: ${CUDA_FOUND}")
message(STATUS "  CUDA Libraries: ${CUDA_LIBRARIES}")
message(STATUS "  CUDA CUFFT Libraries: ${CUDA_CUFFT_LIBRARIES}")

# Add install targets
install(TARGETS imag3d-cuda real3d-cuda
    RUNTIME DESTINATION bin
)

# Add custom targets for convenience
add_custom_target(clean-all
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/CMakeFiles
    COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_BINARY_DIR}/CMakeCache.txt
    COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_BINARY_DIR}/cmake_install.cmake
    COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_BINARY_DIR}/Makefile
    COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_BINARY_DIR}/*.exe
    COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_BINARY_DIR}/*.so
    COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_BINARY_DIR}/*.a
    COMMENT "Cleaning all build artifacts"
)
