# Compiler settings
CUDA_HOME ?= /usr/local/cuda
NVCC = $(CUDA_HOME)/bin/nvcc
CXX = g++

# Auto-detect highest supported C++ standard
# Detect for host compiler first
CXX_STD_CANDIDATES := c++26 c++23 c++20 c++17 c++14 c++11
CXXSTD := $(firstword $(foreach s,$(CXX_STD_CANDIDATES),$(if $(shell $(CXX) -std=$(s) -x c++ -E - </dev/null >/dev/null 2>&1 && echo ok),-std=$(s),)))

NVCCSTD := $(CXXSTD)
ifneq ($(filter -std=c++26 -std=c++23,$(CXXSTD)),)
NVCCSTD := -std=c++20
endif

# Automatically detect GPU compute capability
GPU_COMPUTE_CAPABILITY := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d '.')
ifeq ($(GPU_COMPUTE_CAPABILITY),)
    $(warning Could not detect GPU compute capability, using sm_75 as default)
    GPU_COMPUTE_CAPABILITY := 75
endif
ARCH = -arch=sm_$(GPU_COMPUTE_CAPABILITY)

# Compiler flags
NVCCFLAGS = $(NVCCSTD) -O3 --fmad=true $(ARCH) -Xcompiler -fPIC,-fopenmp -lcufft
CXXFLAGS = $(CXXSTD) -fPIC -fopenmp -O3

# Include directories
INCLUDES = -I. -Isrc/utils -I$(CUDA_HOME)/include

# Source directories
UTILS_DIR = src/utils
IMAG3D_SRC_DIR = src/imag3d-cuda
REAL3D_SRC_DIR = src/real3d-cuda

# Utils CUDA sources (shared between both programs)
UTILS_CUDA_SOURCES = $(UTILS_DIR)/diffint.cu \
                     $(UTILS_DIR)/simpson3d_kernel.cu \
                     $(UTILS_DIR)/simpson3d_integrator.cu

# C++ sources (shared between both programs)
CPP_SOURCES = $(UTILS_DIR)/multiarray.cpp \
			  $(UTILS_DIR)/cfg.cpp \
			  $(UTILS_DIR)/CudaArray.cpp

# Object files for utils (shared)
UTILS_CUDA_OBJECTS = $(UTILS_CUDA_SOURCES:.cu=.o)
CPP_OBJECTS = $(CPP_SOURCES:.cpp=.o)
SHARED_OBJECTS = $(UTILS_CUDA_OBJECTS) $(CPP_OBJECTS)

# Program-specific sources and objects
IMAG3D_MAIN_SOURCE = $(IMAG3D_SRC_DIR)/imag3d-cuda.cu
IMAG3D_MAIN_OBJECT = $(IMAG3D_MAIN_SOURCE:.cu=.o)
IMAG3D_ALL_OBJECTS = $(SHARED_OBJECTS) $(IMAG3D_MAIN_OBJECT)

REAL3D_MAIN_SOURCE = $(REAL3D_SRC_DIR)/real3d-cuda.cu
REAL3D_MAIN_OBJECT = $(REAL3D_MAIN_SOURCE:.cu=.o)
REAL3D_ALL_OBJECTS = $(SHARED_OBJECTS) $(REAL3D_MAIN_OBJECT)

# Target executables
IMAG3D_TARGET = imag3d-cuda
REAL3D_TARGET = real3d-cuda

# Libraries
LIBS = -L$(CUDA_HOME)/lib64 -lcudart -lcufft -Xcompiler -fopenmp 

# Default target - build both programs
all: $(IMAG3D_TARGET) $(REAL3D_TARGET)

# Build imag3d-cuda executable
$(IMAG3D_TARGET): $(IMAG3D_ALL_OBJECTS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -Isrc/imag3d-cuda -o $@ $^ $(LIBS)
	@echo "Build complete: $(IMAG3D_TARGET)"

# Build real3d-cuda executable
$(REAL3D_TARGET): $(REAL3D_ALL_OBJECTS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -Isrc/real3d-cuda -o $@ $^ $(LIBS)
	@echo "Build complete: $(REAL3D_TARGET)"

# Compile CUDA source files
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

# Compile C++ source files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Specific rules for Simpson3D files to ensure proper dependencies
$(UTILS_DIR)/simpson3d_kernel.o: $(UTILS_DIR)/simpson3d_kernel.cu $(UTILS_DIR)/simpson3d_kernel.cuh
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

$(UTILS_DIR)/simpson3d_integrator.o: $(UTILS_DIR)/simpson3d_integrator.cu \
                                     $(UTILS_DIR)/simpson3d_integrator.hpp \
                                     $(UTILS_DIR)/simpson3d_kernel.cuh
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

# Main CUDA files with dependencies on their respective headers
$(IMAG3D_SRC_DIR)/imag3d-cuda.o: $(IMAG3D_SRC_DIR)/imag3d-cuda.cu \
                                 $(IMAG3D_SRC_DIR)/imag3d-cuda.cuh \
                                 $(UTILS_DIR)/simpson3d_integrator.hpp
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -Isrc/imag3d-cuda -c $< -o $@

$(REAL3D_SRC_DIR)/real3d-cuda.o: $(REAL3D_SRC_DIR)/real3d-cuda.cu \
                                 $(REAL3D_SRC_DIR)/real3d-cuda.cuh \
                                 $(UTILS_DIR)/simpson3d_integrator.hpp
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -Isrc/real3d-cuda -c $< -o $@

# Clean build artifacts
clean:
	rm -f $(SHARED_OBJECTS) $(IMAG3D_MAIN_OBJECT) $(REAL3D_MAIN_OBJECT) $(IMAG3D_TARGET) $(REAL3D_TARGET)

# Phony targets (targets that don't create files)
.PHONY: all clean