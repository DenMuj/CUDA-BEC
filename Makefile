

# Compiler settings
NVCC = nvcc
CXX = g++

# Target architecture for GTX 1660 Super (Turing architecture, compute capability 7.5)
ARCH = -arch=sm_75

# Compiler flags
NVCCFLAGS = -std=c++20 -O3 $(ARCH) -Xcompiler -fPIC -lcufft
CXXFLAGS = -std=c++20 -O3 -fPIC

# Include directories
INCLUDES = -I. -Isrc/utils

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
LIBS = -lcudart -lcufft 

# Default target
all: $(IMAG3D_TARGET) $(REAL3D_TARGET)

# Build both executables
$(IMAG3D_TARGET): $(IMAG3D_ALL_OBJECTS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -Isrc/imag3d-cuda -o $@ $^ $(LIBS)
	@echo "Build complete: $(IMAG3D_TARGET)"

$(REAL3D_TARGET): $(REAL3D_ALL_OBJECTS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -Isrc/real3d-cuda -o $@ $^ $(LIBS)
	@echo "Build complete: $(REAL3D_TARGET)"

# Compile CUDA source files with dependency tracking
%.o: %.cu
	@echo "Compiling CUDA: $<"
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

# Compile C++ source files
%.o: %.cpp
	@echo "Compiling C++: $<"
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Specific rules for Simpson3D files to ensure proper dependencies
$(UTILS_DIR)/simpson3d_kernel.o: $(UTILS_DIR)/simpson3d_kernel.cu $(UTILS_DIR)/simpson3d_kernel.cuh
	@echo "Compiling Simpson3D kernel..."
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

$(UTILS_DIR)/simpson3d_integrator.o: $(UTILS_DIR)/simpson3d_integrator.cu \
                                     $(UTILS_DIR)/simpson3d_integrator.hpp \
                                     $(UTILS_DIR)/simpson3d_kernel.cuh
	@echo "Compiling Simpson3D integrator..."
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

# Main CUDA files with dependencies on their respective headers
$(IMAG3D_SRC_DIR)/imag3d-cuda.o: $(IMAG3D_SRC_DIR)/imag3d-cuda.cu \
                                 $(IMAG3D_SRC_DIR)/imag3d-cuda.cuh \
                                 $(UTILS_DIR)/simpson3d_integrator.hpp
	@echo "Compiling imag3d-cuda main file..."
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -Isrc/imag3d-cuda -c $< -o $@

$(REAL3D_SRC_DIR)/real3d-cuda.o: $(REAL3D_SRC_DIR)/real3d-cuda.cu \
                                 $(REAL3D_SRC_DIR)/real3d-cuda.cuh \
                                 $(UTILS_DIR)/simpson3d_integrator.hpp
	@echo "Compiling real3d-cuda main file..."
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -Isrc/real3d-cuda -c $< -o $@

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -f $(SHARED_OBJECTS) $(IMAG3D_MAIN_OBJECT) $(REAL3D_MAIN_OBJECT) $(IMAG3D_TARGET) $(REAL3D_TARGET)
	@echo "Clean complete"

# Deep clean - also removes any generated dependency files
deep-clean: clean
	find . -name "*.d" -type f -delete
	find . -name "*~" -type f -delete

# Rebuild everything
rebuild: clean all

# Debug build (with debug symbols and less optimization)
debug: NVCCFLAGS = -std=c++20 -g -G -O0 $(ARCH) -Xcompiler -fPIC -lineinfo
debug: CXXFLAGS = -std=c++20 -g -O0 -fPIC
debug: $(IMAG3D_TARGET) $(REAL3D_TARGET)
	@echo "Debug build complete"

# Release build with maximum optimization
release: NVCCFLAGS = -std=c++20 -O3 $(ARCH) -Xcompiler -fPIC -use_fast_math
release: CXXFLAGS = -std=c++20 -O3 -fPIC -march=native
release: clean $(IMAG3D_TARGET) $(REAL3D_TARGET)
	@echo "Release build complete"

# Run the programs
run-imag3d: $(IMAG3D_TARGET)
	./$(IMAG3D_TARGET)

run-real3d: $(REAL3D_TARGET)
	./$(REAL3D_TARGET)

# Run with CUDA debugging
run-imag3d-debug: debug
	cuda-gdb ./$(IMAG3D_TARGET)

run-real3d-debug: debug
	cuda-gdb ./$(REAL3D_TARGET)

# Memory check with cuda-memcheck
memcheck-imag3d: debug
	cuda-memcheck ./$(IMAG3D_TARGET)

memcheck-real3d: debug
	cuda-memcheck ./$(REAL3D_TARGET)

# Profile with nvprof (deprecated but still useful)
profile-imag3d-nvprof: $(IMAG3D_TARGET)
	nvprof --print-gpu-trace ./$(IMAG3D_TARGET)

profile-real3d-nvprof: $(REAL3D_TARGET)
	nvprof --print-gpu-trace ./$(REAL3D_TARGET)

# Profile with Nsight Systems (newer profiler)
profile-imag3d-nsys: $(IMAG3D_TARGET)
	nsys profile --stats=true --output=profile_report_imag3d ./$(IMAG3D_TARGET)

profile-real3d-nsys: $(REAL3D_TARGET)
	nsys profile --stats=true --output=profile_report_real3d ./$(REAL3D_TARGET)

# Profile with Nsight Compute for kernel analysis
profile-imag3d-ncu: $(IMAG3D_TARGET)
	ncu --set full -o profile_kernels_imag3d ./$(IMAG3D_TARGET)

profile-real3d-ncu: $(REAL3D_TARGET)
	ncu --set full -o profile_kernels_real3d ./$(REAL3D_TARGET)

# Show compiler and GPU info
info:
	@echo "========================================="
	@echo "Build Configuration Info"
	@echo "========================================="
	@echo "NVCC version:"
	@$(NVCC) --version
	@echo ""
	@echo "GCC version:"
	@$(CXX) --version | head -n 1
	@echo ""
	@echo "Available GPU devices:"
	@nvidia-smi -L
	@echo ""
	@echo "GPU Details:"
	@nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv
	@echo "========================================="

# Check if all source files exist
check-sources:
	@echo "Checking source files..."
	@for file in $(UTILS_CUDA_SOURCES) $(CPP_SOURCES) $(IMAG3D_MAIN_SOURCE) $(REAL3D_MAIN_SOURCE); do \
		if [ -f $$file ]; then \
			echo "✓ $$file"; \
		else \
			echo "✗ $$file - FILE NOT FOUND"; \
		fi \
	done
	@echo ""
	@echo "Checking header files..."
	@for file in $(IMAG3D_SRC_DIR)/imag3d-cuda.cuh \
	            $(REAL3D_SRC_DIR)/real3d-cuda.cuh \
	            $(UTILS_DIR)/diffint.cuh \
	            $(UTILS_DIR)/multiarray.h \
	            $(UTILS_DIR)/simpson3d_kernel.cuh \
	            $(UTILS_DIR)/simpson3d_integrator.hpp; do \
		if [ -f $$file ]; then \
			echo "✓ $$file"; \
		else \
			echo "✗ $$file - FILE NOT FOUND"; \
		fi \
	done

# Install dependencies (Ubuntu/Debian)
install-deps:
	@echo "Installing CUDA development dependencies..."
	sudo apt-get update
	sudo apt-get install nvidia-cuda-toolkit nvidia-cuda-dev build-essential

# Create a simple test for Simpson3D integration
test-simpson: $(UTILS_DIR)/simpson3d_kernel.o $(UTILS_DIR)/simpson3d_integrator.o
	@echo "Creating Simpson3D test program..."
	@echo "#include \"src/utils/simpson3d_integrator.hpp\"" > test_simpson.cpp
	@echo "#include <iostream>" >> test_simpson.cpp
	@echo "#include <memory>" >> test_simpson.cpp
	@echo "int main() {" >> test_simpson.cpp
	@echo "    long N = 65;" >> test_simpson.cpp
	@echo "    double hx = 0.1, hy = 0.1, hz = 0.1;" >> test_simpson.cpp
	@echo "    std::unique_ptr<double[]> f(new double[N*N*N]);" >> test_simpson.cpp
	@echo "    for(long i = 0; i < N*N*N; i++) f[i] = 1.0;" >> test_simpson.cpp
	@echo "    Simpson3DTiledIntegrator integrator(N, N, 32);" >> test_simpson.cpp
	@echo "    double result = integrator.integrate(hx, hy, hz, f.get(), N, N, N);" >> test_simpson.cpp
	@echo "    std::cout << \"Test result: \" << result << std::endl;" >> test_simpson.cpp
	@echo "    return 0;" >> test_simpson.cpp
	@echo "}" >> test_simpson.cpp
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o test_simpson test_simpson.cpp \
	        $(UTILS_DIR)/simpson3d_kernel.o $(UTILS_DIR)/simpson3d_integrator.o $(LIBS)
	./test_simpson
	rm -f test_simpson test_simpson.cpp

# Help target
help:
	@echo "========================================="
	@echo "Makefile targets for CUDA-BEC project"
	@echo "========================================="
	@echo "Build targets:"
	@echo "  make              - Build both programs (default)"
	@echo "  make imag3d-cuda  - Build imag3d-cuda program only"
	@echo "  make real3d-cuda  - Build real3d-cuda program only"
	@echo "  make debug        - Build both with debug symbols"
	@echo "  make release      - Build both with maximum optimization"
	@echo "  make rebuild      - Clean and rebuild both"
	@echo ""
	@echo "Run targets:"
	@echo "  make run-imag3d      - Run imag3d-cuda program"
	@echo "  make run-real3d      - Run real3d-cuda program"
	@echo "  make run-imag3d-debug - Run imag3d-cuda with cuda-gdb"
	@echo "  make run-real3d-debug - Run real3d-cuda with cuda-gdb"
	@echo "  make memcheck-imag3d  - Run imag3d-cuda with cuda-memcheck"
	@echo "  make memcheck-real3d  - Run real3d-cuda with cuda-memcheck"
	@echo ""
	@echo "Profiling targets:"
	@echo "  make profile-imag3d-nvprof - Profile imag3d-cuda with nvprof"
	@echo "  make profile-real3d-nvprof - Profile real3d-cuda with nvprof"
	@echo "  make profile-imag3d-nsys   - Profile imag3d-cuda with Nsight Systems"
	@echo "  make profile-real3d-nsys   - Profile real3d-cuda with Nsight Systems"
	@echo "  make profile-imag3d-ncu    - Profile imag3d-cuda with Nsight Compute"
	@echo "  make profile-real3d-ncu    - Profile real3d-cuda with Nsight Compute"
	@echo ""
	@echo "Utility targets:"
	@echo "  make clean        - Remove build artifacts"
	@echo "  make deep-clean   - Remove all generated files"
	@echo "  make info         - Show compiler and GPU info"
	@echo "  make check-sources - Verify all source files exist"
	@echo "  make test-simpson - Test Simpson3D integration"
	@echo "  make install-deps - Install CUDA dependencies"
	@echo "  make help         - Show this help message"
	@echo "========================================="

# Phony targets (targets that don't create files)
.PHONY: all clean deep-clean rebuild debug release \
        run-imag3d run-real3d run-imag3d-debug run-real3d-debug \
        memcheck-imag3d memcheck-real3d \
        profile-imag3d-nvprof profile-real3d-nvprof \
        profile-imag3d-nsys profile-real3d-nsys \
        profile-imag3d-ncu profile-real3d-ncu \
        info check-sources install-deps test-simpson help \
        imag3d-cuda real3d-cuda

# Automatic dependency generation for better incremental builds
DEPDIR = .deps
DEPFLAGS = -MMD -MP -MF $(DEPDIR)/$*.d

# Create dependency directory
$(shell mkdir -p $(DEPDIR)/$(IMAG3D_SRC_DIR) $(DEPDIR)/$(REAL3D_SRC_DIR) $(DEPDIR)/$(UTILS_DIR))

# Include dependency files if they exist
-include $(patsubst %,$(DEPDIR)/%.d,$(basename $(UTILS_CUDA_SOURCES) $(CPP_SOURCES) $(IMAG3D_MAIN_SOURCE) $(REAL3D_MAIN_SOURCE)))