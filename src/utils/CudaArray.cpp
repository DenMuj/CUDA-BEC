/**
 * @file CudaArray3D.cpp
 * @brief Implementation of CudaArray3D class for CUDA 3D device memory management
 * 
 * This file contains the implementation of all CudaArray3D member functions,
 * including constructors, memory transfer operations, and utility functions.
 * The implementation handles both pitched and linear memory layouts with
 * automatic fallback mechanisms for robustness.
 * 
 * @author CUDA-BEC Project
 * @date 2025
 */

#include "CudaArray.h"
#include <stdexcept>
#include <cstring>

// ========== INTERNAL HELPER FUNCTIONS ==========

/**
 * @brief Reset all member variables to default/empty state
 * 
 * This function is used internally by move operations and destructor
 * to ensure objects are left in a valid but empty state. It resets
 * all pointers to nullptr and counters to zero.
 */
template<typename T>
void CudaArray3D<T>::reset() {
    pitched_ptr = make_cudaPitchedPtr(nullptr, 0, 0, 0);
    d_data_linear = nullptr;
    nx = ny = nz = 0;
    total_elements = 0;
    pitch = 0;
    use_pitched = false;
}

// ========== CONSTRUCTORS AND DESTRUCTOR ==========

/**
 * @brief Construct a 3D CUDA array with automatic memory optimization
 * 
 * This constructor attempts to allocate the most efficient memory layout:
 * - For true 3D arrays (nz > 1, ny > 1): Uses cudaMalloc3D for pitched memory
 * - For 2D arrays (nz = 1, ny > 1): Uses cudaMallocPitch for pitched memory  
 * - For 1D arrays or when pitched fails: Falls back to linear cudaMalloc
 * 
 * The automatic fallback ensures the allocation always succeeds if sufficient
 * memory is available, even on systems with limited pitched memory support.
 */
template<typename T>
CudaArray3D<T>::CudaArray3D(size_t nx_, size_t ny_, size_t nz_, bool use_pitched_memory) 
    : nx(nx_), ny(ny_), nz(nz_), total_elements(nx_ * ny_ * nz_), 
      use_pitched(use_pitched_memory), d_data_linear(nullptr) {
    
    pitched_ptr = make_cudaPitchedPtr(nullptr, 0, 0, 0);
    
    if (total_elements > 0) {
        if (use_pitched && nz > 1 && ny > 1) {
            // ===== TRUE 3D ALLOCATION (nz > 1, ny > 1) =====
            // Use cudaMalloc3D for optimal 3D memory layout with hardware alignment
            cudaExtent extent = make_cudaExtent(nx * sizeof(T), ny, nz);
            cudaError_t error = cudaMalloc3D(&pitched_ptr, extent);
            
            if (error != cudaSuccess) {
                // 3D pitched allocation failed - fallback to linear memory
                use_pitched = false;
                size_t bytes = total_elements * sizeof(T);
                error = cudaMalloc((void**)&d_data_linear, bytes);
                checkCudaError(error, "Failed to allocate device memory");
                pitch = nx * sizeof(T);  // No actual pitch in linear memory
            } else {
                // 3D pitched allocation succeeded
                pitch = pitched_ptr.pitch;
            }
        } else if (use_pitched && ny > 1) {
            // ===== 2D ALLOCATION (nz = 1, ny > 1) =====
            // Use cudaMallocPitch for optimal 2D memory layout
            size_t width = nx * sizeof(T);
            size_t height = ny * nz;  // For 2D: nz=1, so height = ny
            
            T* ptr2d;
            cudaError_t error = cudaMallocPitch((void**)&ptr2d, &pitch, width, height);
            
            if (error != cudaSuccess) {
                // 2D pitched allocation failed - fallback to linear memory
                use_pitched = false;
                size_t bytes = total_elements * sizeof(T);
                error = cudaMalloc((void**)&d_data_linear, bytes);
                checkCudaError(error, "Failed to allocate device memory");
                pitch = nx * sizeof(T);  // No actual pitch in linear memory
            } else {
                // 2D pitched allocation succeeded - wrap in 3D structure for consistency
                pitched_ptr = make_cudaPitchedPtr(ptr2d, pitch, nx * sizeof(T), ny);
            }
        } else {
            // ===== LINEAR ALLOCATION =====
            // For 1D arrays, when pitched memory is not requested, or as fallback
            use_pitched = false;
            size_t bytes = total_elements * sizeof(T);
            cudaError_t error = cudaMalloc((void**)&d_data_linear, bytes);
            checkCudaError(error, "Failed to allocate device memory");
            pitch = nx * sizeof(T);  // Conceptual pitch for consistency
        }
    }
}

/**
 * @brief Construct a 1D CUDA array (special case)
 * 
 * This constructor creates a 1D array by delegating to the 3D constructor
 * with ny=1, nz=1, and pitched memory disabled. Linear memory allocation
 * is always used for 1D arrays since pitched memory provides no benefit.
 */
template<typename T>
CudaArray3D<T>::CudaArray3D(size_t n) : CudaArray3D(n, 1, 1, false) {
    // Delegates to 3D constructor with ny=1, nz=1, and no pitched memory
}

/**
 * @brief Destructor - automatically frees all allocated GPU memory
 * 
 * The destructor ensures proper cleanup of both pitched and linear memory
 * allocations. It uses the appropriate cudaFree call based on the memory
 * type and resets all member variables. RAII ensures no memory leaks.
 */
template<typename T>
CudaArray3D<T>::~CudaArray3D() {
    if (use_pitched && pitched_ptr.ptr != nullptr) {
        cudaFree(pitched_ptr.ptr);  // Free pitched memory
    } else if (d_data_linear != nullptr) {
        cudaFree(d_data_linear);    // Free linear memory
    }
    reset();  // Reset all member variables
}

// ========== MOVE SEMANTICS ==========

/**
 * @brief Move constructor - transfers ownership of GPU memory
 * 
 * Efficiently transfers ownership of GPU memory from another object
 * without copying data. The source object is left in a valid but
 * empty state (all pointers set to nullptr, sizes to zero).
 */
template<typename T>
CudaArray3D<T>::CudaArray3D(CudaArray3D&& other) noexcept 
    : pitched_ptr(other.pitched_ptr), 
      nx(other.nx), ny(other.ny), nz(other.nz),
      total_elements(other.total_elements),
      pitch(other.pitch),
      use_pitched(other.use_pitched),
      d_data_linear(other.d_data_linear) {
    other.reset();  // Leave source object in valid but empty state
}

/**
 * @brief Move assignment operator - transfers ownership of GPU memory
 * 
 * Frees any existing GPU memory held by this object, then transfers
 * ownership from the source object. Provides strong exception safety
 * and prevents self-assignment issues.
 */
template<typename T>
CudaArray3D<T>& CudaArray3D<T>::operator=(CudaArray3D&& other) noexcept {
    if (this != &other) {  // Prevent self-assignment
        // Free existing resources before taking ownership of new ones
        if (use_pitched && pitched_ptr.ptr != nullptr) {
            cudaFree(pitched_ptr.ptr);
        } else if (d_data_linear != nullptr) {
            cudaFree(d_data_linear);
        }
        
        // Transfer ownership of resources from source object
        pitched_ptr = other.pitched_ptr;
        nx = other.nx;
        ny = other.ny;
        nz = other.nz;
        total_elements = other.total_elements;
        pitch = other.pitch;
        use_pitched = other.use_pitched;
        d_data_linear = other.d_data_linear;
        
        // Leave source object in valid but empty state
        other.reset();
    }
    return *this;
}

// ========== DATA TRANSFER OPERATIONS ==========

/**
 * @brief Synchronously copy data from host to device
 * 
 * Copies data from host memory to GPU memory, handling the complexity
 * of different memory layouts automatically. For pitched memory, uses
 * cudaMemcpy3D to properly handle padding. For linear memory, uses
 * simple cudaMemcpy. This is a blocking operation.
 */
template<typename T>
void CudaArray3D<T>::copyFromHost(const T* h_data) {
    if (h_data == nullptr || total_elements == 0) return;  // Safety checks
    
    if (use_pitched) {
        // ===== PITCHED MEMORY COPY =====
        // Use cudaMemcpy3D to handle padding and memory layout automatically
        cudaMemcpy3DParms params = {0};  // Initialize all fields to zero
        
        // Configure source (host) - treat as linear memory
        params.srcPtr = make_cudaPitchedPtr((void*)h_data, 
                                           nx * sizeof(T),  // pitch = width (no padding)
                                           nx * sizeof(T),  // width in bytes
                                           ny);                  // height
        params.srcPos = make_cudaPos(0, 0, 0);  // Start from origin
        
        // Configure destination (device) - use pitched memory layout
        params.dstPtr = pitched_ptr;            // Our pitched memory allocation
        params.dstPos = make_cudaPos(0, 0, 0);  // Start from origin
        
        // Configure copy extent and direction
        params.extent = make_cudaExtent(nx * sizeof(T), ny, nz);  // Full array size
        params.kind = cudaMemcpyHostToDevice;
        
        // Execute the 3D copy operation
        cudaError_t error = cudaMemcpy3D(&params);
        checkCudaError(error, "Failed to copy data from host to device (pitched)");
    } else {
        // ===== LINEAR MEMORY COPY =====
        // Simple contiguous memory copy - no padding to worry about
        size_t bytes = total_elements * sizeof(T);
        cudaError_t error = cudaMemcpy(d_data_linear, h_data, bytes, cudaMemcpyHostToDevice);
        checkCudaError(error, "Failed to copy data from host to device");
    }
}

/**
 * @brief Synchronously copy data from device to host
 * 
 * Copies data from GPU memory to host memory, automatically handling
 * different memory layouts. The host data will be in row-major order
 * regardless of the device memory layout. This is a blocking operation.
 */
template<typename T>
void CudaArray3D<T>::copyToHost(T* h_data) const {
    if (h_data == nullptr || total_elements == 0) return;  // Safety checks
    
    if (use_pitched) {
        // ===== PITCHED MEMORY COPY =====
        // Use cudaMemcpy3D to properly handle padding removal
        cudaMemcpy3DParms params = {0};  // Initialize all fields to zero
        
        // Configure source (device) - use pitched memory layout
        params.srcPtr = pitched_ptr;            // Our pitched memory allocation
        params.srcPos = make_cudaPos(0, 0, 0);  // Start from origin
        
        // Configure destination (host) - treat as linear memory
        params.dstPtr = make_cudaPitchedPtr((void*)h_data,
                                           nx * sizeof(T),  // pitch = width (no padding)
                                           nx * sizeof(T),  // width in bytes
                                           ny);                  // height
        params.dstPos = make_cudaPos(0, 0, 0);  // Start from origin
        
        // Configure copy extent and direction
        params.extent = make_cudaExtent(nx * sizeof(T), ny, nz);  // Full array size
        params.kind = cudaMemcpyDeviceToHost;
        
        // Execute the 3D copy operation (removes padding automatically)
        cudaError_t error = cudaMemcpy3D(&params);
        checkCudaError(error, "Failed to copy data from device to host (pitched)");
    } else {
        // ===== LINEAR MEMORY COPY =====
        // Simple contiguous memory copy - no padding to remove
        size_t bytes = total_elements * sizeof(T);
        cudaError_t error = cudaMemcpy(h_data, d_data_linear, bytes, cudaMemcpyDeviceToHost);
        checkCudaError(error, "Failed to copy data from device to host");
    }
}

/**
 * @brief Asynchronously copy data from host to device
 * 
 * Non-blocking version of copyFromHost(). The copy operation is queued
 * in the specified CUDA stream, allowing overlap with other operations.
 * Use cudaStreamSynchronize() to wait for completion.
 */
template<typename T>
void CudaArray3D<T>::copyFromHostAsync(const T* h_data, cudaStream_t stream) {
    if (h_data == nullptr || total_elements == 0) return;  // Safety checks
    
    if (use_pitched) {
        // ===== ASYNC PITCHED MEMORY COPY =====
        // Use cudaMemcpy3DAsync for non-blocking operation
        cudaMemcpy3DParms params = {0};  // Initialize all fields to zero
        
        // Configure source, destination, extent, and direction (same as sync version)
        params.srcPtr = make_cudaPitchedPtr((void*)h_data,
                                           nx * sizeof(T),  // pitch = width
                                           nx * sizeof(T),  // width in bytes
                                           ny);                  // height
        params.srcPos = make_cudaPos(0, 0, 0);
        params.dstPtr = pitched_ptr;
        params.dstPos = make_cudaPos(0, 0, 0);
        params.extent = make_cudaExtent(nx * sizeof(T), ny, nz);
        params.kind = cudaMemcpyHostToDevice;
        
        // Execute async 3D copy operation in specified stream
        cudaError_t error = cudaMemcpy3DAsync(&params, stream);
        checkCudaError(error, "Failed to copy data from host to device async (pitched)");
    } else {
        // ===== ASYNC LINEAR MEMORY COPY =====
        // Simple async contiguous memory copy
        size_t bytes = total_elements * sizeof(T);
        cudaError_t error = cudaMemcpyAsync(d_data_linear, h_data, bytes, 
                                           cudaMemcpyHostToDevice, stream);
        checkCudaError(error, "Failed to copy data from host to device async");
    }
}

/**
 * @brief Asynchronously copy data from device to host
 * 
 * Non-blocking version of copyToHost(). The copy operation is queued
 * in the specified CUDA stream, allowing overlap with other operations.
 * Use cudaStreamSynchronize() to wait for completion.
 */
template<typename T>
void CudaArray3D<T>::copyToHostAsync(T* h_data, cudaStream_t stream) const {
    if (h_data == nullptr || total_elements == 0) return;  // Safety checks
    
    if (use_pitched) {
        // ===== ASYNC PITCHED MEMORY COPY =====
        // Use cudaMemcpy3DAsync for non-blocking operation
        cudaMemcpy3DParms params = {0};  // Initialize all fields to zero
        
        // Configure source, destination, extent, and direction (same as sync version)
        params.srcPtr = pitched_ptr;
        params.srcPos = make_cudaPos(0, 0, 0);
        params.dstPtr = make_cudaPitchedPtr((void*)h_data,
                                           nx * sizeof(T),  // pitch = width
                                           nx * sizeof(T),  // width in bytes
                                           ny);                  // height
        params.dstPos = make_cudaPos(0, 0, 0);
        params.extent = make_cudaExtent(nx * sizeof(T), ny, nz);
        params.kind = cudaMemcpyDeviceToHost;
        
        // Execute async 3D copy operation in specified stream
        cudaError_t error = cudaMemcpy3DAsync(&params, stream);
        checkCudaError(error, "Failed to copy data from device to host async (pitched)");
    } else {
        // ===== ASYNC LINEAR MEMORY COPY =====
        // Simple async contiguous memory copy
        size_t bytes = total_elements * sizeof(T);
        cudaError_t error = cudaMemcpyAsync(h_data, d_data_linear, bytes,
                                           cudaMemcpyDeviceToHost, stream);
        checkCudaError(error, "Failed to copy data from device to host async");
    }
}

// ========== UTILITY OPERATIONS ==========

/**
 * @brief Set all bytes in the array to a specific value
 * 
 * Efficiently initializes all memory to the specified byte value using
 * optimized CUDA memset functions. Note: This operates on bytes, not
 * double values! Use value=0 for zero-initialization.
 */
template<typename T>
void CudaArray3D<T>::memset(int value) {
    if (total_elements == 0) return;  // Nothing to set for empty arrays
    
    if (use_pitched) {
        // ===== PITCHED MEMORY MEMSET =====
        // Use cudaMemset3D to handle padding correctly
        cudaExtent extent = make_cudaExtent(nx * sizeof(T), ny, nz);
        cudaError_t error = cudaMemset3D(pitched_ptr, value, extent);
        checkCudaError(error, "Failed to memset device memory (pitched)");
    } else {
        // ===== LINEAR MEMORY MEMSET =====
        // Simple contiguous memory initialization
        size_t bytes = total_elements * sizeof(T);
        cudaError_t error = cudaMemset(d_data_linear, value, bytes);
        checkCudaError(error, "Failed to memset device memory");
    }
}

/**
 * @brief Check CUDA error codes and throw exceptions on failure
 * 
 * Centralized error checking that provides meaningful error messages
 * by combining the user-provided message with CUDA's detailed error
 * description. Throws std::runtime_error for any CUDA error.
 */
template<typename T>
void CudaArray3D<T>::checkCudaError(cudaError_t error, const char* msg) const {
    if (error != cudaSuccess) {
        // Combine user message with CUDA error description for better debugging
        std::string errorMsg = std::string(msg) + ": " + cudaGetErrorString(error);
        throw std::runtime_error(errorMsg);
    }
}

// ========== EXPLICIT TEMPLATE INSTANTIATIONS ==========
// These ensure the template is compiled for specific types used in the project

// Include CUDA complex types for instantiation
#include <cuComplex.h>
#include <cufft.h>

// Real number types
template class CudaArray3D<double>;
template class CudaArray3D<float>;

// Complex number types for CUDA/cuFFT
template class CudaArray3D<cuDoubleComplex>;
template class CudaArray3D<cuFloatComplex>;
// Note: cufftDoubleComplex and cuDoubleComplex might be the same type (typedef)
// Note: cufftComplex and cuFloatComplex might be the same type (typedef)
// So we only instantiate the fundamental types to avoid duplicates
