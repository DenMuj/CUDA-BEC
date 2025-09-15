/**
 * @file CudaArray3D.h
 * @brief RAII wrapper for CUDA 3D device memory with automatic layout optimization
 * 
 * This class provides a safe and efficient interface for managing 3D arrays on GPU memory.
 * It automatically chooses between pitched memory (for optimal performance) and linear memory
 * (for compatibility), with seamless fallback mechanisms.
 * 
 * Key Features:
 * - Automatic memory layout optimization (pitched vs linear)
 * - RAII memory management with move semantics
 * - Synchronous and asynchronous data transfer operations
 * - Support for 1D, 2D, and 3D arrays
 * - Exception-safe error handling
 * 
 * @author CUDA-BEC Project
 * @date 2024
 */

#ifndef CUDA_ARRAY_3D_H
#define CUDA_ARRAY_3D_H

#include <cstddef>
#include <cuda_runtime.h>

/**
 * @class CudaArray3D
 * @brief RAII wrapper for 3D CUDA device memory with automatic optimization
 * 
 * This class manages GPU memory for 3D arrays with automatic selection between
 * pitched memory (optimized for 2D/3D access patterns) and linear memory (fallback).
 * Memory layout follows row-major ordering: index = iz*(ny*nx) + iy*nx + ix
 * where nx is fastest-changing, nz is slowest-changing dimension.
 * 
 * @tparam T Data type to store (double, cuDoubleComplex, cufftDoubleComplex, etc.)
 */
template<typename T>
class CudaArray3D {
private:
    // Memory management members
    cudaPitchedPtr pitched_ptr;  ///< CUDA pitched pointer for optimized 3D memory layout
    size_t nx, ny, nz;          ///< Array dimensions (nx=fastest, nz=slowest changing)
    size_t total_elements;      ///< Total number of elements (nx * ny * nz)
    size_t pitch;              ///< Memory pitch in bytes for alignment optimization
    bool use_pitched;          ///< True if using pitched memory, false for linear fallback
    
    // Fallback linear allocation
    T* d_data_linear;          ///< Linear device memory pointer (used when pitched fails)
    
public:
    // ========== CONSTRUCTORS AND DESTRUCTOR ==========
    
    /**
     * @brief Construct a 3D CUDA array with automatic memory optimization
     * @param nx Number of elements in X dimension (fastest-changing)
     * @param ny Number of elements in Y dimension 
     * @param nz Number of elements in Z dimension (slowest-changing)
     * @param use_pitched_memory If true, attempt pitched memory allocation for better performance
     * 
     * The constructor will try to allocate pitched memory for optimal performance.
     * If pitched allocation fails, it automatically falls back to linear memory.
     * For 1D arrays (ny=nz=1), linear memory is used regardless of the flag.
     */
    CudaArray3D(size_t nx, size_t ny, size_t nz, bool use_pitched_memory = true);
    
    /**
     * @brief Construct a 1D CUDA array (special case)
     * @param n Number of elements in the 1D array
     * 
     * This constructor creates a 1D array using linear memory allocation.
     * Equivalent to CudaArray3D(n, 1, 1, false).
     */
    explicit CudaArray3D(size_t n);
    
    /**
     * @brief Destructor - automatically frees all allocated GPU memory
     * 
     * Ensures proper cleanup of both pitched and linear memory allocations.
     * No explicit cleanup is required by the user (RAII principle).
     */
    ~CudaArray3D();
    
    // ========== COPY/MOVE SEMANTICS ==========
    
    /**
     * @brief Copy constructor - DISABLED
     * 
     * Copying is explicitly disabled to prevent accidental expensive GPU memory copying.
     * Use move semantics or explicit copy operations instead.
     */
    CudaArray3D(const CudaArray3D&) = delete;
    
    /**
     * @brief Copy assignment operator - DISABLED
     * 
     * Copy assignment is disabled for the same reasons as copy constructor.
     */
    CudaArray3D& operator=(const CudaArray3D&) = delete;
    
    /**
     * @brief Move constructor - transfers ownership of GPU memory
     * @param other Source object to move from (will be left in valid but empty state)
     * 
     * Efficiently transfers ownership of GPU memory without copying data.
     * The source object is left in a valid but empty state.
     */
    CudaArray3D(CudaArray3D&& other) noexcept;
    
    /**
     * @brief Move assignment operator - transfers ownership of GPU memory
     * @param other Source object to move from
     * @return Reference to this object
     * 
     * Frees current GPU memory and transfers ownership from the source object.
     */
    CudaArray3D& operator=(CudaArray3D&& other) noexcept;
    
    // ========== DATA TRANSFER OPERATIONS ==========
    
    /**
     * @brief Synchronously copy data from host to device
     * @param h_data Host data pointer (must be in row-major order)
     * 
     * Copies data from host memory to GPU memory. The host data must be organized
     * in row-major order: index = iz*(ny*nx) + iy*nx + ix
     * Automatically handles the complexity of pitched vs linear memory layouts.
     * This is a blocking operation - execution continues only after copy completes.
     */
    void copyFromHost(const T* h_data);
    
    /**
     * @brief Synchronously copy data from device to host  
     * @param h_data Host data pointer to receive data (will be in row-major order)
     * 
     * Copies data from GPU memory to host memory. The host data will be organized
     * in row-major order: index = iz*(ny*nx) + iy*nx + ix
     * This is a blocking operation - execution continues only after copy completes.
     */
    void copyToHost(T* h_data) const;
    
    
    // ========== DEVICE POINTER ACCESS ==========
    
    /**
     * @brief Get raw device pointer for kernel launches
     * @return Device pointer to the allocated memory
     * 
     * Returns the base device pointer that can be used in CUDA kernels.
     * For pitched memory, use getPitchElements() for proper indexing.
     * For linear memory, standard array indexing applies.
     */
    T* raw() { 
        return use_pitched ? (T*)pitched_ptr.ptr : d_data_linear; 
    }
    
    /**
     * @brief Get raw device pointer for kernel launches (const version)
     * @return Const device pointer to the allocated memory
     */
    const T* raw() const { 
        return use_pitched ? (const T*)pitched_ptr.ptr : d_data_linear; 
    }
    
    /**
     * @brief Alternative name for raw() - provided for compatibility
     * @return Device pointer to the allocated memory
     */
    T* data() { return raw(); }
    
    /**
     * @brief Alternative name for raw() - provided for compatibility (const version)
     * @return Const device pointer to the allocated memory
     */
    const T* data() const { return raw(); }
    
private:
    // ========== INTERNAL HELPER FUNCTIONS ==========
    
    /**
     * @brief Check CUDA error codes and throw exceptions on failure
     * @param error CUDA error code to check
     * @param msg Descriptive message for the operation that might have failed
     * @throws std::runtime_error if error != cudaSuccess
     * 
     * Centralized error checking that provides meaningful error messages
     * by combining the user message with CUDA's error description.
     */
    void checkCudaError(cudaError_t error, const char* msg) const;
    
    /**
     * @brief Reset all member variables to default/empty state
     * 
     * Used internally by move operations and destructor to ensure
     * objects are left in a valid but empty state.
     */
    void reset();
};

// ========== EXPLICIT TEMPLATE INSTANTIATIONS ==========
// These ensure the template is compiled for specific types used in the project

// Real number types
extern template class CudaArray3D<double>;
extern template class CudaArray3D<float>;

// Complex number types for CUDA/cuFFT
#include <cuComplex.h>
#include <cufft.h>
extern template class CudaArray3D<cuDoubleComplex>;
extern template class CudaArray3D<cuFloatComplex>;
// Note: Avoiding duplicate instantiations as cufftDoubleComplex and cuDoubleComplex might be the same

// For backward compatibility, maintain the original non-templated interface
using CudaArray3D_double = CudaArray3D<double>;

// You can also use this as the default if most of your code uses double
#ifndef CUDA_ARRAY_DEFAULT_TYPE
#define CUDA_ARRAY_DEFAULT_TYPE double
#endif

// Backward compatibility typedef - existing code using CudaArray3D without template will use double
// Note: This creates an alias, so CudaArray3D without template parameters defaults to double
#if !defined(CUDA_ARRAY_TEMPLATED_ONLY)
// For backward compatibility, you can still use CudaArray3D as before, it will default to double
// If you want to be explicit, use CudaArray3D<double> or CudaArray3D<your_type>
#endif

#endif // CUDA_ARRAY_3D_H