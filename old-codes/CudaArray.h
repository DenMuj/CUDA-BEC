// // CudaArray.h
// #ifndef CUDA_ARRAY_H
// #define CUDA_ARRAY_H

// #include <cstddef>
// #include <cuda_runtime.h>

// class CudaArray {
// private:
//     double* d_data;      // Device pointer
//     size_t size;         // Number of elements
//     size_t bytes;        // Size in bytes
    
// public:
//     // Constructor - allocates device memory
//     explicit CudaArray(size_t n);
    
//     // Destructor - deallocates device memory
//     ~CudaArray();
    
//     // Disable copy constructor and assignment operator (for safety)
//     // You can implement them properly if needed
//     CudaArray(const CudaArray&) = delete;
//     CudaArray& operator=(const CudaArray&) = delete;
    
//     // Enable move semantics
//     CudaArray(CudaArray&& other) noexcept;
//     CudaArray& operator=(CudaArray&& other) noexcept;
    
//     // Copy data from host to device
//     void copyFromHost(const double* h_data);
    
//     // Copy data from device to host
//     void copyToHost(double* h_data) const;
    
//     // Copy data asynchronously (optional, requires stream)
//     void copyFromHostAsync(const double* h_data, cudaStream_t stream = 0);
//     void copyToHostAsync(double* h_data, cudaStream_t stream = 0) const;
    
//     // Get raw device pointer (for kernel launches)
//     double* raw() { return d_data; }
//     const double* raw() const { return d_data; }
    
//     // Get size information
//     size_t getSize() const { return size; }
//     size_t getBytes() const { return bytes; }
    
//     // Set all elements to a value
//     void memset(int value = 0);
    
//     // Resize array (will deallocate and reallocate)
//     void resize(size_t new_size);
    
//     // Check if allocation was successful
//     bool isValid() const { return d_data != nullptr; }
    
// private:
//     // Helper function to check CUDA errors
//     void checkCudaError(cudaError_t error, const char* msg) const;
// };

// #endif // CUDA_ARRAY_H


// CudaArray3D.h
#ifndef CUDA_ARRAY_3D_H
#define CUDA_ARRAY_3D_H

#include <cstddef>
#include <cuda_runtime.h>

class CudaArray3D {
private:
    cudaPitchedPtr pitched_ptr;  // CUDA pitched pointer for 3D memory
    size_t nx, ny, nz;          // Dimensions (nx fastest, nz slowest)
    size_t total_elements;      // Total number of elements (nx * ny * nz)
    size_t pitch;              // Pitch in bytes for memory alignment
    bool use_pitched;          // Flag to indicate if using pitched memory
    
    // For fallback linear allocation
    double* d_data_linear;     // Used only if pitched allocation fails
    
public:
    // Constructor for 3D array with optional pitched memory
    CudaArray3D(size_t nx, size_t ny, size_t nz, bool use_pitched_memory = true);
    
    // Constructor for 1D array (special case)
    explicit CudaArray3D(size_t n);
    
    // Destructor
    ~CudaArray3D();
    
    // Disable copy constructor and assignment operator
    CudaArray3D(const CudaArray3D&) = delete;
    CudaArray3D& operator=(const CudaArray3D&) = delete;
    
    // Enable move semantics
    CudaArray3D(CudaArray3D&& other) noexcept;
    CudaArray3D& operator=(CudaArray3D&& other) noexcept;
    
    // Copy data from host to device
    // h_data must be in row-major order: index = iz*(ny*nx) + iy*nx + ix
    void copyFromHost(const double* h_data);
    
    // Copy data from device to host
    // h_data will be in row-major order: index = iz*(ny*nx) + iy*nx + ix
    void copyToHost(double* h_data) const;
    
    // Async versions
    void copyFromHostAsync(const double* h_data, cudaStream_t stream = 0);
    void copyToHostAsync(double* h_data, cudaStream_t stream = 0) const;
    
    // Get raw device pointer for kernel launches
    // This returns the base pointer that can be used with proper pitch calculations
    double* raw() { 
        return use_pitched ? (double*)pitched_ptr.ptr : d_data_linear; 
    }
    const double* raw() const { 
        return use_pitched ? (const double*)pitched_ptr.ptr : d_data_linear; 
    }
    
    // Alternative name for compatibility
    double* data() { return raw(); }
    const double* data() const { return raw(); }
    
    // Get pitch information (in elements, not bytes)
    size_t getPitchElements() const { 
        return use_pitched ? (pitch / sizeof(double)) : nx; 
    }
    
    // Get pitch in bytes
    size_t getPitchBytes() const { 
        return use_pitched ? pitch : (nx * sizeof(double)); 
    }
    
    // Get slice pitch (distance between z-slices in bytes)
    size_t getSlicePitch() const {
        return use_pitched ? pitched_ptr.pitch * ny : (nx * ny * sizeof(double));
    }
    
    // Get dimensions
    size_t getSizeX() const { return nx; }
    size_t getSizeY() const { return ny; }
    size_t getSizeZ() const { return nz; }
    size_t getTotalElements() const { return total_elements; }
    
    // Calculate linear index from 3D coordinates (for host-side calculations)
    // This gives you the offset in elements for accessing as a 1D array
    size_t getLinearIndex(size_t ix, size_t iy, size_t iz) const {
        if (use_pitched) {
            // With pitched memory, we need to account for padding
            size_t pitch_elements = pitch / sizeof(double);
            return iz * (pitch_elements * ny) + iy * pitch_elements + ix;
        } else {
            // Standard row-major ordering
            return iz * (ny * nx) + iy * nx + ix;
        }
    }
    
    // Get pointer to a specific (x,y,z) element
    // Useful for kernel launches that need element addresses
    double* getElementPtr(size_t ix, size_t iy, size_t iz) {
        if (use_pitched) {
            char* base = (char*)pitched_ptr.ptr;
            char* slice = base + iz * pitched_ptr.pitch * ny;
            char* row = slice + iy * pitched_ptr.pitch;
            return (double*)(row) + ix;
        } else {
            return d_data_linear + getLinearIndex(ix, iy, iz);
        }
    }
    
    // Set all elements to zero
    void memset(int value = 0);
    
    // Check if allocation was successful
    bool isValid() const { 
        return use_pitched ? (pitched_ptr.ptr != nullptr) : (d_data_linear != nullptr); 
    }
    
    // Check if using pitched memory
    bool isPitched() const { return use_pitched; }
    
    // Get cudaPitchedPtr for advanced CUDA operations
    cudaPitchedPtr getPitchedPtr() const { return pitched_ptr; }
    
private:
    // Helper function to check CUDA errors
    void checkCudaError(cudaError_t error, const char* msg) const;
    
    // Helper to reset all members
    void reset();
};

#endif // CUDA_ARRAY_3D_H