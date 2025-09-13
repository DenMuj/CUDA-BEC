// // CudaArray.cpp
// #include "CudaArray.h"
// #include <iostream>
// #include <stdexcept>

// // Constructor
// CudaArray::CudaArray(size_t n) : size(n), bytes(n * sizeof(double)), d_data(nullptr) {
//     if (size > 0) {
//         cudaError_t error = cudaMalloc((void**)&d_data, bytes);
//         checkCudaError(error, "Failed to allocate device memory");
//     }
// }

// // Destructor
// CudaArray::~CudaArray() {
//     if (d_data != nullptr) {
//         cudaFree(d_data);
//         d_data = nullptr;
//     }
// }

// // Move constructor
// CudaArray::CudaArray(CudaArray&& other) noexcept 
//     : d_data(other.d_data), size(other.size), bytes(other.bytes) {
//     other.d_data = nullptr;
//     other.size = 0;
//     other.bytes = 0;
// }

// // Move assignment operator
// CudaArray& CudaArray::operator=(CudaArray&& other) noexcept {
//     if (this != &other) {
//         // Free existing resource
//         if (d_data != nullptr) {
//             cudaFree(d_data);
//         }
        
//         // Move resources from other
//         d_data = other.d_data;
//         size = other.size;
//         bytes = other.bytes;
        
//         // Reset other
//         other.d_data = nullptr;
//         other.size = 0;
//         other.bytes = 0;
//     }
//     return *this;
// }

// // Copy from host to device
// void CudaArray::copyFromHost(const double* h_data) {
//     if (d_data != nullptr && h_data != nullptr && size > 0) {
//         cudaError_t error = cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);
//         checkCudaError(error, "Failed to copy data from host to device");
//     }
// }

// // Copy from device to host
// void CudaArray::copyToHost(double* h_data) const {
//     if (d_data != nullptr && h_data != nullptr && size > 0) {
//         cudaError_t error = cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost);
//         checkCudaError(error, "Failed to copy data from device to host");
//     }
// }

// // Async copy from host to device
// void CudaArray::copyFromHostAsync(const double* h_data, cudaStream_t stream) {
//     if (d_data != nullptr && h_data != nullptr && size > 0) {
//         cudaError_t error = cudaMemcpyAsync(d_data, h_data, bytes, 
//                                            cudaMemcpyHostToDevice, stream);
//         checkCudaError(error, "Failed to copy data from host to device (async)");
//     }
// }

// // Async copy from device to host
// void CudaArray::copyToHostAsync(double* h_data, cudaStream_t stream) const {
//     if (d_data != nullptr && h_data != nullptr && size > 0) {
//         cudaError_t error = cudaMemcpyAsync(h_data, d_data, bytes, 
//                                            cudaMemcpyDeviceToHost, stream);
//         checkCudaError(error, "Failed to copy data from device to host (async)");
//     }
// }

// // Set all bytes to a value
// void CudaArray::memset(int value) {
//     if (d_data != nullptr && size > 0) {
//         cudaError_t error = cudaMemset(d_data, value, bytes);
//         checkCudaError(error, "Failed to memset device memory");
//     }
// }

// // Resize the array
// void CudaArray::resize(size_t new_size) {
//     if (new_size != size) {
//         // Free old memory
//         if (d_data != nullptr) {
//             cudaFree(d_data);
//             d_data = nullptr;
//         }
        
//         // Update size
//         size = new_size;
//         bytes = new_size * sizeof(double);
        
//         // Allocate new memory
//         if (size > 0) {
//             cudaError_t error = cudaMalloc((void**)&d_data, bytes);
//             checkCudaError(error, "Failed to allocate device memory during resize");
//         }
//     }
// }

// // Error checking helper
// void CudaArray::checkCudaError(cudaError_t error, const char* msg) const {
//     if (error != cudaSuccess) {
//         std::string errorMsg = std::string(msg) + ": " + cudaGetErrorString(error);
//         throw std::runtime_error(errorMsg);
//     }
// }


// CudaArray3D.cpp
#include "CudaArray.h"
#include <iostream>
#include <stdexcept>
#include <cstring>

// Helper to reset all members
void CudaArray3D::reset() {
    pitched_ptr = make_cudaPitchedPtr(nullptr, 0, 0, 0);
    d_data_linear = nullptr;
    nx = ny = nz = 0;
    total_elements = 0;
    pitch = 0;
    use_pitched = false;
}

// Constructor for 3D array
CudaArray3D::CudaArray3D(size_t nx_, size_t ny_, size_t nz_, bool use_pitched_memory) 
    : nx(nx_), ny(ny_), nz(nz_), total_elements(nx_ * ny_ * nz_), 
      use_pitched(use_pitched_memory), d_data_linear(nullptr) {
    
    pitched_ptr = make_cudaPitchedPtr(nullptr, 0, 0, 0);
    
    if (total_elements > 0) {
        if (use_pitched && nz > 1 && ny > 1) {
            // Use cudaMalloc3D for true 3D arrays
            cudaExtent extent = make_cudaExtent(nx * sizeof(double), ny, nz);
            cudaError_t error = cudaMalloc3D(&pitched_ptr, extent);
            
            if (error != cudaSuccess) {
                // Fallback to linear allocation
                use_pitched = false;
                size_t bytes = total_elements * sizeof(double);
                error = cudaMalloc((void**)&d_data_linear, bytes);
                checkCudaError(error, "Failed to allocate device memory");
                pitch = nx * sizeof(double);
            } else {
                pitch = pitched_ptr.pitch;
            }
        } else if (use_pitched && ny > 1) {
            // Use cudaMallocPitch for 2D arrays (nz == 1)
            size_t width = nx * sizeof(double);
            size_t height = ny * nz;  // Treat as 2D with height = ny * nz
            
            double* ptr2d;
            cudaError_t error = cudaMallocPitch((void**)&ptr2d, &pitch, width, height);
            
            if (error != cudaSuccess) {
                // Fallback to linear allocation
                use_pitched = false;
                size_t bytes = total_elements * sizeof(double);
                error = cudaMalloc((void**)&d_data_linear, bytes);
                checkCudaError(error, "Failed to allocate device memory");
                pitch = nx * sizeof(double);
            } else {
                // Set up pitched_ptr for 2D allocation
                pitched_ptr = make_cudaPitchedPtr(ptr2d, pitch, nx * sizeof(double), ny);
            }
        } else {
            // For 1D arrays or when pitched memory is not requested, use linear allocation
            use_pitched = false;
            size_t bytes = total_elements * sizeof(double);
            cudaError_t error = cudaMalloc((void**)&d_data_linear, bytes);
            checkCudaError(error, "Failed to allocate device memory");
            pitch = nx * sizeof(double);
        }
    }
}

// Constructor for 1D array (special case)
CudaArray3D::CudaArray3D(size_t n) : CudaArray3D(n, 1, 1, false) {
    // Delegates to 3D constructor with ny=1, nz=1, and no pitched memory
}

// Destructor
CudaArray3D::~CudaArray3D() {
    if (use_pitched && pitched_ptr.ptr != nullptr) {
        cudaFree(pitched_ptr.ptr);
    } else if (d_data_linear != nullptr) {
        cudaFree(d_data_linear);
    }
    reset();
}

// Move constructor
CudaArray3D::CudaArray3D(CudaArray3D&& other) noexcept 
    : pitched_ptr(other.pitched_ptr), 
      nx(other.nx), ny(other.ny), nz(other.nz),
      total_elements(other.total_elements),
      pitch(other.pitch),
      use_pitched(other.use_pitched),
      d_data_linear(other.d_data_linear) {
    other.reset();
}

// Move assignment operator
CudaArray3D& CudaArray3D::operator=(CudaArray3D&& other) noexcept {
    if (this != &other) {
        // Free existing resources
        if (use_pitched && pitched_ptr.ptr != nullptr) {
            cudaFree(pitched_ptr.ptr);
        } else if (d_data_linear != nullptr) {
            cudaFree(d_data_linear);
        }
        
        // Move resources from other
        pitched_ptr = other.pitched_ptr;
        nx = other.nx;
        ny = other.ny;
        nz = other.nz;
        total_elements = other.total_elements;
        pitch = other.pitch;
        use_pitched = other.use_pitched;
        d_data_linear = other.d_data_linear;
        
        // Reset other
        other.reset();
    }
    return *this;
}

// Copy from host to device
void CudaArray3D::copyFromHost(const double* h_data) {
    if (h_data == nullptr || total_elements == 0) return;
    
    if (use_pitched) {
        // For pitched memory, we need to copy row by row
        cudaMemcpy3DParms params = {0};
        
        // Source (host) - linear memory
        params.srcPtr = make_cudaPitchedPtr((void*)h_data, 
                                           nx * sizeof(double),  // pitch = width for linear memory
                                           nx * sizeof(double),  // width
                                           ny);                  // height
        params.srcPos = make_cudaPos(0, 0, 0);
        
        // Destination (device) - pitched memory
        params.dstPtr = pitched_ptr;
        params.dstPos = make_cudaPos(0, 0, 0);
        
        // Copy extent
        params.extent = make_cudaExtent(nx * sizeof(double), ny, nz);
        params.kind = cudaMemcpyHostToDevice;
        
        cudaError_t error = cudaMemcpy3D(&params);
        checkCudaError(error, "Failed to copy data from host to device (pitched)");
    } else {
        // For linear memory, simple copy
        size_t bytes = total_elements * sizeof(double);
        cudaError_t error = cudaMemcpy(d_data_linear, h_data, bytes, cudaMemcpyHostToDevice);
        checkCudaError(error, "Failed to copy data from host to device");
    }
}

// Copy from device to host
void CudaArray3D::copyToHost(double* h_data) const {
    if (h_data == nullptr || total_elements == 0) return;
    
    if (use_pitched) {
        // For pitched memory, we need to copy row by row
        cudaMemcpy3DParms params = {0};
        
        // Source (device) - pitched memory
        params.srcPtr = pitched_ptr;
        params.srcPos = make_cudaPos(0, 0, 0);
        
        // Destination (host) - linear memory
        params.dstPtr = make_cudaPitchedPtr((void*)h_data,
                                           nx * sizeof(double),  // pitch = width for linear memory
                                           nx * sizeof(double),  // width
                                           ny);                  // height
        params.dstPos = make_cudaPos(0, 0, 0);
        
        // Copy extent
        params.extent = make_cudaExtent(nx * sizeof(double), ny, nz);
        params.kind = cudaMemcpyDeviceToHost;
        
        cudaError_t error = cudaMemcpy3D(&params);
        checkCudaError(error, "Failed to copy data from device to host (pitched)");
    } else {
        // For linear memory, simple copy
        size_t bytes = total_elements * sizeof(double);
        cudaError_t error = cudaMemcpy(d_data_linear, h_data, bytes, cudaMemcpyDeviceToHost);
        checkCudaError(error, "Failed to copy data from device to host");
    }
}

// Async copy from host to device
void CudaArray3D::copyFromHostAsync(const double* h_data, cudaStream_t stream) {
    if (h_data == nullptr || total_elements == 0) return;
    
    if (use_pitched) {
        // For pitched memory, use cudaMemcpy3DAsync
        cudaMemcpy3DParms params = {0};
        
        params.srcPtr = make_cudaPitchedPtr((void*)h_data,
                                           nx * sizeof(double),
                                           nx * sizeof(double),
                                           ny);
        params.srcPos = make_cudaPos(0, 0, 0);
        params.dstPtr = pitched_ptr;
        params.dstPos = make_cudaPos(0, 0, 0);
        params.extent = make_cudaExtent(nx * sizeof(double), ny, nz);
        params.kind = cudaMemcpyHostToDevice;
        
        cudaError_t error = cudaMemcpy3DAsync(&params, stream);
        checkCudaError(error, "Failed to copy data from host to device async (pitched)");
    } else {
        size_t bytes = total_elements * sizeof(double);
        cudaError_t error = cudaMemcpyAsync(d_data_linear, h_data, bytes, 
                                           cudaMemcpyHostToDevice, stream);
        checkCudaError(error, "Failed to copy data from host to device async");
    }
}

// Async copy from device to host
void CudaArray3D::copyToHostAsync(double* h_data, cudaStream_t stream) const {
    if (h_data == nullptr || total_elements == 0) return;
    
    if (use_pitched) {
        cudaMemcpy3DParms params = {0};
        
        params.srcPtr = pitched_ptr;
        params.srcPos = make_cudaPos(0, 0, 0);
        params.dstPtr = make_cudaPitchedPtr((void*)h_data,
                                           nx * sizeof(double),
                                           nx * sizeof(double),
                                           ny);
        params.dstPos = make_cudaPos(0, 0, 0);
        params.extent = make_cudaExtent(nx * sizeof(double), ny, nz);
        params.kind = cudaMemcpyDeviceToHost;
        
        cudaError_t error = cudaMemcpy3DAsync(&params, stream);
        checkCudaError(error, "Failed to copy data from device to host async (pitched)");
    } else {
        size_t bytes = total_elements * sizeof(double);
        cudaError_t error = cudaMemcpyAsync(h_data, d_data_linear, bytes,
                                           cudaMemcpyDeviceToHost, stream);
        checkCudaError(error, "Failed to copy data from device to host async");
    }
}

// Set all bytes to a value
void CudaArray3D::memset(int value) {
    if (total_elements == 0) return;
    
    if (use_pitched) {
        // For pitched memory, use cudaMemset3D
        cudaExtent extent = make_cudaExtent(nx * sizeof(double), ny, nz);
        cudaError_t error = cudaMemset3D(pitched_ptr, value, extent);
        checkCudaError(error, "Failed to memset device memory (pitched)");
    } else {
        size_t bytes = total_elements * sizeof(double);
        cudaError_t error = cudaMemset(d_data_linear, value, bytes);
        checkCudaError(error, "Failed to memset device memory");
    }
}

// Error checking helper
void CudaArray3D::checkCudaError(cudaError_t error, const char* msg) const {
    if (error != cudaSuccess) {
        std::string errorMsg = std::string(msg) + ": " + cudaGetErrorString(error);
        throw std::runtime_error(errorMsg);
    }
}
