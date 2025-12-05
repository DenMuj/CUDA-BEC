/**
 * @file simpson3d_kernel.cu
 * @brief Implementation of Simpson 3D tiled reduction kernel
 *
 * This file contains the implementation of the Simpson 3D tiled reduction kernel,
 * including the kernel function and the launch function.
 */

 #include "simpson3d_kernel.cuh"
 #include "cuda_error_check.cuh"
 #include <algorithm>
 #include <cmath>
 #include <cstdio>
 #include <cuda_runtime.h>
 
 /**
  * @brief Simple helper function to get GPU SM count
  * @return Number of streaming multiprocessors on the current GPU
  */
 inline int getGPUSMCount() {
     int smCount;
     int device;
     CUDA_CHECK(cudaGetDevice(&device));
     CUDA_CHECK(cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device));
     return smCount;
 }
 
 /**
  * @brief Calculate optimal grid size for 3D reduction kernels with grid-stride loops
  * @param smCount: Number of SMs on GPU
  * @param Nx, Ny, Nz: Problem dimensions
  * @param blockSize: Block dimensions
  * @param blocksPerSM: Target blocks per SM (default 3 for reduction kernels)
  * @return dim3 grid size
  */
 inline dim3 getOptimalGridReduction3D(int smCount, long Nx, long Ny, long Nz, dim3 blockSize, int blocksPerSM = 4) {
     int stride = 1;
     int gridX = (Nx + blockSize.x * stride - 1) / (blockSize.x * stride);  // with stride
     int gridY = (Ny + blockSize.y * stride - 1) / (blockSize.y * stride);
     int gridZ = (Nz + blockSize.z * stride - 1) / (blockSize.z * stride);
 
     // Ensure we have enough blocks for good occupancy
     int minBlocks = smCount * blocksPerSM;
     int totalBlocks = gridX * gridY * gridZ;
 
     if (totalBlocks < minBlocks) {
         // Scale up if too few blocks
         double scale = std::pow((double)minBlocks / totalBlocks, 1.0/3.0);  // Cube root for 3D
         gridX = std::max(1, (int)(gridX * scale));
         gridY = std::max(1, (int)(gridY * scale));
         gridZ = std::max(1, (int)(gridZ * scale));
     }
 
     // Cap maximum to avoid too many blocks competing for atomic adds
     gridX = std::min(gridX, 16);
     gridY = std::min(gridY, 16);
     gridZ = std::min(gridZ, 8);
 
     return dim3(gridX, gridY, gridZ);
 }
 
/**
 * @brief Kernel function for Simpson 3D tiled reduction
 * @param f Pointer to function values (DEVICE memory, may be padded)
 * @param partial_sums Pointer to partial sums (DEVICE memory)
 * @param Nx Logical number of points in X direction (for weighting)
 * @param Ny Number of points in Y direction
 * @param Nz Number of points in Z direction
 * @param tile_size_z Number of z-slices per tile
 * @param z_start Starting z-index for the current tile
 * @param f_Nx Actual X dimension of array f (Nx for unpadded, Nx+2 for padded)
 */
__global__ void simpson3d_tiled_reduce(double *f, double *partial_sums, long Nx, long Ny, long Nz,
                                       long tile_size_z, long z_start, long f_Nx) {
    extern __shared__ double shared[];
    double *sum_data = shared;

    // Calculate thread ID within block
    long tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;

    // Calculate global 3D strides for grid-stride loop
    long stride_x = blockDim.x * gridDim.x;
    long stride_y = blockDim.y * gridDim.y;
    long stride_z = blockDim.z * gridDim.z;

    // Initial indices
    long idx_start = blockIdx.x * blockDim.x + threadIdx.x;
    long idy_start = blockIdx.y * blockDim.y + threadIdx.y;
    long idz_start = blockIdx.z * blockDim.z + threadIdx.z;

    double local_sum = 0.0;

    // Grid-stride loop: each thread processes multiple points
    for (long idz_local = idz_start; idz_local < tile_size_z; idz_local += stride_z) {
        long idz_global = z_start + idz_local;
        if (idz_global >= Nz) continue;

        double weight_z = (idz_global == 0 || idz_global == Nz - 1) ? 1.0
                          : (idz_global % 2 == 1)                   ? 4.0
                                                                    : 2.0;

        for (long idy = idy_start; idy < Ny; idy += stride_y) {
            double weight_y = (idy == 0 || idy == Ny - 1) ? 1.0 : (idy % 2 == 1) ? 4.0 : 2.0;
            double weight_yz = weight_y * weight_z;

            for (long idx = idx_start; idx < Nx; idx += stride_x) {
                double weight_x = (idx == 0 || idx == Nx - 1) ? 1.0 : (idx % 2 == 1) ? 4.0 : 2.0;

                // Use actual array dimension for indexing (f_Nx accounts for padding)
                long linear_idx = idz_local * f_Nx * Ny + idy * f_Nx + idx;

                // Accumulate weighted value
                local_sum = fma(f[linear_idx], weight_x * weight_yz, local_sum);
            }
        }
    }
 
    // ============ WARP SHUFFLE REDUCTION ============
    // First, reduce within each warp using shuffle intrinsics
    double val = local_sum;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }

    // Lane 0 of each warp now has the partial sum for that warp
    int warp_id = tid >> 5;   // tid / 32
    int lane_id = tid & 31;   // tid % 32
    int num_warps = (blockDim.x * blockDim.y * blockDim.z) >> 5;  // block_size / 32

    // Lane 0 of each warp writes its partial sum to shared memory
    if (lane_id == 0) {
        sum_data[warp_id] = val;
    }
    __syncthreads();

    // First warp reduces all the partial sums from each warp
    if (warp_id == 0) {
        // Load partial sum (or 0 if this lane has no corresponding warp)
        val = (lane_id < num_warps) ? sum_data[lane_id] : 0.0;

        // Full warp shuffle reduction
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }

        // Thread 0 has the final block sum, add to global result
        if (lane_id == 0) {
            atomicAdd(&partial_sums[0], val);
        }
    }
 }

/**
 * @brief Kernel function for Simpson 3D tiled reduction with complex array cast to double
 * @param f Pointer to function values (DEVICE memory) - complex array cast to double
 * @param partial_sums Pointer to partial sums (DEVICE memory)
 * @param Nx Number of points in X direction
 * @param Ny Number of points in Y direction
 * @param Nz Number of points in Z direction
 * @param tile_size_z Number of z-slices per tile
 * @param z_start Starting z-index for the current tile
 */
__global__ void simpson3d_tiled_reduce_complex(double *f, double *partial_sums, long Nx, long Ny, long Nz, long f_Nx,
                                               long tile_size_z, long z_start) {
    extern __shared__ double shared[];
    double *sum_data = shared;

    // Calculate thread ID within block
    long tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;

    // Calculate global 3D strides for grid-stride loop
    long stride_x = blockDim.x * gridDim.x;
    long stride_y = blockDim.y * gridDim.y;
    long stride_z = blockDim.z * gridDim.z;

    // Initial indices
    long idx_start = blockIdx.x * blockDim.x + threadIdx.x;
    long idy_start = blockIdx.y * blockDim.y + threadIdx.y;
    long idz_start = blockIdx.z * blockDim.z + threadIdx.z;

    double local_sum = 0.0;

    // Grid-stride loop: each thread processes multiple points
    for (long idz_local = idz_start; idz_local < tile_size_z; idz_local += stride_z) {
        long idz_global = z_start + idz_local;
        if (idz_global >= Nz) continue;

        double weight_z = (idz_global == 0 || idz_global == Nz - 1) ? 1.0
                          : (idz_global % 2 == 1)                   ? 4.0
                                                                     : 2.0;

        for (long idy = idy_start; idy < Ny; idy += stride_y) {
            double weight_y = (idy == 0 || idy == Ny - 1) ? 1.0 : (idy % 2 == 1) ? 4.0 : 2.0;
            double weight_yz = weight_y * weight_z;

            for (long idx = idx_start; idx < Nx; idx += stride_x) {
                double weight_x = (idx == 0 || idx == Nx - 1) ? 1.0 : (idx % 2 == 1) ? 4.0 : 2.0;

                // Use local index for accessing the tile data in memory
                // When complex array is cast to double, each element is 2 doubles (real, imag)
                // Access the real part at index 2*linear_idx
                // With padding, stride per z-plane is f_Nx * Ny (f_Nx may be Nx+2 for padded arrays)
                long linear_idx = idz_local * f_Nx * Ny + idy * f_Nx + idx;
                long double_idx = 2 * linear_idx;

                // Accumulate weighted value (read from real part)
                local_sum = fma(f[double_idx], weight_x * weight_yz, local_sum);
            }
        }
    }

   // ============ WARP SHUFFLE REDUCTION ============
   // First, reduce within each warp using shuffle intrinsics
   double val = local_sum;
   #pragma unroll
   for (int offset = 16; offset > 0; offset >>= 1) {
       val += __shfl_down_sync(0xFFFFFFFF, val, offset);
   }

   // Lane 0 of each warp now has the partial sum for that warp
   int warp_id = tid >> 5;   // tid / 32
   int lane_id = tid & 31;   // tid % 32
   int num_warps = (blockDim.x * blockDim.y * blockDim.z) >> 5;  // block_size / 32

   // Lane 0 of each warp writes its partial sum to shared memory
   if (lane_id == 0) {
       sum_data[warp_id] = val;
   }
   __syncthreads();

   // First warp reduces all the partial sums from each warp
   if (warp_id == 0) {
       // Load partial sum (or 0 if this lane has no corresponding warp)
       val = (lane_id < num_warps) ? sum_data[lane_id] : 0.0;

       // Full warp shuffle reduction
       #pragma unroll
       for (int offset = 16; offset > 0; offset >>= 1) {
           val += __shfl_down_sync(0xFFFFFFFF, val, offset);
       }

       // Thread 0 has the final block sum, add to global result
       if (lane_id == 0) {
           atomicAdd(&partial_sums[0], val);
       }
   }
}


/**
 * @brief Launch the Simpson 3D kernel
  * @param d_f Pointer to function values (DEVICE memory)
  * @param d_partial_sum Pointer to partial sums (DEVICE memory)
  * @param Nx Number of points in X direction
  * @param Ny Number of points in Y direction
  * @param Nz Number of points in Z direction
  * @param tile_size_z Number of z-slices per tile
  * @param z_start Starting z-index for the current tile
  * @param current_tile_z Number of z-slices in the current tile
  */
 void launchSimpson3DKernel(double *d_f, double *d_partial_sum, long Nx, long Ny, long Nz,
                            long tile_size_z, long z_start, long current_tile_z, long f_Nx) {
     static int smCount = getGPUSMCount();
     dim3 blockSize(32, 4, 2); // 256 threads per block
 
     // Use SM-aware grid sizing with grid-stride loops
     dim3 gridSize = getOptimalGridReduction3D(smCount, Nx, Ny, current_tile_z, blockSize, 2);
 
    // Calculate shared memory size for warp shuffle reduction
    // Only need space for one partial sum per warp (max 32 warps for 1024 threads)
    int num_warps = (blockSize.x * blockSize.y * blockSize.z + 31) / 32;
    size_t shared_mem_size = num_warps * sizeof(double);
 
     // Launch kernel
     simpson3d_tiled_reduce<<<gridSize, blockSize, shared_mem_size>>>(d_f, d_partial_sum, Nx, Ny, Nz,
                                                                     current_tile_z, z_start, f_Nx);
     CUDA_CHECK_KERNEL("simpson3d_tiled_reduce");
 }

/**
 * @brief Launch the Simpson 3D kernel for complex array cast to double
 * @param d_f Pointer to function values (DEVICE memory) - complex array cast to double
 * @param d_partial_sum Pointer to partial sums (DEVICE memory)
 * @param Nx Number of points in X direction
 * @param Ny Number of points in Y direction
 * @param Nz Number of points in Z direction
 * @param tile_size_z Number of z-slices per tile
 * @param z_start Starting z-index for the current tile
 * @param current_tile_z Number of z-slices in the current tile
 */
void launchSimpson3DKernelComplex(double *d_f, double *d_partial_sum, long Nx, long Ny, long Nz, long f_Nx,
                                 long tile_size_z, long z_start, long current_tile_z) {
    static int smCount = getGPUSMCount();
    dim3 blockSize(32, 4, 2); // 256 threads per block

    // Use SM-aware grid sizing with grid-stride loops
    dim3 gridSize = getOptimalGridReduction3D(smCount, Nx, Ny, current_tile_z, blockSize, 2);

   // Calculate shared memory size for warp shuffle reduction
   // Only need space for one partial sum per warp (max 32 warps for 1024 threads)
   int num_warps = (blockSize.x * blockSize.y * blockSize.z + 31) / 32;
   size_t shared_mem_size = num_warps * sizeof(double);

    // Launch kernel
    // When complex array is cast to double, pointer offset needs to account for 2 doubles per element
    // With padding, stride per z-plane is 2 * f_Nx * Ny (f_Nx may be Nx+2 for padded arrays)
    simpson3d_tiled_reduce_complex<<<gridSize, blockSize, shared_mem_size>>>(d_f + 2 * z_start * f_Nx * Ny, d_partial_sum, Nx, Ny, Nz, f_Nx,
                                                                             current_tile_z, z_start);
    CUDA_CHECK_KERNEL("simpson3d_tiled_reduce_complex");
}

/**
 * @brief Kernel function for Simpson 3D tiled reduction with on-the-fly |psi|^2 computation from complex array
 * @param f Pointer to function values (DEVICE memory, complex array, unpadded, uses Nx for indexing)
 * @param partial_sums Pointer to partial sums (DEVICE memory)
 * @param Nx Number of points in X direction (also used for indexing since array is unpadded)
 * @param Ny Number of points in Y direction
 * @param Nz Number of points in Z direction
 * @param tile_size_z Number of z-slices per tile
 * @param z_start Starting z-index for the current tile
 */
__global__ void simpson3d_tiled_reduce_complex_norm(const cuDoubleComplex *__restrict__ f, double *partial_sums, long Nx, long Ny, long Nz,
                                                     long tile_size_z, long z_start) {
    extern __shared__ double shared[];
    double *sum_data = shared;

    // Calculate thread ID within block
    long tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;

    // Calculate global 3D strides for grid-stride loop
    long stride_x = blockDim.x * gridDim.x;
    long stride_y = blockDim.y * gridDim.y;
    long stride_z = blockDim.z * gridDim.z;

    // Initial indices
    long idx_start = blockIdx.x * blockDim.x + threadIdx.x;
    long idy_start = blockIdx.y * blockDim.y + threadIdx.y;
    long idz_start = blockIdx.z * blockDim.z + threadIdx.z;

    double local_sum = 0.0;

    // Grid-stride loop: each thread processes multiple points
    for (long idz_local = idz_start; idz_local < tile_size_z; idz_local += stride_z) {
        long idz_global = z_start + idz_local;
        if (idz_global >= Nz) continue;

        double weight_z = (idz_global == 0 || idz_global == Nz - 1) ? 1.0
                          : (idz_global % 2 == 1)                   ? 4.0
                                                                    : 2.0;

        for (long idy = idy_start; idy < Ny; idy += stride_y) {
            double weight_y = (idy == 0 || idy == Ny - 1) ? 1.0 : (idy % 2 == 1) ? 4.0 : 2.0;
            double weight_yz = weight_y * weight_z;

            for (long idx = idx_start; idx < Nx; idx += stride_x) {
                double weight_x = (idx == 0 || idx == Nx - 1) ? 1.0 : (idx % 2 == 1) ? 4.0 : 2.0;

                // Use Nx for indexing since array is unpadded
                long linear_idx = idz_local * Nx * Ny + idy * Nx + idx;

                // Read complex value
                cuDoubleComplex psi = __ldg(&f[linear_idx]);

                // Compute |psi|^2 = real^2 + imag^2 using FMA for both squares
                double imag2 = fma(psi.y, psi.y, 0.0);
                double psi_squared = fma(psi.x, psi.x, imag2);

                // Accumulate weighted value
                local_sum = fma(psi_squared, weight_x * weight_yz, local_sum);
            }
        }
    }

    // ============ WARP SHUFFLE REDUCTION ============
    // First, reduce within each warp using shuffle intrinsics
    double val = local_sum;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }

    // Lane 0 of each warp now has the partial sum for that warp
    int warp_id = tid >> 5;   // tid / 32
    int lane_id = tid & 31;   // tid % 32
    int num_warps = (blockDim.x * blockDim.y * blockDim.z) >> 5;  // block_size / 32

    // Lane 0 of each warp writes its partial sum to shared memory
    if (lane_id == 0) {
        sum_data[warp_id] = val;
    }
    __syncthreads();

    // First warp reduces all the partial sums from each warp
    if (warp_id == 0) {
        // Load partial sum (or 0 if this lane has no corresponding warp)
        val = (lane_id < num_warps) ? sum_data[lane_id] : 0.0;

        // Full warp shuffle reduction
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }

        // Thread 0 has the final block sum, add to global result
        if (lane_id == 0) {
            atomicAdd(&partial_sums[0], val);
        }
    }
}

/**
 * @brief Launch the Simpson 3D kernel with on-the-fly |psi|^2 computation from complex array
 * @param d_f Pointer to function values (DEVICE memory, complex array, unpadded, uses Nx for indexing)
 * @param d_partial_sum Pointer to partial sums (DEVICE memory)
 * @param Nx Number of points in X direction (also used for indexing)
 * @param Ny Number of points in Y direction
 * @param Nz Number of points in Z direction
 * @param tile_size_z Number of z-slices per tile
 * @param z_start Starting z-index for the current tile
 * @param current_tile_z Number of z-slices in the current tile
 */
void launchSimpson3DKernelComplexNorm(const cuDoubleComplex *d_f, double *d_partial_sum, long Nx, long Ny, long Nz,
                                       long tile_size_z, long z_start, long current_tile_z) {
    static int smCount = getGPUSMCount();
    dim3 blockSize(32, 4, 2); // 256 threads per block

    // Use SM-aware grid sizing with grid-stride loops
    dim3 gridSize = getOptimalGridReduction3D(smCount, Nx, Ny, current_tile_z, blockSize, 2);

    // Calculate shared memory size for warp shuffle reduction
    // Only need space for one partial sum per warp (max 32 warps for 1024 threads)
    int num_warps = (blockSize.x * blockSize.y * blockSize.z + 31) / 32;
    size_t shared_mem_size = num_warps * sizeof(double);

    // Launch kernel
    // Note: d_f is already offset to point to the start of the current tile by the caller
    // With unpadded array, stride per z-plane is Nx * Ny (complex elements)
    simpson3d_tiled_reduce_complex_norm<<<gridSize, blockSize, shared_mem_size>>>(d_f, d_partial_sum, Nx, Ny, Nz,
                                                                                   current_tile_z, z_start);
    CUDA_CHECK_KERNEL("simpson3d_tiled_reduce_complex_norm");
}

/**
 * @brief Kernel function for Simpson 3D tiled reduction with on-the-fly squaring
 * @param f Pointer to function values (DEVICE memory, unpadded, uses Nx for indexing)
 * @param partial_sums Pointer to partial sums (DEVICE memory)
 * @param Nx Number of points in X direction (also used for indexing since array is unpadded)
 * @param Ny Number of points in Y direction
 * @param Nz Number of points in Z direction
 * @param tile_size_z Number of z-slices per tile
 * @param z_start Starting z-index for the current tile
 */
__global__ void simpson3d_tiled_reduce_norm(const double *__restrict__ f, double *partial_sums, long Nx, long Ny, long Nz,
                                            long tile_size_z, long z_start) {
    extern __shared__ double shared[];
    double *sum_data = shared;

    // Calculate thread ID within block
    long tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;

    // Calculate global 3D strides for grid-stride loop
    long stride_x = blockDim.x * gridDim.x;
    long stride_y = blockDim.y * gridDim.y;
    long stride_z = blockDim.z * gridDim.z;

    // Initial indices
    long idx_start = blockIdx.x * blockDim.x + threadIdx.x;
    long idy_start = blockIdx.y * blockDim.y + threadIdx.y;
    long idz_start = blockIdx.z * blockDim.z + threadIdx.z;

    double local_sum = 0.0;

    // Grid-stride loop: each thread processes multiple points
    for (long idz_local = idz_start; idz_local < tile_size_z; idz_local += stride_z) {
        long idz_global = z_start + idz_local;
        if (idz_global >= Nz) continue;

        double weight_z = (idz_global == 0 || idz_global == Nz - 1) ? 1.0
                          : (idz_global % 2 == 1)                   ? 4.0
                                                                    : 2.0;

        for (long idy = idy_start; idy < Ny; idy += stride_y) {
            double weight_y = (idy == 0 || idy == Ny - 1) ? 1.0 : (idy % 2 == 1) ? 4.0 : 2.0;
            double weight_yz = weight_y * weight_z;

            for (long idx = idx_start; idx < Nx; idx += stride_x) {
                double weight_x = (idx == 0 || idx == Nx - 1) ? 1.0 : (idx % 2 == 1) ? 4.0 : 2.0;

                // Use Nx for indexing since array is unpadded
                long linear_idx = idz_local * Nx * Ny + idy * Nx + idx;

                // Square the value on-the-fly and accumulate weighted value
                double val = f[linear_idx];
                local_sum = fma(val * val, weight_x * weight_yz, local_sum);
            }
        }
    }

    // ============ WARP SHUFFLE REDUCTION ============
    // First, reduce within each warp using shuffle intrinsics
    double val = local_sum;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }

    // Lane 0 of each warp now has the partial sum for that warp
    int warp_id = tid >> 5;   // tid / 32
    int lane_id = tid & 31;   // tid % 32
    int num_warps = (blockDim.x * blockDim.y * blockDim.z) >> 5;  // block_size / 32

    // Lane 0 of each warp writes its partial sum to shared memory
    if (lane_id == 0) {
        sum_data[warp_id] = val;
    }
    __syncthreads();

    // First warp reduces all the partial sums from each warp
    if (warp_id == 0) {
        // Load partial sum (or 0 if this lane has no corresponding warp)
        val = (lane_id < num_warps) ? sum_data[lane_id] : 0.0;

        // Full warp shuffle reduction
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }

        // Thread 0 has the final block sum, add to global result
        if (lane_id == 0) {
            atomicAdd(&partial_sums[0], val);
        }
    }
}

/**
 * @brief Launch the Simpson 3D kernel with on-the-fly squaring
 * @param d_f Pointer to function values (DEVICE memory, unpadded, uses Nx for indexing)
 * @param d_partial_sum Pointer to partial sums (DEVICE memory)
 * @param Nx Number of points in X direction (also used for indexing)
 * @param Ny Number of points in Y direction
 * @param Nz Number of points in Z direction
 * @param tile_size_z Number of z-slices per tile
 * @param z_start Starting z-index for the current tile
 * @param current_tile_z Number of z-slices in the current tile
 */
void launchSimpson3DKernelNorm(const double *d_f, double *d_partial_sum, long Nx, long Ny, long Nz,
                                long tile_size_z, long z_start, long current_tile_z) {
    static int smCount = getGPUSMCount();
    dim3 blockSize(32, 4, 2); // 256 threads per block

    // Use SM-aware grid sizing with grid-stride loops
    dim3 gridSize = getOptimalGridReduction3D(smCount, Nx, Ny, current_tile_z, blockSize, 2);

    // Calculate shared memory size for warp shuffle reduction
    // Only need space for one partial sum per warp (max 32 warps for 1024 threads)
    int num_warps = (blockSize.x * blockSize.y * blockSize.z + 31) / 32;
    size_t shared_mem_size = num_warps * sizeof(double);

    // Launch kernel
    // Note: d_f is already offset to point to the start of the current tile by the caller
    simpson3d_tiled_reduce_norm<<<gridSize, blockSize, shared_mem_size>>>(d_f, d_partial_sum, Nx, Ny, Nz,
                                                                           current_tile_z, z_start);
    CUDA_CHECK_KERNEL("simpson3d_tiled_reduce_norm");
}

/**
 * @brief Kernel function for Simpson 3D tiled reduction with on-the-fly RMS calculation
 * @param f Pointer to function values (DEVICE memory, unpadded, uses Nx for indexing)
 * @param partial_sums Pointer to partial sums (DEVICE memory)
 * @param Nx Number of points in X direction (also used for indexing since array is unpadded)
 * @param Ny Number of points in Y direction
 * @param Nz Number of points in Z direction
 * @param tile_size_z Number of z-slices per tile
 * @param z_start Starting z-index for the current tile
 * @param direction Direction for coordinate calculation (0=x, 1=y, 2=z)
 * @param scale Grid spacing for the chosen direction (dx, dy, or dz)
 */
__global__ void simpson3d_tiled_reduce_rms(const double *__restrict__ f, double *partial_sums, long Nx, long Ny, long Nz,
                                           long tile_size_z, long z_start, int direction, double scale) {
    extern __shared__ double shared[];
    double *sum_data = shared;

    // Calculate thread ID within block
    long tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;

    // Calculate global 3D strides for grid-stride loop
    long stride_x = blockDim.x * gridDim.x;
    long stride_y = blockDim.y * gridDim.y;
    long stride_z = blockDim.z * gridDim.z;

    // Initial indices
    long idx_start = blockIdx.x * blockDim.x + threadIdx.x;
    long idy_start = blockIdx.y * blockDim.y + threadIdx.y;
    long idz_start = blockIdx.z * blockDim.z + threadIdx.z;

    double local_sum = 0.0;

    // Grid-stride loop: each thread processes multiple points
    for (long idz_local = idz_start; idz_local < tile_size_z; idz_local += stride_z) {
        long idz_global = z_start + idz_local;
        if (idz_global >= Nz) continue;

        double weight_z = (idz_global == 0 || idz_global == Nz - 1) ? 1.0
                          : (idz_global % 2 == 1)                   ? 4.0
                                                                    : 2.0;

        for (long idy = idy_start; idy < Ny; idy += stride_y) {
            double weight_y = (idy == 0 || idy == Ny - 1) ? 1.0 : (idy % 2 == 1) ? 4.0 : 2.0;
            double weight_yz = weight_y * weight_z;

            for (long idx = idx_start; idx < Nx; idx += stride_x) {
                double weight_x = (idx == 0 || idx == Nx - 1) ? 1.0 : (idx % 2 == 1) ? 4.0 : 2.0;

                // Use Nx for indexing since array is unpadded
                long linear_idx = idz_local * Nx * Ny + idy * Nx + idx;

                // Read psi value and square it
                double psi_val = f[linear_idx];
                double psi_squared = psi_val * psi_val;

                // Compute coordinate^2 based on direction
                double coordinate_squared = 0.0;
                if (direction == 0) {
                    // x direction: x = (ix - Nx/2) * scale
                    double x = (static_cast<double>(idx) - static_cast<double>(Nx) * 0.5) * scale;
                    coordinate_squared = x * x;
                } else if (direction == 1) {
                    // y direction: y = (iy - Ny/2) * scale
                    double y = (static_cast<double>(idy) - static_cast<double>(Ny) * 0.5) * scale;
                    coordinate_squared = y * y;
                } else if (direction == 2) {
                    // z direction: z = (iz_global - Nz/2) * scale
                    double z = (static_cast<double>(idz_global) - static_cast<double>(Nz) * 0.5) * scale;
                    coordinate_squared = z * z;
                }

                // Compute psi^2 * coordinate^2 and accumulate weighted value
                local_sum = fma(psi_squared * coordinate_squared, weight_x * weight_yz, local_sum);
            }
        }
    }

    // ============ WARP SHUFFLE REDUCTION ============
    // First, reduce within each warp using shuffle intrinsics
    double val = local_sum;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }

    // Lane 0 of each warp now has the partial sum for that warp
    int warp_id = tid >> 5;   // tid / 32
    int lane_id = tid & 31;   // tid % 32
    int num_warps = (blockDim.x * blockDim.y * blockDim.z) >> 5;  // block_size / 32

    // Lane 0 of each warp writes its partial sum to shared memory
    if (lane_id == 0) {
        sum_data[warp_id] = val;
    }
    __syncthreads();

    // First warp reduces all the partial sums from each warp
    if (warp_id == 0) {
        // Load partial sum (or 0 if this lane has no corresponding warp)
        val = (lane_id < num_warps) ? sum_data[lane_id] : 0.0;

        // Full warp shuffle reduction
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }

        // Thread 0 has the final block sum, add to global result
        if (lane_id == 0) {
            atomicAdd(&partial_sums[0], val);
        }
    }
}

/**
 * @brief Launch the Simpson 3D kernel with on-the-fly RMS calculation
 * @param d_f Pointer to function values (DEVICE memory, unpadded, uses Nx for indexing)
 * @param d_partial_sum Pointer to partial sums (DEVICE memory)
 * @param Nx Number of points in X direction (also used for indexing)
 * @param Ny Number of points in Y direction
 * @param Nz Number of points in Z direction
 * @param tile_size_z Number of z-slices per tile
 * @param z_start Starting z-index for the current tile
 * @param current_tile_z Number of z-slices in the current tile
 * @param direction Direction for coordinate calculation (0=x, 1=y, 2=z)
 * @param scale Grid spacing for the chosen direction (dx, dy, or dz)
 */
void launchSimpson3DKernelRMS(const double *d_f, double *d_partial_sum, long Nx, long Ny, long Nz,
                               long tile_size_z, long z_start, long current_tile_z, int direction, double scale) {
    static int smCount = getGPUSMCount();
    dim3 blockSize(32, 4, 2); // 256 threads per block

    // Use SM-aware grid sizing with grid-stride loops
    dim3 gridSize = getOptimalGridReduction3D(smCount, Nx, Ny, current_tile_z, blockSize, 2);

    // Calculate shared memory size for warp shuffle reduction
    // Only need space for one partial sum per warp (max 32 warps for 1024 threads)
    int num_warps = (blockSize.x * blockSize.y * blockSize.z + 31) / 32;
    size_t shared_mem_size = num_warps * sizeof(double);

    // Launch kernel
    // Note: d_f is already offset to point to the start of the current tile by the caller
    simpson3d_tiled_reduce_rms<<<gridSize, blockSize, shared_mem_size>>>(d_f, d_partial_sum, Nx, Ny, Nz,
                                                                          current_tile_z, z_start, direction, scale);
    CUDA_CHECK_KERNEL("simpson3d_tiled_reduce_rms");
}

/**
 * @brief Fused kernel for Simpson 3D tiled reduction computing all 3 RMS integrals in one pass
 * @param f Pointer to function values (DEVICE memory, unpadded, uses Nx for indexing)
 * @param partial_sums_x2 Pointer to partial sums for x^2 * psi^2 (DEVICE memory)
 * @param partial_sums_y2 Pointer to partial sums for y^2 * psi^2 (DEVICE memory)
 * @param partial_sums_z2 Pointer to partial sums for z^2 * psi^2 (DEVICE memory)
 * @param Nx Number of points in X direction (also used for indexing since array is unpadded)
 * @param Ny Number of points in Y direction
 * @param Nz Number of points in Z direction
 * @param tile_size_z Number of z-slices per tile
 * @param z_start Starting z-index for the current tile
 * @param scale_x Grid spacing in X direction (dx)
 * @param scale_y Grid spacing in Y direction (dy)
 * @param scale_z Grid spacing in Z direction (dz)
 */
__global__ void simpson3d_tiled_reduce_rms_fused(const double *__restrict__ f, 
                                                  double *partial_sums_x2,
                                                  double *partial_sums_y2,
                                                  double *partial_sums_z2,
                                                  long Nx, long Ny, long Nz,
                                                  long tile_size_z, long z_start,
                                                  double scale_x, double scale_y, double scale_z) {
    // Shared memory for warp-level partial sums (3 arrays for x2, y2, z2)
    extern __shared__ double shared[];
    int num_warps = (blockDim.x * blockDim.y * blockDim.z) >> 5;
    double *sum_data_x2 = shared;
    double *sum_data_y2 = shared + num_warps;
    double *sum_data_z2 = shared + 2 * num_warps;

    // Calculate thread ID within block
    long tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;

    // Calculate global 3D strides for grid-stride loop
    long stride_x = blockDim.x * gridDim.x;
    long stride_y = blockDim.y * gridDim.y;
    long stride_z = blockDim.z * gridDim.z;

    // Initial indices
    long idx_start = blockIdx.x * blockDim.x + threadIdx.x;
    long idy_start = blockIdx.y * blockDim.y + threadIdx.y;
    long idz_start = blockIdx.z * blockDim.z + threadIdx.z;

    // Precompute half dimensions for coordinate calculation
    double half_Nx = static_cast<double>(Nx) * 0.5;
    double half_Ny = static_cast<double>(Ny) * 0.5;
    double half_Nz = static_cast<double>(Nz) * 0.5;

    double local_sum_x2 = 0.0;
    double local_sum_y2 = 0.0;
    double local_sum_z2 = 0.0;

    // Grid-stride loop: each thread processes multiple points
    for (long idz_local = idz_start; idz_local < tile_size_z; idz_local += stride_z) {
        long idz_global = z_start + idz_local;
        if (idz_global >= Nz) continue;

        double weight_z = (idz_global == 0 || idz_global == Nz - 1) ? 1.0
                          : (idz_global % 2 == 1)                   ? 4.0
                                                                    : 2.0;

        // Precompute z coordinate (constant for this z-slice)
        double z = (static_cast<double>(idz_global) - half_Nz) * scale_z;
        double z2 = z * z;

        for (long idy = idy_start; idy < Ny; idy += stride_y) {
            double weight_y = (idy == 0 || idy == Ny - 1) ? 1.0 : (idy % 2 == 1) ? 4.0 : 2.0;
            double weight_yz = weight_y * weight_z;

            // Precompute y coordinate (constant for this y-row)
            double y = (static_cast<double>(idy) - half_Ny) * scale_y;
            double y2 = y * y;

            for (long idx = idx_start; idx < Nx; idx += stride_x) {
                double weight_x = (idx == 0 || idx == Nx - 1) ? 1.0 : (idx % 2 == 1) ? 4.0 : 2.0;
                double weight_xyz = weight_x * weight_yz;

                // Use Nx for indexing since array is unpadded
                long linear_idx = idz_local * Nx * Ny + idy * Nx + idx;

                // Read psi value ONCE and square it
                double psi_val = f[linear_idx];
                double psi2 = psi_val * psi_val;
                double weighted_psi2 = psi2 * weight_xyz;

                // Compute x coordinate
                double x = (static_cast<double>(idx) - half_Nx) * scale_x;
                double x2 = x * x;

                // Accumulate all 3 integrals using FMA
                local_sum_x2 = fma(x2, weighted_psi2, local_sum_x2);
                local_sum_y2 = fma(y2, weighted_psi2, local_sum_y2);
                local_sum_z2 = fma(z2, weighted_psi2, local_sum_z2);
            }
        }
    }

    // ============ WARP SHUFFLE REDUCTION FOR ALL 3 VALUES ============
    // Reduce within each warp using shuffle intrinsics
    double val_x2 = local_sum_x2;
    double val_y2 = local_sum_y2;
    double val_z2 = local_sum_z2;
    
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val_x2 += __shfl_down_sync(0xFFFFFFFF, val_x2, offset);
        val_y2 += __shfl_down_sync(0xFFFFFFFF, val_y2, offset);
        val_z2 += __shfl_down_sync(0xFFFFFFFF, val_z2, offset);
    }

    // Lane 0 of each warp now has the partial sums for that warp
    int warp_id = tid >> 5;   // tid / 32
    int lane_id = tid & 31;   // tid % 32

    // Lane 0 of each warp writes its partial sums to shared memory
    if (lane_id == 0) {
        sum_data_x2[warp_id] = val_x2;
        sum_data_y2[warp_id] = val_y2;
        sum_data_z2[warp_id] = val_z2;
    }
    __syncthreads();

    // First warp reduces all the partial sums from each warp
    if (warp_id == 0) {
        // Load partial sums (or 0 if this lane has no corresponding warp)
        val_x2 = (lane_id < num_warps) ? sum_data_x2[lane_id] : 0.0;
        val_y2 = (lane_id < num_warps) ? sum_data_y2[lane_id] : 0.0;
        val_z2 = (lane_id < num_warps) ? sum_data_z2[lane_id] : 0.0;

        // Full warp shuffle reduction
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            val_x2 += __shfl_down_sync(0xFFFFFFFF, val_x2, offset);
            val_y2 += __shfl_down_sync(0xFFFFFFFF, val_y2, offset);
            val_z2 += __shfl_down_sync(0xFFFFFFFF, val_z2, offset);
        }

        // Thread 0 has the final block sums, add to global results
        if (lane_id == 0) {
            atomicAdd(&partial_sums_x2[0], val_x2);
            atomicAdd(&partial_sums_y2[0], val_y2);
            atomicAdd(&partial_sums_z2[0], val_z2);
        }
    }
}

/**
 * @brief Launch the fused Simpson 3D RMS kernel
 * @param d_f Pointer to function values (DEVICE memory, unpadded, uses Nx for indexing)
 * @param d_partial_sum_x2 Pointer to partial sums for x^2 * psi^2 (DEVICE memory)
 * @param d_partial_sum_y2 Pointer to partial sums for y^2 * psi^2 (DEVICE memory)
 * @param d_partial_sum_z2 Pointer to partial sums for z^2 * psi^2 (DEVICE memory)
 * @param Nx Number of points in X direction (also used for indexing)
 * @param Ny Number of points in Y direction
 * @param Nz Number of points in Z direction
 * @param tile_size_z Number of z-slices per tile
 * @param z_start Starting z-index for the current tile
 * @param current_tile_z Number of z-slices in the current tile
 * @param scale_x Grid spacing in X direction (dx)
 * @param scale_y Grid spacing in Y direction (dy)
 * @param scale_z Grid spacing in Z direction (dz)
 */
void launchSimpson3DKernelRMSFused(const double *d_f, 
                                    double *d_partial_sum_x2,
                                    double *d_partial_sum_y2,
                                    double *d_partial_sum_z2,
                                    long Nx, long Ny, long Nz,
                                    long tile_size_z, long z_start, long current_tile_z,
                                    double scale_x, double scale_y, double scale_z) {
    static int smCount = getGPUSMCount();
    dim3 blockSize(32, 4, 2); // 256 threads per block

    // Use SM-aware grid sizing with grid-stride loops
    dim3 gridSize = getOptimalGridReduction3D(smCount, Nx, Ny, current_tile_z, blockSize, 2);

    // Calculate shared memory size for warp shuffle reduction
    // Need space for 3 partial sums per warp (x2, y2, z2)
    int num_warps = (blockSize.x * blockSize.y * blockSize.z + 31) / 32;
    size_t shared_mem_size = 3 * num_warps * sizeof(double);

    // Launch kernel
    // Note: d_f is already offset to point to the start of the current tile by the caller
    simpson3d_tiled_reduce_rms_fused<<<gridSize, blockSize, shared_mem_size>>>(
        d_f, d_partial_sum_x2, d_partial_sum_y2, d_partial_sum_z2,
        Nx, Ny, Nz, current_tile_z, z_start, scale_x, scale_y, scale_z);
    CUDA_CHECK_KERNEL("simpson3d_tiled_reduce_rms_fused");
}

/**
 * @brief Fused kernel for Simpson 3D tiled reduction computing all 3 RMS integrals in one pass from complex array
 * @param f Pointer to function values (DEVICE memory, complex array, unpadded, uses Nx for indexing)
 * @param partial_sums_x2 Pointer to partial sums for x^2 * |psi|^2 (DEVICE memory)
 * @param partial_sums_y2 Pointer to partial sums for y^2 * |psi|^2 (DEVICE memory)
 * @param partial_sums_z2 Pointer to partial sums for z^2 * |psi|^2 (DEVICE memory)
 * @param Nx Number of points in X direction (also used for indexing since array is unpadded)
 * @param Ny Number of points in Y direction
 * @param Nz Number of points in Z direction
 * @param tile_size_z Number of z-slices per tile
 * @param z_start Starting z-index for the current tile
 * @param scale_x Grid spacing in X direction (dx)
 * @param scale_y Grid spacing in Y direction (dy)
 * @param scale_z Grid spacing in Z direction (dz)
 */
__global__ void simpson3d_tiled_reduce_complex_rms_fused(const cuDoubleComplex *__restrict__ f, 
                                                          double *partial_sums_x2,
                                                          double *partial_sums_y2,
                                                          double *partial_sums_z2,
                                                          long Nx, long Ny, long Nz,
                                                          long tile_size_z, long z_start,
                                                          double scale_x, double scale_y, double scale_z) {
    // Shared memory for warp-level partial sums (3 arrays for x2, y2, z2)
    extern __shared__ double shared[];
    int num_warps = (blockDim.x * blockDim.y * blockDim.z) >> 5;
    double *sum_data_x2 = shared;
    double *sum_data_y2 = shared + num_warps;
    double *sum_data_z2 = shared + 2 * num_warps;

    // Calculate thread ID within block
    long tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;

    // Calculate global 3D strides for grid-stride loop
    long stride_x = blockDim.x * gridDim.x;
    long stride_y = blockDim.y * gridDim.y;
    long stride_z = blockDim.z * gridDim.z;

    // Initial indices
    long idx_start = blockIdx.x * blockDim.x + threadIdx.x;
    long idy_start = blockIdx.y * blockDim.y + threadIdx.y;
    long idz_start = blockIdx.z * blockDim.z + threadIdx.z;

    // Precompute half dimensions for coordinate calculation
    double half_Nx = static_cast<double>(Nx) * 0.5;
    double half_Ny = static_cast<double>(Ny) * 0.5;
    double half_Nz = static_cast<double>(Nz) * 0.5;

    double local_sum_x2 = 0.0;
    double local_sum_y2 = 0.0;
    double local_sum_z2 = 0.0;

    // Grid-stride loop: each thread processes multiple points
    for (long idz_local = idz_start; idz_local < tile_size_z; idz_local += stride_z) {
        long idz_global = z_start + idz_local;
        if (idz_global >= Nz) continue;

        double weight_z = (idz_global == 0 || idz_global == Nz - 1) ? 1.0
                          : (idz_global % 2 == 1)                   ? 4.0
                                                                    : 2.0;

        // Precompute z coordinate (constant for this z-slice)
        double z = (static_cast<double>(idz_global) - half_Nz) * scale_z;
        double z2 = z * z;

        for (long idy = idy_start; idy < Ny; idy += stride_y) {
            double weight_y = (idy == 0 || idy == Ny - 1) ? 1.0 : (idy % 2 == 1) ? 4.0 : 2.0;
            double weight_yz = weight_y * weight_z;

            // Precompute y coordinate (constant for this y-row)
            double y = (static_cast<double>(idy) - half_Ny) * scale_y;
            double y2 = y * y;

            for (long idx = idx_start; idx < Nx; idx += stride_x) {
                double weight_x = (idx == 0 || idx == Nx - 1) ? 1.0 : (idx % 2 == 1) ? 4.0 : 2.0;
                double weight_xyz = weight_x * weight_yz;

                // Use Nx for indexing since array is unpadded
                long linear_idx = idz_local * Nx * Ny + idy * Nx + idx;

                // Read complex psi value ONCE
                cuDoubleComplex psi = __ldg(&f[linear_idx]);
                
                // Compute |psi|^2 = real^2 + imag^2 using FMA
                double imag2 = fma(psi.y, psi.y, 0.0);
                double psi_squared = fma(psi.x, psi.x, imag2);
                double weighted_psi2 = psi_squared * weight_xyz;

                // Compute x coordinate
                double x = (static_cast<double>(idx) - half_Nx) * scale_x;
                double x2 = x * x;

                // Accumulate all 3 integrals using FMA
                local_sum_x2 = fma(x2, weighted_psi2, local_sum_x2);
                local_sum_y2 = fma(y2, weighted_psi2, local_sum_y2);
                local_sum_z2 = fma(z2, weighted_psi2, local_sum_z2);
            }
        }
    }

    // ============ WARP SHUFFLE REDUCTION FOR ALL 3 VALUES ============
    // Reduce within each warp using shuffle intrinsics
    double val_x2 = local_sum_x2;
    double val_y2 = local_sum_y2;
    double val_z2 = local_sum_z2;
    
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val_x2 += __shfl_down_sync(0xFFFFFFFF, val_x2, offset);
        val_y2 += __shfl_down_sync(0xFFFFFFFF, val_y2, offset);
        val_z2 += __shfl_down_sync(0xFFFFFFFF, val_z2, offset);
    }

    // Lane 0 of each warp now has the partial sums for that warp
    int warp_id = tid >> 5;   // tid / 32
    int lane_id = tid & 31;   // tid % 32

    // Lane 0 of each warp writes its partial sums to shared memory
    if (lane_id == 0) {
        sum_data_x2[warp_id] = val_x2;
        sum_data_y2[warp_id] = val_y2;
        sum_data_z2[warp_id] = val_z2;
    }
    __syncthreads();

    // First warp reduces all the partial sums from each warp
    if (warp_id == 0) {
        // Load partial sums (or 0 if this lane has no corresponding warp)
        val_x2 = (lane_id < num_warps) ? sum_data_x2[lane_id] : 0.0;
        val_y2 = (lane_id < num_warps) ? sum_data_y2[lane_id] : 0.0;
        val_z2 = (lane_id < num_warps) ? sum_data_z2[lane_id] : 0.0;

        // Full warp shuffle reduction
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            val_x2 += __shfl_down_sync(0xFFFFFFFF, val_x2, offset);
            val_y2 += __shfl_down_sync(0xFFFFFFFF, val_y2, offset);
            val_z2 += __shfl_down_sync(0xFFFFFFFF, val_z2, offset);
        }

        // Thread 0 has the final block sums, add to global results
        if (lane_id == 0) {
            atomicAdd(&partial_sums_x2[0], val_x2);
            atomicAdd(&partial_sums_y2[0], val_y2);
            atomicAdd(&partial_sums_z2[0], val_z2);
        }
    }
}

/**
 * @brief Launch the fused Simpson 3D complex RMS kernel
 * @param d_f Pointer to function values (DEVICE memory, complex array, unpadded, uses Nx for indexing)
 * @param d_partial_sum_x2 Pointer to partial sums for x^2 * |psi|^2 (DEVICE memory)
 * @param d_partial_sum_y2 Pointer to partial sums for y^2 * |psi|^2 (DEVICE memory)
 * @param d_partial_sum_z2 Pointer to partial sums for z^2 * |psi|^2 (DEVICE memory)
 * @param Nx Number of points in X direction (also used for indexing)
 * @param Ny Number of points in Y direction
 * @param Nz Number of points in Z direction
 * @param tile_size_z Number of z-slices per tile
 * @param z_start Starting z-index for the current tile
 * @param current_tile_z Number of z-slices in the current tile
 * @param scale_x Grid spacing in X direction (dx)
 * @param scale_y Grid spacing in Y direction (dy)
 * @param scale_z Grid spacing in Z direction (dz)
 */
void launchSimpson3DKernelComplexRMSFused(const cuDoubleComplex *d_f, 
                                           double *d_partial_sum_x2,
                                           double *d_partial_sum_y2,
                                           double *d_partial_sum_z2,
                                           long Nx, long Ny, long Nz,
                                           long tile_size_z, long z_start, long current_tile_z,
                                           double scale_x, double scale_y, double scale_z) {
    static int smCount = getGPUSMCount();
    dim3 blockSize(32, 4, 2); // 256 threads per block

    // Use SM-aware grid sizing with grid-stride loops
    dim3 gridSize = getOptimalGridReduction3D(smCount, Nx, Ny, current_tile_z, blockSize, 2);

    // Calculate shared memory size for warp shuffle reduction
    // Need space for 3 partial sums per warp (x2, y2, z2)
    int num_warps = (blockSize.x * blockSize.y * blockSize.z + 31) / 32;
    size_t shared_mem_size = 3 * num_warps * sizeof(double);

    // Launch kernel
    // Note: d_f is already offset to point to the start of the current tile by the caller
    simpson3d_tiled_reduce_complex_rms_fused<<<gridSize, blockSize, shared_mem_size>>>(
        d_f, d_partial_sum_x2, d_partial_sum_y2, d_partial_sum_z2,
        Nx, Ny, Nz, current_tile_z, z_start, scale_x, scale_y, scale_z);
    CUDA_CHECK_KERNEL("simpson3d_tiled_reduce_complex_rms_fused");
}
 
 /**
  * @brief Get the CUDA error string
  * @param error CUDA error
  * @return Error string
  */
 const char *getCudaErrorString(cudaError_t error) { return cudaGetErrorString(error); }
