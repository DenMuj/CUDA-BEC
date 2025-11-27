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
  * @param f Pointer to function values (DEVICE memory)
  * @param partial_sums Pointer to partial sums (DEVICE memory)
  * @param Nx Number of points in X direction
  * @param Ny Number of points in Y direction
  * @param Nz Number of points in Z direction
  * @param tile_size_z Number of z-slices per tile
  * @param z_start Starting z-index for the current tile
  */
 __global__ void simpson3d_tiled_reduce(double *f, double *partial_sums, long Nx, long Ny, long Nz,
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
                 long linear_idx = idz_local * Nx * Ny + idy * Nx + idx;
 
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
     simpson3d_tiled_reduce<<<gridSize, blockSize, shared_mem_size>>>(d_f, d_partial_sum, Nx, Ny, Nz,
                                                                      current_tile_z, z_start);
     CUDA_CHECK_KERNEL("simpson3d_tiled_reduce");
 }
 
 /**
  * @brief Get the CUDA error string
  * @param error CUDA error
  * @return Error string
  */
 const char *getCudaErrorString(cudaError_t error) { return cudaGetErrorString(error); }
