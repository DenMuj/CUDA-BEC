/**
 * @file simpson3d_kernel.cu
 * @brief Implementation of Simpson 3D tiled reduction kernel
 * 
 * This file contains the implementation of the Simpson 3D tiled reduction kernel,
 * including the kernel function and the launch function.
 */

#include "simpson3d_kernel.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

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
__global__ void simpson3d_tiled_reduce(double *f, double *partial_sums, 
                                       long Nx, long Ny, long Nz,
                                       long tile_size_z, long z_start) {
  extern __shared__ double shared[];
  double* sum_data = shared;
  double* comp_data = shared + blockDim.x * blockDim.y * blockDim.z;
  
  // Calculate thread ID within block
  long tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
  
  // Calculate global 3D indices
  long idx = blockIdx.x * blockDim.x + threadIdx.x;
  long idy = blockIdx.y * blockDim.y + threadIdx.y;
  long idz_global = z_start + blockIdx.z * blockDim.z + threadIdx.z;  // Global z index in full volume
  long idz_local = blockIdx.z * blockDim.z + threadIdx.z;  // Local z index within tile
  
  double local_sum = 0.0;
  
  // Check bounds and compute weighted value
  if (idx < Nx && idy < Ny && idz_global < Nz && idz_local < tile_size_z) {
    // Use local index for accessing the tile data in memory
    long linear_idx = idz_local * Nx * Ny + idy * Nx + idx;
    
    // Compute Simpson weights using GLOBAL indices (important for boundary conditions)
    double weight_x = (idx == 0 || idx == Nx - 1) ? 1.0 : (idx % 2 == 1) ? 4.0 : 2.0;
    double weight_y = (idy == 0 || idy == Ny - 1) ? 1.0 : (idy % 2 == 1) ? 4.0 : 2.0;
    double weight_z = (idz_global == 0 || idz_global == Nz - 1) ? 1.0 : (idz_global % 2 == 1) ? 4.0 : 2.0;
    
    local_sum = f[linear_idx] * weight_x * weight_y * weight_z;
  }
  
  // Store in shared memory for reduction
  sum_data[tid] = local_sum;
  comp_data[tid] = 0.0;
  __syncthreads();
  
  // Perform tree-based reduction in shared memory
  long block_size = blockDim.x * blockDim.y * blockDim.z;
  for (unsigned int s = block_size / 2; s > 0; s >>= 1) {
    if (tid < s && tid + s < block_size) {
      double val_a = sum_data[tid] + comp_data[tid];
      double val_b = sum_data[tid + s] + comp_data[tid + s];
      double temp = val_a + val_b;
      double corr;
      if (fabs(val_a) >= fabs(val_b)) {
        corr = (val_a - temp) + val_b;
      } else {
        corr = (val_b - temp) + val_a;
      }
      sum_data[tid] = temp;
      comp_data[tid] = corr;
    }
    __syncthreads();
  }
  
  // Thread 0 adds the block's sum to global result
  if (tid == 0) {
    double block_sum = sum_data[0] + comp_data[0];
    atomicAdd(&partial_sums[0], block_sum);
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
void launchSimpson3DKernel(double* d_f, double* d_partial_sum,
                           long Nx, long Ny, long Nz,
                           long tile_size_z, long z_start,
                           long current_tile_z) {
    // Configure kernel launch parameters
    dim3 blockSize(8, 8, 4);  // 8*8*4 = 256 threads per block
    dim3 gridSize((Nx + blockSize.x - 1) / blockSize.x,
                  (Ny + blockSize.y - 1) / blockSize.y,
                  (current_tile_z + blockSize.z - 1) / blockSize.z);
    
    // Calculate shared memory size for reduction
    size_t shared_mem_size = 2 * blockSize.x * blockSize.y * blockSize.z * sizeof(double);
    
    // Launch kernel
    simpson3d_tiled_reduce<<<gridSize, blockSize, shared_mem_size>>>(
        d_f, d_partial_sum, Nx, Ny, Nz, current_tile_z, z_start);
}

/**
 * @brief Get the CUDA error string
 * @param error CUDA error
 * @return Error string
 */
const char* getCudaErrorString(cudaError_t error) {
    return cudaGetErrorString(error);
}
