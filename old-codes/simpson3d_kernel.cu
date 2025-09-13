// simpson3d_kernel.cu
#include "simpson3d_kernel.cuh"
#include <cuda_runtime.h>
#include <cstdio>

__global__ void simpson3d_tiled_reduce(double *f, double *partial_sums, 
                                       long Nx, long Ny, long Nz,
                                       long tile_size_z, long z_start) {
  extern __shared__ double sdata[];
  
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
  sdata[tid] = local_sum;
  __syncthreads();
  
  // Perform tree-based reduction in shared memory
  long block_size = blockDim.x * blockDim.y * blockDim.z;
  for (unsigned int s = block_size / 2; s > 0; s >>= 1) {
    if (tid < s && tid + s < block_size) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }
  
  // Thread 0 adds the block's sum to global result
  if (tid == 0) {
    atomicAdd(&partial_sums[0], sdata[0]);
  }
}

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
    size_t shared_mem_size = blockSize.x * blockSize.y * blockSize.z * sizeof(double);
    
    // Launch kernel
    simpson3d_tiled_reduce<<<gridSize, blockSize, shared_mem_size>>>(
        d_f, d_partial_sum, Nx, Ny, Nz, current_tile_z, z_start);
}

const char* getCudaErrorString(cudaError_t error) {
    return cudaGetErrorString(error);
}