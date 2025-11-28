// simpson3d_kernel.cuh
#ifndef SIMPSON3D_KERNEL_CUH
#define SIMPSON3D_KERNEL_CUH

#include <cuda_runtime.h>

/**
 * @brief CUDA kernel for tiled Simpson 3D integration with reduction
 *
 * This kernel processes a tile of the 3D data, applies Simpson weights,
 * and performs a parallel reduction to sum the weighted values.
 *
 * @param f Input function values for current tile
 * @param partial_sums Output array for accumulating partial sums
 * @param Nx Grid size in X direction
 * @param Ny Grid size in Y direction
 * @param Nz Total grid size in Z direction (full volume)
 * @param tile_size_z Size of current tile in Z direction
 * @param z_start Starting Z index of current tile in full volume
 */
__global__ void simpson3d_tiled_reduce(double *f, double *partial_sums, long Nx, long Ny, long Nz,
                                       long tile_size_z, long z_start);

/**
 * @brief Wrapper function to launch the Simpson 3D kernel
 *
 * @param d_f Device pointer to input data
 * @param d_partial_sum Device pointer to partial sum storage
 * @param Nx Grid size in X direction
 * @param Ny Grid size in Y direction
 * @param Nz Total grid size in Z direction
 * @param tile_size_z Maximum tile size in Z direction
 * @param z_start Starting Z index of current tile
 * @param current_tile_z Actual size of current tile (may be less than tile_size_z for last tile)
 */
void launchSimpson3DKernel(double *d_f, double *d_partial_sum, long Nx, long Ny, long Nz,
                           long tile_size_z, long z_start, long current_tile_z);

/**
 * @brief CUDA kernel for tiled Simpson 3D integration with reduction for complex array cast to double
 *
 * This kernel processes a tile of the 3D data (complex array cast to double), applies Simpson weights,
 * and performs a parallel reduction to sum the weighted values.
 *
 * @param f Input function values for current tile (complex array cast to double)
 * @param partial_sums Output array for accumulating partial sums
 * @param Nx Grid size in X direction
 * @param Ny Grid size in Y direction
 * @param Nz Total grid size in Z direction (full volume)
 * @param tile_size_z Size of current tile in Z direction
 * @param z_start Starting Z index of current tile in full volume
 */
__global__ void simpson3d_tiled_reduce_complex(double *f, double *partial_sums, long Nx, long Ny, long Nz,
                                                long tile_size_z, long z_start);

/**
 * @brief Wrapper function to launch the Simpson 3D kernel for complex array cast to double
 *
 * @param d_f Device pointer to input data (complex array cast to double)
 * @param d_partial_sum Device pointer to partial sum storage
 * @param Nx Grid size in X direction
 * @param Ny Grid size in Y direction
 * @param Nz Total grid size in Z direction
 * @param tile_size_z Maximum tile size in Z direction
 * @param z_start Starting Z index of current tile
 * @param current_tile_z Actual size of current tile (may be less than tile_size_z for last tile)
 */
void launchSimpson3DKernelComplex(double *d_f, double *d_partial_sum, long Nx, long Ny, long Nz,
                                   long tile_size_z, long z_start, long current_tile_z);

/**
 * @brief Get CUDA error string
 * @param error CUDA error code
 * @return Error description string
 */
const char *getCudaErrorString(cudaError_t error);

#endif // SIMPSON3D_KERNEL_CUH