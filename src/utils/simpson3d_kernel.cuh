// simpson3d_kernel.cuh
#ifndef SIMPSON3D_KERNEL_CUH
#define SIMPSON3D_KERNEL_CUH

#include <cuda_runtime.h>
#include <cuComplex.h>

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
                                       long tile_size_z, long z_start, long f_Nx);

/**
 * @brief Wrapper function to launch the Simpson 3D kernel
 *
 * @param d_f Device pointer to input data (may be padded)
 * @param d_partial_sum Device pointer to partial sum storage
 * @param Nx Logical grid size in X direction (for weighting)
 * @param Ny Grid size in Y direction
 * @param Nz Total grid size in Z direction
 * @param tile_size_z Maximum tile size in Z direction
 * @param z_start Starting Z index of current tile
 * @param current_tile_z Actual size of current tile (may be less than tile_size_z for last tile)
 * @param f_Nx Actual X dimension of array f (Nx for unpadded, Nx+2 for padded)
 */
void launchSimpson3DKernel(double *d_f, double *d_partial_sum, long Nx, long Ny, long Nz,
                           long tile_size_z, long z_start, long current_tile_z, long f_Nx);

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
__global__ void simpson3d_tiled_reduce_complex(double *f, double *partial_sums, long Nx, long Ny, long Nz, long f_Nx,
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
void launchSimpson3DKernelComplex(double *d_f, double *d_partial_sum, long Nx, long Ny, long Nz, long f_Nx,
                                   long tile_size_z, long z_start, long current_tile_z);

/**
 * @brief CUDA kernel for tiled Simpson 3D integration with reduction, computing |psi|^2 on-the-fly from complex array
 *
 * This kernel processes a tile of the 3D data (complex array), computes |psi|^2 = real^2 + imag^2,
 * applies Simpson weights, and performs a parallel reduction to sum the weighted squared values.
 *
 * @param f Input function values for current tile (complex array, unpadded, uses Nx for indexing)
 * @param partial_sums Output array for accumulating partial sums
 * @param Nx Grid size in X direction (also used for indexing since array is unpadded)
 * @param Ny Grid size in Y direction
 * @param Nz Total grid size in Z direction (full volume)
 * @param tile_size_z Size of current tile in Z direction
 * @param z_start Starting Z index of current tile in full volume
 */
__global__ void simpson3d_tiled_reduce_complex_norm(const cuDoubleComplex *f, double *partial_sums, long Nx, long Ny, long Nz,
                                                     long tile_size_z, long z_start);

/**
 * @brief Wrapper function to launch the Simpson 3D kernel with on-the-fly |psi|^2 computation from complex array
 *
 * @param d_f Device pointer to input data (complex array, unpadded, uses Nx for indexing)
 * @param d_partial_sum Device pointer to partial sum storage
 * @param Nx Grid size in X direction (also used for indexing)
 * @param Ny Grid size in Y direction
 * @param Nz Total grid size in Z direction
 * @param tile_size_z Maximum tile size in Z direction
 * @param z_start Starting Z index of current tile
 * @param current_tile_z Actual size of current tile (may be less than tile_size_z for last tile)
 */
void launchSimpson3DKernelComplexNorm(const cuDoubleComplex *d_f, double *d_partial_sum, long Nx, long Ny, long Nz,
                                      long tile_size_z, long z_start, long current_tile_z);

/**
 * @brief CUDA kernel for tiled Simpson 3D integration with reduction, squaring input on-the-fly
 *
 * This kernel processes a tile of the 3D data, squares each value, applies Simpson weights,
 * and performs a parallel reduction to sum the weighted squared values.
 *
 * @param f Input function values for current tile (unpadded, uses Nx for indexing)
 * @param partial_sums Output array for accumulating partial sums
 * @param Nx Grid size in X direction (also used for indexing since array is unpadded)
 * @param Ny Grid size in Y direction
 * @param Nz Total grid size in Z direction (full volume)
 * @param tile_size_z Size of current tile in Z direction
 * @param z_start Starting Z index of current tile in full volume
 */
__global__ void simpson3d_tiled_reduce_norm(const double *f, double *partial_sums, long Nx, long Ny, long Nz,
                                            long tile_size_z, long z_start);

/**
 * @brief Wrapper function to launch the Simpson 3D kernel with on-the-fly squaring
 *
 * @param d_f Device pointer to input data (unpadded, uses Nx for indexing)
 * @param d_partial_sum Device pointer to partial sum storage
 * @param Nx Grid size in X direction (also used for indexing)
 * @param Ny Grid size in Y direction
 * @param Nz Total grid size in Z direction
 * @param tile_size_z Maximum tile size in Z direction
 * @param z_start Starting Z index of current tile
 * @param current_tile_z Actual size of current tile (may be less than tile_size_z for last tile)
 */
void launchSimpson3DKernelNorm(const double *d_f, double *d_partial_sum, long Nx, long Ny, long Nz,
                                long tile_size_z, long z_start, long current_tile_z);

/**
 * @brief CUDA kernel for tiled Simpson 3D integration with reduction, computing psi^2 * coordinate^2 on-the-fly
 *
 * This kernel processes a tile of the 3D data, computes psi^2 * coordinate^2 (where coordinate depends on direction),
 * applies Simpson weights, and performs a parallel reduction to sum the weighted values.
 *
 * @param f Input function values for current tile (unpadded, uses Nx for indexing)
 * @param partial_sums Output array for accumulating partial sums
 * @param Nx Grid size in X direction (also used for indexing since array is unpadded)
 * @param Ny Grid size in Y direction
 * @param Nz Total grid size in Z direction (full volume)
 * @param tile_size_z Size of current tile in Z direction
 * @param z_start Starting Z index of current tile in full volume
 * @param direction Direction for coordinate calculation (0=x, 1=y, 2=z)
 * @param scale Grid spacing for the chosen direction (dx, dy, or dz)
 */
__global__ void simpson3d_tiled_reduce_rms(const double *f, double *partial_sums, long Nx, long Ny, long Nz,
                                           long tile_size_z, long z_start, int direction, double scale);

/**
 * @brief Wrapper function to launch the Simpson 3D kernel with on-the-fly RMS calculation
 *
 * @param d_f Device pointer to input data (unpadded, uses Nx for indexing)
 * @param d_partial_sum Device pointer to partial sum storage
 * @param Nx Grid size in X direction (also used for indexing)
 * @param Ny Grid size in Y direction
 * @param Nz Total grid size in Z direction
 * @param tile_size_z Maximum tile size in Z direction
 * @param z_start Starting Z index of current tile
 * @param current_tile_z Actual size of current tile (may be less than tile_size_z for last tile)
 * @param direction Direction for coordinate calculation (0=x, 1=y, 2=z)
 * @param scale Grid spacing for the chosen direction (dx, dy, or dz)
 */
void launchSimpson3DKernelRMS(const double *d_f, double *d_partial_sum, long Nx, long Ny, long Nz,
                               long tile_size_z, long z_start, long current_tile_z, int direction, double scale);

/**
 * @brief CUDA kernel for tiled Simpson 3D integration computing all 3 RMS integrals in one pass
 *
 * This kernel processes a tile of the 3D data, computes psi^2 * x^2, psi^2 * y^2, and psi^2 * z^2
 * simultaneously, applies Simpson weights, and performs parallel reductions for all three values.
 * This is 3x more efficient than calling the single-direction RMS kernel three times.
 *
 * @param f Input function values for current tile (unpadded, uses Nx for indexing)
 * @param partial_sums_x2 Output array for accumulating x^2 * psi^2 partial sums
 * @param partial_sums_y2 Output array for accumulating y^2 * psi^2 partial sums
 * @param partial_sums_z2 Output array for accumulating z^2 * psi^2 partial sums
 * @param Nx Grid size in X direction (also used for indexing since array is unpadded)
 * @param Ny Grid size in Y direction
 * @param Nz Total grid size in Z direction (full volume)
 * @param tile_size_z Size of current tile in Z direction
 * @param z_start Starting Z index of current tile in full volume
 * @param scale_x Grid spacing in X direction (dx)
 * @param scale_y Grid spacing in Y direction (dy)
 * @param scale_z Grid spacing in Z direction (dz)
 */
__global__ void simpson3d_tiled_reduce_rms_fused(const double *f, 
                                                  double *partial_sums_x2,
                                                  double *partial_sums_y2,
                                                  double *partial_sums_z2,
                                                  long Nx, long Ny, long Nz,
                                                  long tile_size_z, long z_start,
                                                  double scale_x, double scale_y, double scale_z);

/**
 * @brief Wrapper function to launch the fused Simpson 3D RMS kernel
 *
 * @param d_f Device pointer to input data (unpadded, uses Nx for indexing)
 * @param d_partial_sum_x2 Device pointer to partial sum storage for x^2 * psi^2
 * @param d_partial_sum_y2 Device pointer to partial sum storage for y^2 * psi^2
 * @param d_partial_sum_z2 Device pointer to partial sum storage for z^2 * psi^2
 * @param Nx Grid size in X direction (also used for indexing)
 * @param Ny Grid size in Y direction
 * @param Nz Total grid size in Z direction
 * @param tile_size_z Maximum tile size in Z direction
 * @param z_start Starting Z index of current tile
 * @param current_tile_z Actual size of current tile (may be less than tile_size_z for last tile)
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
                                    double scale_x, double scale_y, double scale_z);

/**
 * @brief CUDA kernel for tiled Simpson 3D integration computing all 3 RMS integrals in one pass from complex array
 *
 * This kernel processes a tile of the 3D complex data, computes |psi|^2 * x^2, |psi|^2 * y^2, and |psi|^2 * z^2
 * simultaneously (where |psi|^2 = real^2 + imag^2), applies Simpson weights, and performs parallel reductions
 * for all three values. This is 3x more efficient than calling the single-direction RMS kernel three times.
 *
 * @param f Input function values for current tile (complex array, unpadded, uses Nx for indexing)
 * @param partial_sums_x2 Output array for accumulating x^2 * |psi|^2 partial sums
 * @param partial_sums_y2 Output array for accumulating y^2 * |psi|^2 partial sums
 * @param partial_sums_z2 Output array for accumulating z^2 * |psi|^2 partial sums
 * @param Nx Grid size in X direction (also used for indexing since array is unpadded)
 * @param Ny Grid size in Y direction
 * @param Nz Total grid size in Z direction (full volume)
 * @param tile_size_z Size of current tile in Z direction
 * @param z_start Starting Z index of current tile in full volume
 * @param scale_x Grid spacing in X direction (dx)
 * @param scale_y Grid spacing in Y direction (dy)
 * @param scale_z Grid spacing in Z direction (dz)
 */
__global__ void simpson3d_tiled_reduce_complex_rms_fused(const cuDoubleComplex *f, 
                                                          double *partial_sums_x2,
                                                          double *partial_sums_y2,
                                                          double *partial_sums_z2,
                                                          long Nx, long Ny, long Nz,
                                                          long tile_size_z, long z_start,
                                                          double scale_x, double scale_y, double scale_z);

/**
 * @brief Wrapper function to launch the fused Simpson 3D complex RMS kernel
 *
 * @param d_f Device pointer to input data (complex array, unpadded, uses Nx for indexing)
 * @param d_partial_sum_x2 Device pointer to partial sum storage for x^2 * |psi|^2
 * @param d_partial_sum_y2 Device pointer to partial sum storage for y^2 * |psi|^2
 * @param d_partial_sum_z2 Device pointer to partial sum storage for z^2 * |psi|^2
 * @param Nx Grid size in X direction (also used for indexing)
 * @param Ny Grid size in Y direction
 * @param Nz Total grid size in Z direction
 * @param tile_size_z Maximum tile size in Z direction
 * @param z_start Starting Z index of current tile
 * @param current_tile_z Actual size of current tile (may be less than tile_size_z for last tile)
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
                                           double scale_x, double scale_y, double scale_z);

/**
 * @brief Get CUDA error string
 * @param error CUDA error code
 * @return Error description string
 */
const char *getCudaErrorString(cudaError_t error);

#endif // SIMPSON3D_KERNEL_CUH