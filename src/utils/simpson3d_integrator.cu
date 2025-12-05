/**
 * @file simpson3d_integrator.cu
 * @brief Implementation of Simpson3DTiledIntegrator class for 3D integration using Simpson's rule
 * with a tiled approach
 *
 * This file contains the implementation of all Simpson3DTiledIntegrator member functions,
 * including constructors, memory management, and integration operations.
 */
#include "simpson3d_integrator.hpp"
#include "simpson3d_kernel.cuh"
#include "cuda_error_check.cuh"
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <iostream>

/**
 * @brief Implementation class (hidden from public interface)
 */
class Simpson3DTiledIntegratorImpl {
  public:
    double *d_partial_sum;    // Device memory for accumulating results
    double *d_partial_sum_y2; // Device memory for y^2 partial sum (fused RMS)
    double *d_partial_sum_z2; // Device memory for z^2 partial sum (fused RMS)
    double h_tile_sum;        // Host memory for tile sum
    double h_tile_sums[3];    // Host memory for fused RMS tile sums [x2, y2, z2]
    long tile_size_z;         // Number of z-slices per tile
    long cached_Nx;           // Cached grid dimensions
    long cached_Ny;

    /**
     * @brief Constructor
     * @param Nx Grid size in X direction
     * @param Ny Grid size in Y direction
     * @param tile_z Number of z-slices to process per tile (default: 32)
     */
    Simpson3DTiledIntegratorImpl(long Nx, long Ny, long tile_z)
        : tile_size_z(tile_z), cached_Nx(Nx), cached_Ny(Ny), h_tile_sum(0.0) {

        // Allocate memory for partial sums (d_partial_sum also used for x^2 in fused)
        CUDA_CHECK(cudaMalloc(&d_partial_sum, sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_partial_sum_y2, sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_partial_sum_z2, sizeof(double)));
    }

    /**
     * @brief Destructor
     */
    ~Simpson3DTiledIntegratorImpl() {
        cudaFree(d_partial_sum);
        cudaFree(d_partial_sum_y2);
        cudaFree(d_partial_sum_z2);
    }

    /**
     * @brief Integrate the function using device memory
     * @param hx Step size in X direction
     * @param hy Step size in Y direction
     * @param hz Step size in Z direction
     * @param d_f_full Pointer to function values (DEVICE memory, may be padded)
     * @param Nx Logical X dimension (for weighting)
     * @param Ny Y dimension
     * @param Nz Z dimension
     * @param f_Nx Actual X dimension of d_f_full (Nx for unpadded, Nx+2 for padded)
     */
    double integrateDevice(double hx, double hy, double hz, double *d_f_full, long Nx, long Ny,
                           long Nz, long f_Nx) {
        double total_sum = 0.0;

        // Process the volume in tiles along the Z direction
        for (long z_start = 0; z_start < Nz; z_start += tile_size_z) {
            // Calculate the actual size of this tile (last tile might be smaller)
            long current_tile_z = std::min(tile_size_z, Nz - z_start);

            // Reset the partial sum for this tile
            CUDA_CHECK(cudaMemset(d_partial_sum, 0, sizeof(double)));

            // Launch kernel for this tile
            // Pointer offset must account for actual array dimension (f_Nx) if padded
            launchSimpson3DKernel(d_f_full + z_start * f_Nx * Ny, d_partial_sum, Nx, Ny, Nz,
                                  tile_size_z, z_start, current_tile_z, f_Nx);

            // Debug mode: sync and check for kernel execution errors
            CUDA_SYNC_CHECK("integrateDevice kernel");

            // Copy result back to host (implicitly syncs)
            CUDA_CHECK(cudaMemcpy(&h_tile_sum, d_partial_sum, sizeof(double), cudaMemcpyDeviceToHost));

            // Accumulate the result
            total_sum += h_tile_sum;
        }

        // Apply Simpson's rule scaling factor
        return total_sum * hx * hy * hz / 27.0;
    }

    /**
     * @brief Integrate the function using device memory (complex array cast to double)
     * @param hx Step size in X direction
     * @param hy Step size in Y direction
     * @param hz Step size in Z direction
     * @param d_f_full Pointer to function values (DEVICE memory) - complex array cast to double
     */
    double integrateDeviceComplex(double hx, double hy, double hz, double *d_f_full, long Nx, long Ny,
                                  long Nz, long f_Nx) {
        double total_sum = 0.0;
        
        // Default to Nx if f_Nx not specified (backward compatibility)
        if (f_Nx == 0) f_Nx = Nx;

        // Process the volume in tiles along the Z direction
        for (long z_start = 0; z_start < Nz; z_start += tile_size_z) {
            // Calculate the actual size of this tile (last tile might be smaller)
            long current_tile_z = std::min(tile_size_z, Nz - z_start);

            // Reset the partial sum for this tile
            CUDA_CHECK(cudaMemset(d_partial_sum, 0, sizeof(double)));

            // Launch kernel for this tile
            // When complex array is cast to double, pointer offset needs to account for 2 doubles per element
            // With padding, stride per z-plane is 2 * f_Nx * Ny
            launchSimpson3DKernelComplex(d_f_full, d_partial_sum, Nx, Ny, Nz, f_Nx,
                                         tile_size_z, z_start, current_tile_z);

            // Debug mode: sync and check for kernel execution errors
            CUDA_SYNC_CHECK("integrateDeviceComplex kernel");

            // Copy result back to host (implicitly syncs)
            CUDA_CHECK(cudaMemcpy(&h_tile_sum, d_partial_sum, sizeof(double), cudaMemcpyDeviceToHost));

            // Accumulate the result
            total_sum += h_tile_sum;
        }

        // Apply Simpson's rule scaling factor
        return total_sum * hx * hy * hz / 27.0;
    }

    /**
     * @brief Integrate |psi|^2 using device memory (computes on-the-fly from complex array)
     * @param hx Step size in X direction
     * @param hy Step size in Y direction
     * @param hz Step size in Z direction
     * @param d_f_full Pointer to function values (DEVICE memory, complex array, unpadded, uses Nx for indexing)
     * @param Nx Logical X dimension (for weighting and indexing)
     * @param Ny Y dimension
     * @param Nz Z dimension
     */
    double integrateDeviceComplexNorm(double hx, double hy, double hz, const cuDoubleComplex *d_f_full, long Nx, long Ny, long Nz) {
        double total_sum = 0.0;

        // Process the volume in tiles along the Z direction
        for (long z_start = 0; z_start < Nz; z_start += tile_size_z) {
            // Calculate the actual size of this tile (last tile might be smaller)
            long current_tile_z = std::min(tile_size_z, Nz - z_start);

            // Reset the partial sum for this tile
            CUDA_CHECK(cudaMemset(d_partial_sum, 0, sizeof(double)));

            // Launch kernel for this tile with on-the-fly |psi|^2 computation
            // Pointer offset accounts for unpadded array (uses Nx for indexing)
            // With unpadded array, stride per z-plane is Nx * Ny (complex elements)
            launchSimpson3DKernelComplexNorm(d_f_full + z_start * Nx * Ny, d_partial_sum, Nx, Ny, Nz,
                                              tile_size_z, z_start, current_tile_z);

            // Debug mode: sync and check for kernel execution errors
            CUDA_SYNC_CHECK("integrateDeviceComplexNorm kernel");

            // Copy result back to host (implicitly syncs)
            CUDA_CHECK(cudaMemcpy(&h_tile_sum, d_partial_sum, sizeof(double), cudaMemcpyDeviceToHost));

            // Accumulate the result
            total_sum += h_tile_sum;
        }

        // Apply Simpson's rule scaling factor
        return total_sum * hx * hy * hz / 27.0;
    }

    /**
     * @brief Integrate the squared function using device memory (squares on-the-fly)
     * @param hx Step size in X direction
     * @param hy Step size in Y direction
     * @param hz Step size in Z direction
     * @param d_f_full Pointer to function values (DEVICE memory, unpadded, uses Nx for indexing)
     * @param Nx Logical X dimension (for weighting and indexing)
     * @param Ny Y dimension
     * @param Nz Z dimension
     */
    double integrateDeviceNorm(double hx, double hy, double hz, const double *d_f_full, long Nx, long Ny, long Nz) {
        double total_sum = 0.0;

        // Process the volume in tiles along the Z direction
        for (long z_start = 0; z_start < Nz; z_start += tile_size_z) {
            // Calculate the actual size of this tile (last tile might be smaller)
            long current_tile_z = std::min(tile_size_z, Nz - z_start);

            // Reset the partial sum for this tile
            CUDA_CHECK(cudaMemset(d_partial_sum, 0, sizeof(double)));

            // Launch kernel for this tile with on-the-fly squaring
            // Pointer offset accounts for unpadded array (uses Nx for indexing)
            launchSimpson3DKernelNorm(d_f_full + z_start * Nx * Ny, d_partial_sum, Nx, Ny, Nz,
                                      tile_size_z, z_start, current_tile_z);

            // Debug mode: sync and check for kernel execution errors
            CUDA_SYNC_CHECK("integrateDeviceNorm kernel");

            // Copy result back to host (implicitly syncs)
            CUDA_CHECK(cudaMemcpy(&h_tile_sum, d_partial_sum, sizeof(double), cudaMemcpyDeviceToHost));

            // Accumulate the result
            total_sum += h_tile_sum;
        }

        // Apply Simpson's rule scaling factor
        return total_sum * hx * hy * hz / 27.0;
    }

    /**
     * @brief Integrate psi^2 * coordinate^2 using device memory (computes on-the-fly)
     * @param hx Step size in X direction
     * @param hy Step size in Y direction
     * @param hz Step size in Z direction
     * @param d_f_full Pointer to function values (DEVICE memory, unpadded, uses Nx for indexing)
     * @param Nx Logical X dimension (for weighting and indexing)
     * @param Ny Y dimension
     * @param Nz Z dimension
     * @param direction Direction for coordinate calculation (0=x, 1=y, 2=z)
     * @param scale Grid spacing for the chosen direction (dx, dy, or dz)
     */
    double integrateDeviceRMS(double hx, double hy, double hz, const double *d_f_full, long Nx, long Ny, long Nz,
                             int direction, double scale) {
        double total_sum = 0.0;

        // Process the volume in tiles along the Z direction
        for (long z_start = 0; z_start < Nz; z_start += tile_size_z) {
            // Calculate the actual size of this tile (last tile might be smaller)
            long current_tile_z = std::min(tile_size_z, Nz - z_start);

            // Reset the partial sum for this tile
            CUDA_CHECK(cudaMemset(d_partial_sum, 0, sizeof(double)));

            // Launch kernel for this tile with on-the-fly RMS calculation
            // Pointer offset accounts for unpadded array (uses Nx for indexing)
            launchSimpson3DKernelRMS(d_f_full + z_start * Nx * Ny, d_partial_sum, Nx, Ny, Nz,
                                     tile_size_z, z_start, current_tile_z, direction, scale);

            // Debug mode: sync and check for kernel execution errors
            CUDA_SYNC_CHECK("integrateDeviceRMS kernel");

            // Copy result back to host (implicitly syncs)
            CUDA_CHECK(cudaMemcpy(&h_tile_sum, d_partial_sum, sizeof(double), cudaMemcpyDeviceToHost));

            // Accumulate the result
            total_sum += h_tile_sum;
        }

        // Apply Simpson's rule scaling factor
        return total_sum * hx * hy * hz / 27.0;
    }

    /**
     * @brief Integrate all 3 RMS components in a single pass (3x more efficient)
     * @param hx Step size in X direction
     * @param hy Step size in Y direction
     * @param hz Step size in Z direction
     * @param d_f_full Pointer to function values (DEVICE memory, unpadded, uses Nx for indexing)
     * @param Nx Logical X dimension (for weighting and indexing)
     * @param Ny Y dimension
     * @param Nz Z dimension
     * @param scale_x Grid spacing in X direction (dx)
     * @param scale_y Grid spacing in Y direction (dy)
     * @param scale_z Grid spacing in Z direction (dz)
     * @param[out] result_x2 Integrated value of psi^2 * x^2
     * @param[out] result_y2 Integrated value of psi^2 * y^2
     * @param[out] result_z2 Integrated value of psi^2 * z^2
     */
    void integrateDeviceRMSFused(double hx, double hy, double hz, const double *d_f_full, long Nx, long Ny, long Nz,
                                 double scale_x, double scale_y, double scale_z,
                                 double &result_x2, double &result_y2, double &result_z2) {
        double total_sum_x2 = 0.0;
        double total_sum_y2 = 0.0;
        double total_sum_z2 = 0.0;

        // Process the volume in tiles along the Z direction
        for (long z_start = 0; z_start < Nz; z_start += tile_size_z) {
            // Calculate the actual size of this tile (last tile might be smaller)
            long current_tile_z = std::min(tile_size_z, Nz - z_start);

            // Reset all partial sums for this tile
            CUDA_CHECK(cudaMemset(d_partial_sum, 0, sizeof(double)));
            CUDA_CHECK(cudaMemset(d_partial_sum_y2, 0, sizeof(double)));
            CUDA_CHECK(cudaMemset(d_partial_sum_z2, 0, sizeof(double)));

            // Launch fused kernel for this tile - computes all 3 RMS components in one pass
            // Pointer offset accounts for unpadded array (uses Nx for indexing)
            launchSimpson3DKernelRMSFused(d_f_full + z_start * Nx * Ny, 
                                          d_partial_sum, d_partial_sum_y2, d_partial_sum_z2,
                                          Nx, Ny, Nz, tile_size_z, z_start, current_tile_z,
                                          scale_x, scale_y, scale_z);

            // Debug mode: sync and check for kernel execution errors
            CUDA_SYNC_CHECK("integrateDeviceRMSFused kernel");

            // Copy all 3 results back to host (implicitly syncs)
            CUDA_CHECK(cudaMemcpy(&h_tile_sums[0], d_partial_sum, sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&h_tile_sums[1], d_partial_sum_y2, sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&h_tile_sums[2], d_partial_sum_z2, sizeof(double), cudaMemcpyDeviceToHost));

            // Accumulate the results
            total_sum_x2 += h_tile_sums[0];
            total_sum_y2 += h_tile_sums[1];
            total_sum_z2 += h_tile_sums[2];
        }

        // Apply Simpson's rule scaling factor
        double scale = hx * hy * hz / 27.0;
        result_x2 = total_sum_x2 * scale;
        result_y2 = total_sum_y2 * scale;
        result_z2 = total_sum_z2 * scale;
    }

    /**
     * @brief Integrate all 3 RMS components in a single pass from complex array (3x more efficient)
     * @param hx Step size in X direction
     * @param hy Step size in Y direction
     * @param hz Step size in Z direction
     * @param d_f_full Pointer to function values (DEVICE memory, complex array, unpadded, uses Nx for indexing)
     * @param Nx Logical X dimension (for weighting and indexing)
     * @param Ny Y dimension
     * @param Nz Z dimension
     * @param scale_x Grid spacing in X direction (dx)
     * @param scale_y Grid spacing in Y direction (dy)
     * @param scale_z Grid spacing in Z direction (dz)
     * @param[out] result_x2 Integrated value of |psi|^2 * x^2
     * @param[out] result_y2 Integrated value of |psi|^2 * y^2
     * @param[out] result_z2 Integrated value of |psi|^2 * z^2
     */
    void integrateDeviceComplexRMSFused(double hx, double hy, double hz, const cuDoubleComplex *d_f_full, long Nx, long Ny, long Nz,
                                        double scale_x, double scale_y, double scale_z,
                                        double &result_x2, double &result_y2, double &result_z2) {
        double total_sum_x2 = 0.0;
        double total_sum_y2 = 0.0;
        double total_sum_z2 = 0.0;

        // Process the volume in tiles along the Z direction
        for (long z_start = 0; z_start < Nz; z_start += tile_size_z) {
            // Calculate the actual size of this tile (last tile might be smaller)
            long current_tile_z = std::min(tile_size_z, Nz - z_start);

            // Reset all partial sums for this tile
            CUDA_CHECK(cudaMemset(d_partial_sum, 0, sizeof(double)));
            CUDA_CHECK(cudaMemset(d_partial_sum_y2, 0, sizeof(double)));
            CUDA_CHECK(cudaMemset(d_partial_sum_z2, 0, sizeof(double)));

            // Launch fused complex kernel for this tile - computes all 3 RMS components in one pass
            // Pointer offset accounts for unpadded array (uses Nx for indexing)
            launchSimpson3DKernelComplexRMSFused(d_f_full + z_start * Nx * Ny, 
                                                 d_partial_sum, d_partial_sum_y2, d_partial_sum_z2,
                                                 Nx, Ny, Nz, tile_size_z, z_start, current_tile_z,
                                                 scale_x, scale_y, scale_z);

            // Debug mode: sync and check for kernel execution errors
            CUDA_SYNC_CHECK("integrateDeviceComplexRMSFused kernel");

            // Copy all 3 results back to host (implicitly syncs)
            CUDA_CHECK(cudaMemcpy(&h_tile_sums[0], d_partial_sum, sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&h_tile_sums[1], d_partial_sum_y2, sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&h_tile_sums[2], d_partial_sum_z2, sizeof(double), cudaMemcpyDeviceToHost));

            // Accumulate the results
            total_sum_x2 += h_tile_sums[0];
            total_sum_y2 += h_tile_sums[1];
            total_sum_z2 += h_tile_sums[2];
        }

        // Apply Simpson's rule scaling factor
        double scale = hx * hy * hz / 27.0;
        result_x2 = total_sum_x2 * scale;
        result_y2 = total_sum_y2 * scale;
        result_z2 = total_sum_z2 * scale;
    }

    /**
     * @brief Set the tile size
     * @param new_tile_size New tile size
     */
    void setTileSize(long new_tile_size) {
        if (new_tile_size <= 0) {
            throw std::invalid_argument("Tile size must be positive");
        }

        tile_size_z = new_tile_size;
    }
};

/**
 * @brief Constructor
 * @param Nx Grid size in X direction
 * @param Ny Grid size in Y direction
 * @param tile_z Number of z-slices to process per tile (default: 32)
 */
Simpson3DTiledIntegrator::Simpson3DTiledIntegrator(long Nx, long Ny, long tile_z) {
    pImpl = new Simpson3DTiledIntegratorImpl(Nx, Ny, tile_z);
}

/**
 * @brief Destructor
 */
Simpson3DTiledIntegrator::~Simpson3DTiledIntegrator() { delete pImpl; }

/**
 * @brief Integrate the function using device memory
 * @param hx Step size in X direction
 * @param hy Step size in Y direction
 * @param hz Step size in Z direction
 * @param d_f Pointer to function values (DEVICE memory, may be padded)
 * @param Nx Logical number of points in X direction (for weighting)
 * @param Ny Number of points in Y direction
 * @param Nz Number of points in Z direction
 * @param f_Nx Actual X dimension of d_f (Nx for unpadded, Nx+2 for padded, defaults to Nx)
 * @return Integrated value
 */
double Simpson3DTiledIntegrator::integrateDevice(double hx, double hy, double hz, double *d_f,
                                                 long Nx, long Ny, long Nz, long f_Nx) {
    // Default to Nx if f_Nx not specified (backward compatibility)
    if (f_Nx == 0) f_Nx = Nx;
    return pImpl->integrateDevice(hx, hy, hz, d_f, Nx, Ny, Nz, f_Nx);
}

/**
 * @brief Integrate the function using device memory (complex array cast to double)
 * @param hx Step size in X direction
 * @param hy Step size in Y direction
 * @param hz Step size in Z direction
 * @param d_f Pointer to function values (DEVICE memory) - complex array cast to double
 * @param Nx Number of points in X direction
 * @param Ny Number of points in Y direction
 * @param Nz Number of points in Z direction
 * @return Integrated value
 */
double Simpson3DTiledIntegrator::integrateDeviceComplex(double hx, double hy, double hz, double *d_f,
                                                         long Nx, long Ny, long Nz, long f_Nx) {
    // Default to Nx if f_Nx not specified (backward compatibility)
    if (f_Nx == 0) f_Nx = Nx;
    return pImpl->integrateDeviceComplex(hx, hy, hz, d_f, Nx, Ny, Nz, f_Nx);
}

/**
 * @brief Integrate |psi|^2 using device memory (computes on-the-fly from complex array)
 * @param hx Step size in X direction
 * @param hy Step size in Y direction
 * @param hz Step size in Z direction
 * @param d_f Pointer to function values (DEVICE memory, complex array, unpadded, uses Nx for indexing)
 * @param Nx Number of points in X direction (also used for indexing)
 * @param Ny Number of points in Y direction
 * @param Nz Number of points in Z direction
 * @return Integrated value of |psi|^2
 */
double Simpson3DTiledIntegrator::integrateDeviceComplexNorm(double hx, double hy, double hz, const cuDoubleComplex *d_f,
                                                             long Nx, long Ny, long Nz) {
    return pImpl->integrateDeviceComplexNorm(hx, hy, hz, d_f, Nx, Ny, Nz);
}

/**
 * @brief Integrate the squared function using device memory (squares input on-the-fly)
 * @param hx Step size in X direction
 * @param hy Step size in Y direction
 * @param hz Step size in Z direction
 * @param d_f Pointer to function values (DEVICE memory, unpadded, uses Nx for indexing)
 * @param Nx Number of points in X direction (also used for indexing)
 * @param Ny Number of points in Y direction
 * @param Nz Number of points in Z direction
 * @return Integrated value of f^2
 */
double Simpson3DTiledIntegrator::integrateDeviceNorm(double hx, double hy, double hz, const double *d_f,
                                                      long Nx, long Ny, long Nz) {
    return pImpl->integrateDeviceNorm(hx, hy, hz, d_f, Nx, Ny, Nz);
}

/**
 * @brief Integrate psi^2 * coordinate^2 using device memory (computes on-the-fly)
 * @param hx Step size in X direction
 * @param hy Step size in Y direction
 * @param hz Step size in Z direction
 * @param d_f Pointer to function values (DEVICE memory, unpadded, uses Nx for indexing)
 * @param Nx Number of points in X direction (also used for indexing)
 * @param Ny Number of points in Y direction
 * @param Nz Number of points in Z direction
 * @param direction Direction for coordinate calculation (0=x, 1=y, 2=z)
 * @param scale Grid spacing for the chosen direction (dx, dy, or dz)
 * @return Integrated value of psi^2 * coordinate^2
 */
double Simpson3DTiledIntegrator::integrateDeviceRMS(double hx, double hy, double hz, const double *d_f,
                                                    long Nx, long Ny, long Nz, int direction, double scale) {
    return pImpl->integrateDeviceRMS(hx, hy, hz, d_f, Nx, Ny, Nz, direction, scale);
}

/**
 * @brief Integrate all 3 RMS components in a single pass (3x more efficient)
 * @param hx Step size in X direction
 * @param hy Step size in Y direction
 * @param hz Step size in Z direction
 * @param d_f Pointer to function values (DEVICE memory, unpadded, uses Nx for indexing)
 * @param Nx Number of points in X direction (also used for indexing)
 * @param Ny Number of points in Y direction
 * @param Nz Number of points in Z direction
 * @param scale_x Grid spacing in X direction (dx)
 * @param scale_y Grid spacing in Y direction (dy)
 * @param scale_z Grid spacing in Z direction (dz)
 * @param[out] result_x2 Integrated value of psi^2 * x^2
 * @param[out] result_y2 Integrated value of psi^2 * y^2
 * @param[out] result_z2 Integrated value of psi^2 * z^2
 */
void Simpson3DTiledIntegrator::integrateDeviceRMSFused(double hx, double hy, double hz, const double *d_f,
                                                       long Nx, long Ny, long Nz,
                                                       double scale_x, double scale_y, double scale_z,
                                                       double &result_x2, double &result_y2, double &result_z2) {
    pImpl->integrateDeviceRMSFused(hx, hy, hz, d_f, Nx, Ny, Nz, scale_x, scale_y, scale_z,
                                    result_x2, result_y2, result_z2);
}

/**
 * @brief Integrate all 3 RMS components in a single pass from complex array (3x more efficient)
 * @param hx Step size in X direction
 * @param hy Step size in Y direction
 * @param hz Step size in Z direction
 * @param d_f Pointer to function values (DEVICE memory, complex array, unpadded, uses Nx for indexing)
 * @param Nx Number of points in X direction (also used for indexing)
 * @param Ny Number of points in Y direction
 * @param Nz Number of points in Z direction
 * @param scale_x Grid spacing in X direction (dx)
 * @param scale_y Grid spacing in Y direction (dy)
 * @param scale_z Grid spacing in Z direction (dz)
 * @param[out] result_x2 Integrated value of |psi|^2 * x^2
 * @param[out] result_y2 Integrated value of |psi|^2 * y^2
 * @param[out] result_z2 Integrated value of |psi|^2 * z^2
 */
void Simpson3DTiledIntegrator::integrateDeviceComplexRMSFused(double hx, double hy, double hz, const cuDoubleComplex *d_f,
                                                              long Nx, long Ny, long Nz,
                                                              double scale_x, double scale_y, double scale_z,
                                                              double &result_x2, double &result_y2, double &result_z2) {
    pImpl->integrateDeviceComplexRMSFused(hx, hy, hz, d_f, Nx, Ny, Nz, scale_x, scale_y, scale_z,
                                           result_x2, result_y2, result_z2);
}

/**
 * @brief Set the tile size
 * @param new_tile_size New tile size
 */
void Simpson3DTiledIntegrator::setTileSize(long new_tile_size) {
    pImpl->setTileSize(new_tile_size);
}
