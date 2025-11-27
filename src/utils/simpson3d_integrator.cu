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
#include <iostream>

/**
 * @brief Implementation class (hidden from public interface)
 */
class Simpson3DTiledIntegratorImpl {
  public:
    double *d_partial_sum; // Device memory for accumulating results
    double h_tile_sum;     // Host memory for tile sum
    long tile_size_z;      // Number of z-slices per tile
    long cached_Nx;        // Cached grid dimensions
    long cached_Ny;

    /**
     * @brief Constructor
     * @param Nx Grid size in X direction
     * @param Ny Grid size in Y direction
     * @param tile_z Number of z-slices to process per tile (default: 32)
     */
    Simpson3DTiledIntegratorImpl(long Nx, long Ny, long tile_z)
        : tile_size_z(tile_z), cached_Nx(Nx), cached_Ny(Ny), h_tile_sum(0.0) {

        // Allocate memory for partial sum
        CUDA_CHECK(cudaMalloc(&d_partial_sum, sizeof(double)));
    }

    /**
     * @brief Destructor
     */
    ~Simpson3DTiledIntegratorImpl() {
        cudaFree(d_partial_sum);
    }

    /**
     * @brief Integrate the function using device memory
     * @param hx Step size in X direction
     * @param hy Step size in Y direction
     * @param hz Step size in Z direction
     * @param d_f_full Pointer to function values (DEVICE memory)
     */
    double integrateDevice(double hx, double hy, double hz, double *d_f_full, long Nx, long Ny,
                           long Nz) {
        double total_sum = 0.0;

        // Process the volume in tiles along the Z direction
        for (long z_start = 0; z_start < Nz; z_start += tile_size_z) {
            // Calculate the actual size of this tile (last tile might be smaller)
            long current_tile_z = std::min(tile_size_z, Nz - z_start);

            // Reset the partial sum for this tile
            CUDA_CHECK(cudaMemset(d_partial_sum, 0, sizeof(double)));

            // Launch kernel for this tile (pass pointer directly with offset - no D2D copy needed)
            launchSimpson3DKernel(d_f_full + z_start * Nx * Ny, d_partial_sum, Nx, Ny, Nz,
                                  tile_size_z, z_start, current_tile_z);

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
 * @param d_f Pointer to function values (DEVICE memory)
 * @param Nx Number of points in X direction
 * @param Ny Number of points in Y direction
 * @param Nz Number of points in Z direction
 * @return Integrated value
 */
double Simpson3DTiledIntegrator::integrateDevice(double hx, double hy, double hz, double *d_f,
                                                 long Nx, long Ny, long Nz) {
    return pImpl->integrateDevice(hx, hy, hz, d_f, Nx, Ny, Nz);
}

/**
 * @brief Set the tile size
 * @param new_tile_size New tile size
 */
void Simpson3DTiledIntegrator::setTileSize(long new_tile_size) {
    pImpl->setTileSize(new_tile_size);
}
