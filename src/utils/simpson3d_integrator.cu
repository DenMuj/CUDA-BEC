// simpson3d_integrator.cu - Modified version with device memory support
#include "simpson3d_integrator.hpp"
#include "simpson3d_kernel.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>

// Implementation class (hidden from public interface)
class Simpson3DTiledIntegratorImpl {
public:
    double *d_f;           // Device memory for current tile
    double *d_partial_sum; // Device memory for accumulating results
    double *h_tile_sum_pinned; // Pinned host memory for async transfers
    cudaStream_t stream;   // CUDA stream for async operations
    long tile_size_z;      // Number of z-slices per tile
    long max_tile_points;  // Maximum points in a tile
    long cached_Nx;        // Cached grid dimensions
    long cached_Ny;
    
    Simpson3DTiledIntegratorImpl(long Nx, long Ny, long tile_z) 
        : tile_size_z(tile_z), cached_Nx(Nx), cached_Ny(Ny) {
        
        // Allocate memory for one tile
        max_tile_points = Nx * Ny * tile_size_z;
        cudaMalloc(&d_f, max_tile_points * sizeof(double));
        cudaMalloc(&d_partial_sum, sizeof(double));
        
        // Allocate pinned host memory for async transfers
        cudaHostAlloc(&h_tile_sum_pinned, sizeof(double), cudaHostAllocDefault);
        
        // Create CUDA stream for async operations
        cudaStreamCreate(&stream);
        
        // Check for allocation errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA allocation error: " << cudaGetErrorString(err) << std::endl;
            throw std::runtime_error("Failed to allocate GPU memory");
        }
    }
    
    ~Simpson3DTiledIntegratorImpl() {
        cudaFree(d_f);
        cudaFree(d_partial_sum);
        cudaFreeHost(h_tile_sum_pinned);
        cudaStreamDestroy(stream);
    }
    
    // Original function - copies from host memory
    double integrate(double hx, double hy, double hz, double *h_f, 
                    long Nx, long Ny, long Nz) {
        double total_sum = 0.0;
        
        // Process the volume in tiles along the Z direction
        for (long z_start = 0; z_start < Nz; z_start += tile_size_z) {
            // Calculate the actual size of this tile (last tile might be smaller)
            long current_tile_z = std::min(tile_size_z, Nz - z_start);
            long tile_points = Nx * Ny * current_tile_z;
            
            // Copy this tile's data from host to device
            cudaMemcpy(d_f, h_f + z_start * Nx * Ny, 
                      tile_points * sizeof(double), cudaMemcpyHostToDevice);
            
            // Reset the partial sum for this tile
            cudaMemset(d_partial_sum, 0, sizeof(double));
            
            // Launch kernel for this tile
            launchSimpson3DKernel(d_f, d_partial_sum, Nx, Ny, Nz, 
                                 tile_size_z, z_start, current_tile_z);
            
            // Check for kernel launch errors
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                std::cerr << "Kernel launch error: " << cudaGetErrorString(err) << std::endl;
                return 0.0;
            }
            
            // Check for kernel execution errors
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                std::cerr << "Kernel execution error: " << cudaGetErrorString(err) << std::endl;
                return 0.0;
            }
            
            // Use async transfer with pinned memory for better performance
            cudaMemcpyAsync(h_tile_sum_pinned, d_partial_sum, sizeof(double), 
                           cudaMemcpyDeviceToHost, stream);
            
            // Wait for the async transfer to complete
            cudaStreamSynchronize(stream);
            
            // Accumulate the result
            total_sum += *h_tile_sum_pinned;
        }
        
        // Apply Simpson's rule scaling factor
        return total_sum * hx * hy * hz / 27.0;
    }
    
    // New function - works with device memory directly
    double integrateDevice(double hx, double hy, double hz, double *d_f_full, 
                          long Nx, long Ny, long Nz) {
        double total_sum = 0.0;
        
        // Process the volume in tiles along the Z direction
        for (long z_start = 0; z_start < Nz; z_start += tile_size_z) {
            // Calculate the actual size of this tile (last tile might be smaller)
            long current_tile_z = std::min(tile_size_z, Nz - z_start);
            long tile_points = Nx * Ny * current_tile_z;
            
            // Copy this tile's data from device to device (GPU to GPU copy)
            cudaMemcpy(d_f, d_f_full + z_start * Nx * Ny, 
                      tile_points * sizeof(double), cudaMemcpyDeviceToDevice);
            
            // Reset the partial sum for this tile
            cudaMemset(d_partial_sum, 0, sizeof(double));
            
            // Launch kernel for this tile
            launchSimpson3DKernel(d_f, d_partial_sum, Nx, Ny, Nz, 
                                 tile_size_z, z_start, current_tile_z);
            
            // Check for kernel launch errors
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                std::cerr << "Kernel launch error: " << cudaGetErrorString(err) << std::endl;
                return 0.0;
            }
            
            // Check for kernel execution errors
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                std::cerr << "Kernel execution error: " << cudaGetErrorString(err) << std::endl;
                return 0.0;
            }
            
            // Use async transfer with pinned memory for better performance
            cudaMemcpyAsync(h_tile_sum_pinned, d_partial_sum, sizeof(double), 
                           cudaMemcpyDeviceToHost, stream);
            
            // Wait for the async transfer to complete
            cudaStreamSynchronize(stream);
            
            // Accumulate the result
            total_sum += *h_tile_sum_pinned;
        }
        
        // Apply Simpson's rule scaling factor
        return total_sum * hx * hy * hz / 27.0;
    }
    
    void setTileSize(long new_tile_size) {
        if (new_tile_size <= 0) {
            throw std::invalid_argument("Tile size must be positive");
        }
        
        tile_size_z = new_tile_size;
        // Reallocate if new size is larger
        long new_max_points = cached_Nx * cached_Ny * tile_size_z;
        if (new_max_points > max_tile_points) {
            cudaFree(d_f);
            max_tile_points = new_max_points;
            cudaMalloc(&d_f, max_tile_points * sizeof(double));
            
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                throw std::runtime_error("Failed to reallocate GPU memory");
            }
        }
    }
};

// Public interface implementation
Simpson3DTiledIntegrator::Simpson3DTiledIntegrator(long Nx, long Ny, long tile_z) {
    pImpl = new Simpson3DTiledIntegratorImpl(Nx, Ny, tile_z);
}

Simpson3DTiledIntegrator::~Simpson3DTiledIntegrator() {
    delete pImpl;
}

// Host memory version
double Simpson3DTiledIntegrator::integrate(double hx, double hy, double hz, 
                                          double* h_f, long Nx, long Ny, long Nz) {
    return pImpl->integrate(hx, hy, hz, h_f, Nx, Ny, Nz);
}

// Device memory version
double Simpson3DTiledIntegrator::integrateDevice(double hx, double hy, double hz,
                                                double* d_f, long Nx, long Ny, long Nz) {
    return pImpl->integrateDevice(hx, hy, hz, d_f, Nx, Ny, Nz);
}

void Simpson3DTiledIntegrator::setTileSize(long new_tile_size) {
    pImpl->setTileSize(new_tile_size);
}

size_t Simpson3DTiledIntegrator::getMemoryUsage(long Nx, long Ny) const {
    return (Nx * Ny * pImpl->tile_size_z + 1) * sizeof(double);
}

long Simpson3DTiledIntegrator::getTileSize() const {
    return pImpl->tile_size_z;
}