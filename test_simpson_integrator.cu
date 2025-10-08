/**
 * @file test_simpson_integrator.cu
 * @brief Test program for Simpson3DTiledIntegrator using f(x,y,z) = x² + y² + z²
 * 
 * This program demonstrates the usage of Simpson3DTiledIntegrator to compute
 * the 3D integral of f(x,y,z) = x² + y² + z² over a cuboid domain.
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include "src/utils/simpson3d_integrator.hpp"

// CUDA kernel to fill the 3D array with f(x,y,z) = x² + y² + z²
__global__ void fillFunctionKernel(double* d_f, 
                                    double x_min, double y_min, double z_min,
                                    double dx, double dy, double dz,
                                    long Nx, long Ny, long Nz) {
    // Calculate global indices
    long ix = blockIdx.x * blockDim.x + threadIdx.x;
    long iy = blockIdx.y * blockDim.y + threadIdx.y;
    long iz = blockIdx.z * blockDim.z + threadIdx.z;
    
    // Boundary check
    if (ix < Nx && iy < Ny && iz < Nz) {
        // Calculate physical coordinates
        double x = x_min + ix * dx;
        double y = y_min + iy * dy;
        double z = z_min + iz * dz;
        
        // Calculate function value: f(x,y,z) = x² + y² + z²
        double f_val = x*x*x* y*y + z*z;
        
        // Store in column-major order (z changes fastest)
        long idx = ix * Ny * Nz + iy * Nz + iz;
        d_f[idx] = f_val;
    }
}

// Analytical solution for the integral of x² + y² + z²
// over [x_min, x_max] × [y_min, y_max] × [z_min, z_max]
double analyticalIntegral(double x_min, double x_max,
                         double y_min, double y_max,
                         double z_min, double z_max) {
    // ∫∫∫ (x² + y² + z²) dx dy dz
    // = ∫∫ [x³/3]_{x_min}^{x_max} + x(y² + z²)|_{x_min}^{x_max} dy dz
    
    double Lx = x_max - x_min;
    double Ly = y_max - y_min;
    double Lz = z_max - z_min;
    
    // Integral of x² over [x_min, x_max]
    double int_x2 = (x_max*x_max*x_max - x_min*x_min*x_min) / 3.0;
    
    // Integral of y² over [y_min, y_max]
    double int_y2 = (y_max*y_max*y_max - y_min*y_min*y_min) / 3.0;
    
    // Integral of z² over [z_min, z_max]
    double int_z2 = (z_max*z_max*z_max - z_min*z_min*z_min) / 3.0;
    
    // Total integral
    double integral = int_x2 * Ly * Lz + Lx * int_y2 * Lz + Lx * Ly * int_z2;
    
    return integral;
}

int main(int argc, char* argv[]) {
    std::cout << "========================================" << std::endl;
    std::cout << "Simpson 3D Integration Test" << std::endl;
    std::cout << "Function: f(x,y,z) = x² + y² + z²" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Define integration domain
    double x_min = 0.0, x_max = 1.0;
    double y_min = 0.0, y_max = 2.0;
    double z_min = 0.0, z_max = 3.0;
    
    // Grid parameters
    long Nx = 136;  // Must be odd for Simpson's rule
    long Ny = 136;
    long Nz = 136;
    
    // Calculate step sizes
    double dx = (x_max - x_min) / (Nx - 1);
    double dy = (y_max - y_min) / (Ny - 1);
    double dz = (z_max - z_min) / (Nz - 1);
    
    std::cout << "\nDomain:" << std::endl;
    std::cout << "  x ∈ [" << x_min << ", " << x_max << "]" << std::endl;
    std::cout << "  y ∈ [" << y_min << ", " << y_max << "]" << std::endl;
    std::cout << "  z ∈ [" << z_min << ", " << z_max << "]" << std::endl;
    
    std::cout << "\nGrid parameters:" << std::endl;
    std::cout << "  Nx = " << Nx << ", dx = " << dx << std::endl;
    std::cout << "  Ny = " << Ny << ", dy = " << dy << std::endl;
    std::cout << "  Nz = " << Nz << ", dz = " << dz << std::endl;
    std::cout << "  Total points: " << Nx * Ny * Nz << std::endl;
    
    // Allocate device memory for the function values
    double* d_f;
    long total_points = Nx * Ny * Nz;
    size_t size = total_points * sizeof(double);
    
    cudaError_t err = cudaMalloc(&d_f, size);
    if (err != cudaSuccess) {
        std::cerr << "Error allocating device memory: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    
    // Configure kernel launch parameters
    dim3 blockSize(8, 8, 8);
    dim3 gridSize((Nx + blockSize.x - 1) / blockSize.x,
                  (Ny + blockSize.y - 1) / blockSize.y,
                  (Nz + blockSize.z - 1) / blockSize.z);
    
    std::cout << "\nFilling array with function values..." << std::endl;
    fillFunctionKernel<<<gridSize, blockSize>>>(d_f, x_min, y_min, z_min,
                                                 dx, dy, dz, Nx, Ny, Nz);
    
    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch error: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_f);
        return 1;
    }
    
    // Synchronize to ensure kernel completion
    cudaDeviceSynchronize();
    
    // Create Simpson3DTiledIntegrator
    long tile_size_z = Nz;  // Process entire z-dimension at once for this test
    std::cout << "Creating Simpson3DTiledIntegrator with tile_size_z = " << tile_size_z << std::endl;
    
    Simpson3DTiledIntegrator integrator(Nx, Ny, tile_size_z);
    
    // Perform integration using device memory
    std::cout << "\nPerforming 3D integration..." << std::endl;
    double numerical_result = integrator.integrateDevice(dx, dy, dz, d_f, Nx, Ny, Nz);
    
    // Calculate analytical result
    double analytical_result = analyticalIntegral(x_min, x_max, y_min, y_max, z_min, z_max);
    
    // Calculate relative error
    double relative_error = std::abs(numerical_result - analytical_result) / std::abs(analytical_result);
    
    // Display results
    std::cout << "\n========================================" << std::endl;
    std::cout << "RESULTS" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::fixed << std::setprecision(12);
    std::cout << "Numerical result:  " << numerical_result << std::endl;
    std::cout << "Analytical result: " << analytical_result << std::endl;
    std::cout << "Absolute error:    " << std::abs(numerical_result - analytical_result) << std::endl;
    std::cout << std::scientific << std::setprecision(4);
    std::cout << "Relative error:    " << relative_error << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Test with different tile sizes
    std::cout << "\n\nTesting with different tile sizes:" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    
    std::vector<long> tile_sizes = {16, 32, 64, Nz};
    for (long tile_z : tile_sizes) {
        if (tile_z <= Nz) {
            Simpson3DTiledIntegrator test_integrator(Nx, Ny, tile_z);
            double result = test_integrator.integrateDevice(dx, dy, dz, d_f, Nx, Ny, Nz);
            double error = std::abs(result - analytical_result) / std::abs(analytical_result);
            std::cout << "Tile size " << std::setw(3) << tile_z << ": " 
                      << std::fixed << std::setprecision(12) << result 
                      << " (rel. error: " << std::scientific << std::setprecision(2) 
                      << error << ")" << std::endl;
        }
    }
    
    // Clean up
    cudaFree(d_f);
    
    std::cout << "\nTest completed successfully!" << std::endl;
    
    return 0;
}

