/**
 * @file diffint.cu
 * @brief Implementation of richardson extrapolation formula for calculation of space derivatives.
 * 
 * This file contains the implementation of the richardson extrapolation formula for calculation of space derivatives.
 */

#include "diffint.cuh"
#include "cuda_error_check.cuh"
#include <cuComplex.h>

/**
 * @brief Richardson extrapolation formula for calculation of space derivatives.
 * @param hx Space step
 * @param hy Space step
 * @param hz Space step
 * @param f Function values
 * @param f_res Array with the first derivatives of the function
 */
 void diff(double hx, double hy, double hz, double* f, double* __restrict__ f_res, long nx, long ny, long nz, int par, long f_res_nx) {
    // Get raw pointers and size from MultiArray
   double* f_ptr = f;
   double* f_res_ptr = f_res;
   
   // Precompute kinetic energy factor
   const double kin_energy_factor = 1.0 / (2.0 * (3.0 - par));
   
   dim3 blockSize(8, 8, 8);
    dim3 gridSize((nx + blockSize.x - 1) / blockSize.x,
                  (ny + blockSize.y - 1) / blockSize.y,
                  (nz + blockSize.z - 1) / blockSize.z);
    
    // Kernel still uses raw pointers
   diff_kernel<<<gridSize, blockSize>>>(hx, hy, hz, f_ptr, f_res_ptr, nx, ny, nz, kin_energy_factor, f_res_nx);
   CUDA_CHECK_KERNEL("diff_kernel");
}

/**
 * @brief Kernel function for the Richardson extrapolation formula for calculation of space derivatives.
 * @param hx Space step
 * @param hy Space step
 * @param hz Space step
 * @param psi Function values
 * @param f_res Array with the first derivatives of the function
 */
__global__ void diff_kernel(
   double hx, double hy, double hz,
   double* __restrict__ psi,
   double* __restrict__ f_res,
   long nx, long ny, long nz, double kin_energy_factor, long f_res_nx) {
   
   int i = threadIdx.x + blockDim.x * blockIdx.x;
   int j = threadIdx.y + blockDim.y * blockIdx.y;
   int k = threadIdx.z + blockDim.z * blockIdx.z;
   
   if (i >= nx || j >= ny || k >= nz) return;
   
   // Index for writing to f_res (may be padded, uses f_res_nx)
   // Note: Reading from psi uses nx (unpadded) - indices calculated inline in derivative calculations
   int f_res_idx = k * f_res_nx * ny + j * f_res_nx + i;
   
   // Precompute reciprocals
   const double inv_12hx = 1.0 / (12.0 * hx);
   const double inv_2hx = 1.0 / (2.0 * hx);
   const double inv_12hy = 1.0 / (12.0 * hy);
   const double inv_2hy = 1.0 / (2.0 * hy);
   const double inv_12hz = 1.0 / (12.0 * hz);
   const double inv_2hz = 1.0 / (2.0 * hz);
   
   double dpsidx = 0.0, dpsidy = 0.0, dpsidz = 0.0;
   
   // X-derivative (∂Ψ/∂x) - reading from psi (unpadded, uses nx)
   if (i > 1 && i < nx - 2) {
       // 4th order central difference
       int idx_m2 = k * nx * ny + j * nx + (i - 2);
       int idx_m1 = k * nx * ny + j * nx + (i - 1);
       int idx_p1 = k * nx * ny + j * nx + (i + 1);
       int idx_p2 = k * nx * ny + j * nx + (i + 2);
       dpsidx = (__ldg(&psi[idx_m2]) - 8.0 * __ldg(&psi[idx_m1]) + 8.0 * __ldg(&psi[idx_p1]) - __ldg(&psi[idx_p2])) * inv_12hx;
   } else if (i == 1) {
       // 2nd order forward difference
       int idx_0 = k * nx * ny + j * nx + 0;
       int idx_2 = k * nx * ny + j * nx + 2;
       dpsidx = (__ldg(&psi[idx_2]) - __ldg(&psi[idx_0])) * inv_2hx;
   } else if (i == nx - 2) {
       // 2nd order backward difference
       int idx_m2 = k * nx * ny + j * nx + (nx - 3);
       int idx_0 = k * nx * ny + j * nx + (nx - 1);
       dpsidx = (__ldg(&psi[idx_0]) - __ldg(&psi[idx_m2])) * inv_2hx;
   }
   // Boundary points (i=0, i=nx-1) have dpsidx = 0.0
   
   // Y-derivative (∂Ψ/∂y) - reading from psi (unpadded, uses nx)
   if (j > 1 && j < ny - 2) {
       // 4th order central difference
       int idx_m2 = k * nx * ny + (j - 2) * nx + i;
       int idx_m1 = k * nx * ny + (j - 1) * nx + i;
       int idx_p1 = k * nx * ny + (j + 1) * nx + i;
       int idx_p2 = k * nx * ny + (j + 2) * nx + i;
       dpsidy = (__ldg(&psi[idx_m2]) - 8.0 * __ldg(&psi[idx_m1]) + 8.0 * __ldg(&psi[idx_p1]) - __ldg(&psi[idx_p2])) * inv_12hy;
   } else if (j == 1) {
       int idx_0 = k * nx * ny + 0 * nx + i;
       int idx_2 = k * nx * ny + 2 * nx + i;
       dpsidy = (__ldg(&psi[idx_2]) - __ldg(&psi[idx_0])) * inv_2hy;
   } else if (j == ny - 2) {
       int idx_m2 = k * nx * ny + (ny - 3) * nx + i;
       int idx_0 = k * nx * ny + (ny - 1) * nx + i;
       dpsidy = (__ldg(&psi[idx_0]) - __ldg(&psi[idx_m2])) * inv_2hy;
   }
   
   // Z-derivative (∂Ψ/∂z) - reading from psi (unpadded, uses nx)
   if (k > 1 && k < nz - 2) {
       // 4th order central difference
       int idx_m2 = (k - 2) * nx * ny + j * nx + i;
       int idx_m1 = (k - 1) * nx * ny + j * nx + i;
       int idx_p1 = (k + 1) * nx * ny + j * nx + i;
       int idx_p2 = (k + 2) * nx * ny + j * nx + i;
       dpsidz = (__ldg(&psi[idx_m2]) - 8.0 * __ldg(&psi[idx_m1]) + 8.0 * __ldg(&psi[idx_p1]) - __ldg(&psi[idx_p2])) * inv_12hz;
   } else if (k == 1) {
       int idx_0 = 0 * nx * ny + j * nx + i;
       int idx_2 = 2 * nx * ny + j * nx + i;
       dpsidz = (__ldg(&psi[idx_2]) - __ldg(&psi[idx_0])) * inv_2hz;
   } else if (k == nz - 2) {
       int idx_m2 = (nz - 3) * nx * ny + j * nx + i;
       int idx_0 = (nz - 1) * nx * ny + j * nx + i;
       dpsidz = (__ldg(&psi[idx_0]) - __ldg(&psi[idx_m2])) * inv_2hz;
   }
   
   // Compute kinetic energy density: |∇Ψ|²
   // Writing to f_res (may be padded, uses f_res_nx)
   f_res[f_res_idx] = (dpsidx * dpsidx + dpsidy * dpsidy + dpsidz * dpsidz) * kin_energy_factor;
}

/**
 * @brief Kernel function for the Richardson extrapolation formula for calculation of space derivatives for complex functions.
 * @param hx Space step
 * @param hy Space step
 * @param hz Space step
 * @param f Function values for complex functions
 * @param f_res Array with the first derivatives of the function
 */
void diff_complex(double hx, double hy, double hz, cuDoubleComplex* f, double* __restrict__ f_res, long nx, long ny, long nz, int par) {
    // Get raw pointers and size from MultiArray
   cuDoubleComplex* f_ptr = f;
   double* f_res_ptr = f_res;
   
   // Precompute kinetic energy factor
   const double kin_energy_factor = 1.0 / (2.0 * (3.0 - par));
    
   dim3 blockSize(8, 8, 8);
    dim3 gridSize((nx + blockSize.x - 1) / blockSize.x,
                  (ny + blockSize.y - 1) / blockSize.y,
                  (nz + blockSize.z - 1) / blockSize.z);
    
    // Kernel still uses raw pointers
   diff_kernel_complex<<<gridSize, blockSize>>>(hx, hy, hz, f_ptr, f_res_ptr, nx, ny, nz, kin_energy_factor);
   CUDA_CHECK_KERNEL("diff_kernel_complex");
}

/**
 * @brief Kernel function for the Richardson extrapolation formula for calculation of space derivatives for complex functions.
 * @param hx Space step
 * @param hy Space step
 * @param hz Space step
 * @param psi Function values for complex functions
 * @param f_res Array with the first derivatives of the function
 */
__global__ void diff_kernel_complex(
    double hx, double hy, double hz,
    cuDoubleComplex* __restrict__ psi,
    double* __restrict__ f_res,
    long nx, long ny, long nz, double kin_energy_factor) {
    
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;
    int k = threadIdx.z + blockDim.z * blockIdx.z;
    
    if (i >= nx || j >= ny || k >= nz) return;
    
    int idx = k * nx * ny + j * nx + i;
    
    // Precompute reciprocals
    const double inv_12hx = 1.0 / (12.0 * hx);
    const double inv_2hx = 1.0 / (2.0 * hx);
    const double inv_12hy = 1.0 / (12.0 * hy);
    const double inv_2hy = 1.0 / (2.0 * hy);
    const double inv_12hz = 1.0 / (12.0 * hz);
    const double inv_2hz = 1.0 / (2.0 * hz);
    
    // Complex derivatives
    cuDoubleComplex dpsidx = make_cuDoubleComplex(0.0, 0.0);
    cuDoubleComplex dpsidy = make_cuDoubleComplex(0.0, 0.0);
    cuDoubleComplex dpsidz = make_cuDoubleComplex(0.0, 0.0);
    
    // X-derivative (∂Ψ/∂x) - operating on complex values
    if (i > 1 && i < nx - 2) {
        int idx_m2 = k * nx * ny + j * nx + (i - 2);
        int idx_m1 = k * nx * ny + j * nx + (i - 1);
        int idx_p1 = k * nx * ny + j * nx + (i + 1);
        int idx_p2 = k * nx * ny + j * nx + (i + 2);
        
        // Complex arithmetic: (f[i-2] - 8*f[i-1] + 8*f[i+1] - f[i+2]) / (12*h)
        cuDoubleComplex term1 = __ldg(&psi[idx_m2]);
        cuDoubleComplex term2 = cuCmul(make_cuDoubleComplex(8.0, 0.0), __ldg(&psi[idx_m1]));
        cuDoubleComplex term3 = cuCmul(make_cuDoubleComplex(8.0, 0.0), __ldg(&psi[idx_p1]));
        cuDoubleComplex term4 = __ldg(&psi[idx_p2]);
        
        cuDoubleComplex numerator = cuCsub(cuCadd(cuCsub(term1, term2), term3), term4);
        dpsidx = cuCmul(numerator, make_cuDoubleComplex(inv_12hx, 0.0));
        
    } else if (i == 1) {
        int idx_0 = k * nx * ny + j * nx + 0;
        int idx_2 = k * nx * ny + j * nx + 2;
        cuDoubleComplex numerator = cuCsub(__ldg(&psi[idx_2]), __ldg(&psi[idx_0]));
        dpsidx = cuCmul(numerator, make_cuDoubleComplex(inv_2hx, 0.0));
        
    } else if (i == nx - 2) {
        int idx_m2 = k * nx * ny + j * nx + (nx - 3);
        int idx_0 = k * nx * ny + j * nx + (nx - 1);
        cuDoubleComplex numerator = cuCsub(__ldg(&psi[idx_0]), __ldg(&psi[idx_m2]));
        dpsidx = cuCmul(numerator, make_cuDoubleComplex(inv_2hx, 0.0));
    }
    
    // Y-derivative (∂Ψ/∂y) - similar structure
    if (j > 1 && j < ny - 2) {
        int idx_m2 = k * nx * ny + (j - 2) * nx + i;
        int idx_m1 = k * nx * ny + (j - 1) * nx + i;
        int idx_p1 = k * nx * ny + (j + 1) * nx + i;
        int idx_p2 = k * nx * ny + (j + 2) * nx + i;
        
        cuDoubleComplex term1 = __ldg(&psi[idx_m2]);
        cuDoubleComplex term2 = cuCmul(make_cuDoubleComplex(8.0, 0.0), __ldg(&psi[idx_m1]));
        cuDoubleComplex term3 = cuCmul(make_cuDoubleComplex(8.0, 0.0), __ldg(&psi[idx_p1]));
        cuDoubleComplex term4 = __ldg(&psi[idx_p2]);
        
        cuDoubleComplex numerator = cuCsub(cuCadd(cuCsub(term1, term2), term3), term4);
        dpsidy = cuCmul(numerator, make_cuDoubleComplex(inv_12hy, 0.0));
        
    } else if (j == 1) {
        int idx_0 = k * nx * ny + 0 * nx + i;
        int idx_2 = k * nx * ny + 2 * nx + i;
        cuDoubleComplex numerator = cuCsub(__ldg(&psi[idx_2]), __ldg(&psi[idx_0]));
        dpsidy = cuCmul(numerator, make_cuDoubleComplex(inv_2hy, 0.0));
        
    } else if (j == ny - 2) {
        int idx_m2 = k * nx * ny + (ny - 3) * nx + i;
        int idx_0 = k * nx * ny + (ny - 1) * nx + i;
        cuDoubleComplex numerator = cuCsub(__ldg(&psi[idx_0]), __ldg(&psi[idx_m2]));
        dpsidy = cuCmul(numerator, make_cuDoubleComplex(inv_2hy, 0.0));
    }
    
    // Z-derivative (∂Ψ/∂z) - similar structure
    if (k > 1 && k < nz - 2) {
        int idx_m2 = (k - 2) * nx * ny + j * nx + i;
        int idx_m1 = (k - 1) * nx * ny + j * nx + i;
        int idx_p1 = (k + 1) * nx * ny + j * nx + i;
        int idx_p2 = (k + 2) * nx * ny + j * nx + i;
        
        cuDoubleComplex term1 = __ldg(&psi[idx_m2]);
        cuDoubleComplex term2 = cuCmul(make_cuDoubleComplex(8.0, 0.0), __ldg(&psi[idx_m1]));
        cuDoubleComplex term3 = cuCmul(make_cuDoubleComplex(8.0, 0.0), __ldg(&psi[idx_p1]));
        cuDoubleComplex term4 = __ldg(&psi[idx_p2]);
        
        cuDoubleComplex numerator = cuCsub(cuCadd(cuCsub(term1, term2), term3), term4);
        dpsidz = cuCmul(numerator, make_cuDoubleComplex(inv_12hz, 0.0));
        
    } else if (k == 1) {
        int idx_0 = 0 * nx * ny + j * nx + i;
        int idx_2 = 2 * nx * ny + j * nx + i;
        cuDoubleComplex numerator = cuCsub(__ldg(&psi[idx_2]), __ldg(&psi[idx_0]));
        dpsidz = cuCmul(numerator, make_cuDoubleComplex(inv_2hz, 0.0));
        
    } else if (k == nz - 2) {
        int idx_m2 = (nz - 3) * nx * ny + j * nx + i;
        int idx_0 = (nz - 1) * nx * ny + j * nx + i;
        cuDoubleComplex numerator = cuCsub(__ldg(&psi[idx_0]), __ldg(&psi[idx_m2]));
        dpsidz = cuCmul(numerator, make_cuDoubleComplex(inv_2hz, 0.0));
    }
    
    // Now compute |∇Ψ|² = |∂Ψ/∂x|² + |∂Ψ/∂y|² + |∂Ψ/∂z|²
    double grad_mag_squared = cuCabs(dpsidx) * cuCabs(dpsidx) + 
                             cuCabs(dpsidy) * cuCabs(dpsidy) + 
                             cuCabs(dpsidz) * cuCabs(dpsidz);
    
    f_res[idx] = grad_mag_squared * kin_energy_factor;
}
 
 /**
 *    Gauss-Legendre N-point quadrature formula.
 */
void gauleg(double x1, double x2, MultiArray<double>& x, MultiArray<double>& w) {
    // Get raw pointers and size from MultiArray
    double* x_ptr = x.raw();
    double* w_ptr = w.raw();
    long N = x.size();
    
    long m, j, i;
    double z1, z, xm, xl, pp, p3, p2, p1;

    m = (N + 1) / 2;
    xm = 0.5 * (x2 + x1);
    xl = 0.5 * (x2 - x1);
    
    for(i = 1; i <= m; i++) {
       z = cos(4. * atan(1.) * (i - 0.25) / (N + 0.5));
       do {
          p1 = 1.;
          p2 = 0.;
          for(j = 1; j <= N; j++) {
             p3 = p2;
             p2 = p1;
             p1 = ((2. * j - 1.) * z * p2 - (j - 1.) * p3) / j;
          }
          pp = N * (z * p1 - p2) / (z * z - 1.);
          z1 = z;
          z = z1 - p1 / pp;
       } while (fabs(z - z1) > GAULEG_EPS);
       
       // Use raw pointers for array access
       x_ptr[i] = xm - xl * z;
       x_ptr[N + 1 - i] = xm + xl * z;
       w_ptr[i] = 2. * xl / ((1. - z * z) * pp * pp);
       w_ptr[N + 1 - i] = w_ptr[i];
    }

    return;
}

/**
 * @file diffint.cu
 * @brief Implementation of spatial 1D integration with Simpson's rule.
 * @param h Space step
 * @param f Array with the function values
 * @param N Number of integration points
 * @return Integrated value
 */

__host__ double simpint(double h, double *f, long N) {
    long cnti;
    double sumi = 0., sumj = 0., sumk = 0.;

    for (cnti = 1; cnti < N - 1; cnti += 2) {
        sumi += f[cnti];
        sumj += f[cnti - 1];
        sumk += f[cnti + 1];
    }

    double sum = sumj + 4.0 * sumi + sumk;
    if (N % 2 == 0)
        sum += (5.0 * f[N - 1] + 8.0 * f[N - 2] - f[N - 3]) / 4.0;

    return sum * h / 3.0;
}