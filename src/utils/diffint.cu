#include "diffint.cuh"
#include <cuComplex.h>


/**
 *    Richardson extrapolation formula for calculation of space derivatives.
 *    h  - space step
 *    f  - array with the function values
 *    df - array with the first derivatives of the function
 */
 void diff(double hx, double hy, double hz, double* f, double* __restrict__ f_res, long nx, long ny, long nz, int par) {
    // Get raw pointers and size from MultiArray
   double* f_ptr = f;
   double* f_res_ptr = f_res;
    
   dim3 blockSize(8, 8, 8);
    dim3 gridSize((nx + blockSize.x - 1) / blockSize.x,
                  (ny + blockSize.y - 1) / blockSize.y,
                  (nz + blockSize.z - 1) / blockSize.z);
    
    // Kernel still uses raw pointers
   diff_kernel<<<gridSize, blockSize>>>(hx, hy, hz, f_ptr, f_res_ptr, nx, ny, nz, par);
}

__global__ void diff_kernel(
   double hx, double hy, double hz,
   double* __restrict__ psi,
   double* __restrict__ f_res,
   long nx, long ny, long nz, int par) {
   
   int i = threadIdx.x + blockDim.x * blockIdx.x;
   int j = threadIdx.y + blockDim.y * blockIdx.y;
   int k = threadIdx.z + blockDim.z * blockIdx.z;
   
   if (i >= nx || j >= ny || k >= nz) return;
   
   int idx = k * nx * ny + j * nx + i;  // 3D to 1D index
   
   // Precompute reciprocals
   const double inv_12hx = 1.0 / (12.0 * hx);
   const double inv_2hx = 1.0 / (2.0 * hx);
   const double inv_12hy = 1.0 / (12.0 * hy);
   const double inv_2hy = 1.0 / (2.0 * hy);
   const double inv_12hz = 1.0 / (12.0 * hz);
   const double inv_2hz = 1.0 / (2.0 * hz);
   
   double dpsidx = 0.0, dpsidy = 0.0, dpsidz = 0.0;
   
   // X-derivative (∂Ψ/∂x)
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
   
   // Y-derivative (∂Ψ/∂y)
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
   
   // Z-derivative (∂Ψ/∂z)
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
   double con = 1/(2.0 * (3-par));
   f_res[idx] = (dpsidx * dpsidx + dpsidy * dpsidy + dpsidz * dpsidz)*con;
}

void diff_complex(double hx, double hy, double hz, cuDoubleComplex* f, double* __restrict__ f_res, long nx, long ny, long nz, int par) {
    // Get raw pointers and size from MultiArray
   cuDoubleComplex* f_ptr = f;
   double* f_res_ptr = f_res;
    
   dim3 blockSize(8, 8, 8);
    dim3 gridSize((nx + blockSize.x - 1) / blockSize.x,
                  (ny + blockSize.y - 1) / blockSize.y,
                  (nz + blockSize.z - 1) / blockSize.z);
    
    // Kernel still uses raw pointers
   diff_kernel_complex<<<gridSize, blockSize>>>(hx, hy, hz, f_ptr, f_res_ptr, nx, ny, nz, par);
}

__global__ void diff_kernel_complex(
    double hx, double hy, double hz,
    cuDoubleComplex* __restrict__ psi,
    double* __restrict__ f_res,
    long nx, long ny, long nz, int par) {
    
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
    
    double con = 1.0 / (2.0 * (3 - par));
    f_res[idx] = grad_mag_squared * con;
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