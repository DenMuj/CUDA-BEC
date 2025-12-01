#ifndef DIFFINT_H
#define DIFFINT_H

#include <cuda_runtime.h>
#include <math.h>
#include <cuComplex.h>
#include "multiarray.h"

#define GAULEG_EPS   1.e-12

/**
 * @brief Kernel function for the Richardson extrapolation formula for calculation of space derivatives.
 * @param hx Space step
 * @param hy Space step
 * @param hz Space step
 * @param psi Function values
 * @param f_res Array with the first derivatives of the function
 * @param kin_energy_factor Precomputed kinetic energy factor 1/(2*(3-par))
 */
__global__ void diff_kernel(double hx, double hy, double hz,
    double* __restrict__ psi,
    double* __restrict__ f_res,
    long nx, long ny, long nz, double kin_energy_factor, long f_res_nx);

/**
 * @brief Richardson extrapolation formula for calculation of space derivatives.
 * @param hx Space step
 * @param hy Space step
 * @param hz Space step
 * @param f Function values
 * @param f_res Array with the first derivatives of the function
 * @param nx X dimension of input array f
 * @param ny Y dimension
 * @param nz Z dimension
 * @param par Parameter
 * @param f_res_nx X dimension of output array f_res (defaults to nx if not padded)
 */
void diff(double hx, double hy, double hz, double* f, double* __restrict__ f_res, long nx, long ny, long nz, int par, long f_res_nx);

/**
 * @brief Kernel function for the Richardson extrapolation formula for calculation of space derivatives for complex functions.
 * @param hx Space step
 * @param hy Space step
 * @param hz Space step
 * @param f Function values for complex functions
 * @param f_res Array with the first derivatives of the function
 */
__global__ void diff_kernel_complex(double hx, double hy, double hz, cuDoubleComplex* __restrict__ f, double* __restrict__ f_res, long nx, long ny, long nz, double kin_energy_factor);


/**
 * @brief Richardson extrapolation formula for calculation of space derivatives for complex functions.
 * @param hx Space step
 * @param hy Space step
 * @param hz Space step
 * @param f Function values for complex functions
 * @param f_res Array with the first derivatives of the function
 */
void diff_complex(double hx, double hy, double hz, cuDoubleComplex* f, double* __restrict__ f_res, long nx, long ny, long nz, int par);


void gauleg(double x1, double x2, double *x, double *w, long N);

/**
 * @brief Spatial 1D integration with Simpson's rule.
 * @param h Space step
 * @param f Array with the function values
 * @param N Number of integration points
 * @return Integrated value
 */
__host__ double simpint(double h, double *f, long N);

#endif