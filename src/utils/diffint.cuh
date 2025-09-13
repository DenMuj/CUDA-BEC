#ifndef DIFFINT_H
#define DIFFINT_H

#include <cuda_runtime.h>
#include <math.h>
#include <cuComplex.h>
#include "multiarray.h"

#define GAULEG_EPS   1.e-12

__global__ void diff_kernel(double hx, double hy, double hz,
    double* __restrict__ psi,
    double* __restrict__ f_res,
    long nx, long ny, long nz, int par);
void diff(double h, double *f, double* __restrict__ f_res, long nx, long ny, long nz, int par);
__global__ void diff_kernel_complex(double hx, double hy, double hz, cuDoubleComplex* __restrict__ f, double* __restrict__ f_res, long nx, long ny, long nz, int par);
void diff_complex(double hx, double hy, double hz, cuDoubleComplex* f, double* __restrict__ f_res, long nx, long ny, long nz, int par);
void gauleg(double x1, double x2, double *x, double *w, long N);


#endif