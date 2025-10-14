#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cufft.h>
#include <cuComplex.h>
#include "../utils/diffint.cuh"
#include "../utils/multiarray.h"
#include "../utils/simpson3d_integrator.hpp"
#include "../utils/cfg.h"
#include "../utils/CudaArray.h"
#include <complex>
#include <memory>
#include <iomanip>
#include <iostream>

#define BOHR_RADIUS        5.2917720859e-11
#define MAX_FILENAME_SIZE  256
// #define TILE_SIZE 128
enum OutputFlags { DEN_X = 1 << 0, DEN_Y = 1 << 1, DEN_Z = 1 << 2, DEN_XY = 1 << 3, DEN_XZ = 1 << 4, DEN_YZ     = 1 << 5, DEN_XY0 = 1 << 6, DEN_X0Z = 1 << 7, DEN_0YZ = 1 << 8, DEN_XYZ = 1 << 9, ARG_XY0 = 1 << 10, ARG_X0Z = 1 <<     11, ARG_0YZ = 1 << 12 };
int outflags = 0;

double pi;

const char *input, *input_type, *output, *muoutput, *rmsout, *Niterout, *finalpsi;
long outstpx, outstpy, outstpz;



long Nx, Ny, Nz;
long Nx2, Ny2, Nz2;
__constant__ long d_Nx, d_Ny, d_Nz;

double edd, h2, h4, q3, q5, norm_psi2, norm_psi3, sx, sy, sz, murel, muend;
int QF, QDEPL;
int opt, optms;
int MS;
double Na;
long Niter, Nsnap;
double Nad;
double g, gd, g3;
double aho, as, add;
double dx, dy, dz, dx2, dy2, dz2;
double dt;
__constant__ double d_dt;
double vnu, vlambda, vgamma;
double par;
double cutoff;
double mx,my,mz,mt;
cuDoubleComplex minusAx, minusAy, minusAz;
__constant__ cuDoubleComplex d_minusAx, d_minusAy, d_minusAz;



void readpar();
void initpsi(double  *psi, MultiArray<double>& x2, MultiArray<double>& y2, MultiArray<double>& z2, MultiArray<double>& x, MultiArray<double>& y, MultiArray<double>& z);
void initpot(MultiArray<double>& pot, MultiArray<double>& x2, MultiArray<double>& y2, MultiArray<double>& z2);


void compute_rms_values(
    const CudaArray3D<cuDoubleComplex>& d_psi,
    CudaArray3D<double>& d_work_array,
    Simpson3DTiledIntegrator& integ,
    double* h_rms_pinned);

__global__ void compute_single_weighted_psi_squared(
        const cuDoubleComplex* __restrict__ psi,
        double* result,
        int direction,
        const double discretiz);

void calc_d_psi2(const cuDoubleComplex *d_psi, double *d_psi2);
__global__ void compute_d_psi2(
    const cuDoubleComplex* __restrict__ d_psi, 
     double* __restrict__ d_psi2);

void gencoef(MultiArray<cuDoubleComplex>& calphax, MultiArray<cuDoubleComplex>& cgammax, MultiArray<cuDoubleComplex>& calphay, MultiArray<cuDoubleComplex>& cgammay, MultiArray<cuDoubleComplex>& calphaz, MultiArray<cuDoubleComplex>& cgammaz, cuDoubleComplex& Ax0, cuDoubleComplex& Ay0, cuDoubleComplex& Az0, cuDoubleComplex& Ax0r, cuDoubleComplex& Ay0r, cuDoubleComplex& Az0r, cuDoubleComplex& Ax, cuDoubleComplex& Ay, cuDoubleComplex& Az);
void diff(double hx, double hy, double hz, double* f, double* __restrict__ f_res, long nx, long ny, long nz, int par);
void diff_complex(double hx, double hy, double hz, cuDoubleComplex* f, double* __restrict__ f_res, long nx, long ny, long nz, int par);
extern __global__ void diff_kernel(double hx, double hy, double hz, double* __restrict__ f, double* __restrict__ f_res, long nx, long ny, long nz, int par);
extern __global__ void diff_kernel_complex(double hx, double hy, double hz, cuDoubleComplex* __restrict__ f, double* __restrict__ f_res, long nx, long ny, long nz, int par);

void calcnorm(CudaArray3D<cuDoubleComplex>& d_psi, CudaArray3D<double>& d_psi2, double& norm, Simpson3DTiledIntegrator& integ);
__global__ void multiply_by_norm(cuDoubleComplex* __restrict__ d_psi, const double norm);

void calcnu(CudaArray3D<cuDoubleComplex>& d_psi, CudaArray3D<double>& d_psi2, CudaArray3D<double>& d_pot, double g, double gd, double h2);
__global__ void calcnu_kernel(cuDoubleComplex* __restrict__ d_psi, double* __restrict__ d_psi2, const double* __restrict__ pot, const double g, const double gd, const double h2);

void calclux(CudaArray3D<cuDoubleComplex>& d_psi, cuDoubleComplex* d_cbeta, CudaArray3D<cuDoubleComplex>& d_calphax, CudaArray3D<cuDoubleComplex>& d_cgammax, cuDoubleComplex d_Ax0r, cuDoubleComplex d_Ax);
__global__ void calclux_kernel(
   cuDoubleComplex* __restrict__ psi, 
   cuDoubleComplex* __restrict__ cbeta, 
   const cuDoubleComplex* __restrict__ d_calphax, 
   const cuDoubleComplex* __restrict__ d_cgammax,
   const cuDoubleComplex d_Ax, const cuDoubleComplex d_Ax0r
   );

void calcluy(CudaArray3D<cuDoubleComplex>& d_psi, cuDoubleComplex* d_cbeta, CudaArray3D<cuDoubleComplex>& d_calphay, CudaArray3D<cuDoubleComplex>& d_cgammay, cuDoubleComplex Ay0r, cuDoubleComplex Ay);
__global__ void calcluy_kernel(
   cuDoubleComplex* __restrict__ psi, 
   cuDoubleComplex* __restrict__ cbeta, 
   const cuDoubleComplex* __restrict__ d_calphay, 
   const cuDoubleComplex* __restrict__ d_cgammay,
   const cuDoubleComplex Ay0r, const cuDoubleComplex Ay
   );

void calcluz(CudaArray3D<cuDoubleComplex>& d_psi, cuDoubleComplex* d_cbeta, CudaArray3D<cuDoubleComplex>& d_calphaz, CudaArray3D<cuDoubleComplex>& d_cgammaz, cuDoubleComplex Az0r, cuDoubleComplex Az);
__global__ void calcluz_kernel(
   cuDoubleComplex* __restrict__ psi, 
   cuDoubleComplex* __restrict__ cbeta, 
   const cuDoubleComplex* __restrict__ d_calphaz, 
   const cuDoubleComplex* __restrict__ d_cgammaz,
   const cuDoubleComplex Az0r, const cuDoubleComplex Az
   );

void initpotdd(MultiArray<double> &potdd, MultiArray<double> &kx, MultiArray<double> &ky, MultiArray<double> &kz, MultiArray<double> &kx2, MultiArray<double> &ky2, MultiArray<double> &kz2);

void calc_psid2_potdd(cufftHandle forward_plan, cufftHandle backward_plan, cuDoubleComplex* d_psi, double* d_psi2_real, cufftDoubleComplex * d_psi2_fft,const double* potdd);
__global__ void compute_psid2_potdd(cufftDoubleComplex * d_psi2_fft, 
  const double* __restrict__ potdd);


__global__ void calcpsidd2_boundaries(double *psidd2);

void calcmuen(MultiArray<double>& muen,CudaArray3D<cuDoubleComplex> &d_psi, CudaArray3D<double> &d_psi2, CudaArray3D<double> &d_pot, CudaArray3D<double> &d_psi2dd, CudaArray3D<double> &d_potdd, cufftDoubleComplex * d_psi2_fft, cufftHandle forward_plan, cufftHandle backward_plan,Simpson3DTiledIntegrator &integ, const double g, const double gd, const double h2);


__global__ void calcmuen_fused_contact(const cuDoubleComplex *__restrict__ d_psi, double *__restrict__ d_result, double g);
__global__ void calcmuen_fused_potential(const cuDoubleComplex *__restrict__ d_psi, double *__restrict__ d_result, const double *__restrict__ d_pot);
__global__ void calcmuen_fused_dipolar(const cuDoubleComplex *__restrict__ d_psi, double *__restrict__ d_result, const double *__restrict__ d_psidd2, const double gd);
__global__ void calcmuen_fused_h2(const cuDoubleComplex *__restrict__ d_psi, double *__restrict__ d_result, const double h2);
void calcmuen_kin(CudaArray3D<cuDoubleComplex> &d_psi, CudaArray3D<double> &d_work_array, int par);

void save_psi_from_gpu(cuDoubleComplex *psi, cuDoubleComplex *d_psi, const char *filename, long Nx, long Ny, long Nz);
void read_psi_from_file_complex(cuDoubleComplex *psi, const char *filename, long Nx, long Ny, long Nz);

void rms_output(FILE *filerms);
void mu_output(FILE *filemu);

void outdenx(cuDoubleComplex *psi, MultiArray<double> &x, MultiArray<double> &tmpy, MultiArray<double> &tmpz, FILE *file);
void outdeny(cuDoubleComplex *psi, MultiArray<double> &y, MultiArray<double> &tmpx, MultiArray<double> &tmpz, FILE *file);
void outdenz(cuDoubleComplex *psi, MultiArray<double> &z, MultiArray<double> &tmpx, MultiArray<double> &tmpy, FILE *file);
void outdenxy(cuDoubleComplex *psi, MultiArray<double> &x, MultiArray<double> &y, MultiArray<double> &tmpz, FILE *file);
void outdenxz(cuDoubleComplex *psi, MultiArray<double> &x, MultiArray<double> &z, MultiArray<double> &tmpx, FILE *file);
void outdenyz(cuDoubleComplex *psi, MultiArray<double> &y, MultiArray<double> &z, MultiArray<double> &tmpx, FILE *file);
void outpsi2xy(cuDoubleComplex *psi, MultiArray<double> &x, MultiArray<double> &y, FILE *file);
void outpsi2xz(cuDoubleComplex *psi, MultiArray<double> &x, MultiArray<double> &z, FILE *file);
void outpsi2yz(cuDoubleComplex *psi, MultiArray<double> &y, MultiArray<double> &z, FILE *file);