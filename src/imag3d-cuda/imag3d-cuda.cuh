#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <cmath>
#include <chrono>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cufft.h>
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

// Function declerations
void readpar();
void initpsi(double  *psi, MultiArray<double>& x2, MultiArray<double>& y2, MultiArray<double>& z2);
void initpot(MultiArray<double>& pot, MultiArray<double>& x2, MultiArray<double>& y2, MultiArray<double>& z2);


void compute_rms_values(
    const double *d_psi,                 // Device: 3D psi array
    double *d_work_array,  // Single work array instead of 3
    const double *d_x2,
    const double *d_y2,
    const double *d_z2,
    Simpson3DTiledIntegrator& integ,
    double* h_rms_pinned); // Output RMS values in pinned memory [rms_x, rms_y, rms_z]

__global__ void compute_single_weighted_psi_squared(
        const double* __restrict__ psi,
        const double* __restrict__ coord_squared,  // x2, y2, or z2
        double* result,
        int direction);

void calc_d_psi2(const double *d_psi, double *d_psi2);
__global__ void compute_d_psi2(
    const double* __restrict__ d_psi, 
     double* __restrict__ d_psi2);

void gencoef(MultiArray<double>& calphax, MultiArray<double>& cgammax, MultiArray<double>& calphay, MultiArray<double>& cgammay, MultiArray<double>& calphaz, MultiArray<double>& cgammaz, double& Ax0, double& Ay0, double& Az0, double& Ax0r, double& Ay0r, double& Az0r, double& Ax, double& Ay, double& Az);
void diff(double hx, double hy, double hz, double* f, double* __restrict__ f_res, long nx, long ny, long nz, int par);
extern __global__ void diff_kernel(double hx, double hy, double hz, double* __restrict__ f, double* __restrict__ f_res, long nx, long ny, long nz, int par);

void calcnorm(double *d_psi, double *d_psi2, double& norm, Simpson3DTiledIntegrator& integ);
__global__ void multiply_by_norm(double* __restrict__ d_psi, const double norm);

void calcnu(double *d_psi, double *d_psi2, double *d_pot, double g, double gd);
__global__ void calcnu_kernel(double* __restrict__ d_psi, double* __restrict__ d_psi2, const double* __restrict__ pot, const double g, const double gd);

void calclux(double *d_psi, double *d_cbeta, double *d_calphax, double *d_cgammax, double d_Ax0r, double d_Ax);
__global__ void calclux_kernel(
   double* __restrict__ psi, 
   double* __restrict__ cbeta, 
   const double* __restrict__ d_calphax, 
   const double* __restrict__ d_cgammax,
   const double d_Ax, const double d_Ax0r
   );

void calcluy(double *d_psi, double *d_cbeta, double *d_calphay, double *d_cgammay, double d_Ay, double d_Ay0r);
__global__ void calcluy_kernel(
   double* __restrict__ psi, 
   double* __restrict__ cbeta, 
   const double* __restrict__ d_calphay, 
   const double* __restrict__ d_cgammay,
   const double d_Ay, const double d_Ay0r
   );

void calcluz(double *d_psi, double *d_cbeta, double *d_calphaz, double *d_cgammaz, double d_Az, double d_Az0r);
__global__ void calcluz_kernel(
   double* __restrict__ psi, 
   double* __restrict__ cbeta, 
   const double* __restrict__ d_calphaz, 
   const double* __restrict__ d_cgammaz,
   const double d_Az, const double d_Az0r
   );

void initpotdd(MultiArray<double> &potdd, MultiArray<double> &kx, MultiArray<double> &ky, MultiArray<double> &kz, MultiArray<double> &kx2, MultiArray<double> &ky2, MultiArray<double> &kz2);

void calc_psid2_potdd(cufftHandle forward_plan, cufftHandle backward_plan, const double* d_psi, double* d_psi2_real, cufftDoubleComplex * d_psi2_fft,const double* potdd);
__global__ void compute_psid2_potdd(cufftDoubleComplex * d_psi2_fft, 
  const double* __restrict__ potdd);


__global__ void calcpsidd2_boundaries(double *psidd2);

void calcmuen(double *muen,double *d_psi, double *d_psi2, double *d_pot, double *d_psi2dd, double *d_potdd, cufftDoubleComplex * d_psi2_fft, cufftHandle forward_plan, cufftHandle backward_plan,Simpson3DTiledIntegrator &integ, const double g, const double gd);

// Optimized fused kernels
__global__ void calcmuen_fused_contact(const double *__restrict__ d_psi, double *__restrict__ d_result, double g);
__global__ void calcmuen_fused_potential(const double *__restrict__ d_psi, double *__restrict__ d_result, const double *__restrict__ d_pot);
__global__ void calcmuen_fused_dipolar(const double *__restrict__ d_psi, double *__restrict__ d_result, const double *__restrict__ d_psidd2, const double gd);

// Original kernels (kept for reference)
__global__ void calcmuen_kernel_con(double *__restrict__ d_psi2, double g);
__global__ void calcmuen_kernel_pot(double *__restrict__ d_psi2, double *__restrict__ d_pot);
__global__ void calcmuen_kernel_potdd(double * __restrict__ d_psi2, double *__restrict__ d_psidd2, const double gd);
void calcmuen_kin(double *d_psi, double *d_work_array, int par);

void save_psi_from_gpu(double *psi, double *d_psi, const char *filename, long Nx, long Ny, long Nz);
void read_psi_from_file(double *psi, const char *filename, long Nx, long Ny, long Nz);

void rms_output(FILE *filerms);
void mu_output(FILE *filemu);