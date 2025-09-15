#include "imag3d-cuda.cuh"


int main(int argc, char **argv) {
  if ((argc != 3) || (strcmp(*(argv + 1), "-i") != 0)) {
    fprintf(stderr, "Usage: %s -i <input parameter file> \n", *argv);
    exit(EXIT_FAILURE);
  }

  if (!cfg_init(argv[2])) {
    fprintf(stderr, "Wrong input parameter file.\n");
    exit(EXIT_FAILURE);
  }
  pi = M_PI;
  FILE *filerms;
  FILE *filemu;
  char filename[MAX_FILENAME_SIZE];
  readpar();
  if (opt == 2)
    par = 2.;
  else
    par = 1.;

  g = par * g;
  gd = par * gd;

  gd *= MS;
  edd = (4. * pi / 3.) * gd / g;
  Nad = Na;

  Nx2 = Nx / 2;
  Ny2 = Ny / 2;
  Nz2 = Nz / 2;

  cudaMemcpyToSymbol(d_Nx, &Nx, sizeof(long));
  cudaMemcpyToSymbol(d_Ny, &Ny, sizeof(long));
  cudaMemcpyToSymbol(d_Nz, &Nz, sizeof(long));
  cudaMemcpyToSymbol(d_dt, &dt, sizeof(double));

  // Initialize squared grid spacings
  dx2 = dx * dx;
  dy2 = dy * dy;
  dz2 = dz * dz;

  MultiArray<double> muen(4);
  MultiArray<double> x(Nx), y(Ny), z(Nz);
  MultiArray<double> x2(Nx), y2(Ny), z2(Nz);
  MultiArray<double> kx(Nx), ky(Ny), kz(Nz);
  MultiArray<double> kx2(Nx), ky2(Ny), kz2(Nz);

  double mutotold, mutotnew;

  // Allocation of crank-nicolson coefficients
  MultiArray<double> calphax(Nx - 1), cgammax(Nx - 1);
  MultiArray<double> calphay(Ny - 1), cgammay(Ny - 1);
  MultiArray<double> calphaz(Nz - 1), cgammaz(Nz - 1);
  double Ax0, Ay0, Az0, Ax0r, Ay0r, Az0r, Ax, Ay, Az;

  // Allocation of crank-nicolson coefficients on device
  CudaArray3D<double> d_calphax(Nx - 1);
  CudaArray3D<double> d_cgammax(Nx - 1);
  CudaArray3D<double> d_calphay(Ny - 1);
  CudaArray3D<double> d_cgammay(Ny - 1);
  CudaArray3D<double> d_calphaz(Nz - 1);
  CudaArray3D<double> d_cgammaz(Nz - 1);

  //CudaArray d_cbeta(Nx * Ny * Nz);

  // Allocation of the wave function norm
  double norm;

  long TILE_SIZE = Nz;
  Simpson3DTiledIntegrator integ(Nx, Ny, TILE_SIZE);
  MultiArray<double> pot(Nz, Ny, Nx);
  //MultiArray<double> psi(Nz, Ny, Nx);
  double *psi;
  cudaMallocHost(&psi, Nz * Ny * Nx * sizeof(double));
  MultiArray<double> psi2(Nz, Ny, Nx);
  MultiArray<double> potdd(Nz, Ny, Nx);
  
  CudaArray3D<double> d_psi(Nx, Ny, Nz, true);
  CudaArray3D<double> d_pot(Nx, Ny, Nz, true);
  CudaArray3D<double> d_x2(Nx);
  CudaArray3D<double> d_y2(Ny);
  CudaArray3D<double> d_z2(Nz);
  CudaArray3D<double> d_work_array(Nx, Ny, Nz, true);
  CudaArray3D<double> d_potdd(Nx, Ny, Nz, true);
  CudaArray3D<double> d_psi2dd(Nx, Ny, Nz, true);
  

  // FFT arrays
  cufftDoubleComplex *d_psi2_fft;
  cudaMalloc(&d_psi2_fft, Nz * Ny * (Nx2 +1) * sizeof(cufftDoubleComplex));

  //Create plan for FFT of 3D array
  cufftHandle forward_plan, backward_plan;
  cufftResult res = cufftPlan3d(&forward_plan, Nz, Ny, Nx, CUFFT_D2Z);
    if (res != CUFFT_SUCCESS) {
        std::cerr << "CUFFT error: Plan creation failed - Forward plan" << std::endl;
        return -1;
    }

  res = cufftPlan3d(&backward_plan, Nz, Ny, Nx, CUFFT_Z2D);
  if (res != CUFFT_SUCCESS) {
    std::cerr << "CUFFT error: Plan creation failed - Backward plan" << std::endl;
    return -1;
  }

  // Allocate pinned memory for RMS results for better performance
  double *h_rms_pinned;
  cudaHostAlloc(&h_rms_pinned, 3 * sizeof(double), cudaHostAllocDefault);
  
  if (rmsout != NULL) {
    sprintf(filename, "%s.txt", rmsout);
    filerms = fopen(filename, "w");
  } else filerms = NULL;
  if (muoutput != NULL) {
    sprintf(filename, "%s.txt", muoutput);
    filemu = fopen(filename, "w");
  } else filemu = NULL;

  initpsi(psi, x2, y2, z2);
  initpot(pot, x2, y2, z2);
  initpotdd(potdd,kx,ky,kz,kx2,ky2,kz2);
  gencoef(calphax, cgammax, calphay, cgammay, calphaz, cgammaz, Ax0, Ay0, Az0,
          Ax0r, Ay0r, Az0r, Ax, Ay, Az);

  // Copy coefficients to device (do this once during initialization)
  d_calphax.copyFromHost(calphax.raw());
  d_cgammax.copyFromHost(cgammax.raw());
  d_calphay.copyFromHost(calphay.raw());
  d_cgammay.copyFromHost(cgammay.raw());
  d_calphaz.copyFromHost(calphaz.raw());
  d_cgammaz.copyFromHost(cgammaz.raw());

  // Copy psi data to device
  d_psi.copyFromHost(psi);
  d_x2.copyFromHost(x2.raw());
  d_y2.copyFromHost(y2.raw());
  d_z2.copyFromHost(z2.raw());
  d_pot.copyFromHost(pot.raw());
  d_potdd.copyFromHost(potdd.raw());

  if (rmsout != NULL) {
    rms_output(filerms);
  }
  if(muoutput != NULL) {
    mu_output(filemu);
  }
  compute_rms_values(d_psi.raw(), d_work_array.raw(), d_x2.data(), d_y2.data(), d_z2.data(), integ, h_rms_pinned);
  if(muoutput != NULL) {
    calcmuen(muen.raw(),d_psi.data(),d_work_array.data(), d_pot.data(), d_psi2dd.data(), d_potdd.data(), d_psi2_fft, forward_plan, backward_plan, integ, g, gd);
    fprintf(filemu, "%-9d %-19.16le %-19.16le %-19.16le %-19.16le %-19.16le\n", 0, muen[0]+muen[1]+muen[2]+muen[3], muen[3], muen[1], muen[0], muen[2]);
    fflush(filemu);
    mutotold = muen[0]+muen[1]+muen[2]+muen[3];
  }
  if(rmsout != NULL) {
    double rms_r = sqrt(h_rms_pinned[0]*h_rms_pinned[0] + h_rms_pinned[1]*h_rms_pinned[1] + h_rms_pinned[2]*h_rms_pinned[2]);
    fprintf(filerms, "%-9d %-19.16le %-19.16le %-19.16le %-19.16le\n", 0, rms_r, h_rms_pinned[0], h_rms_pinned[1], h_rms_pinned[2]);
    fflush(filerms);
  }

  double nsteps;
  nsteps = Niter / Nsnap;
  auto start = std::chrono::high_resolution_clock::now();
  for (long snap = 1; snap <= Nsnap; snap++) {
    for(long j = 0; j<nsteps; j++){
      calc_psid2_potdd(forward_plan, backward_plan, d_psi.raw(), d_work_array.raw(), d_psi2_fft, d_potdd.raw());
      calcnu(d_psi.raw(), d_work_array.raw(), d_pot.raw(), g, gd);
      calclux(d_psi.raw(), d_work_array.raw(), d_calphax.raw(), d_cgammax.raw(), Ax0r, Ax);
      calcluy(d_psi.raw(), d_work_array.raw(), d_calphay.raw(), d_cgammay.raw(), Ay0r, Ay);
      calcluz(d_psi.raw(), d_work_array.raw(), d_calphaz.raw(), d_cgammaz.raw(), Az0r, Az);
      calcnorm(d_psi.raw(), d_work_array.raw(), norm, integ);
    }

    compute_rms_values(d_psi.raw(), d_work_array.raw(), d_x2.data(), d_y2.data(), d_z2.data(), integ, h_rms_pinned);
    
    if(rmsout != NULL) {
      double rms_r = sqrt(h_rms_pinned[0]*h_rms_pinned[0] + h_rms_pinned[1]*h_rms_pinned[1] + h_rms_pinned[2]*h_rms_pinned[2]);
      fprintf(filerms, "%-9li %-19.16le %-19.16le %-19.16le %-19.16le\n", snap, rms_r, h_rms_pinned[0], h_rms_pinned[1], h_rms_pinned[2]);
      fflush(filerms);
    }
    calcmuen(muen.raw(),d_psi.data(),d_work_array.data(), d_pot.data(), d_psi2dd.data(), d_potdd.data(), d_psi2_fft, forward_plan, backward_plan, integ, g, gd);
    if(muoutput != NULL) {
      fprintf(filemu, "%-9li %-19.16le %-19.16le %-19.16le %-19.16le %-19.16le\n", snap, muen[0]+muen[1]+muen[2]+muen[3], muen[3], muen[1], muen[0], muen[2]);
      fflush(filemu);
    }
    mutotnew = muen[0]+muen[1]+muen[2]+muen[3];
    if (fabs((mutotold - mutotnew) / mutotnew) < murel) break;
    mutotold = mutotnew;
    if (mutotnew > muend) break;

  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;
  if (rmsout != NULL) {
    fprintf(filerms, "-------------------------------------------------------------------\n\n");
    fprintf(filerms, "Total time on GPU: %f seconds\n", duration.count());
    fprintf(filerms, "-------------------------------------------------------------------\n\n");
    fclose(filerms);
  }  
  if(muoutput != NULL) {
    fprintf(filemu, "---------------------------------------------------------------------------------\n\n");
    fprintf(filemu, "Total time on GPU: %f seconds\n", duration.count());
    fprintf(filemu, "---------------------------------------------------------------------------------\n\n");
    fclose(filemu);
  }
  // Save FINALPSI
  if (finalpsi != NULL) {
    sprintf(filename, "%s.bin", finalpsi);
    save_psi_from_gpu(psi, d_psi.raw(), filename, Nx, Ny, Nz);
  }
//   std::cout << "Psi before reading: " << psi[0] << std::endl;
//   // Find min, max, and count non-zero elements asfsafas
// double min_val = psi[0], max_val = psi[0];
// int non_zero_count = 0;
// int total_elements = Nx * Ny * Nz;

// for(int i = 0; i < total_elements; i++) {
//     if(psi[i] != 0.0) non_zero_count++;
//     if(psi[i] < min_val) min_val = psi[i];
//     if(psi[i] > max_val) max_val = psi[i];
// }

// std::cout << "Total elements BEFORE ReAD: " << total_elements << std::endl;
// std::cout << "Non-zero elements BEFORE READ: " << non_zero_count << std::endl;
// std::cout << "Min value BEFORE READ: " << min_val << std::endl;
// std::cout << "Max value BEFORE READ: " << max_val << std::endl;
//   for(int i = 0; i < Nx * Ny * Nz; i++) {psi[i]=0;}
//   read_psi_from_file(psi, filename, Nx, Ny, Nz);
//   // Find min, max, and count non-zero elements
// double min_val1 = psi[0], max_val1 = psi[0];
// int non_zero_count1 = 0;
// int total_elements1 = Nx * Ny * Nz;

// for(int i = 0; i < total_elements; i++) {
//     if(psi[i] != 0.0) non_zero_count1++;
//     if(psi[i] < min_val1) min_val1 = psi[i];
//     if(psi[i] > max_val1) max_val1 = psi[i];
// }

// std::cout << "Total elements: " << total_elements << std::endl;
// std::cout << "Non-zero elements: " << non_zero_count1 << std::endl;
// std::cout << "Min value: " << min_val1 << std::endl;
// std::cout << "Max value: " << max_val1 << std::endl;
//   d_psi.copyFromHost(psi);
//   compute_rms_values(d_psi, d_work_array, d_x2, d_y2, d_z2, integ, h_rms_pinned);
//  std::cout << "rms_x: " << std::setprecision(16) << h_rms_pinned[0] << " rms_y: " << std::setprecision(16) << h_rms_pinned[1] << " rms_z: " << std::setprecision(16) << h_rms_pinned[2] << std::endl;

  // Cleanup pinned memory
  cudaFreeHost(h_rms_pinned);
  cudaFreeHost(psi);
  cudaFree(d_psi2_fft);
  cufftDestroy(forward_plan);
  cufftDestroy(backward_plan);
  return 0;
}

/**
 *    Reading input parameters from the configuration file.
 */
void readpar(void) {
  const char *cfg_tmp;

  if ((cfg_tmp = cfg_read("OPTION")) == NULL) {
    fprintf(stderr, "OPTION is not defined in the configuration file\n");
    exit(EXIT_FAILURE);
  }
  opt = atol(cfg_tmp);
  if ((cfg_tmp = cfg_read("OPTION_MICROWAVE_SHIELDING")) == NULL) {
    fprintf(stderr, "OPTION_MICROWAVE_SHIELDING is not defined in the configuration file\n");
    exit(EXIT_FAILURE);
  }
  optms = atol(cfg_tmp);
  if (optms == 0) {
    MS = 1;
  } else {
    MS = - 1;
  }
   
  if ((cfg_tmp = cfg_read("NATOMS")) == NULL) {
    fprintf(stderr, "NATOMS is not defined in the configuration file.\n");
    exit(EXIT_FAILURE);
  }
  Na = atof(cfg_tmp);

  if ((cfg_tmp = cfg_read("AHO")) == NULL) {
    fprintf(stderr, "AHO is not defined in the configuration file.\n");
    exit(EXIT_FAILURE);
  }
  aho = atof(cfg_tmp);

  if ((cfg_tmp = cfg_read("G")) == NULL) {
    if ((cfg_tmp = cfg_read("AS")) == NULL) {
      fprintf(stderr, "AS is not defined in the configuration file.\n");
      exit(EXIT_FAILURE);
    }
    as = atof(cfg_tmp);

    g = 4. * pi * as * Na * BOHR_RADIUS / aho;
  } else {
    g = atof(cfg_tmp);
  }

  if ((cfg_tmp = cfg_read("GDD")) == NULL) {
    if ((cfg_tmp = cfg_read("ADD")) == NULL) {
      fprintf(stderr, "ADD is not defined in the configuration file.\n");
      exit(EXIT_FAILURE);
    }
    add = atof(cfg_tmp);

    gd = 3. * add * Na * BOHR_RADIUS / aho;
  } else {
    gd = atof(cfg_tmp);
  }

  if ((cfg_tmp = cfg_read("QF")) == NULL) {
    QF = 0;
  } else
    QF = atol(cfg_tmp);

  if ((cfg_tmp = cfg_read("QDEPL")) == NULL) {
    QDEPL = 0;
  } else
    QDEPL = atol(cfg_tmp);

  if ((cfg_tmp = cfg_read("SX")) == NULL)
    sx = 0.;
  else
    sx = atof(cfg_tmp);

  if ((cfg_tmp = cfg_read("SY")) == NULL)
    sy = 0.;
  else
    sy = atof(cfg_tmp);

  if ((cfg_tmp = cfg_read("SZ")) == NULL)
    sz = 0.;
  else
    sz = atof(cfg_tmp);

  if ((cfg_tmp = cfg_read("NX")) == NULL) {
    fprintf(stderr, "NX is not defined in the configuration file.\n");
    exit(EXIT_FAILURE);
  }
  Nx = atol(cfg_tmp);

  if ((cfg_tmp = cfg_read("NY")) == NULL) {
    fprintf(stderr, "NY is not defined in the configuration file.\n");
    exit(EXIT_FAILURE);
  }
  Ny = atol(cfg_tmp);

  if ((cfg_tmp = cfg_read("NZ")) == NULL) {
    fprintf(stderr, "Nz is not defined in the configuration file.\n");
    exit(EXIT_FAILURE);
  }
  Nz = atol(cfg_tmp);

  if ((cfg_tmp = cfg_read("MX")) == NULL)
    mx = 10;
  else
    mx = atoi(cfg_tmp);

  if ((cfg_tmp = cfg_read("MY")) == NULL)
    my = 10;
  else
    my = atoi(cfg_tmp);

  if ((cfg_tmp = cfg_read("MZ")) == NULL)
    mz = 10;
  else
    mz = atoi(cfg_tmp);

  if ((cfg_tmp = cfg_read("MT")) == NULL)
    mt = 10;
  else
    mt = atoi(cfg_tmp);

  if ((cfg_tmp = cfg_read("DX")) == NULL)
    dx = sx * mx * 2 / Nx;
  else
    dx = atof(cfg_tmp);

  if ((cfg_tmp = cfg_read("DY")) == NULL)
    dy = sy * my * 2 / Ny;
  else
    dy = atof(cfg_tmp);
  if ((cfg_tmp = cfg_read("DZ")) == NULL)
    dz = sz * mz * 2 / Nz;
  else
    dz = atof(cfg_tmp);

  if ((cfg_tmp = cfg_read("DT")) == NULL) {
    dt = dy * dy / mt;
  } else
    dt = atof(cfg_tmp);

  if ((cfg_tmp = cfg_read("MUREL")) == NULL) {
    fprintf(stderr, "MUREL is not defined in the configuration file.\n");
    exit(EXIT_FAILURE);
  }
  murel = atof(cfg_tmp);

  if ((cfg_tmp = cfg_read("MUEND")) == NULL) {
    fprintf(stderr, "MUEND is not defined in the configuration file.\n");
    exit(EXIT_FAILURE);
  }
  muend = atof(cfg_tmp);

  if ((cfg_tmp = cfg_read("GAMMA")) == NULL) {
    fprintf(stderr, "GAMMA is not defined in the configuration file.\n");
    exit(EXIT_FAILURE);
  }
  vgamma = atof(cfg_tmp);

  if ((cfg_tmp = cfg_read("NU")) == NULL) {
    fprintf(stderr, "NU is not defined in the configuration file.\n");
    exit(EXIT_FAILURE);
  }
  vnu = atof(cfg_tmp);

  if ((cfg_tmp = cfg_read("LAMBDA")) == NULL) {
    fprintf(stderr, "LAMBDA is not defined in the configuration file.\n");
    exit(EXIT_FAILURE);
  }
  vlambda = atof(cfg_tmp);

  if ((cfg_tmp = cfg_read("NITER")) == NULL) {
    fprintf(stderr, "NITER is not defined in the configuration file.\n");
    exit(EXIT_FAILURE);
  }
  Niter = atol(cfg_tmp);
  if ((cfg_tmp = cfg_read("NSNAP")) == NULL)
    Nsnap = 1;
  else
    Nsnap = atol(cfg_tmp);

  if ((cfg_tmp = cfg_read("CUTOFF")) == NULL)
    cutoff = Ny * dy / 2;
  else
    cutoff = atof(cfg_tmp);

  input = cfg_read("INPUT");
  input_type = cfg_read("INPUT_TYPE");
  output = cfg_read("OUTPUT");
  muoutput = cfg_read("MUOUTPUT");
  rmsout = cfg_read("RMSOUT");
  Niterout = cfg_read("NITEROUT");
  finalpsi = cfg_read("FINALPSI");

  if (Niterout != NULL) {
    if ((cfg_tmp = cfg_read("OUTFLAGS")) == NULL) {
      fprintf(stderr, "OUTFLAGS is not defined in the configuration file.\n");
      exit(EXIT_FAILURE);
    }
    outflags = atoi(cfg_tmp);
  } else
    outflags = 0;

  if ((Niterout != NULL) || (finalpsi != NULL)) {
    if ((cfg_tmp = cfg_read("OUTSTPX")) == NULL) {
      fprintf(stderr, "OUTSTPX is not defined in the configuration file.\n");
      exit(EXIT_FAILURE);
    }
    outstpx = atol(cfg_tmp);
    if ((cfg_tmp = cfg_read("OUTSTPY")) == NULL) {
      fprintf(stderr, "OUTSTPY is not defined in the configuration file.\n");
      exit(EXIT_FAILURE);
    }
    outstpy = atol(cfg_tmp);

    if ((cfg_tmp = cfg_read("OUTSTPZ")) == NULL) {
      fprintf(stderr, "OUTSTPZ is not defined in the configuration file.\n");
      exit(EXIT_FAILURE);
    }
    outstpz = atol(cfg_tmp);
  }

  return;
}

// function to compute RMS values
void compute_rms_values(const double *d_psi, // Device: 3D psi array
                        double *d_work_array, const double *d_x2,
                        const double *d_y2, const double *d_z2,
                        Simpson3DTiledIntegrator &integ, 
                        double *h_rms_pinned) // Output RMS values in pinned memory [rms_x, rms_y, rms_z]clean
{
  // Check for CUDA errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA error after memcpy: %s\n", cudaGetErrorString(err));
  }

  // Configure kernel launch parameters
  dim3 blockSize(8, 8, 4); // Adjust based on your GPU
  dim3 gridSize((Nx + blockSize.x - 1) / blockSize.x,
                (Ny + blockSize.y - 1) / blockSize.y,
                (Nz + blockSize.z - 1) / blockSize.z);

  // Compute x^2 * psi^2
  compute_single_weighted_psi_squared<<<gridSize, blockSize>>>(
      d_psi, d_x2, d_work_array,
      0 // 0 for x direction
       );

  cudaDeviceSynchronize();
  double x2_integral =
      integ.integrateDevice(dx, dy, dz, d_work_array, Nx, Ny, Nz);

  // Compute y^2 * psi^2 (reuse d_work_array)
  compute_single_weighted_psi_squared<<<gridSize, blockSize>>>(
      d_psi, d_y2, d_work_array,
      1 // 1 for y direction
  );
  cudaDeviceSynchronize();
  double y2_integral =
      integ.integrateDevice(dx, dy, dz, d_work_array, Nx, Ny, Nz);

  // Compute z^2 * psi^2 (reuse d_work_array)
  compute_single_weighted_psi_squared<<<gridSize, blockSize>>>(
      d_psi, d_z2, d_work_array,
      2 // 2 for z direction
  );
  cudaDeviceSynchronize();
  double z2_integral =
      integ.integrateDevice(dx, dy, dz, d_work_array, Nx, Ny, Nz);

  // Calculate RMS values and store in pinned memory
  h_rms_pinned[0] = sqrt(x2_integral); // rms_x
  h_rms_pinned[1] = sqrt(y2_integral); // rms_y
  h_rms_pinned[2] = sqrt(z2_integral); // rms_z
}

__global__ void compute_single_weighted_psi_squared(
    const double *__restrict__ psi,
    const double *__restrict__ coord_squared, // x2, y2, or z2
    double *result,
    int direction) // 0=x, 1=y, 2=z
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;
  int idz = blockIdx.z * blockDim.z + threadIdx.z;

  if (idx >= d_Nx || idy >= d_Ny || idz >= d_Nz)
    return;

  int linear_idx = idz * d_Ny * d_Nx + idy * d_Nx + idx;
  double psi_val = __ldg(&psi[linear_idx]); // Read-only cache for psi
  double psi_squared = psi_val * psi_val;

  double weight = 0.0;
  if (direction == 0)
    weight = __ldg(&coord_squared[idx]); // x^2 - read-only cache
  else if (direction == 1)
    weight = __ldg(&coord_squared[idy]); // y^2 - read-only cache
  else if (direction == 2)
    weight = __ldg(&coord_squared[idz]); // z^2 - read-only cache

  result[linear_idx] = weight * psi_squared;
}

void calc_d_psi2(const double *d_psi, double *d_psi2) {
  dim3 threadsPerBlock(8, 8, 8);
  dim3 numBlocks((Nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (Ny + threadsPerBlock.y - 1) / threadsPerBlock.y,
                 (Nz + threadsPerBlock.z - 1) / threadsPerBlock.z);
  compute_d_psi2<<<numBlocks, threadsPerBlock>>>(d_psi, d_psi2);
  return;
}

__global__ void compute_d_psi2(const double *__restrict__ d_psi,
                               double *__restrict__ d_psi2) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;
  int idz = blockIdx.z * blockDim.z + threadIdx.z;

  if (idx >= d_Nx || idy >= d_Ny || idz >= d_Nz)
    return;

  int linear_idx = idz * d_Ny * d_Nx + idy * d_Nx + idx;

  double psi_val = d_psi[linear_idx];
  d_psi2[linear_idx] = psi_val * psi_val;
}

void initpsi(double *psi, MultiArray<double> &x2, MultiArray<double> &y2, MultiArray<double> &z2) {
  long cnti, cntj, cntk;
  double cpsi;
  double tmp;
  cpsi = sqrt(2. * pi * sqrt(2. * pi) * sx * sy * sz);
  double val;
  for (cnti = 0; cnti < Nx; cnti++) {
    val = (cnti - Nx2) * dx;
    // x[cnti] = val;
    x2[cnti] = val * val;
  }

  for (cntj = 0; cntj < Ny; cntj++) {
    val = (cntj - Ny2) * dy;
    // y[cntj] = val;
    y2[cntj] = val * val;
  }

  for (cntk = 0; cntk < Nz; cntk++) {
    val = (cntk - Nz2) * dz;
    // z[cntk] = val;
    z2[cntk] = val * val;
  }

  for (cntk = 0; cntk < Nz; cntk++) {
    for (cntj = 0; cntj < Ny; cntj++) {
      for (cnti = 0; cnti < Nx; cnti++) {
        tmp = exp(-0.25 * (x2[cnti] / (sx * sx) + y2[cntj] / (sy * sy) +
                           z2[cntk] / (sz * sz)));
        psi[cntk * Ny * Nx + cntj * Nx + cnti] = tmp / cpsi;
      }
    }
  }
  return;
}

void initpot(MultiArray<double> &pot, MultiArray<double> &x2, MultiArray<double> &y2, MultiArray<double> &z2) {
  long cnti, cntj, cntk;
  double vgamma2 = vgamma * vgamma;
  double vnu2 = vnu * vnu;
  double vlambda2 = vlambda * vlambda;
  double val;
  for (cntk = 0; cntk < Nz; cntk++) {
    for (cntj = 0; cntj < Ny; cntj++) {
      for (cnti = 0; cnti < Nx; cnti++) {
        val = 0.5 * par *
              (vgamma2 * x2[cnti] + vnu2 * y2[cntj] + vlambda2 * z2[cntk]);
        pot(cntk, cntj, cnti) = val;
      }
    }
  }
  return;
}

/**
 *    Initialization of the dipolar potential.
 *    kx  - array with the space mesh values in the x-direction in the K-space
 *    ky  - array with the space mesh values in the y-direction in the K-space
 *    kz  - array with the space mesh values in the z-direction in the K-space
 *    kx2 - array with the squared space mesh values in the x-direction in the
 *          K-space
 *    ky2 - array with the squared space mesh values in the y-direction in the
 *          K-space
 *    kz2 - array with the squared space mesh values in the z-direction in the
 *          K-space
 */
 void initpotdd(MultiArray<double> &potdd, MultiArray<double> &kx, MultiArray<double> &ky, MultiArray<double> &kz, MultiArray<double> &kx2, MultiArray<double> &ky2, MultiArray<double> &kz2) {
  long cnti, cntj, cntk;
  double dkx, dky, dkz, xk, tmp;

  dkx = 2. * pi / (Nx * dx);
  dky = 2. * pi / (Ny * dy);
  dkz = 2. * pi / (Nz * dz);

  for (cnti = 0; cnti < Nx2; cnti ++) kx[cnti] = cnti * dkx;
  for (cnti = 0; cnti < Nx2; cnti ++) kx[cnti + Nx2] = (cnti - Nx2) * dkx;
  for (cntj = 0; cntj < Ny2; cntj ++) ky[cntj] = cntj * dky;
  for (cntj = 0; cntj < Ny2; cntj ++) ky[cntj + Ny2] = (cntj - Ny2) * dky;
  for (cntk = 0; cntk < Nz2; cntk ++) kz[cntk] = cntk * dkz;
  for (cntk = 0; cntk < Nz2; cntk ++) kz[cntk + Nz2] = (cntk - Nz2) * dkz;

  for (cnti = 0; cnti < Nx; cnti ++) kx2[cnti] = kx[cnti] * kx[cnti];
  for (cntj = 0; cntj < Ny; cntj ++) ky2[cntj] = ky[cntj] * ky[cntj];
  for (cntk = 0; cntk < Nz; cntk ++) kz2[cntk] = kz[cntk] * kz[cntk];



  for (cntk = 0; cntk < Nz; cntk ++) {
     for (cntj = 0; cntj < Ny; cntj ++) {
        for (cnti = 0; cnti < Nx; cnti ++) {

           xk = sqrt(kz2[cntk] + kx2[cnti] + ky2[cntj]);
           tmp = 1. + 3. * cos(xk * cutoff) / (xk * xk * cutoff * cutoff) - 3. * sin(xk * cutoff) / (xk * xk * xk * cutoff * cutoff * cutoff);
           potdd(cntk, cntj, cnti) = (4. * pi * (3. * kz2[cntk] / (kx2[cnti] + ky2[cntj] + kz2[cntk]) - 1.) / 3.) * tmp;
           
          }
     }
  }
  potdd(0,0,0) = 0.;
 

  return;
}

// Calculation of the wave function norm and normalization on device
void calcnorm(double *d_psi, double *d_psi2, double &norm,
              Simpson3DTiledIntegrator &integ) {
  calc_d_psi2(d_psi, d_psi2);
  double raw_norm =
      integ.integrateDevice(dx, dy, dz, d_psi2, Nx, Ny, Nz);
  norm = 1.0 / sqrt(raw_norm);

  // Apply normalization
  dim3 threadsPerBlock(8, 8, 8);
  dim3 numBlocks((Nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (Ny + threadsPerBlock.y - 1) / threadsPerBlock.y,
                 (Nz + threadsPerBlock.z - 1) / threadsPerBlock.z);

  multiply_by_norm<<<numBlocks, threadsPerBlock>>>(d_psi, norm);
  cudaDeviceSynchronize(); // Ensure completion
}

// Multiplication of the wave function by the norm on device
__global__ void multiply_by_norm(double *__restrict__ d_psi, const double norm) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;
  int idz = blockIdx.z * blockDim.z + threadIdx.z;

  if (idx >= d_Nx || idy >= d_Ny || idz >= d_Nz)
    return;

  long linear_idx = idz * d_Ny * d_Nx + idy * d_Nx + idx;
  d_psi[linear_idx] *= norm;
}

__global__ void compute_psid2_potdd(cufftDoubleComplex * d_psi2_fft, 
  const double* __restrict__ potdd) {
  int grid_stride_z = gridDim.z * blockDim.z;
  int grid_stride_y = gridDim.y * blockDim.y;
  int grid_stride_x = gridDim.x * blockDim.x;
  int tdz=blockIdx.z * blockDim.z + threadIdx.z;
  int tdy=blockIdx.y * blockDim.y + threadIdx.y;
  int tdx=blockIdx.x * blockDim.x + threadIdx.x;
  
  // Grid-stride loop implementation
  for (int idz = tdz; idz < d_Nz; idz += grid_stride_z) {
    for (int idy = tdy; idy < d_Ny; idy += grid_stride_y) {
      for (int idx = tdx; idx < d_Nx/2+1; idx += grid_stride_x) {

  // Calculate linear index for FFT array (R2C format)
        int fft_idx = idz * d_Ny * (d_Nx/2 + 1) + idy * (d_Nx/2 + 1) + idx;
        int pot_idx = idz * d_Ny * d_Nx + idy * d_Nx + idx;

        double val = __ldg(&potdd[pot_idx]); // Read-only cache for dipolar potential
        d_psi2_fft[fft_idx].x *= val;
        d_psi2_fft[fft_idx].y *= val;
      }
    }
  }
}

__global__ void calcpsidd2_boundaries(double *psidd2) {
  long cnti, cntj, cntk;

  int grid_stride_y = gridDim.y * blockDim.y;
  int grid_stride_x = gridDim.x * blockDim.x;

  int tdy=blockIdx.y * blockDim.y + threadIdx.y;
  int tdx=blockIdx.x * blockDim.x + threadIdx.x;
  // Memory layout: fastest changing = x, slowest = z
  // Index calculation: idx = k * (d_Nx * d_Ny) + j * d_Nx + i
  
  // First loop: Copy from first x-slice to last x-slice
  // (j=0 to j=d_Ny-1, k=0 to k=d_Nz-1)
  for (cntj = tdy; cntj < d_Ny; cntj += grid_stride_y) {
     for (cntk = tdx; cntk < d_Nz; cntk += grid_stride_x) {
        long first_idx = cntk * (d_Nx * d_Ny) + cntj * d_Nx + 0;           // i=0 (first x-slice)
        long last_idx = cntk * (d_Nx * d_Ny) + cntj * d_Nx + (d_Nx - 1);   // i=d_Nx-1 (last x-slice)
        
        psidd2[last_idx] = psidd2[first_idx];
     }
  }

  // Second loop: Copy from first y-slice to last y-slice  
  // (i=0 to i=d_Nx-1, k=0 to k=d_Nz-1)
  for (cnti = tdy; cnti < d_Nx; cnti += grid_stride_y) {
     for (cntk = tdx; cntk < d_Nz; cntk += grid_stride_x) {
        long first_idx = cntk * (d_Nx * d_Ny) + 0 * d_Nx + cnti;           // j=0 (first y-slice)
        long last_idx = cntk * (d_Nx * d_Ny) + (d_Ny - 1) * d_Nx + cnti;   // j=d_Ny-1 (last y-slice)
        
        psidd2[last_idx] = psidd2[first_idx];
     }
  }

  // Third loop: Copy from first z-slice to last z-slice
  // (i=0 to i=d_Nx-1, j=0 to j=d_Ny-1)
  for (cnti = tdy; cnti < d_Nx; cnti += grid_stride_y) {
     for (cntj = tdx; cntj < d_Ny; cntj += grid_stride_x) {
        long first_idx = 0 * (d_Nx * d_Ny) + cntj * d_Nx + cnti;           // k=0 (first z-slice)
        long last_idx = (d_Nz - 1) * (d_Nx * d_Ny) + cntj * d_Nx + cnti;   // k=d_Nz-1 (last z-slice)
        
        psidd2[last_idx] = psidd2[first_idx];
     }
  }
}

void calc_psid2_potdd(cufftHandle forward_plan, cufftHandle backward_plan, const double* d_psi, double* d_psi2_real, cufftDoubleComplex * d_psi2_fft,const double* potdd) {
  calc_d_psi2(d_psi, d_psi2_real);
  cufftExecD2Z(forward_plan, (cufftDoubleReal*)d_psi2_real, d_psi2_fft);
  
  dim3 threadsPerBlock(8, 8, 8);
  dim3 numBlocks((Nx/2 + 1 + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (Ny + threadsPerBlock.y - 1) / threadsPerBlock.y,
                 (Nz + threadsPerBlock.z - 1) / threadsPerBlock.z);
  compute_psid2_potdd<<<numBlocks, threadsPerBlock>>>(d_psi2_fft, potdd);

  cufftExecZ2D(backward_plan, d_psi2_fft, (cufftDoubleReal*)d_psi2_real);
  calcpsidd2_boundaries<<<numBlocks, threadsPerBlock>>>(d_psi2_real);
  return;
}

/**
 *    Crank-Nicolson scheme coefficients generation.
 */
void gencoef(MultiArray<double> &calphax, MultiArray<double> &cgammax, MultiArray<double> &calphay,
             MultiArray<double> &cgammay, MultiArray<double> &calphaz, MultiArray<double> &cgammaz,
             double &Ax0, double &Ay0, double &Az0, double &Ax0r, double &Ay0r,
             double &Az0r, double &Ax, double &Ay, double &Az) {
  long cnti;

  Ax0 = 1. + dt / dx2 / (3. - par);
  Ay0 = 1. + dt / dy2 / (3. - par);
  Az0 = 1. + dt / dz2 / (3. - par);

  Ax0r = 1. - dt / dx2 / (3. - par);
  Ay0r = 1. - dt / dy2 / (3. - par);
  Az0r = 1. - dt / dz2 / (3. - par);

  Ax = -0.5 * dt / dx2 / (3. - par);
  Ay = -0.5 * dt / dy2 / (3. - par);
  Az = -0.5 * dt / dz2 / (3. - par);

  calphax[Nx - 2] = 0.;
  cgammax[Nx - 2] = -1. / Ax0;
  for (cnti = Nx - 2; cnti > 0; cnti--) {
    calphax[cnti - 1] = Ax * cgammax[cnti];
    cgammax[cnti - 1] = -1. / (Ax0 + Ax * calphax[cnti - 1]);
  }

  calphay[Ny - 2] = 0.;
  cgammay[Ny - 2] = -1. / Ay0;
  for (cnti = Ny - 2; cnti > 0; cnti--) {
    calphay[cnti - 1] = Ay * cgammay[cnti];
    cgammay[cnti - 1] = -1. / (Ay0 + Ay * calphay[cnti - 1]);
  }

  calphaz[Nz - 2] = 0.;
  cgammaz[Nz - 2] = -1. / Az0;
  for (cnti = Nz - 2; cnti > 0; cnti--) {
    calphaz[cnti - 1] = Az * cgammaz[cnti];
    cgammaz[cnti - 1] = -1. / (Az0 + Az * calphaz[cnti - 1]);
  }

  return;
}

/**
 *    Time propagation with respect to H1 (part of the Hamiltonian without
 *    spatial derivatives).
 *    psi    - array with the wave function values
 *
 */

void calcnu(double *d_psi, double *d_psi2, double *d_pot, double g,
            double gd) {
  //calc_d_psi2(d_psi, d_psi2);

  dim3 threadsPerBlock(8, 8, 8);
  dim3 numBlocks((Nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (Ny + threadsPerBlock.y - 1) / threadsPerBlock.y,
                 (Nz + threadsPerBlock.z - 1) / threadsPerBlock.z);
  calcnu_kernel<<<numBlocks, threadsPerBlock>>>(
      d_psi, d_psi2, d_pot, g, gd);
  cudaDeviceSynchronize();
  return;
}

__global__ void calcnu_kernel(double *__restrict__ d_psi,
                              double *__restrict__ d_psi2,
                              const double *__restrict__ d_pot, const double g, const double gd) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;
  int idz = blockIdx.z * blockDim.z + threadIdx.z;
  
  if (idx >= d_Nx || idy >= d_Ny || idz >= d_Nz)
    return;

  int linear_idx = idz * d_Ny * d_Nx + idy * d_Nx + idx;
  double psi_val = d_psi[linear_idx];
  double psi2dd = __ldg(&d_psi2[linear_idx])/((double)(d_Nx * d_Ny * d_Nz)) * gd; // Read-only cache for psi2dd
  double tmp = d_dt * ( psi_val * psi_val * g+ psi2dd + __ldg(&d_pot[linear_idx])); // Read-only cache for potential
  //double tmp = dt * (psi2dd);
  d_psi[linear_idx] *= exp(-tmp);
}

/**
 *    Time propagation with respect to H2 (x-part of the Laplacian).
 *    psi   - array with the wave function values
 *    cbeta - Crank-Nicolson scheme coefficients
 */
void calclux(double *d_psi, double *d_cbeta, double *d_calphax,
             double *d_cgammax, double Ax0r, double Ax) {

  dim3 threadsPerBlock(16, 16); // 2D blocks for y-z planes
  dim3 numBlocks((Nz + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (Ny + threadsPerBlock.y - 1) / threadsPerBlock.y);

  calclux_kernel<<<numBlocks, threadsPerBlock>>>(
      d_psi, d_cbeta, d_calphax, d_cgammax, Ax0r, Ax);

  //cudaDeviceSynchronize();
  return;
}

__global__ void calclux_kernel(double *__restrict__ psi,
                               double *__restrict__ cbeta,
                               const double *__restrict__ calphax,
                               const double *__restrict__ cgammax,
                               const double Ax0r, const double Ax
                               ) {

  int cntj = blockIdx.y * blockDim.y + threadIdx.y;
  int cntk = blockIdx.x * blockDim.x + threadIdx.x;

  if (cntj >= d_Ny || cntk >= d_Nz)
    return;

  // Base offset for this (j,k) y-z position
  const long base_offset = cntk * d_Ny * d_Nx + cntj * d_Nx;

  // Forward elimination: fill cbeta array
  // Boundary condition: cbeta[nx-2] = psi[nx-1]
  cbeta[base_offset + d_Nx - 2] = psi[base_offset + d_Nx - 1];

  // Thomas algorithm forward sweep
  for (int cnti = d_Nx - 2; cnti > 0; cnti--) {
    double c = -Ax * psi[base_offset + cnti + 1] +
               Ax0r * psi[base_offset + cnti] -
               Ax * psi[base_offset + cnti - 1]; 
    cbeta[base_offset + cnti - 1] =
        __ldg(&cgammax[cnti]) * (Ax * cbeta[base_offset + cnti] - c); // Read-only cache for cgamma
  }

  // Boundary condition
  psi[base_offset + 0] = 0.0;

  // Back substitution: update psi values
  for (int cnti = 0; cnti < d_Nx - 2; cnti++) {
    psi[base_offset + cnti + 1] =
        fma(__ldg(&calphax[cnti]), psi[base_offset + cnti], cbeta[base_offset + cnti]); // Read-only cache for calpha
  }

  // Boundary condition
  psi[base_offset + d_Nx - 1] = 0.0;
}

/**
 *    Time propagation with respect to H3 (y-part of the Laplacian).
 *    d_psi   - device array with the wave function values
 *    d_cbeta - device array for Crank-Nicolson temporary values
 *    d_calphay - device array with alpha coefficients for y-direction
 *    d_cgammay - device array with gamma coefficients for y-direction
 *    Ay0r - diagonal coefficient for right-hand side
 *    Ay - off-diagonal coefficient
 */
void calcluy(double *d_psi, double *d_cbeta, double *d_calphay,
             double *d_cgammay, double Ay0r, double Ay) {

  dim3 threadsPerBlock(16, 16); // 2D blocks for x-z planes
  dim3 numBlocks((Nz + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (Nx + threadsPerBlock.y - 1) / threadsPerBlock.y);

  calcluy_kernel<<<numBlocks, threadsPerBlock>>>(
      d_psi, d_cbeta, d_calphay, d_cgammay, Ay0r,
      Ay);

  cudaDeviceSynchronize();
  return;
}

__global__ void calcluy_kernel(double *__restrict__ psi,
                               double *__restrict__ cbeta,
                               const double *__restrict__ calphay,
                               const double *__restrict__ cgammay,
                               const double Ay0r, const double Ay
                               ) {

  int cnti = blockIdx.y * blockDim.y + threadIdx.y;
  int cntk = blockIdx.x * blockDim.x + threadIdx.x;

  if (cnti >= d_Nx || cntk >= d_Nz)
    return;

  // Forward elimination: fill cbeta array
  // Boundary condition: cbeta[ny-2] = psi[ny-1]
  long idx = cntk * d_Ny * d_Nx + (d_Ny - 2) * d_Nx + cnti;
  long psi_idx = cntk * d_Ny * d_Nx + (d_Ny - 1) * d_Nx + cnti;
  cbeta[idx] = psi[psi_idx];

  // Thomas algorithm forward sweep (working in y-direction)
  for (int cntj = d_Ny - 2; cntj > 0; cntj--) {
    long base_offset = cntk * d_Ny * d_Nx + cnti;

    double c = -Ay * psi[base_offset + (cntj + 1) * d_Nx] +
               Ay0r * psi[base_offset + cntj * d_Nx] -
               Ay * psi[base_offset + (cntj - 1) * d_Nx];

    cbeta[base_offset + (cntj - 1) * d_Nx] =
        __ldg(&cgammay[cntj]) * (Ay * cbeta[base_offset + cntj * d_Nx] - c); // Read-only cache for cgamma
  }

  // Boundary condition
  psi[cntk * d_Ny * d_Nx + 0 * d_Nx + cnti] = 0.0;

  // Back substitution: update psi values
  for (int cntj = 0; cntj < d_Ny - 2; cntj++) {
    long base_offset = cntk * d_Ny * d_Nx + cnti;

    psi[base_offset + (cntj + 1) * d_Nx] =
        fma(__ldg(&calphay[cntj]), psi[base_offset + cntj * d_Nx], // Read-only cache for calpha
            cbeta[base_offset + cntj * d_Nx]);
  }

  // Boundary condition
  psi[cntk * d_Ny * d_Nx + (d_Ny - 1) * d_Nx + cnti] = 0.0;
}

/**
 *    Time propagation with respect to H4 (z-part of the Laplacian).
 *    d_psi   - device array with the wave function values
 *    d_cbeta - device array for Crank-Nicolson temporary values
 *    d_calphaz - device array with alpha coefficients for z-direction
 *    d_cgammaz - device array with gamma coefficients for z-direction
 *    Az0r - diagonal coefficient for right-hand side
 *    Az - off-diagonal coefficient
 */
void calcluz(double *d_psi, double *d_cbeta, double *d_calphaz,
             double *d_cgammaz, double Az0r, double Az) {

  dim3 threadsPerBlock(16, 16); // 2D blocks for x-y planes
  dim3 numBlocks((Ny + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (Nx + threadsPerBlock.y - 1) / threadsPerBlock.y);

  calcluz_kernel<<<numBlocks, threadsPerBlock>>>(
      d_psi, d_cbeta, d_calphaz, d_cgammaz, Az0r, Az);

  cudaDeviceSynchronize();
  return;
}

__global__ void calcluz_kernel(double *__restrict__ psi,
                               double *__restrict__ cbeta,
                               const double *__restrict__ calphaz,
                               const double *__restrict__ cgammaz,
                               const double Az0r, const double Az
                               ) {

  int cntj = blockIdx.x * blockDim.x + threadIdx.x;
  int cnti = blockIdx.y * blockDim.y + threadIdx.y;

  if (cnti >= d_Nx || cntj >= d_Ny)
    return;

  // Base offset for this (i,j) x-y position - points to z=0
  const long base_offset = cntj * d_Nx + cnti;
  const long stride = d_Ny * d_Nx; // stride to move in z-direction

  // Forward elimination: fill cbeta array
  // Boundary condition: cbeta[nz-2] = psi[nz-1]
  cbeta[base_offset + (d_Nz - 2) * stride] = psi[base_offset + (d_Nz - 1) * stride];

  // Thomas algorithm forward sweep (working in z-direction)
  for (int cntk = d_Nz - 2; cntk > 0; cntk--) {
    double c = -Az * psi[base_offset + (cntk + 1) * stride] +
               Az0r * psi[base_offset + cntk * stride] -
               Az * psi[base_offset + (cntk - 1) * stride];

    cbeta[base_offset + (cntk - 1) * stride] =
        __ldg(&cgammaz[cntk]) * (Az * cbeta[base_offset + cntk * stride] - c); // Read-only cache for cgamma
  }

  // Boundary condition
  psi[base_offset + 0 * stride] = 0.0;

  // Back substitution: update psi values
  for (int cntk = 0; cntk < d_Nz - 2; cntk++) {
    psi[base_offset + (cntk + 1) * stride] =
        fma(__ldg(&calphaz[cntk]), psi[base_offset + cntk * stride], // Read-only cache for calpha
            cbeta[base_offset + cntk * stride]);
  }

  // Boundary condition
  psi[base_offset + (d_Nz - 1) * stride] = 0.0;
}

// HIGHLY OPTIMIZED VERSION: Uses fused kernels to eliminate redundant calculations
void calcmuen(double *muen,double *d_psi, double *d_psi2, double *d_pot, double *d_psi2dd, double *d_potdd, cufftDoubleComplex * d_psi2_fft, cufftHandle forward_plan, cufftHandle backward_plan, Simpson3DTiledIntegrator &integ, const double g, const double gd){
  
  dim3 threadsPerBlock(8, 8, 8);
  dim3 numBlocks((Nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (Ny + threadsPerBlock.y - 1) / threadsPerBlock.y,
                 (Nz + threadsPerBlock.z - 1) / threadsPerBlock.z);

  // Step 1: Contact energy - Calculate ψ² and immediately compute 0.5 * g * ψ⁴
  calcmuen_fused_contact<<<numBlocks, threadsPerBlock>>>(d_psi, d_psi2, g);
  muen[0] = integ.integrateDevice(dx, dy, dz, d_psi2, Nx, Ny, Nz);

  // Step 2: Potential energy - Calculate ψ² and 0.5 * ψ² * V in one kernel
  calcmuen_fused_potential<<<numBlocks, threadsPerBlock>>>(d_psi, d_psi2, d_pot);
  muen[1] = integ.integrateDevice(dx, dy, dz, d_psi2, Nx, Ny, Nz);
  
  // Step 3: Dipolar energy - requires FFT computation first
  calc_psid2_potdd(forward_plan, backward_plan, d_psi, d_psi2dd, d_psi2_fft, d_potdd);
  calcmuen_fused_dipolar<<<numBlocks, threadsPerBlock>>>(d_psi, d_psi2, d_psi2dd, gd);
  muen[2] = integ.integrateDevice(dx, dy, dz, d_psi2, Nx, Ny, Nz)/((double)Nx * Ny * Nz);

  // Step 4: Kinetic energy - calculate gradients and kinetic energy density directly
  calcmuen_kin(d_psi, d_psi2, par);
  muen[3] = integ.integrateDevice(dx, dy, dz, d_psi2, Nx, Ny, Nz);

  return;
}
// FUSED KERNEL: Calculate ψ² and contact energy in one pass
__global__ void calcmuen_fused_contact(const double *__restrict__ d_psi, double *__restrict__ d_result, double g){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;
  int idz = blockIdx.z * blockDim.z + threadIdx.z;

  if (idx >= d_Nx || idy >= d_Ny || idz >= d_Nz)
    return;

  int linear_idx = idz * d_Ny * d_Nx + idy * d_Nx + idx;
  double psi_val = d_psi[linear_idx];
  double psi2_val = psi_val * psi_val;
  double psi4_val = psi2_val * psi2_val;
  d_result[linear_idx] = 0.5 * psi4_val * g;
}

// FUSED KERNEL: Calculate ψ² and potential energy in one pass
__global__ void calcmuen_fused_potential(const double *__restrict__ d_psi, double *__restrict__ d_result, const double *__restrict__ d_pot){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;
  int idz = blockIdx.z * blockDim.z + threadIdx.z;

  if (idx >= d_Nx || idy >= d_Ny || idz >= d_Nz)
    return;

  int linear_idx = idz * d_Ny * d_Nx + idy * d_Nx + idx;
  double psi_val = __ldg(&d_psi[linear_idx]); // Read-only cache for psi
  double psi2_val = psi_val * psi_val;
  d_result[linear_idx] = 0.5 * psi2_val * __ldg(&d_pot[linear_idx]); // Read-only cache for potential
}

// FUSED KERNEL: Calculate ψ² and dipolar energy in one pass
__global__ void calcmuen_fused_dipolar(const double *__restrict__ d_psi, double *__restrict__ d_result, const double *__restrict__ d_psidd2, const double gd){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;
  int idz = blockIdx.z * blockDim.z + threadIdx.z;

  if (idx >= d_Nx || idy >= d_Ny || idz >= d_Nz)
    return;

  int linear_idx = idz * d_Ny * d_Nx + idy * d_Nx + idx;
  double psi_val = __ldg(&d_psi[linear_idx]); // Read-only cache for psi
  double psi2_val = psi_val * psi_val;
  double psidd2_val = __ldg(&d_psidd2[linear_idx]); // Read-only cache for dipolar psi squared
  d_result[linear_idx] = 0.5 * psi2_val * psidd2_val * gd;
}

void calcmuen_kin(double *d_psi, double *d_work_array, int par){
  diff(dx, dy, dz, d_psi, d_work_array, Nx, Ny, Nz, par);
}

void rms_output(FILE *filerms){
  fprintf(filerms, "\n**********************************************\n");
  if (cfg_read("G") != NULL) {
    fprintf(filerms, "Contact: G = %.6le, G * par = %.6le\n", g / par, g);
  } else {
    fprintf(filerms, "Contact: Natoms = %.11le, as = %.6le * a0, G = %.6le, G * par = %.6le\n", Nad, as, g / par, g);
  }
  if(optms == 0) {
    fprintf(filerms, "Regular ");
  } else {
    fprintf(filerms, "Microwave-shielded ");
  }
  if (cfg_read("GDD") != NULL) {
    fprintf(filerms, "DDI: GD = %.6le, GD * par = %.6le, edd = %.6le\n", gd / par, gd, edd);
  } else {
    fprintf(filerms, "DDI: add = %.6le * a0, GD = %.6le, GD * par = %.6le, edd = %.6le\n", add, gd / par, gd, edd);
  }
  fprintf(filerms, "     Dipolar cutoff Scut = %.6le,\n\n",cutoff);
  if (QF == 1) {
    fprintf(filerms, "QF = 1: h2 = %.6le, h4 = %.6le\n        q3 = %.6le, q5 = %.6le\n\n", h2, h4, q3, q5);
  } else  fprintf(filerms, "QF = 0\n\n");
  fprintf(filerms, "Trap parameters:\nGAMMA = %.6le, NU = %.6le, LAMBDA = %.6le\n", vgamma, vnu, vlambda);
  fprintf(filerms, "Space discretization: NX = %li, NY = %li, NZ = %li\n", Nx, Ny, Nz);
  fprintf(filerms, "                      DX = %.16le, DY = %.16le, DZ = %.16le, mx = %.2le, my = %.2le, mz = %.2le\n", dx, dy, dz, mx, my, mz);
  if (cfg_read("AHO") != NULL) fprintf(filerms, "      Unit of length: aho = %.6le m\n", aho);
  fprintf(filerms, "\nTime discretization: NITER = %li (NSNAP = %li)\n", Niter, Nsnap);
  fprintf(filerms, "                     DT = %.6le, mt = %.2le\n\n",  dt, mt);
  if (input != NULL) {
    fprintf(filerms, "file %s\n", input);
  } else {
    fprintf(filerms, "Gaussian\n               SX = %.6le, SY = %.6le, SZ = %.6le\n", sx, sy, sz);
  }
  fprintf(filerms, "MUREL = %.6le, MUEND=%.6le\n\n", murel, muend);
  fprintf(filerms, "-------------------------------------------------------------------\n");
  fprintf(filerms, "Snap      <r>            <x>            <y>            <z>\n");
  fprintf(filerms, "-------------------------------------------------------------------\n");
  fflush(filerms);
  
  
}
void mu_output(FILE *filemu){
  fprintf(filemu, "\n**********************************************\n");
  if (cfg_read("G") != NULL) {
    fprintf(filemu, "Contact: G = %.6le, G * par = %.6le\n", g / par, g);
  } else {
    fprintf(filemu, "Contact: Natoms = %.11le, as = %.6le * a0, G = %.6le, G * par = %.6le\n", Nad, as, g / par, g);
  }
  if(optms == 0) {
    fprintf(filemu, "Regular ");
  } else {
    fprintf(filemu, "Microwave-shielded ");
  }
  if (cfg_read("GDD") != NULL) {
    fprintf(filemu, "DDI: GD = %.6le, GD * par = %.6le, edd = %.6le\n", gd / par, gd, edd);
  } else {
    fprintf(filemu, "DDI: add = %.6le * a0, GD = %.6le, GD * par = %.6le, edd = %.6le\n", add, gd / par, gd, edd);
  }
  fprintf(filemu, "     Dipolar cutoff Scut = %.6le,\n\n",cutoff);
  if (QF == 1) {
    fprintf(filemu, "QF = 1: h2 = %.6le, h4 = %.6le\n        q3 = %.6le, q5 = %.6le\n\n", h2, h4, q3, q5);
  } else  fprintf(filemu, "QF = 0\n\n");
  fprintf(filemu, "Trap parameters:\nGAMMA = %.6le, NU = %.6le, LAMBDA = %.6le\n", vgamma, vnu, vlambda);
  fprintf(filemu, "Space discretization: NX = %li, NY = %li, NZ = %li\n", Nx, Ny, Nz);
  fprintf(filemu, "                      DX = %.16le, DY = %.16le, DZ = %.16le, mx = %.2le, my = %.2le, mz = %.2le\n", dx, dy, dz, mx, my, mz);
  if (cfg_read("AHO") != NULL) fprintf(filemu, "      Unit of length: aho = %.6le m\n", aho);
  fprintf(filemu, "\nTime discretization: NITER = %li (NSNAP = %li)\n", Niter, Nsnap);
  fprintf(filemu, "                     DT = %.6le, mt = %.2le\n\n",  dt, mt);
  if (input != NULL) {
    fprintf(filemu, "file %s\n", input);
  } else {
    fprintf(filemu, "Gaussian\n               SX = %.6le, SY = %.6le, SZ = %.6le\n", sx, sy, sz);
  }
  fprintf(filemu, "MUREL = %.6le, MUEND=%.6le\n\n", murel, muend);
  fprintf(filemu, "---------------------------------------------------------------------------------\n");
  fprintf(filemu, "Snap      mu           Kin             Pot            Contact            DDI\n");
  fprintf(filemu, "---------------------------------------------------------------------------------\n");
  fflush(filemu);

}

void save_psi_from_gpu(double *psi, double *d_psi, const char *filename, long Nx, long Ny, long Nz) {
  // Allocate host memory
  size_t total_size = Nx * Ny * Nz;

  // Copy from device to host
  cudaMemcpy(psi, d_psi, total_size * sizeof(double), cudaMemcpyDeviceToHost);
  
  // Check for CUDA errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
      fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
      return;
  }
  
  // Write to binary file
  FILE *file = fopen(filename, "wb");
  if (file == NULL) {
      fprintf(stderr, "Failed to open file %s\n", filename);
      return;
  }
  
  // Write the entire array
  size_t written = fwrite(psi, sizeof(double), total_size, file);
  if (written != total_size) {
      fprintf(stderr, "Failed to write all data: wrote %zu of %zu elements\n", 
              written, total_size);
  }
  
  fclose(file);
}

void read_psi_from_file(double *psi, const char *filename, long Nx, long Ny, long Nz) {
  size_t total_size = Nx * Ny * Nz;

  
  // Open file
  FILE *file = fopen(filename, "rb");  // Note: "rb" for binary read
  if (file == NULL) {
      fprintf(stderr, "Failed to open file %s\n", filename);
      free(psi);     
  }
  
  // Read data
  size_t read_count = fread(psi, sizeof(double), total_size, file);
  if (read_count != total_size) {
      fprintf(stderr, "Failed to read all data: read %zu of %zu elements\n", 
              read_count, total_size);
      fclose(file);
  }
  fclose(file);
}