#include "imag3d-cuda.cuh"
#include "../utils/cuda_error_check.cuh"

// Use constant memory for tridiagonal coefficients when sizes are modest.
#define CGALPHA_MAX 1024
__constant__ double cgammax_c[CGALPHA_MAX];
__constant__ double calphax_c[CGALPHA_MAX];
__constant__ double cgammay_c[CGALPHA_MAX];
__constant__ double calphay_c[CGALPHA_MAX];
__constant__ double cgammaz_c[CGALPHA_MAX];
__constant__ double calphaz_c[CGALPHA_MAX];

/**
 * @brief Simple helper function to get GPU SM count
 * @return Number of streaming multiprocessors on the current GPU
 */
inline int getGPUSMCount() {
    int smCount;
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device));
    return smCount;
}

/**
 * @brief Calculate optimal grid size for 3D kernels with grid-stride loops
 * @param smCount: Number of SMs on GPU
 * @param Nx, Ny, Nz: Problem dimensions
 * @param blockSize: Block dimensions
 * @param blocksPerSM: Minimum blocks per SM for occupancy 
 * @return dim3 grid size
 */
inline dim3 getOptimalGrid3D(int smCount, long Nx, long Ny, long Nz, dim3 blockSize, int blocksPerSM = 2) {
    // Calculate grid needed to cover the problem with some stride
    int stride = 1;
    int gridX = (Nx + blockSize.x * stride - 1) / (blockSize.x * stride);  // with stride
    int gridY = (Ny + blockSize.y * stride - 1) / (blockSize.y * stride);
    int gridZ = (Nz + blockSize.z * stride - 1) / (blockSize.z * stride);
    
    // Ensure minimum blocks for occupancy
    int minBlocks = smCount * blocksPerSM;
    int totalBlocks = gridX * gridY * gridZ;
    
    if (totalBlocks < minBlocks) {
        // Scale up if too few blocks
        double scale = sqrt((double)minBlocks / totalBlocks);
        gridX = (int)(gridX * scale) + 1;
        gridY = (int)(gridY * scale) + 1;
    }
    
    return dim3(gridX, gridY, gridZ);
}

/**
 * @brief Calculate optimal grid size for 2D kernels with grid-stride loops
 * @param smCount: Number of SMs on GPU
 * @param Nx, Ny: Problem dimensions for the 2D sweep
 * @param blockSize: Block dimensions
 * @param blocksPerSM: Minimum blocks per SM for occupancy
 * @return dim3 grid size
 */
inline dim3 getOptimalGrid2D(int smCount, long Nx, long Ny, dim3 blockSize, int blocksPerSM = 2) {
    // Calculate grid needed to cover the problem with some stride
    int stride = 1;
    int gridX = (Nx + blockSize.x * stride - 1) / (blockSize.x * stride);  //with stride
    int gridY = (Ny + blockSize.y * stride - 1) / (blockSize.y * stride);
    
    // Ensure minimum blocks for occupancy
    int minBlocks = smCount * blocksPerSM;
    int totalBlocks = gridX * gridY;
    
    if (totalBlocks < minBlocks) {
        double scale = sqrt((double)minBlocks / totalBlocks);
        gridX = (int)(gridX * scale) + 1;
        gridY = (int)(gridY * scale) + 1;
    }
    
    return dim3(gridX, gridY, 1);
}

int main(int argc, char **argv) 
{
    if ((argc != 3) || (strcmp(*(argv + 1), "-i") != 0)) 
    {
        std::fprintf(stderr, "Usage: %s -i <input parameter file> \n", *argv);
        exit(EXIT_FAILURE);
    }

    if (!cfg_init(argv[2])) 
    {
        std::fprintf(stderr, "Wrong input parameter file.\n");
        exit(EXIT_FAILURE);
    }
    pi = M_PI;
    FILE *filerms;
    FILE *filemu;
    FILE *file;
    char filename[MAX_FILENAME_SIZE];
    
    readpar();

    if (opt == 2)
        par = 2.;
    else
        par = 1.;

    g = par * g;
    gd = par * gd;
    //gd *= MS;
    edd = (4. * pi / 3.) * gd / g;
    Nad = Na;

    if (fabs(edd) < 1e-10) 
    {
        q5 = 1.;
    } 
    else 
    {
        if (fabs(edd - 1.) < 1e-10) 
        {
            q5 = 3. * sqrt(3.) / 2.;
        } 
        else 
        {
            std::complex<double> sqrt_edd = std::sqrt(std::complex<double>(edd, 0.0));
            std::complex<double> sqrt_1_plus_2edd =
                std::sqrt(std::complex<double>(1.0 + 2.0 * edd, 0.0));
            std::complex<double> sqrt_3 = std::sqrt(std::complex<double>(3.0, 0.0));

            std::complex<double> term1 = 6.0 * sqrt_1_plus_2edd * (11.0 + edd * (4.0 + 9.0 * edd));

            std::complex<double> log_term1 = std::log(std::complex<double>(1.0 - edd, 0.0));
            std::complex<double> log_term2 = std::log(-sqrt_3 * sqrt_edd + sqrt_1_plus_2edd);

            std::complex<double> term2 =
                (5.0 * sqrt_3 * std::pow(std::complex<double>(-1.0 + edd, 0.0), 3.0) *
                 (log_term1 - 2.0 * log_term2)) /
                sqrt_edd;
            q5 = (term1 - term2).real() / 96.0;
        }
    }

    q5 *= QF;
    h2 = 32. * sqrt(pi) * pow(as * BOHR_RADIUS / aho, 2.5) * pow(Nad, 1.5) * (4. * q5) / 3.;
    h2 = par * h2;

    Nx2 = Nx / 2;
    Ny2 = Ny / 2;
    Nz2 = Nz / 2;

    // Initialize squared grid spacings
    dx2 = dx * dx;
    dy2 = dy * dy;
    dz2 = dz * dz;

    // Copy constants Nx, Ny, Nz (number of grid points), dt (time step) to device
    CUDA_CHECK(cudaMemcpyToSymbol(d_Nx, &Nx, sizeof(long)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_Ny, &Ny, sizeof(long)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_Nz, &Nz, sizeof(long)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_dt, &dt, sizeof(double)));
    
    // Copy constants for on-the-fly potential calculation
    CUDA_CHECK(cudaMemcpyToSymbol(d_dx, &dx, sizeof(double)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_dy, &dy, sizeof(double)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_dz, &dz, sizeof(double)));
    double vgamma2 = vgamma * vgamma;
    double vnu2 = vnu * vnu;
    double vlambda2 = vlambda * vlambda;
    CUDA_CHECK(cudaMemcpyToSymbol(d_vgamma2, &vgamma2, sizeof(double)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_vnu2, &vnu2, sizeof(double)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_vlambda2, &vlambda2, sizeof(double)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_par, &par, sizeof(double)));

    // Get GPU SM count for optimal kernel launch configuration
    int smCount = getGPUSMCount();
    // Allocate memory for psi (pinned memory) and squared wave function (psi2)
    double *psi;
    CUDA_CHECK(cudaMallocHost(&psi, Nz * Ny * Nx * sizeof(double)));
    MultiArray<double> psi2(Nz, Ny, Nx);

    // Allocation of the wave function norm
    double norm;

    // Allocate memory for trap potential (pot) and dipole potential (potdd)
    MultiArray<double> pot(Nz, Ny, Nx);
    MultiArray<double> potdd(Nz, Ny, Nx);

    // Allocate memory for x2, y2, z2, kx, ky, kz, kx2, ky2, kz2
    MultiArray<double> x(Nx), y(Ny), z(Nz);
    MultiArray<double> x2(Nx), y2(Ny), z2(Nz);
    MultiArray<double> kx(Nx), ky(Ny), kz(Nz);
    MultiArray<double> kx2(Nx), ky2(Ny), kz2(Nz);

    // Allocate memory that will hold all chemical potential contributions
    MultiArray<double> muen(5);

    // Initialize total chemical potential of the current and next time step
    double mutotold, mutotnew;

    // Allocation of crank-nicolson coefficients
    MultiArray<double> calphax(Nx - 1), cgammax(Nx - 1);
    MultiArray<double> calphay(Ny - 1), cgammay(Ny - 1);
    MultiArray<double> calphaz(Nz - 1), cgammaz(Nz - 1);
    double Ax0, Ay0, Az0, Ax0r, Ay0r, Az0r, Ax, Ay, Az;

    // Initialize temporary arrays for density
    MultiArray<double> tmpx(Nx), tmpy(Ny), tmpz(Nz);

    // Setup Simpson3DTiledIntegrator for integration
    long TILE_SIZE = Nz/2;
    Simpson3DTiledIntegrator integ(Nx, Ny, TILE_SIZE);

    // Allocation of crank-nicolson coefficients on device
    CudaArray3D<double> d_calphax(Nx - 1);
    CudaArray3D<double> d_cgammax(Nx - 1);
    CudaArray3D<double> d_calphay(Ny - 1);
    CudaArray3D<double> d_cgammay(Ny - 1);
    CudaArray3D<double> d_calphaz(Nz - 1);
    CudaArray3D<double> d_cgammaz(Nz - 1);

    // Allocate memory for psi on device
    CudaArray3D<double> d_psi(Nx, Ny, Nz);

    // Allocate memory for work array on device
    CudaArray3D<double> d_work_array(Nx+2, Ny, Nz);

    cufftDoubleComplex* d_complex = (cufftDoubleComplex*)d_work_array.raw();

    // Allocate memory for trap potential (d_pot) and dipole potential (d_potdd)
    // d_pot is only allocated if initialize_pot == 1 (precomputed mode)
    // When initialize_pot == 0, potential is computed on-the-fly to save GPU memory
    double *d_pot_ptr = nullptr;
    if (initialize_pot == 1) 
    {
        CUDA_CHECK(cudaMalloc(&d_pot_ptr, Nx * Ny * Nz * sizeof(double)));
    }
    CudaArray3D<double> d_potdd(Nx, Ny, Nz);

    // FFT arrays - now using in-place FFT with d_work_array (d_complex)

    // Create plan for FFT of 3D array with explicit work area management
    // Using cufftMakePlanMany for in-place-like R2C FFT support
    cufftHandle forward_plan, backward_plan;
    size_t forward_worksize, backward_worksize;
    void *forward_work = nullptr;
    void *backward_work = nullptr;

    // For in-place R2C FFT, we need explicit strides
    // Memory layout: row-major, index = iz*(Ny*Nx) + iy*Nx + ix (Nx fastest, Nz slowest)
    // FFT dimensions: (Nz, Ny, Nx) - processed as Nz batches of (Ny, Nx) 2D FFTs
    int fft_size[3] = {(int)Nz, (int)Ny, (int)Nx};  // Dimensions: [slowest, middle, fastest]
    
    // Input strides for real data (row-major: iz*(Ny*Nx) + iy*Nx + ix)
    int inembed[3] = {(int)Nz, (int)Ny, (int)(Nx+2)};  // Input embedding
    int istride = 1;  // Stride within a row (contiguous)
    int idist = 1;   // Distance between z-planes (batches)
    
    // Output strides for complex data (row-major: iz*(Ny*(Nx/2+1)) + iy*(Nx/2+1) + ix)
    int onembed[3] = {(int)Nz, (int)Ny, (int)(Nx/2 + 1)};  // Output embedding
    int ostride = 1;  // Stride within a row (contiguous)
    int odist = 1;  // Distance between z-planes (batches)

    // Create forward plan (D2Z) with explicit strides for in-place-like operation
    CUFFT_CHECK(cufftCreate(&forward_plan));
    CUFFT_CHECK(cufftSetAutoAllocation(forward_plan, 0));  // Disable auto-allocation for manual work area
    CUFFT_CHECK(cufftMakePlanMany(forward_plan, 3, fft_size,
                                   inembed, istride, idist,  // Input layout
                                   onembed, ostride, odist,  // Output layout
                                   CUFFT_D2Z, 1, &forward_worksize));

    // Create backward plan (Z2D) - reverse the input/output
    CUFFT_CHECK(cufftCreate(&backward_plan));
    CUFFT_CHECK(cufftSetAutoAllocation(backward_plan, 0));  // Disable auto-allocation for manual work area
    CUFFT_CHECK(cufftMakePlanMany(backward_plan, 3, fft_size,
                                   onembed, ostride, odist,  // Input layout (complex)
                                   inembed, istride, idist,  // Output layout (real)
                                   CUFFT_Z2D, 1, &backward_worksize));

    // Allocate a single shared work area for both forward and backward plans
    {
        size_t shared_worksize =
            (forward_worksize > backward_worksize) ? forward_worksize : backward_worksize;
        if (shared_worksize > 0) 
        {
            CUDA_CHECK(cudaMalloc(&forward_work, shared_worksize));
            CUFFT_CHECK(cufftSetWorkArea(forward_plan, forward_work));
            CUFFT_CHECK(cufftSetWorkArea(backward_plan, forward_work));
        }
    }

    // Allocate pinned memory for RMS results
    double *h_rms_pinned;
    CUDA_CHECK(cudaHostAlloc(&h_rms_pinned, 3 * sizeof(double), cudaHostAllocDefault));

    // Initialize RMS output file that will store root mean square values <r>, <x>, <y>, <z>
    if (rmsout != NULL) 
    {
        sprintf(filename, "%s.txt", rmsout);
        filerms = fopen(filename, "w");
    } 
    else
    {
        filerms = NULL;
    }

    // Initialize chemical potential output file that will store chemical potential values, total
    // chemical pot., kinetic, trap, contact, dipole and quantum fluctuation terms
    // if (muoutput != NULL) 
    // {
    //     sprintf(filename, "%s.txt", muoutput);
    //     filemu = fopen(filename, "w");
    // } 
    // else
    // {
    //     filemu = NULL;
    // }

    // Initialize psi function
    initpsi(psi, x2, y2, z2, x, y, z);

    // Initialize trap potential (pot) and dipole potential (potdd)
    initpot(pot, x2, y2, z2);
    initpotdd(potdd, kx, ky, kz, kx2, ky2, kz2);

    // Generate coefficients
    gencoef(calphax, cgammax, calphay, cgammay, calphaz, cgammaz, Ax0, Ay0, Az0, Ax0r, Ay0r, Az0r,
            Ax, Ay, Az);

    // Copy coefficients to device
    d_calphax.copyFromHost(calphax.raw());
    d_cgammax.copyFromHost(cgammax.raw());
    d_calphay.copyFromHost(calphay.raw());
    d_cgammay.copyFromHost(cgammay.raw());
    d_calphaz.copyFromHost(calphaz.raw());
    d_cgammaz.copyFromHost(cgammaz.raw());

    // Also copy coefficients to constant memory for warp-broadcast
    if (Nx - 1 <= CGALPHA_MAX) 
    {
        CUDA_CHECK(cudaMemcpyToSymbol(calphax_c, calphax.raw(), (Nx - 1) * sizeof(double)));
        CUDA_CHECK(cudaMemcpyToSymbol(cgammax_c, cgammax.raw(), (Nx - 1) * sizeof(double)));
    }
    if (Ny - 1 <= CGALPHA_MAX) 
    {
        CUDA_CHECK(cudaMemcpyToSymbol(calphay_c, calphay.raw(), (Ny - 1) * sizeof(double)));
        CUDA_CHECK(cudaMemcpyToSymbol(cgammay_c, cgammay.raw(), (Ny - 1) * sizeof(double)));
    }
    if (Nz - 1 <= CGALPHA_MAX) 
    {
        CUDA_CHECK(cudaMemcpyToSymbol(calphaz_c, calphaz.raw(), (Nz - 1) * sizeof(double)));
        CUDA_CHECK(cudaMemcpyToSymbol(cgammaz_c, cgammaz.raw(), (Nz - 1) * sizeof(double)));
    }

    // Copy psi data to device
    d_psi.copyFromHost(psi);

    // Copy trap potential to device only if using precomputed mode
    if (initialize_pot == 1) 
    {
        CUDA_CHECK(cudaMemcpy(d_pot_ptr, pot.raw(), Nx * Ny * Nz * sizeof(double), cudaMemcpyHostToDevice));
    }
    // Copy dipole potential to device
    d_potdd.copyFromHost(potdd.raw());

    if (rmsout != NULL) 
    {
        rms_output(filerms);
    }
    // if (muoutput != NULL) 
    // {
    //     mu_output(filemu);
    // }

    // Compute wave function norm
    calcnorm(d_psi.raw(), norm, integ);

    // Compute RMS values
    calcrms(d_psi.raw(), d_work_array.raw(), integ, h_rms_pinned);
    if (rmsout != NULL) 
    {
        double rms_r = sqrt(h_rms_pinned[0] * h_rms_pinned[0] + h_rms_pinned[1] * h_rms_pinned[1] +
                            h_rms_pinned[2] * h_rms_pinned[2]);
        std::fprintf(filerms, "%-9d %-19.16le %-19.16le %-19.16le %-19.16le\n", 0, rms_r,
                     h_rms_pinned[0], h_rms_pinned[1], h_rms_pinned[2]);
        fflush(filerms);
    }

    // Compute chemical potential terms
    // calcmuen(muen.raw(), d_psi.raw(), d_work_array.raw(), d_pot_ptr, d_work_array.raw(),
    //             d_potdd.raw(), forward_plan, backward_plan, integ, g, gd, h2);
    // if (muoutput != NULL) 
    // {
    //     std::fprintf(filemu, "%-9d %-19.16le %-19.16le %-19.16le %-19.16le %-19.16le %-19.16le\n",
    //                 0, muen[0] + muen[1] + muen[2] + muen[3] + muen[4], muen[3], muen[1], muen[0],
    //                 muen[2], muen[4]);
                    
    //     fflush(filemu);
    //     mutotold = muen[0] + muen[1] + muen[2] + muen[3];
    // }

    if (Niterout != NULL) 
    {
        char itername[10];
        sprintf(itername, "-%06d-", 0);
        if (outflags & DEN_X) 
        {
            // Open binary file for writing
            sprintf(filename, "%s%s1d_x.bin", Niterout, itername);
            file = fopen(filename, "wb");
            if (file == NULL) 
            {
                std::fprintf(stderr, "Failed to open file %s\n", filename);
                exit(EXIT_FAILURE);
            }
            outdenx(psi, x, tmpy, tmpz, file);
            fclose(file);
        }
        if (outflags & DEN_Y) 
        {
            // Open binary file for writing
            sprintf(filename, "%s%s1d_y.bin", Niterout, itername);
            file = fopen(filename, "wb");
            if (file == NULL) 
            {
                std::fprintf(stderr, "Failed to open file %s\n", filename);
                exit(EXIT_FAILURE);
            }
            outdeny(psi, y, tmpx, tmpz, file);
            fclose(file);
        }
        if (outflags & DEN_Z) 
        {
            // Open binary file for writing
            sprintf(filename, "%s%s1d_z.bin", Niterout, itername);
            file = fopen(filename, "wb");
            if (file == NULL) 
            {
                std::fprintf(stderr, "Failed to open file %s\n", filename);
                exit(EXIT_FAILURE);
            }
            outdenz(psi, z, tmpx, tmpy, file);
            fclose(file);
        }
        if (outflags & DEN_XY) 
        {
            // Open binary file for writing
            sprintf(filename, "%s%s2d_xy.bin", Niterout, itername);
            file = fopen(filename, "wb");
            if (file == NULL) 
            {
                std::fprintf(stderr, "Failed to open file %s\n", filename);
                exit(EXIT_FAILURE);
            }
            outdenxy(psi, x, y, tmpz, file);
            fclose(file);
        }
        if (outflags & DEN_XZ) 
        {
            // Open binary file for writing
            sprintf(filename, "%s%s2d_xz.bin", Niterout, itername);
            file = fopen(filename, "wb");
            if (file == NULL) 
            {
                std::fprintf(stderr, "Failed to open file %s\n", filename);
                exit(EXIT_FAILURE);
            }
            outdenxz(psi, x, z, tmpy, file);
            fclose(file);
        }
        if (outflags & DEN_YZ) 
        {
            // Open binary file for writing
            sprintf(filename, "%s%s2d_yz.bin", Niterout, itername);
            file = fopen(filename, "wb");
            if (file == NULL) 
            {
                fprintf(stderr, "Failed to open file %s\n", filename);
                exit(EXIT_FAILURE);
            }
            outdenyz(psi, y, z, tmpx, file);
            fclose(file);
        }
        if (outflags & DEN_XY0) 
        {
            sprintf(filename, "%s%s3d_xy0.bin", Niterout, itername);
            file = fopen(filename, "wb");
            if (file == NULL) 
            {
                std::fprintf(stderr, "Failed to open file %s\n", filename);
                exit(EXIT_FAILURE);
            }
            outpsi2xy(psi, x, y, file);
            fclose(file);
        }
        if (outflags & DEN_X0Z) 
        {
            // Open binary file for writing
            sprintf(filename, "%s%s3d_x0z.bin", Niterout, itername);
            file = fopen(filename, "wb");
            if (file == NULL) 
            {
                std::fprintf(stderr, "Failed to open file %s\n", filename);
                exit(EXIT_FAILURE);
            }
            outpsi2xz(psi, x, z, file);
            fclose(file);
        }
        if (outflags & DEN_0YZ) 
        {
            // Open binary file for writing
            sprintf(filename, "%s%s3d_0yz.bin", Niterout, itername);
            file = fopen(filename, "wb");
            if (file == NULL) 
            {
                std::fprintf(stderr, "Failed to open file %s\n", filename);
                exit(EXIT_FAILURE);
            }
            outpsi2yz(psi, y, z, file);
            fclose(file);
        }
    }

    // Main loop that does the evolution of the wave function
    long nsteps;
    nsteps = Niter / Nsnap;
    // CUDA events for GPU timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (long snap = 1; snap <= Nsnap; snap++) 
    {
        for (long j = 0; j < nsteps; j++) 
        {
            calcpsidd2(forward_plan, backward_plan, d_psi.raw(), d_work_array.raw(),
                             d_complex, d_potdd.raw());
            calcnu(d_psi.raw(), d_work_array.raw(), d_pot_ptr, g, gd, h2);
            calclux(d_psi.raw(), d_work_array.raw(), d_calphax.raw(), d_cgammax.raw(), Ax0r, Ax);
            calcluy(d_psi.raw(), d_work_array.raw(), d_calphay.raw(), d_cgammay.raw(), Ay0r, Ay);
            calcluz(d_psi.raw(), d_work_array.raw(), d_calphaz.raw(), d_cgammaz.raw(), Az0r, Az);
            calcnorm(d_psi.raw(), norm, integ);
        }

        calcrms(d_psi.raw(), d_work_array.raw(), integ, h_rms_pinned);

        if (rmsout != NULL) 
        {
            // Compute RMS values
            double rms_r =
                sqrt(h_rms_pinned[0] * h_rms_pinned[0] + h_rms_pinned[1] * h_rms_pinned[1] +
                     h_rms_pinned[2] * h_rms_pinned[2]);
            std::fprintf(filerms, "%-9li %-19.16le %-19.16le %-19.16le %-19.16le\n", snap, rms_r,
                         h_rms_pinned[0], h_rms_pinned[1], h_rms_pinned[2]);
            fflush(filerms);
        }
        // Compute chemical potential terms
        // calcmuen(muen.raw(), d_psi.raw(), d_work_array.raw(), d_pot_ptr, d_work_array.raw(),
        //          d_potdd.raw(), forward_plan, backward_plan, integ, g, gd, h2);
        // if (muoutput != NULL) 
        // {
        //     std::fprintf(filemu,
        //                  "%-9li %-19.16le %-19.16le %-19.16le %-19.16le %-19.16le %-19.16le\n",
        //                  snap, muen[0] + muen[1] + muen[2] + muen[3] + muen[4], muen[3], muen[1],
        //                  muen[0], muen[2], muen[4]);
        //     fflush(filemu);
        // }
        if (Niterout != NULL) 
        {
            // Move d_psi to host, host is pinned memory
            CUDA_CHECK(cudaMemcpy(psi, d_psi.data(), Nx * Ny * Nz * sizeof(double), cudaMemcpyDeviceToHost));
            char itername[32]; // Increased buffer size to prevent overflow
            sprintf(itername, "-%06li-", snap);
            if (outflags & DEN_X) 
            {
                // Open binary file for writing
                sprintf(filename, "%s%s1d_x.bin", Niterout, itername);
                file = fopen(filename, "wb");
                if (file == NULL) 
                {
                    std::fprintf(stderr, "Failed to open file %s\n", filename);
                    exit(EXIT_FAILURE);
                }
                outdenx(psi, x, tmpy, tmpz, file);
                fclose(file);
            }
            if (outflags & DEN_Y) 
            {
                // Open binary file for writing
                sprintf(filename, "%s%s1d_y.bin", Niterout, itername);
                file = fopen(filename, "wb");
                if (file == NULL) 
                {
                    std::fprintf(stderr, "Failed to open file %s\n", filename);
                    exit(EXIT_FAILURE);
                }
                outdeny(psi, y, tmpx, tmpz, file);
                fclose(file);
            }
            if (outflags & DEN_Z) 
            {
                // Open binary file for writing
                sprintf(filename, "%s%s1d_z.bin", Niterout, itername);
                file = fopen(filename, "wb");
                if (file == NULL) 
                {
                    std::fprintf(stderr, "Failed to open file %s\n", filename);
                    exit(EXIT_FAILURE);
                }
                outdenz(psi, z, tmpx, tmpy, file);
                fclose(file);
            }
            if (outflags & DEN_XY) 
            {
                // Open binary file for writing
                sprintf(filename, "%s%s2d_xy.bin", Niterout, itername);
                file = fopen(filename, "wb");
                if (file == NULL) 
                {
                    std::fprintf(stderr, "Failed to open file %s\n", filename);
                    exit(EXIT_FAILURE);
                }
                outdenxy(psi, x, y, tmpz, file);
                fclose(file);
            }
            if (outflags & DEN_XZ) 
            {
                // Open binary file for writing
                sprintf(filename, "%s%s2d_xz.bin", Niterout, itername);
                file = fopen(filename, "wb");
                if (file == NULL) 
                {
                    std::fprintf(stderr, "Failed to open file %s\n", filename);
                    exit(EXIT_FAILURE);
                }
                outdenxz(psi, x, z, tmpy, file);
                fclose(file);
            }
            if (outflags & DEN_YZ) 
            {
                // Open binary file for writing
                sprintf(filename, "%s%s2d_yz.bin", Niterout, itername);
                file = fopen(filename, "wb");
                if (file == NULL) 
                {
                    std::fprintf(stderr, "Failed to open file %s\n", filename);
                    exit(EXIT_FAILURE);
                }
                outdenyz(psi, y, z, tmpx, file);
                fclose(file);
            }
            if (outflags & DEN_XY0) 
            {
                // Open binary file for writing
                sprintf(filename, "%s%s3d_xy0.bin", Niterout, itername);
                file = fopen(filename, "wb");
                if (file == NULL) 
                {
                    std::fprintf(stderr, "Failed to open file %s\n", filename);
                    exit(EXIT_FAILURE);
                }
                outpsi2xy(psi, x, y, file);
                fclose(file);
            }
            if (outflags & DEN_X0Z) 
            {
                // Open binary file for writing
                sprintf(filename, "%s%s3d_x0z.bin", Niterout, itername);
                file = fopen(filename, "wb");
                if (file == NULL) 
                {
                    std::fprintf(stderr, "Failed to open file %s\n", filename);
                    exit(EXIT_FAILURE);
                }
                outpsi2xz(psi, x, z, file);
                fclose(file);
            }
            if (outflags & DEN_0YZ) 
            {
                // Open binary file for writing
                sprintf(filename, "%s%s3d_0yz.bin", Niterout, itername);
                file = fopen(filename, "wb");
                if (file == NULL) 
                {
                    std::fprintf(stderr, "Failed to open file %s\n", filename);
                    exit(EXIT_FAILURE);
                }
                outpsi2yz(psi, y, z, file);
                fclose(file);
            }
        }
        // mutotnew = muen[0] + muen[1] + muen[2] + muen[3] + muen[4];
        // if (fabs((mutotold - mutotnew) / mutotnew) < murel)
        //     break;
        // mutotold = mutotnew;
        // if (mutotnew > muend)
        //     break;
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float gpu_time_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_time_ms, start, stop));
    double gpu_time_seconds = gpu_time_ms / 1000.0;
    if (rmsout != NULL) 
    {
        std::fprintf(filerms, "--------------------------------------------------------------------------------------------------------\n");
        std::fprintf(filerms, "Total time on GPU: %f seconds\n", gpu_time_seconds);
        std::fprintf(filerms, "--------------------------------------------------------------------------------------------------------\n");
        fclose(filerms);
    }
    // if (muoutput != NULL) 
    // {
    //     std::fprintf(filemu, "-------------------------------------------------------------------------------------------------------------------------------------------------------\n");
    //     std::fprintf(filemu, "Total time on GPU: %f seconds\n", gpu_time_seconds);
    //     std::fprintf(filemu, "-------------------------------------------------------------------------------------------------------------------------------------------------------\n");
    //     fclose(filemu);
    // }
    // Save FINALPSI
    if (finalpsi != NULL) 
    {
        // Open binary file for writing
        sprintf(filename, "%s.bin", finalpsi);
        save_psi_from_gpu(psi, d_psi.raw(), filename, Nx, Ny, Nz);

        // // Save z,y,x psi arrays to .txt files
        // sprintf(filename, "%s_zryy.txt", finalpsi);
        // file = fopen(filename, "w");
        // if(file == NULL) {
        //   std::fprintf(stderr, "Failed to open file %s\n", filename);
        //   exit(EXIT_FAILURE);
        // }
        // for(long cntk = 0; cntk < Nz; cntk++){
        //   for(long cntj = 0; cntj < Ny; cntj++){
        //     //for(long cnti = 0; cnti < Nx; cnti++){
        //       double ro= sqrt(x[cntj]*x[cntj] + y[cntj]*y[cntj]);
        //       fprintf(file, "%.16le %.16le %.16le\n", z[cntk], ro, psi[cntk * Ny * Nx + cntj * Nx
        //       + cntj]);
        //     //}
        //   }
        // }
        // fclose(file);
    }

    // Cleanup pinned memory
    cudaFreeHost(h_rms_pinned);
    cudaFreeHost(psi);

    // Cleanup FFT plans and work areas
    if (forward_work)
        cudaFree(forward_work);
    if (backward_work)
        cudaFree(backward_work);
    cufftDestroy(forward_plan);
    cufftDestroy(backward_plan);
    
    // Cleanup trap potential if it was allocated
    if (d_pot_ptr)
        cudaFree(d_pot_ptr);
    
    return 0;
}

/**
 * @brief Reading input parameters from the configuration file.
 */
void readpar(void) 
{
    const char *cfg_tmp;

    if ((cfg_tmp = cfg_read("OPTION")) == NULL) 
    {
        std::fprintf(stderr, "OPTION is not defined in the configuration file\n");
        exit(EXIT_FAILURE);
    }
    opt = atol(cfg_tmp);
    // if ((cfg_tmp = cfg_read("OPTION_MICROWAVE_SHIELDING")) == NULL) 
    // {
    //     std::fprintf(stderr,
    //                 "OPTION_MICROWAVE_SHIELDING is not defined in the configuration file\n");
    //     exit(EXIT_FAILURE);
    // }
    // optms = atol(cfg_tmp);
    // if (optms == 0) 
    // {
    //     MS = 1;
    // } else 
    // {
    //     MS = -1;
    // }

    if ((cfg_tmp = cfg_read("NATOMS")) == NULL) 
    {
        std::fprintf(stderr, "NATOMS is not defined in the configuration file.\n");
        exit(EXIT_FAILURE);
    }
    Na = atof(cfg_tmp);

    if ((cfg_tmp = cfg_read("AHO")) == NULL) 
    {
        std::fprintf(stderr, "AHO is not defined in the configuration file.\n");
        exit(EXIT_FAILURE);
    }
    aho = atof(cfg_tmp);

    if ((cfg_tmp = cfg_read("AS")) == NULL) 
    {
        std::fprintf(stderr, "AS is not defined in the configuration file.\n");
            exit(EXIT_FAILURE);
    }
    as = atof(cfg_tmp);
    g = 4. * pi * as * Na * BOHR_RADIUS / aho;

    if ((cfg_tmp = cfg_read("ADD")) == NULL) 
    {
        std::fprintf(stderr, "ADD is not defined in the configuration file.\n");
        exit(EXIT_FAILURE);
    }
    add = atof(cfg_tmp);
    gd = 3. * add * Na * BOHR_RADIUS / aho;

    if ((cfg_tmp = cfg_read("QF")) == NULL) 
    {
        QF = 0;
    } 
    else
    {
        QF = atol(cfg_tmp);
    }

    if ((cfg_tmp = cfg_read("SX")) == NULL)
    {
        sx = 0.;
    } 
    else
    {
        sx = atof(cfg_tmp);
    }

    if ((cfg_tmp = cfg_read("SY")) == NULL)
    {
        sy = 0.;
    } 
    else
    {
        sy = atof(cfg_tmp);
    }

    if ((cfg_tmp = cfg_read("SZ")) == NULL)
    {
        sz = 0.;
    } 
    else
    {
        sz = atof(cfg_tmp);
    }


    if ((cfg_tmp = cfg_read("NX")) == NULL) 
    {
        std::fprintf(stderr, "NX is not defined in the configuration file.\n");
        exit(EXIT_FAILURE);
    }
    Nx = atol(cfg_tmp);

    if ((cfg_tmp = cfg_read("NY")) == NULL) 
    {
        std::fprintf(stderr, "NY is not defined in the configuration file.\n");
        exit(EXIT_FAILURE);
    }
    Ny = atol(cfg_tmp);

    if ((cfg_tmp = cfg_read("NZ")) == NULL) 
    {
        std::fprintf(stderr, "Nz is not defined in the configuration file.\n");
        exit(EXIT_FAILURE);
    }
    Nz = atol(cfg_tmp);

    if ((cfg_tmp = cfg_read("MX")) == NULL)
    {
        mx = 10;
    } 
    else
    {
        mx = atoi(cfg_tmp);
    }

    if ((cfg_tmp = cfg_read("MY")) == NULL)
    {
        my = 10;
    } 
    else
    {
        my = atoi(cfg_tmp);
    }

    if ((cfg_tmp = cfg_read("MZ")) == NULL)
    {
        mz = 10;
    } 
    else
    {
        mz = atoi(cfg_tmp);
    }

    if ((cfg_tmp = cfg_read("MT")) == NULL)
    {
        mt = 10;
    } 
    else
    {
        mt = atoi(cfg_tmp);
    }

    if ((cfg_tmp = cfg_read("DX")) == NULL)
    {
        dx = sx * mx * 2 / Nx;
    } 
    else
    {
        dx = atof(cfg_tmp);
    }

    if ((cfg_tmp = cfg_read("DY")) == NULL)
    {
        dy = sy * my * 2 / Ny;
    } 
    else
    {
        dy = atof(cfg_tmp);
    }

    if ((cfg_tmp = cfg_read("DZ")) == NULL)
    {
        dz = sz * mz * 2 / Nz;
    } 
    else
    {
        dz = atof(cfg_tmp);
    }

    double dmin = std::min(dx, std::min(dy, dz));

    if ((cfg_tmp = cfg_read("DT")) == NULL) 
    {
        dt = dmin * dmin / mt;
    } 
    else
    {
        dt = atof(cfg_tmp);
    }

    if ((cfg_tmp = cfg_read("MUREL")) == NULL) 
    {
        std::fprintf(stderr, "MUREL is not defined in the configuration file.\n");
        exit(EXIT_FAILURE);
    }
    murel = atof(cfg_tmp);

    if ((cfg_tmp = cfg_read("MUEND")) == NULL) 
    {
        std::fprintf(stderr, "MUEND is not defined in the configuration file.\n");
        exit(EXIT_FAILURE);
    }
    muend = atof(cfg_tmp);

    if ((cfg_tmp = cfg_read("GAMMA")) == NULL) 
    {
        std::fprintf(stderr, "GAMMA is not defined in the configuration file.\n");
        exit(EXIT_FAILURE);
    }
    vgamma = atof(cfg_tmp);

    if ((cfg_tmp = cfg_read("NU")) == NULL) 
    {
        std::fprintf(stderr, "NU is not defined in the configuration file.\n");
        exit(EXIT_FAILURE);
    }
    vnu = atof(cfg_tmp);

    if ((cfg_tmp = cfg_read("LAMBDA")) == NULL) 
    {
        std::fprintf(stderr, "LAMBDA is not defined in the configuration file.\n");
        exit(EXIT_FAILURE);
    }
    vlambda = atof(cfg_tmp);

    if ((cfg_tmp = cfg_read("NITER")) == NULL) 
    {
        std::fprintf(stderr, "NITER is not defined in the configuration file.\n");
        exit(EXIT_FAILURE);
    }
    Niter = atol(cfg_tmp);
    if ((cfg_tmp = cfg_read("NSNAP")) == NULL)
    {
        Nsnap = 1;
    }
    else
    {
        Nsnap = atol(cfg_tmp);
    }

    
    if ((cfg_tmp = cfg_read("CUTOFF")) == NULL)
    {
        if(dmin == dx)
        {
            cutoff = Nx * dx / 2;
        }
        else if(dmin == dy)
        {
            cutoff = Ny * dy / 2;
        }
        else if(dmin == dz)
        {
            cutoff = Nz * dz / 2;
        }
    } 
    else
    {
        cutoff = atof(cfg_tmp);
    }

    input = cfg_read("INPUT");
    input_type = cfg_read("INPUT_TYPE");
    output = cfg_read("OUTPUT");
    muoutput = cfg_read("MUOUTPUT");
    rmsout = cfg_read("RMSOUT");
    Niterout = cfg_read("NITEROUT");
    finalpsi = cfg_read("FINALPSI");

    if (Niterout != NULL) {
        if ((cfg_tmp = cfg_read("OUTFLAGS")) == NULL) 
        {
            std::fprintf(stderr, "OUTFLAGS is not defined in the configuration file.\n");
            exit(EXIT_FAILURE);
        }
        outflags = atoi(cfg_tmp);
    } 
    else
    {
        outflags = 0;
    }

    // Read INITIALIZE_POT flag (default is 1 = precomputed, 0 = on-the-fly)
    if ((cfg_tmp = cfg_read("INITIALIZE_POT")) == NULL)
    {
        initialize_pot = 1; // Default: precompute and store potential on GPU
    }
    else
    {
        initialize_pot = atoi(cfg_tmp);
    }

    return;
}

/**
 * @brief Function to compute RMS values on device
 * @param d_psi: Device: 3D psi array
 * @param d_work_array:  Work array (temporary buffer)
 * @param integ: Simpson3DTiledIntegrator
 * @param h_rms_pinned: Output RMS values in pinned memory [rms_x, rms_y, rms_z]
 */
void calcrms(
    const double *d_psi, // Device: 3D psi array
    double *d_work_array, Simpson3DTiledIntegrator &integ,
    double *h_rms_pinned) // Output RMS values in pinned memory [rms_x, rms_y, rms_z]
{
    // Compute all 3 RMS integrals in a single pass over the data
    // d_psi is unpadded (uses Nx for indexing)
    
    double x2_integral, y2_integral, z2_integral;
    integ.integrateDeviceRMSFused(dx, dy, dz, d_psi, Nx, Ny, Nz, dx, dy, dz,
                                   x2_integral, y2_integral, z2_integral);

    // Calculate RMS values and store in pinned memory
    h_rms_pinned[0] = sqrt(x2_integral); // rms_x
    h_rms_pinned[1] = sqrt(y2_integral); // rms_y
    h_rms_pinned[2] = sqrt(z2_integral); // rms_z
}

/**
 * @brief Kernel to compute single weighted psi squared
 * @param psi: Device: 3D psi array
 * @param result: Result array (may be padded)
 * @param direction: Direction (0=x, 1=y, 2=z)
 * @param scale: Grid spacing for the chosen direction (dx, dy, dz)
 * @param result_Nx: X dimension of result array (Nx for unpadded, Nx+2 for padded)
 */
__global__ void compute_single_weighted_psi_squared(const double *__restrict__ psi, double *result,
                                                    int direction, // 0=x, 1=y, 2=z
                                                    const double scale,
                                                    long result_Nx) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;
    
    int stride_x = gridDim.x * blockDim.x;
    int stride_y = gridDim.y * blockDim.y;
    int stride_z = gridDim.z * blockDim.z;

    for (int iz = idz; iz < d_Nz; iz += stride_z) 
    {
        for (int iy = idy; iy < d_Ny; iy += stride_y) 
        {
            for (int ix = idx; ix < d_Nx; ix += stride_x) 
            {
                // Index for reading from psi (unpadded, uses d_Nx)
                int psi_linear_idx = iz * d_Ny * d_Nx + iy * d_Nx + ix;
                // Index for writing to result (may be padded, uses result_Nx)
                int result_linear_idx = iz * d_Ny * result_Nx + iy * result_Nx + ix;
                
                double psi_val = __ldg(&psi[psi_linear_idx]); // Read-only cache for psi
                double psi_squared = psi_val * psi_val;

                double weight = 0.0;
                if (direction == 0) 
                {
                    double x = (static_cast<double>(ix) - static_cast<double>(d_Nx) * 0.5) * scale;
                    weight = x * x;
                } 
                else if (direction == 1) 
                {
                    double y = (static_cast<double>(iy) - static_cast<double>(d_Ny) * 0.5) * scale;
                    weight = y * y;
                } 
                else if (direction == 2) 
                {
                    double z = (static_cast<double>(iz) - static_cast<double>(d_Nz) * 0.5) * scale;
                    weight = z * z;
                }

                result[result_linear_idx] = weight * psi_squared;
            }
        }
    }
}

/**
 * @brief Function to compute squared wave function on device
 * @param d_psi: Device: 3D psi array
 * @param d_psi2: Device: 3D psi2 array (may be padded)
 * @param psi2_Nx: X dimension of d_psi2 array (Nx for unpadded, Nx+2 for padded)
 */
void calc_d_psi2(const double *d_psi, double *d_psi2, long psi2_Nx) 
{
    static int smCount = getGPUSMCount();
    dim3 threadsPerBlock(8, 8, 8);  // threads per block
    dim3 numBlocks = getOptimalGrid3D(smCount, Nx, Ny, Nz, threadsPerBlock, 2);
    compute_d_psi2<<<numBlocks, threadsPerBlock>>>(d_psi, d_psi2, psi2_Nx);
    CUDA_CHECK_KERNEL("compute_d_psi2");
    return;
}

/**
 * @brief Kernel to compute squared wave function
 * @param d_psi: Device: 3D psi array
 * @param d_psi2: Device: 3D psi2 array (may be padded)
 * @param psi2_Nx: X dimension of d_psi2 array (Nx for unpadded, Nx+2 for padded)
 */
__global__ void compute_d_psi2(const double *__restrict__ d_psi, double *__restrict__ d_psi2, long psi2_Nx) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;
    
    int stride_x = gridDim.x * blockDim.x;
    int stride_y = gridDim.y * blockDim.y;
    int stride_z = gridDim.z * blockDim.z;

    for (int iz = idz; iz < d_Nz; iz += stride_z) 
    {
        for (int iy = idy; iy < d_Ny; iy += stride_y) 
        {
            for (int ix = idx; ix < d_Nx; ix += stride_x) 
            {
                // Index for reading from d_psi (unpadded, uses d_Nx)
                int psi_linear_idx = iz * d_Ny * d_Nx + iy * d_Nx + ix;
                // Index for writing to d_psi2 (may be padded, uses psi2_Nx)
                int psi2_linear_idx = iz * d_Ny * psi2_Nx + iy * psi2_Nx + ix;
                
                double psi_val = d_psi[psi_linear_idx];
                d_psi2[psi2_linear_idx] = psi_val * psi_val;
            }
        }
    }
}

/**
 * @brief Function to initialize wave function
 * @param psi: Device: 3D psi array
 * @param x2: x2 array
 * @param y2: y2 array
 * @param z2: z2 array
 * @param x: x array
 * @param y: y array
 * @param z: z array
 */
void initpsi(double *psi, MultiArray<double> &x2, MultiArray<double> &y2, MultiArray<double> &z2,
             MultiArray<double> &x, MultiArray<double> &y, MultiArray<double> &z) 
{
    long cnti, cntj, cntk;
    double cpsi;
    double tmp;
    cpsi = sqrt(2. * pi * sqrt(2. * pi) * sx * sy * sz);

    #pragma omp parallel for private(cnti)
    for (cnti = 0; cnti < Nx; cnti++) 
    {
        x[cnti] = (cnti - Nx2) * dx;
        x2[cnti] = x[cnti] * x[cnti];
    }

    #pragma omp parallel for private(cntj)
    for (cntj = 0; cntj < Ny; cntj++) 
    {
        y[cntj] = (cntj - Ny2) * dy;
        y2[cntj] = y[cntj] * y[cntj];
    }

    #pragma omp parallel for private(cntk)
    for (cntk = 0; cntk < Nz; cntk++) 
    {
        z[cntk] = (cntk - Nz2) * dz;
        z2[cntk] = z[cntk] * z[cntk];
    }

    #pragma omp parallel for private(cnti, cntj, cntk, tmp)
    for (cntk = 0; cntk < Nz; cntk++) 
    {
        for (cntj = 0; cntj < Ny; cntj++) 
        {
            for (cnti = 0; cnti < Nx; cnti++) 
            {
                tmp = exp(-0.25 *
                          (x2[cnti] / (sx * sx) + y2[cntj] / (sy * sy) + z2[cntk] / (sz * sz)));
                psi[cntk * Ny * Nx + cntj * Nx + cnti] = tmp / cpsi;
            }
        }
    }
    return;
}

/**
 * @brief Function to initialize trap potential
 * @param pot: Device: 3D trap potential array
 * @param x2: x2 array
 * @param y2: y2 array
 * @param z2: z2 array
 */
void initpot(MultiArray<double> &pot, MultiArray<double> &x2, MultiArray<double> &y2,
             MultiArray<double> &z2) 
{
    long cnti, cntj, cntk;
    double vgamma2 = vgamma * vgamma;
    double vnu2 = vnu * vnu;
    double vlambda2 = vlambda * vlambda;

    #pragma omp parallel for private(cnti, cntj, cntk)
    for (cntk = 0; cntk < Nz; cntk++) 
    {
        for (cntj = 0; cntj < Ny; cntj++) 
        {
            for (cnti = 0; cnti < Nx; cnti++) 
            {
                pot(cntk, cntj, cnti) =
                    0.5 * par * (vgamma2 * x2[cnti] + vnu2 * y2[cntj] + vlambda2 * z2[cntk]);
            }
        }
    }
    return;
}

/**
 * @brief Function to initialize dipolar potential
 * @param potdd: Device: 3D dipolar potential array
 * @param kx: Device: 3D kx array
 * @param ky: Device: 3D ky array
 * @param kz: Device: 3D kz array
 * @param kx2: Device: 3D kx2 array
 * @param ky2: Device: 3D ky2 array
 * @param kz2: Device: 3D kz2 array
 */
void initpotdd(MultiArray<double> &potdd, MultiArray<double> &kx, MultiArray<double> &ky,
               MultiArray<double> &kz, MultiArray<double> &kx2, MultiArray<double> &ky2,
               MultiArray<double> &kz2) 
{
    long cnti, cntj, cntk;
    double dkx, dky, dkz, xk, tmp;

    dkx = 2. * pi / (Nx * dx);
    dky = 2. * pi / (Ny * dy);
    dkz = 2. * pi / (Nz * dz);

    for (cnti = 0; cnti <= Nx2; cnti++)
        kx[cnti] = cnti * dkx;
    for (cnti = Nx2 + 1; cnti < Nx; cnti++)
        kx[cnti] = (cnti - Nx) * dkx;
    for (cntj = 0; cntj <= Ny2; cntj++)
        ky[cntj] = cntj * dky;
    for (cntj = Ny2 + 1; cntj < Ny; cntj++)
        ky[cntj] = (cntj - Ny) * dky;
    for (cntk = 0; cntk <= Nz2; cntk++)
        kz[cntk] = cntk * dkz;
    for (cntk = Nz2 + 1; cntk < Nz; cntk++)
        kz[cntk] = (cntk - Nz) * dkz;

    for (cnti = 0; cnti < Nx; cnti++)
        kx2[cnti] = kx[cnti] * kx[cnti];
    for (cntj = 0; cntj < Ny; cntj++)
        ky2[cntj] = ky[cntj] * ky[cntj];
    for (cntk = 0; cntk < Nz; cntk++)
        kz2[cntk] = kz[cntk] * kz[cntk];

    #pragma omp parallel for private(cnti, cntj, cntk, tmp, xk)
    for (cntk = 0; cntk < Nz; cntk++) 
    {
        for (cntj = 0; cntj < Ny; cntj++) 
        {
            for (cnti = 0; cnti < Nx; cnti++) 
            {
                xk = sqrt(kz2[cntk] + kx2[cnti] + ky2[cntj]);
                tmp = 1. + 3. * cos(xk * cutoff) / (xk * xk * cutoff * cutoff) -
                      3. * sin(xk * cutoff) / (xk * xk * xk * cutoff * cutoff * cutoff);
                potdd(cntk, cntj, cnti) =
                    (4. * pi * (3. * kz2[cntk] / (kx2[cnti] + ky2[cntj] + kz2[cntk]) - 1.) / 3.) *
                    tmp;
            }
        }
    }
    potdd(0, 0, 0) = 0.;

    return;
}

/**
 * @brief Function to calculate the wave function norm and normalization on device
 * @param d_psi: Device: 3D psi array
 * @param d_psi2: Device: 3D psi2 array
 * @param norm: Wave function norm
 * @param integ: Simpson3DTiledIntegrator
 */
void calcnorm(double *d_psi, double &norm, Simpson3DTiledIntegrator &integ) 
{
    // Square and integrate in a single kernel pass (no need for d_psi2 buffer)
    // d_psi is unpadded (uses Nx for indexing)
    double raw_norm = integ.integrateDeviceNorm(dx, dy, dz, d_psi, Nx, Ny, Nz);
    norm = 1.0 / sqrt(raw_norm);

    // Apply normalization
    static int smCount = getGPUSMCount();
    dim3 threadsPerBlock(8, 8, 8);  // threads per block
    dim3 numBlocks = getOptimalGrid3D(smCount, Nx, Ny, Nz, threadsPerBlock, 4);

    multiply_by_norm<<<numBlocks, threadsPerBlock>>>(d_psi, norm);
    CUDA_CHECK_KERNEL("multiply_by_norm");
}

/**
 * @brief Kernel to multiply the wave function by the norm on device
 * @param d_psi: Device: 3D psi array
 * @param norm: Wave function norm
 */
__global__ void multiply_by_norm(double *__restrict__ d_psi, const double norm) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;
    
    int stride_x = gridDim.x * blockDim.x;
    int stride_y = gridDim.y * blockDim.y;
    int stride_z = gridDim.z * blockDim.z;

    for (int iz = idz; iz < d_Nz; iz += stride_z) 
    {
        for (int iy = idy; iy < d_Ny; iy += stride_y) 
        {
            for (int ix = idx; ix < d_Nx; ix += stride_x) 
            {
                long linear_idx = iz * d_Ny * d_Nx + iy * d_Nx + ix;
                d_psi[linear_idx] *= norm;
            }
        }
    }
}

/**
 * @brief Kernel to compute the squared wave function multiplied by the dipolar potential
 * @param d_psi2_fft: Device: 3D psi2 array in FFT format (in-place FFT buffer)
 * @param potdd: Device: 3D dipolar potential array
 */
__global__ void compute_psid2_potdd(cufftDoubleComplex *d_psi2_fft,
                                    const double *__restrict__ potdd) 
{
    int grid_stride_z = gridDim.z * blockDim.z;
    int grid_stride_y = gridDim.y * blockDim.y;
    int grid_stride_x = gridDim.x * blockDim.x;
    int tdz = blockIdx.z * blockDim.z + threadIdx.z;
    int tdy = blockIdx.y * blockDim.y + threadIdx.y;
    int tdx = blockIdx.x * blockDim.x + threadIdx.x;

    // Grid-stride loop implementation
    for (int idz = tdz; idz < d_Nz; idz += grid_stride_z) 
    {
        for (int idy = tdy; idy < d_Ny; idy += grid_stride_y) 
        {
            for (int idx = tdx; idx < d_Nx / 2 + 1; idx += grid_stride_x) 
            {

                // Calculate linear index for FFT array (R2C format)
                int fft_idx = idz * d_Ny * (d_Nx / 2 + 1) + idy * (d_Nx / 2 + 1) + idx;
                int pot_idx = idz * d_Ny * d_Nx + idy * d_Nx + idx;

                double val = __ldg(&potdd[pot_idx]); // Read-only cache for dipolar potential
                d_psi2_fft[fft_idx].x *= val;
                d_psi2_fft[fft_idx].y *= val;
            }
        }
    }
}

/**
 * @brief Kernel to compute the boundaries of the FFT psi2 array
 * @param psidd2: Device: 3D psi2 array multiplied by the dipolar potential (padded as Nx+2, Ny, Nz)
 * @param psidd2_Nx: X dimension of psidd2 array (Nx+2 for padded)
 */
__global__ void calcpsidd2_boundaries(double *psidd2, long psidd2_Nx) 
{
    long cnti, cntj, cntk;

    int grid_stride_y = gridDim.y * blockDim.y;
    int grid_stride_x = gridDim.x * blockDim.x;

    int tdy = blockIdx.y * blockDim.y + threadIdx.y;
    int tdx = blockIdx.x * blockDim.x + threadIdx.x;

    for (cntj = tdy; cntj < d_Ny; cntj += grid_stride_y) 
    {
        for (cntk = tdx; cntk < d_Nz; cntk += grid_stride_x) 
        {
            // Use padded dimension for indexing
            long first_idx = cntk * (psidd2_Nx * d_Ny) + cntj * psidd2_Nx + 0; // i=0 (first x-slice)
            long last_idx =
                cntk * (psidd2_Nx * d_Ny) + cntj * psidd2_Nx + (d_Nx - 1); // i=d_Nx-1 (last x-slice)

            psidd2[last_idx] = psidd2[first_idx];
        }
    }

    for (cnti = tdy; cnti < d_Nx; cnti += grid_stride_y) 
    {
        for (cntk = tdx; cntk < d_Nz; cntk += grid_stride_x) 
        {
            // Use padded dimension for indexing
            long first_idx = cntk * (psidd2_Nx * d_Ny) + 0 * psidd2_Nx + cnti; // j=0 (first y-slice)
            long last_idx =
                cntk * (psidd2_Nx * d_Ny) + (d_Ny - 1) * psidd2_Nx + cnti; // j=d_Ny-1 (last y-slice)

            psidd2[last_idx] = psidd2[first_idx];
        }
    }

    for (cnti = tdy; cnti < d_Nx; cnti += grid_stride_y) 
    {
        for (cntj = tdx; cntj < d_Ny; cntj += grid_stride_x) 
        {
            // Use padded dimension for indexing
            long first_idx = 0 * (psidd2_Nx * d_Ny) + cntj * psidd2_Nx + cnti; // k=0 (first z-slice)
            long last_idx =
                (d_Nz - 1) * (psidd2_Nx * d_Ny) + cntj * psidd2_Nx + cnti; // k=d_Nz-1 (last z-slice)

            psidd2[last_idx] = psidd2[first_idx];
        }
    }
}

/**
 * @brief Function to compute the FFT of the squared wave function multiplied by the dipolar
 * potential (in-place FFT)
 * @param forward_plan: Forward FFT plan
 * @param backward_plan: Backward FFT plan
 * @param d_psi: Device: 3D psi array
 * @param d_psi2_real: Device: 3D psi2 array (also used for in-place FFT)
 * @param d_psi2_fft: Device: Complex view of d_psi2_real for in-place FFT
 * @param potdd: Device: 3D dipolar potential array
 */
void calcpsidd2(cufftHandle forward_plan, cufftHandle backward_plan, const double *d_psi,
                      double *d_psi2_real, cufftDoubleComplex *d_psi2_fft, const double *potdd) 
{
    // d_psi2_real is d_work_array or d_psi2dd, both are padded (Nx+2, Ny, Nz)
    calc_d_psi2(d_psi, d_psi2_real, Nx + 2);
    // In-place-like FFT: using same buffer with different pointer types
    // cuFFT requires different pointer values, so we cast to ensure they're treated as different
    // The plan (cufftMakePlanMany) knows about the overlapping memory layout
    CUFFT_CHECK(cufftExecD2Z(forward_plan, (cufftDoubleReal *)d_psi2_real, d_psi2_fft));

    static int smCount = getGPUSMCount();
    dim3 threadsPerBlock(8, 8, 8);  // threads per block
    dim3 numBlocks = getOptimalGrid3D(smCount, Nx, Ny, Nz, threadsPerBlock, 2);
    compute_psid2_potdd<<<numBlocks, threadsPerBlock>>>(d_psi2_fft, potdd);
    CUDA_CHECK_KERNEL("compute_psid2_potdd");

    // In-place FFT: same memory location for input (complex) and output (real)
    CUFFT_CHECK(cufftExecZ2D(backward_plan, d_psi2_fft, (cufftDoubleReal *)d_psi2_real));
    
    // Boundary kernel uses 2D grid
    // d_psi2_real is padded (Nx+2, Ny, Nz)
    dim3 blockSize2D(64, 8);
    dim3 numBlocks2D = getOptimalGrid2D(smCount, Ny, Nz, blockSize2D, 4);
    calcpsidd2_boundaries<<<numBlocks2D, blockSize2D>>>(d_psi2_real, Nx + 2);
    CUDA_CHECK_KERNEL("calcpsidd2_boundaries");
    return;
}

/**
 * @brief Function to generate the Crank-Nicolson scheme coefficients
 * @param calphax: Host: 3D alpha x coefficient
 * @param cgammax: Host: 3D gamma x coefficient
 * @param calphay: Host: 3D alpha y coefficient
 * @param cgammay: Host: 3D gamma y coefficient
 * @param calphaz: Host: 3D alpha z coefficient
 * @param cgammaz: Host: 3D gamma z coefficient
 * @param Ax0: Device: Ax0 coefficient
 * @param Ay0: Host: Ay0 coefficient
 * @param Az0: Host: Az0 coefficient
 * @param Ax0r: Host: Ax0r coefficient
 * @param Ay0r: Host: Ay0r coefficient
 * @param Az0r: Host: Az0r coefficient
 * @param Ax: Host: Ax coefficient
 * @param Ay: Host: Ay coefficient
 * @param Az: Host: Az coefficient
 */
void gencoef(MultiArray<double> &calphax, MultiArray<double> &cgammax, MultiArray<double> &calphay,
             MultiArray<double> &cgammay, MultiArray<double> &calphaz, MultiArray<double> &cgammaz,
             double &Ax0, double &Ay0, double &Az0, double &Ax0r, double &Ay0r, double &Az0r,
             double &Ax, double &Ay, double &Az) 
{
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
    for (cnti = Nx - 2; cnti > 0; cnti--) 
    {
        calphax[cnti - 1] = Ax * cgammax[cnti];
        cgammax[cnti - 1] = -1. / (Ax0 + Ax * calphax[cnti - 1]);
    }

    calphay[Ny - 2] = 0.;
    cgammay[Ny - 2] = -1. / Ay0;
    for (cnti = Ny - 2; cnti > 0; cnti--) 
    {
        calphay[cnti - 1] = Ay * cgammay[cnti];
        cgammay[cnti - 1] = -1. / (Ay0 + Ay * calphay[cnti - 1]);
    }

    calphaz[Nz - 2] = 0.;
    cgammaz[Nz - 2] = -1. / Az0;
    for (cnti = Nz - 2; cnti > 0; cnti--) 
    {
        calphaz[cnti - 1] = Az * cgammaz[cnti];
        cgammaz[cnti - 1] = -1. / (Az0 + Az * calphaz[cnti - 1]);
    }

    return;
}

/**
 * @brief Function to compute the time propagation with respect to H1 (part of the Hamiltonian
 * without spatial derivatives).
 * @param d_psi: Device: 3D psi array
 * @param d_psi2: Device: 3D psi2 array
 * @param d_pot: Device: 3D trap potential array (can be nullptr if initialize_pot == 0)
 * @param g: Host to Device: g coefficient for contact interaction term
 * @param gd: Host to Device: gd coefficient for dipolar interaction term
 * @param h2: Host to Device: h2 coefficient for quantum fluctuation term
 */
void calcnu(double *d_psi, double *d_psi2, double *d_pot, double g, double gd, double h2) 
{
    // Precompute ratio_gd on host (constant for all threads)
    double ratio_gd = gd / ((double)(Nx * Ny * Nz));

    static int smCount = getGPUSMCount();
    dim3 threadsPerBlock(8, 8, 8);  // threads per block
    dim3 numBlocks = getOptimalGrid3D(smCount, Nx, Ny, Nz, threadsPerBlock, 2);
    
    // d_psi2 is d_work_array which is padded (Nx+2, Ny, Nz)
    long psi2_Nx = Nx + 2;
    
    if (initialize_pot == 1) 
    {
        // Use precomputed potential from GPU memory
        calcnu_kernel<<<numBlocks, threadsPerBlock>>>(d_psi, d_psi2, d_pot, g, ratio_gd, h2, psi2_Nx);
        CUDA_CHECK_KERNEL("calcnu_kernel");
    } 
    else 
    {
        // Compute potential on-the-fly
        calcnu_kernel_onthefly<<<numBlocks, threadsPerBlock>>>(d_psi, d_psi2, g, ratio_gd, h2, psi2_Nx);
        CUDA_CHECK_KERNEL("calcnu_kernel_onthefly");
    }
    CUDA_SYNC_CHECK("calcnu");  // Only active in debug mode
    return;
}

/**
 * @brief Device function to compute trap potential on-the-fly from grid indices
 * @param ix: x index
 * @param iy: y index
 * @param iz: z index
 * @return Trap potential value at (ix, iy, iz)
 */
__device__ __forceinline__ double compute_pot_onthefly(int ix, int iy, int iz) 
{
    double x = (static_cast<double>(ix) - static_cast<double>(d_Nx) * 0.5) * d_dx;
    double y = (static_cast<double>(iy) - static_cast<double>(d_Ny) * 0.5) * d_dy;
    double z = (static_cast<double>(iz) - static_cast<double>(d_Nz) * 0.5) * d_dz;
    return 0.5 * d_par * (d_vgamma2 * x * x + d_vnu2 * y * y + d_vlambda2 * z * z);
}

/**
 * @brief Kernel to compute the time propagation with respect to H1 (part of the Hamiltonian without
 *    spatial derivatives).
 * @param d_psi: Device: 3D psi array
 * @param d_psi2: Device: 3D psi2 array (may be padded)
 * @param d_pot: Device: 3D trap potential array
 * @param g: Host: g coefficient for contact interaction term
 * @param ratio_gd: Precomputed ratio gd/(Nx*Ny*Nz) for dipolar interaction term
 * @param h2: Host: h2 coefficient for quantum fluctuation term
 * @param psi2_Nx: X dimension of d_psi2 array (Nx for unpadded, Nx+2 for padded)
 */
__global__ void calcnu_kernel(double *__restrict__ d_psi, double *__restrict__ d_psi2,
                              const double *__restrict__ d_pot, const double g,
                              const double ratio_gd, const double h2, long psi2_Nx) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;
    
    int stride_x = gridDim.x * blockDim.x;
    int stride_y = gridDim.y * blockDim.y;
    int stride_z = gridDim.z * blockDim.z;

    for (int iz = idz; iz < d_Nz; iz += stride_z) 
    {
        for (int iy = idy; iy < d_Ny; iy += stride_y) 
        {
            for (int ix = idx; ix < d_Nx; ix += stride_x) 
            {
                // Index for reading from d_psi (unpadded, uses d_Nx)
                int psi_linear_idx = iz * d_Ny * d_Nx + iy * d_Nx + ix;
                // Index for reading from d_psi2 (may be padded, uses psi2_Nx)
                int psi2_linear_idx = iz * d_Ny * psi2_Nx + iy * psi2_Nx + ix;
                // Index for reading from d_pot (unpadded, uses d_Nx)
                int pot_linear_idx = iz * d_Ny * d_Nx + iy * d_Nx + ix;
                
                double psi_val = d_psi[psi_linear_idx];
                double psi_val2 = fma(psi_val, psi_val, 0.0);        // psi^2
                //double psi_val3 = fma(psi_val2, fabs(psi_val), 0.0); // psi^3
                double psi2dd = __ldg(&d_psi2[psi2_linear_idx]) * ratio_gd;
                double pot_val = __ldg(&d_pot[pot_linear_idx]);
                double temp1 = fma(psi_val2, g, psi2dd);
                //double temp2 = fma(psi_val3, h2, pot_val);
                //double sum = temp1 + temp2;
                double sum = temp1 + pot_val;
                double tmp = fma(d_dt, sum, 0.0);
                // double tmp = d_dt * (psi_val3*h2);
                d_psi[psi_linear_idx] *= exp(-tmp);
            }
        }
    }
}

/**
 * @brief Kernel to compute H1 time propagation with on-the-fly potential calculation.
 *        Saves GPU memory by computing trap potential from grid indices instead of reading from array.
 * @param d_psi: Device: 3D psi array
 * @param d_psi2: Device: 3D psi2 array (contains psi2dd after calcpsidd2, may be padded)
 * @param g: Contact interaction coefficient
 * @param ratio_gd: Precomputed ratio gd/(Nx*Ny*Nz) for dipolar interaction term
 * @param h2: Quantum fluctuation coefficient
 * @param psi2_Nx: X dimension of d_psi2 array (Nx for unpadded, Nx+2 for padded)
 */
__global__ void calcnu_kernel_onthefly(double *__restrict__ d_psi, double *__restrict__ d_psi2,
                                       const double g, const double ratio_gd, const double h2, long psi2_Nx) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;
    
    int stride_x = gridDim.x * blockDim.x;
    int stride_y = gridDim.y * blockDim.y;
    int stride_z = gridDim.z * blockDim.z;

    for (int iz = idz; iz < d_Nz; iz += stride_z) 
    {
        for (int iy = idy; iy < d_Ny; iy += stride_y) 
        {
            for (int ix = idx; ix < d_Nx; ix += stride_x) 
            {
                // Index for reading from d_psi (unpadded, uses d_Nx)
                int psi_linear_idx = iz * d_Ny * d_Nx + iy * d_Nx + ix;
                // Index for reading from d_psi2 (may be padded, uses psi2_Nx)
                int psi2_linear_idx = iz * d_Ny * psi2_Nx + iy * psi2_Nx + ix;
                
                double psi_val = d_psi[psi_linear_idx];
                double psi_val2 = fma(psi_val, psi_val, 0.0);        // psi^2
                double psi2dd = __ldg(&d_psi2[psi2_linear_idx]) * ratio_gd;
                
                // Compute potential on-the-fly instead of reading from global memory
                double pot_val = compute_pot_onthefly(ix, iy, iz);
                
                double temp1 = fma(psi_val2, g, psi2dd);
                //double temp2 = fma(psi_val3, h2, pot_val);
                //double sum = temp1 + temp2;
                double sum = temp1 + pot_val;
                double tmp = fma(d_dt, sum, 0.0);
                d_psi[psi_linear_idx] *= exp(-tmp);
            }
        }
    }
}

/**
 * @brief Function to compute the time propagation with respect to H2 (x-part of the Laplacian).
 * @param d_psi: Device: 3D psi array
 * @param d_cbeta: Device: 3D cbeta array
 * @param d_calphax: Device: 3D calphax array
 * @param d_cgammax: Device: 3D cgammax array
 * @param Ax0r: Host to Device: Ax0r coefficient
 * @param Ax: Host to Device: Ax coefficient
 */
void calclux(double *d_psi, double *d_cbeta, double *d_calphax, double *d_cgammax, double Ax0r,
             double Ax) 
{
    // Use grid-stride approach with optimal block count
    static int smCount = getGPUSMCount();
    dim3 threadsPerBlock(64, 8);  // threads per block
    // For batched kernel, effective Ny dimension is Ny/BATCH_SIZE (16)
    long effectiveNy = (Ny + 15) / 16;
    dim3 numBlocks = getOptimalGrid2D(smCount, effectiveNy, Nz, threadsPerBlock, 4);

    calclux_kernel<<<numBlocks, threadsPerBlock>>>(d_psi, d_cbeta, d_calphax, d_cgammax, Ax0r, Ax);
    CUDA_CHECK_KERNEL("calclux_kernel");
    CUDA_SYNC_CHECK("calclux");  // Only active in debug mode
    return;
}

/**
 * @brief Kernel to compute the time propagation with respect to H2 (x-part of the Laplacian).
 * @param d_psi: Device: 3D psi array
 * @param d_cbeta: Device: 3D cbeta array
 * @param d_calphax: Device: 3D calphax array
 * @param d_cgammax: Device: 3D cgammax array
 * @param Ax0r: Host to Device: Ax0r coefficient
 * @param Ax: Host to Device: Ax coefficient
 */
__global__ void calclux_kernel(double *__restrict__ psi, double *__restrict__ cbeta,
                               const double *__restrict__ calphax,
                               const double *__restrict__ cgammax, const double Ax0r,
                               const double Ax) 
{
    // Batched processing: each thread handles BATCH_SIZE Y-lines
    constexpr int BATCH_SIZE = 16;
    
    // Map y to threadIdx.x for better locality across warp; z to threadIdx.y
    int cntj_base_init = (blockIdx.x * blockDim.x + threadIdx.x) * BATCH_SIZE;
    int cntk_init = blockIdx.y * blockDim.y + threadIdx.y;
    
    int stride_j = gridDim.x * blockDim.x * BATCH_SIZE;
    int stride_k = gridDim.y * blockDim.y;

    // Grid-stride loop over z
    for (int cntk = cntk_init; cntk < d_Nz; cntk += stride_k) 
    {
        // Grid-stride loop over y batches
        for (int cntj_base = cntj_base_init; cntj_base < d_Ny; cntj_base += stride_j) 
        {
            // Process BATCH_SIZE consecutive Y-lines
            #pragma unroll
            for (int batch = 0; batch < BATCH_SIZE; batch++) {
                int cntj = cntj_base + batch;
                
                if (cntj >= d_Ny)
                    break;

                // Base offset for this (j,k) y-z position
                const long base_offset = cntk * d_Ny * d_Nx + cntj * d_Nx;

                // Forward elimination: fill cbeta array
                // Boundary condition: cbeta[nx-2] = psi[nx-1]
                long idx_i = base_offset + (d_Nx - 2);
                long idx_ip1 = base_offset + (d_Nx - 1);
                cbeta[idx_i] = psi[idx_ip1];

                // Algorithm forward sweep with rolling window in x
                double psi_ip1 = __ldg(&psi[idx_ip1]);
                double psi_i = __ldg(&psi[idx_i]);
                for (int cnti = d_Nx - 2; cnti > 0; cnti--) 
                {
                    long idx_im1 = idx_i - 1;
                    double psi_im1 = __ldg(&psi[idx_im1]);

                    double c = fma(Ax0r, psi_i, fma(-Ax, psi_im1, -Ax * psi_ip1));
                    double gamma = (d_Nx - 1 <= CGALPHA_MAX) ? cgammax_c[cnti] : __ldg(&cgammax[cnti]);
                    cbeta[idx_im1] = gamma * fma(Ax, cbeta[idx_i], -c);

                    // Roll window down in x
                    psi_ip1 = psi_i;
                    psi_i = psi_im1;
                    idx_i = idx_im1;
                }

                // Boundary condition
                psi[base_offset + 0] = 0.0;

                // Back substitution: update psi values
                long idx_i_bs = base_offset; // i = 0
                for (int cnti = 0; cnti < d_Nx - 2; cnti++) 
                {
                    long idx_ip1_bs = idx_i_bs + 1; // i+1
                    double alpha = (d_Nx - 1 <= CGALPHA_MAX) ? calphax_c[cnti] : __ldg(&calphax[cnti]);
                    psi[idx_ip1_bs] = fma(alpha, psi[idx_i_bs], cbeta[idx_i_bs]);
                    idx_i_bs = idx_ip1_bs;
                }

                // Boundary condition
                psi[base_offset + d_Nx - 1] = 0.0;
            }
        }
    }
}

/**
 * @brief Function to compute the time propagation with respect to H3 (y-part of the Laplacian).
 * @param d_psi: Device: 3D psi array
 * @param d_cbeta: Device: 3D cbeta array
 * @param d_calphay: Device: 3D calphay array
 * @param d_cgammay: Device: 3D cgammay array
 * @param Ay0r: Host to Device: Ay0r coefficient
 * @param Ay: Host to Device: Ay coefficient
 */
void calcluy(double *d_psi, double *d_cbeta, double *d_calphay, double *d_cgammay, double Ay0r,
             double Ay) 
{
    // Use grid-stride approach with optimal block count
    static int smCount = getGPUSMCount();
    dim3 threadsPerBlock(64, 8);  // threads per block, x-major for coalescing
    dim3 numBlocks = getOptimalGrid2D(smCount, Nx, Nz, threadsPerBlock, 2);

    calcluy_kernel<<<numBlocks, threadsPerBlock>>>(d_psi, d_cbeta, d_calphay, d_cgammay, Ay0r, Ay);
    CUDA_CHECK_KERNEL("calcluy_kernel");
    CUDA_SYNC_CHECK("calcluy");  // Only active in debug mode
    return;
}

/**
 * @brief Kernel to compute the time propagation with respect to H3 (y-part of the Laplacian).
 * @param d_psi: Device: 3D psi array
 * @param d_cbeta: Device: 3D cbeta array
 * @param d_calphay: Device: 3D calphay array
 * @param d_cgammay: Device: 3D cgammay array
 * @param Ay0r: Host to Device: Ay0r coefficient
 * @param Ay: Host to Device: Ay coefficient
 */
__global__ void calcluy_kernel(double *__restrict__ psi, double *__restrict__ cbeta,
                               const double *__restrict__ calphay,
                               const double *__restrict__ cgammay, const double Ay0r,
                               const double Ay) 
{
    // Map x to threadIdx.x for coalesced global memory access within a warp
    int cnti = blockIdx.x * blockDim.x + threadIdx.x; // x (fastest)
    int cntk = blockIdx.y * blockDim.y + threadIdx.y; // z
    
    int stride_i = gridDim.x * blockDim.x;
    int stride_k = gridDim.y * blockDim.y;

    // Grid-stride loop over z
    for (int k = cntk; k < d_Nz; k += stride_k) 
    {
        // Grid-stride loop over x
        for (int i = cnti; i < d_Nx; i += stride_i) 
        {
            // Forward elimination: fill cbeta array
            // Boundary condition: cbeta[ny-2] = psi[ny-1]
            long base_offset = k * d_Ny * d_Nx + i;
            long idx = base_offset + (d_Ny - 2) * d_Nx;     // j = Ny-2
            long idx_jp1 = base_offset + (d_Ny - 1) * d_Nx; // j+1 = Ny-1
            cbeta[idx] = psi[idx_jp1];

            // Algorithm forward sweep (y-direction) with rolling window
            double psi_jp1 = __ldg(&psi[idx_jp1]);
            double psi_j = __ldg(&psi[idx]);
            for (int cntj = d_Ny - 2; cntj > 0; cntj--) 
            {
                long idx_jm1 = idx - d_Nx; // j-1
                double psi_jm1 = __ldg(&psi[idx_jm1]);

                double c = fma(Ay0r, psi_j, fma(-Ay, psi_jm1, -Ay * psi_jp1));
                double gamma = (d_Ny - 1 <= CGALPHA_MAX) ? cgammay_c[cntj] : __ldg(&cgammay[cntj]);
                cbeta[idx_jm1] = gamma * fma(Ay, cbeta[idx], -c);

                // Roll window down in y
                psi_jp1 = psi_j;
                psi_j = psi_jm1;
                idx = idx_jm1;
            }

            // Boundary condition
            psi[base_offset + 0] = 0.0;

            // Back substitution: update psi values
            long idx_j = base_offset; // j = 0
            for (int cntj = 0; cntj < d_Ny - 2; cntj++) 
            {
                long idx_jp1_bs = idx_j + d_Nx; // j+1
                double alpha = (d_Ny - 1 <= CGALPHA_MAX) ? calphay_c[cntj] : __ldg(&calphay[cntj]);
                psi[idx_jp1_bs] = fma(alpha, psi[idx_j], cbeta[idx_j]);
                idx_j = idx_jp1_bs;
            }

            // Boundary condition
            psi[base_offset + (d_Ny - 1) * d_Nx] = 0.0;
        }
    }
}

/**
 * @brief Function to compute the time propagation with respect to H4 (z-part of the Laplacian).
 * @param d_psi: Device: 3D psi array
 * @param d_cbeta: Device: 3D cbeta array
 * @param d_calphaz: Device: 3D calphaz array
 * @param d_cgammaz: Device: 3D cgammaz array
 * @param Az0r: Host to Device: Az0r coefficient
 * @param Az: Host to Device: Az coefficient
 */
void calcluz(double *d_psi, double *d_cbeta, double *d_calphaz, double *d_cgammaz, double Az0r,
             double Az) 
{
    // Use grid-stride approach with optimal block count
    static int smCount = getGPUSMCount();
    dim3 threadsPerBlock(64, 8);  // threads per block, x-major for coalescing
    dim3 numBlocks = getOptimalGrid2D(smCount, Nx, Ny, threadsPerBlock, 2);

    calcluz_kernel<<<numBlocks, threadsPerBlock>>>(d_psi, d_cbeta, d_calphaz, d_cgammaz, Az0r, Az);
    CUDA_CHECK_KERNEL("calcluz_kernel");
    CUDA_SYNC_CHECK("calcluz");  // Only active in debug mode
    return;
}

/**
 * @brief Kernel to compute the time propagation with respect to H4 (z-part of the Laplacian).
 * @param d_psi: Device: 3D psi array
 * @param d_cbeta: Device: 3D cbeta array
 * @param d_calphaz: Device: 3D calphaz array
 * @param d_cgammaz: Device: 3D cgammaz array
 * @param Az0r: Host to Device: Az0r coefficient
 * @param Az: Host to Device: Az coefficient
 */
__global__ void calcluz_kernel(double *__restrict__ psi, double *__restrict__ cbeta,
                               const double *__restrict__ calphaz,
                               const double *__restrict__ cgammaz, const double Az0r,
                               const double Az) 
{
    int cnti = blockIdx.x * blockDim.x + threadIdx.x; // x (fastest)
    int cntj = blockIdx.y * blockDim.y + threadIdx.y; // y
    
    int stride_i = gridDim.x * blockDim.x;
    int stride_j = gridDim.y * blockDim.y;

    // Grid-stride loop over y
    for (int j = cntj; j < d_Ny; j += stride_j) 
    {
        // Grid-stride loop over x
        for (int i = cnti; i < d_Nx; i += stride_i) 
        {
            // Base offset for this (i,j) x-y position - points to z=0
            const long base_offset = j * d_Nx + i;
            const long stride = d_Ny * d_Nx; // stride to move in z-direction

            // Forward elimination: fill cbeta array
            // Boundary condition: cbeta[nz-2] = psi[nz-1]
            cbeta[base_offset + (d_Nz - 2) * stride] = psi[base_offset + (d_Nz - 1) * stride];

            // Algorithm forward sweep (working in z-direction) with rolling window to reduce global loads
            long idx_kp1 = base_offset + (d_Nz - 1) * stride; // k+1
            long idx_k = base_offset + (d_Nz - 2) * stride;   // k
            double psi_kp1 = __ldg(&psi[idx_kp1]);
            double psi_k = __ldg(&psi[idx_k]);
            for (int cntk = d_Nz - 2; cntk > 0; cntk--) 
            {
                long idx_km1 = idx_k - stride; // k-1
                double psi_km1 = __ldg(&psi[idx_km1]);

                double c = fma(Az0r, psi_k, fma(-Az, psi_km1, -Az * psi_kp1));
                double gamma = (d_Nz - 1 <= CGALPHA_MAX) ? cgammaz_c[cntk] : __ldg(&cgammaz[cntk]);
                cbeta[idx_km1] = gamma * fma(Az, cbeta[idx_k], -c);

                // Roll window down in z
                psi_kp1 = psi_k;
                psi_k = psi_km1;
                idx_k = idx_km1;
            }

            // Boundary condition
            psi[base_offset + 0 * stride] = 0.0;

            // Back substitution: update psi values
            long idx_k_bs = base_offset; // k = 0
            for (int cntk = 0; cntk < d_Nz - 2; cntk++) 
            {
                long idx_kp1_bs = idx_k_bs + stride; // k+1
                double alpha = (d_Nz - 1 <= CGALPHA_MAX) ? calphaz_c[cntk] : __ldg(&calphaz[cntk]);
                psi[idx_kp1_bs] = fma(alpha, psi[idx_k_bs], cbeta[idx_k_bs]);
                idx_k_bs = idx_kp1_bs;
            }

            // Boundary condition
            psi[base_offset + (d_Nz - 1) * stride] = 0.0;
        }
    }
}

/**
 * @brief Function to compute the chemical potential of the system and its contributions
 * @param muen: Host: 3D muen array that stores the chemical potential of the system and its
 * contributions
 * @param d_psi: Device: 3D psi array
 * @param d_psi2: Device: 3D psi2 array
 * @param d_pot: Device: 3D trap potential array
 * @param d_psi2dd: Device: 3D psi2dd array (used for in-place FFT, must be size (Nx+2)*Ny*Nz)
 * @param d_potdd: Device: 3D dipolar potential array
 * @param forward_plan: FFT forward plan
 * @param backward_plan: FFT backward plan
 * @param integ: Simpson3DTiledIntegrator
 * @param g: Host to Device: g coefficient for contact interaction term
 * @param gd: Host to Device: gd coefficient for dipolar interaction term
 * @param h2: Host to Device: h2 coefficient for quantum fluctuation term
 */
void calcmuen(double *muen, double *d_psi, double *d_psi2, double *d_pot, double *d_psi2dd,
              double *d_potdd, cufftHandle forward_plan,
              cufftHandle backward_plan, Simpson3DTiledIntegrator &integ, const double g,
              const double gd, const double h2) 
{
    // Precompute constants
    const double inv_NxNyNz = 1.0 / ((double)Nx * Ny * Nz);
    const double half_g = 0.5 * g;
    const double half_gd = 0.5 * gd;
    const double half_h2 = 0.5 * h2;

    static int smCount = getGPUSMCount();
    dim3 threadsPerBlock(8, 8, 8);  // threads per block
    dim3 numBlocks = getOptimalGrid3D(smCount, Nx, Ny, Nz, threadsPerBlock, 2);

    // Step 1: Contact energy - Calculate 0.5 * g * ψ⁴
    calcmuen_fused_contact<<<numBlocks, threadsPerBlock>>>(d_psi, d_psi2, half_g);
    CUDA_CHECK_KERNEL("calcmuen_fused_contact");
    muen[0] = integ.integrateDevice(dx, dy, dz, d_psi2, Nx, Ny, Nz, Nx + 2);

    // Step 2: Potential energy - Calculate 0.5 * ψ² * V
    if (initialize_pot == 1) 
    {
        calcmuen_fused_potential<<<numBlocks, threadsPerBlock>>>(d_psi, d_psi2, d_pot);
        CUDA_CHECK_KERNEL("calcmuen_fused_potential");
    } 
    else 
    {
        calcmuen_fused_potential_onthefly<<<numBlocks, threadsPerBlock>>>(d_psi, d_psi2);
        CUDA_CHECK_KERNEL("calcmuen_fused_potential_onthefly");
    }
    muen[1] = integ.integrateDevice(dx, dy, dz, d_psi2, Nx, Ny, Nz, Nx + 2);

    // Step 3: Dipolar energy - requires FFT computation first
    // Use complex view of d_psi2dd for in-place FFT
    calcpsidd2(forward_plan, backward_plan, d_psi, d_psi2dd, (cufftDoubleComplex*)d_psi2dd, d_potdd);
    calcmuen_fused_dipolar<<<numBlocks, threadsPerBlock>>>(d_psi, d_psi2, d_psi2dd, half_gd);
    CUDA_CHECK_KERNEL("calcmuen_fused_dipolar");
    muen[2] = integ.integrateDevice(dx, dy, dz, d_psi2, Nx, Ny, Nz, Nx + 2) * inv_NxNyNz;

    // Step 4: Kinetic energy - calculate gradients and kinetic energy density directly
    calcmuen_kin(d_psi, d_psi2, par);
    muen[3] = integ.integrateDevice(dx, dy, dz, d_psi2, Nx, Ny, Nz, Nx + 2);

    // Step 5: H2 energy - calculate quantum fluctuation energy density
    calcmuen_fused_h2<<<numBlocks, threadsPerBlock>>>(d_psi, d_psi2, half_h2);
    CUDA_CHECK_KERNEL("calcmuen_fused_h2");
    muen[4] = integ.integrateDevice(dx, dy, dz, d_psi2, Nx, Ny, Nz, Nx + 2);

    return;
}
/**
 * @brief Kernel to calculate contact energy term
 * @param d_psi: Device: 3D psi array
 * @param d_result: Device: 3D result array
 * @param half_g: Precomputed 0.5 * g coefficient for contact interaction term
 */
__global__ void calcmuen_fused_contact(const double *__restrict__ d_psi,
                                       double *__restrict__ d_result, double half_g) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;
    
    int stride_x = gridDim.x * blockDim.x;
    int stride_y = gridDim.y * blockDim.y;
    int stride_z = gridDim.z * blockDim.z;

    for (int iz = idz; iz < d_Nz; iz += stride_z) 
    {
        for (int iy = idy; iy < d_Ny; iy += stride_y) 
        {
            for (int ix = idx; ix < d_Nx; ix += stride_x) 
            {
                int linear_idx = iz * d_Ny * d_Nx + iy * d_Nx + ix;
                // d_result is d_work_array (padded), use padded indexing
                int result_idx = iz * d_Ny * (d_Nx + 2) + iy * (d_Nx + 2) + ix;
                double psi_val = d_psi[linear_idx];
                double psi2_val = psi_val * psi_val;
                double psi4_val = psi2_val * psi2_val;
                d_result[result_idx] = psi4_val * half_g;
            }
        }
    }
}

/**
 * @brief Kernel to calculate trap potential energy term
 * @param d_psi: Device: 3D psi array
 * @param d_result: Device: 3D result array
 * @param d_pot: Device: 3D trap potential array
 */
__global__ void calcmuen_fused_potential(const double *__restrict__ d_psi,
                                         double *__restrict__ d_result,
                                         const double *__restrict__ d_pot) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;
    
    int stride_x = gridDim.x * blockDim.x;
    int stride_y = gridDim.y * blockDim.y;
    int stride_z = gridDim.z * blockDim.z;

    for (int iz = idz; iz < d_Nz; iz += stride_z) 
    {
        for (int iy = idy; iy < d_Ny; iy += stride_y) 
        {
            for (int ix = idx; ix < d_Nx; ix += stride_x) 
            {
                int linear_idx = iz * d_Ny * d_Nx + iy * d_Nx + ix;
                // d_result is d_work_array (padded), use padded indexing
                int result_idx = iz * d_Ny * (d_Nx + 2) + iy * (d_Nx + 2) + ix;
                double psi_val = __ldg(&d_psi[linear_idx]); // Read-only cache for psi
                double psi2_val = psi_val * psi_val;
                d_result[result_idx] =
                    0.5 * psi2_val * __ldg(&d_pot[linear_idx]); // Read-only cache for potential
            }
        }
    }
}

/**
 * @brief Kernel to calculate trap potential energy term with on-the-fly potential computation
 * @param d_psi: Device: 3D psi array
 * @param d_result: Device: 3D result array
 */
__global__ void calcmuen_fused_potential_onthefly(const double *__restrict__ d_psi,
                                                   double *__restrict__ d_result) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;
    
    int stride_x = gridDim.x * blockDim.x;
    int stride_y = gridDim.y * blockDim.y;
    int stride_z = gridDim.z * blockDim.z;

    for (int iz = idz; iz < d_Nz; iz += stride_z) 
    {
        for (int iy = idy; iy < d_Ny; iy += stride_y) 
        {
            for (int ix = idx; ix < d_Nx; ix += stride_x) 
            {
                int linear_idx = iz * d_Ny * d_Nx + iy * d_Nx + ix;
                // d_result is d_work_array (padded), use padded indexing
                int result_idx = iz * d_Ny * (d_Nx + 2) + iy * (d_Nx + 2) + ix;
                double psi_val = __ldg(&d_psi[linear_idx]);
                double psi2_val = psi_val * psi_val;
                // Compute potential on-the-fly
                double pot_val = compute_pot_onthefly(ix, iy, iz);
                d_result[result_idx] = 0.5 * psi2_val * pot_val;
            }
        }
    }
}

/**
 * @brief Kernel to calculate dipolar energy term
 * @param d_psi: Device: 3D psi array
 * @param d_result: Device: 3D result array
 * @param d_psidd2: Device: 3D dipolar psi squared array
 * @param half_gd: Precomputed 0.5 * gd coefficient for dipolar interaction term
 */
__global__ void calcmuen_fused_dipolar(const double *__restrict__ d_psi,
                                       double *__restrict__ d_result,
                                       const double *__restrict__ d_psidd2, const double half_gd) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;
    
    int stride_x = gridDim.x * blockDim.x;
    int stride_y = gridDim.y * blockDim.y;
    int stride_z = gridDim.z * blockDim.z;

    for (int iz = idz; iz < d_Nz; iz += stride_z) 
    {
        for (int iy = idy; iy < d_Ny; iy += stride_y) 
        {
            for (int ix = idx; ix < d_Nx; ix += stride_x) 
            {
                int linear_idx = iz * d_Ny * d_Nx + iy * d_Nx + ix;
                // d_result and d_psidd2 are both d_work_array/d_psi2dd (padded), use padded indexing
                int padded_idx = iz * d_Ny * (d_Nx + 2) + iy * (d_Nx + 2) + ix;
                double psi_val = __ldg(&d_psi[linear_idx]); // Read-only cache for psi
                double psi2_val = psi_val * psi_val;
                double psidd2_val = __ldg(&d_psidd2[padded_idx]); // Read-only cache for dipolar psi squared (padded)
                d_result[padded_idx] = psi2_val * psidd2_val * half_gd;
            }
        }
    }
}

/**
 * @brief Kernel to calculate quantum fluctuation energy term
 * @param d_psi: Device: 3D psi array
 * @param d_result: Device: 3D result array
 * @param half_h2: Precomputed 0.5 * h2 coefficient for quantum fluctuation term
 */
__global__ void calcmuen_fused_h2(const double *__restrict__ d_psi, double *__restrict__ d_result,
                                  const double half_h2) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;
    
    int stride_x = gridDim.x * blockDim.x;
    int stride_y = gridDim.y * blockDim.y;
    int stride_z = gridDim.z * blockDim.z;

    for (int iz = idz; iz < d_Nz; iz += stride_z) 
    {
        for (int iy = idy; iy < d_Ny; iy += stride_y) 
        {
            for (int ix = idx; ix < d_Nx; ix += stride_x) 
            {
                int linear_idx = iz * d_Ny * d_Nx + iy * d_Nx + ix;
                // d_result is d_work_array (padded), use padded indexing
                int result_idx = iz * d_Ny * (d_Nx + 2) + iy * (d_Nx + 2) + ix;
                double psi_val = __ldg(&d_psi[linear_idx]); // Read-only cache for psi
                double psi5_val = psi_val * psi_val * psi_val * psi_val * psi_val;
                d_result[result_idx] = psi5_val * half_h2;
            }
        }
    }
}

/**
 * @brief Function to calculate kinetic energy term
 * @param d_psi: Device: 3D psi array
 * @param d_work_array: Device: 3D work array
 * @param par: Host to Device: par coefficient either 1 or 2, defined in Input file
 */
void calcmuen_kin(double *d_psi, double *d_work_array, int par) 
{
    // d_work_array is padded (Nx+2, Ny, Nz)
    diff(dx, dy, dz, d_psi, d_work_array, Nx, Ny, Nz, par, Nx + 2);
}

/**
 * @brief Function to output the rms values
 * @param filerms: File pointer to the rms output file
 */
void rms_output(FILE *filerms) {
    std::fprintf(filerms, "\n**********************************************\n");

    // Print memory mode
    if (initialize_pot == 1) 
    {
        std::fprintf(filerms, "Potential mode: PRECOMPUTED (stored on GPU)\n\n");
    } 
    else 
    {
        std::fprintf(filerms, "Potential mode: ON-THE-FLY (computed in kernels)\n\n");
    }

    std::fprintf(filerms,
                    "Contact: Natoms = %.11le, as = %.6le * a0, G = %.6le, G * par = %.6le\n", Nad, as, g / par, g);

    std::fprintf(filerms, "DDI: add = %.6le * a0, GD = %.6le, GD * par = %.6le, edd = %.6le\n", add, gd / par, gd, edd);
    
    std::fprintf(filerms, "\t\tDipolar cutoff Scut = %.6le,\n\n", cutoff);

    if (QF == 1) 
    {
        std::fprintf(filerms, "Quantum fluctuation: QF = 1: h2 = %.16le,\t\tq5 = %.16le\n", h2, q5);
    } 
    else
    {
        std::fprintf(filerms, "Quantum fluctuation: QF = 0\n\n");
    }

    std::fprintf(filerms, "Trap parameters:\nGAMMA = %.6le, NU = %.6le, LAMBDA = %.6le\n", vgamma,
                 vnu, vlambda);

    std::fprintf(filerms, "\nSpace discretization: NX = %li, NY = %li, NZ = %li\n", Nx, Ny, Nz);

    std::fprintf(filerms,
                "\t\tDX = %.16le, DY = %.16le, DZ = %.16le, mx = %.2le, my = "
                "%.2le, mz = %.2le\n", dx, dy, dz, mx, my, mz);

    std::fprintf(filerms, "\t\tUnit of length: aho = %.6le m\n", aho);

    std::fprintf(filerms, "\nTime discretization: NITER = %li (NSNAP = %li)\n", Niter, Nsnap);

    std::fprintf(filerms, "\t\tDT = %.6le, mt = %.2le\n\n", dt, mt);
    
    std::fprintf(filerms, "Gaussian:\nSX = %.6le, SY = %.6le, SZ = %.6le\n", sx, sy, sz);

    std::fprintf(filerms, "\nMUREL = %.6le, MUEND=%.6le\n\n", murel, muend);

    std::fprintf(filerms, "--------------------------------------------------------------------------------------------------------\n");
    std::fprintf(filerms, "Snap\t\t\t\t<r>\t\t\t\t\t<x>\t\t\t\t\t\t<y>\t\t\t\t\t\t<z>\n");
    std::fprintf(filerms, "--------------------------------------------------------------------------------------------------------\n");
    fflush(filerms);
}

/**
 * @brief Function to output the chemical potential values
 * @param filemu: File pointer to the chemical potential output file
 */
void mu_output(FILE *filemu) {
    std::fprintf(filemu, "\n**********************************************\n");
    std::fprintf(filemu, "Contact: Natoms = %.11le, as = %.6le * a0, G = %.6le, G * par = %.6le\n", Nad, as, g / par, g);
    
    std::fprintf(filemu, "DDI: add = %.6le * a0, GD = %.6le, GD * par = %.6le, edd = %.6le\n", add, gd / par, gd, edd);
    
    std::fprintf(filemu, "\t\tDipolar cutoff Scut = %.6le,\n\n", cutoff);
    
    if (QF == 1) 
    {
        std::fprintf(filemu, "Quantum fluctuation: QF = 1: h2 = %.6le,\t\tq5 = %.6le, \n\n", h2, q5);
    } 
    else
    {
        std::fprintf(filemu, "Quantum fluctuation: QF = 0\n\n");
    }

    std::fprintf(filemu, "Trap parameters:\nGAMMA = %.6le, NU = %.6le, LAMBDA = %.6le\n", vgamma,
                 vnu, vlambda);

    std::fprintf(filemu, "\nSpace discretization: NX = %li, NY = %li, NZ = %li\n", Nx, Ny, Nz);

    std::fprintf(filemu,
                "\t\tDX = %.16le, DY = %.16le, DZ = %.16le, mx = %.2le, my = "
                "%.2le, mz = %.2le\n", dx, dy, dz, mx, my, mz);

    std::fprintf(filemu, "\t\tUnit of length: aho = %.6le m\n", aho);
    
    std::fprintf(filemu, "\nTime discretization: NITER = %li (NSNAP = %li)\n", Niter, Nsnap);

    std::fprintf(filemu, "\t\tDT = %.6le, mt = %.2le\n\n", dt, mt);
    
    std::fprintf(filemu, "Gaussian:\nSX = %.6le, SY = %.6le, SZ = %.6le\n", sx, sy, sz);

    std::fprintf(filemu, "\nMUREL = %.6le, MUEND=%.6le\n\n", murel, muend);

    std::fprintf(
        filemu,
        "-------------------------------------------------------------------------------------------------------------------------------------------------------\n");
        std::fprintf(filemu, "Snap\t\t\t\tmu\t\t\t\t\tKin\t\t\t\t\t\tPot\t\t\t\t\t\tContact\t\t\t\t\t"
            "DDI\t\t\t\t\tQF\n");
    std::fprintf(
        filemu,
        "-------------------------------------------------------------------------------------------------------------------------------------------------------\n");
    fflush(filemu);
}

/**
 * @brief Function to save the psi array from the GPU to a binary file
 * @param psi: Host: 3D psi array
 * @param d_psi: Device: 3D psi array
 * @param filename: Name of the file to save the psi array
 * @param Nx: Host: Nx number of grid points in x direction
 * @param Ny: Host: Ny number of grid points in y direction
 * @param Nz: Host: Nz number of grid points in z direction
 */
void save_psi_from_gpu(double *psi, double *d_psi, const char *filename, long Nx, long Ny,
                       long Nz) 
{
    // Allocate host memory
    size_t total_size = Nx * Ny * Nz;

    // Copy from device to host with error checking
    CUDA_CHECK(cudaMemcpy(psi, d_psi, total_size * sizeof(double), cudaMemcpyDeviceToHost));

    // Write to binary file
    FILE *file = fopen(filename, "wb");
    if (file == NULL) 
    {
        fprintf(stderr, "Failed to open file %s\n", filename);
        return;
    }

    // Write the entire array
    size_t written = fwrite(psi, sizeof(double), total_size, file);
    if (written != total_size) 
    {
        fprintf(stderr, "Failed to write all data: wrote %zu of %zu elements\n", written,
                total_size);
    }

    fclose(file);
}

/**
 * @brief Function to read the psi array from a binary file
 * @param psi: Host: 3D psi array
 * @param filename: Name of the file to read the psi array
 * @param Nx: Host: Nx number of grid points in x direction
 * @param Ny: Host: Ny number of grid points in y direction
 * @param Nz: Host: Nz number of grid points in z direction
 */
void read_psi_from_file(double *psi, const char *filename, long Nx, long Ny, long Nz) 
{
    size_t total_size = Nx * Ny * Nz;

    // Open file
    FILE *file = fopen(filename, "rb"); // Note: "rb" for binary read
    if (file == NULL) 
    {
        fprintf(stderr, "Failed to open file %s\n", filename);
        return;
    }

    // Read data
    size_t read_count = fread(psi, sizeof(double), total_size, file);
    if (read_count != total_size) 
    {
        fprintf(stderr, "Failed to read all data: read %zu of %zu elements\n", read_count,
                total_size);
        fclose(file);
    }
    fclose(file);
}

/**
 * @brief Function to output the integrated density in x direction n(x)
 * @param psi: Host: 3D psi array
 * @param x: Host: x array
 * @param tmpy: Host: temporary array for y direction
 * @param tmpz: Host: temporary array for z direction
 * @param file: File pointer to the output file
 */
void outdenx(double *psi, MultiArray<double> &x, MultiArray<double> &tmpy, MultiArray<double> &tmpz,
             FILE *file) 
{
    for (long cnti = 0; cnti < Nx; cnti++) 
    {
        // For each x position
        for (long cntj = 0; cntj < Ny; cntj++) 
        {
            // Compute |psi|^2
            for (long cntk = 0; cntk < Nz; cntk++) 
            {
                tmpz[cntk] =
                    psi[cntk * Ny * Nx + cntj * Nx + cnti] * psi[cntk * Ny * Nx + cntj * Nx + cnti];
            }
            // Integrate over z -> store in tmpy[cntj]
            tmpy[cntj] = simpint(dz, tmpz.raw(), Nz);
        }
        // Integrate over y to get n(x)
        double n_x = simpint(dy, tmpy.raw(), Ny);

        // Write x position and n(x) to file
        fwrite(&x[cnti], sizeof(double), 1, file);
        fwrite(&n_x, sizeof(double), 1, file);
    }
}

/**
 * @brief Function to output the integrated density in y direction n(y)
 * @param psi: Host: 3D psi array
 * @param y: Host: y array
 * @param tmpx: Host: temporary array for x direction
 * @param tmpz: Host: temporary array for z direction
 * @param file: File pointer to the output file
 */
void outdeny(double *psi, MultiArray<double> &y, MultiArray<double> &tmpx, MultiArray<double> &tmpz,
             FILE *file) 
{
    for (long cntj = 0; cntj < Ny; cntj++) 
    {
        // For each y position
        for (long cnti = 0; cnti < Nx; cnti++) 
        {
            // Compute |psi|^2
            for (long cntk = 0; cntk < Nz; cntk++) 
            {
                tmpz[cntk] =
                    psi[cntk * Ny * Nx + cntj * Nx + cnti] * psi[cntk * Ny * Nx + cntj * Nx + cnti];
            }
            // Integrate over z -> store in tmpx[cnti]
            tmpx[cnti] = simpint(dz, tmpz.raw(), Nz);
        }
        // Integrate over x to get n(y)
        double n_y = simpint(dx, tmpx.raw(), Nx);
        // Write y position and n(y) to file
        fwrite(&y[cntj], sizeof(double), 1, file);
        fwrite(&n_y, sizeof(double), 1, file);
    }
}

/**
 * @brief Function to output the integrated density in z direction n(z)
 * @param psi: Host: 3D psi array
 * @param z: Host: z array
 * @param tmpx: Host: temporary array for x direction
 * @param tmpy: Host: temporary array for y direction
 * @param file: File pointer to the output file
 */
void outdenz(double *psi, MultiArray<double> &z, MultiArray<double> &tmpx, MultiArray<double> &tmpy,
             FILE *file) 
{
    for (long cntk = 0; cntk < Nz; cntk++) 
    {
        // For each z position
        for (long cntj = 0; cntj < Ny; cntj++) 
        {
            // Compute |psi|^2
            for (long cnti = 0; cnti < Nx; cnti++) 
            {
                tmpy[cntj] =
                    fma(psi[cntk * Ny * Nx + cntj * Nx + cnti], psi[cntk * Ny * Nx + cntj * Nx + cnti], 0.0);
            }
            // Integrate over x -> store in tmpy[cntj]
            tmpy[cntj] = simpint(dx, tmpy.raw(), Nx);
        }
        // Integrate over y to get n(z)
        double n_z = simpint(dy, tmpy.raw(), Ny);

        // Write z position and n(z) to file
        fwrite(&z[cntk], sizeof(double), 1, file);
        fwrite(&n_z, sizeof(double), 1, file);
    }
}

/**
 * @brief Function to output the integrated density in x and y direction n(x,y)
 * @param psi: Host: 3D psi array
 * @param x: Host: x array
 * @param y: Host: y array
 * @param tmpz: Host: temporary array for z direction
 * @param file: File pointer to the output file
 */
void outdenxy(double *psi, MultiArray<double> &x, MultiArray<double> &y, MultiArray<double> &tmpz,
              FILE *file) 
{
    for (long cnti = 0; cnti < Nx; cnti++) 
    {
        for (long cntj = 0; cntj < Ny; cntj++) 
        {
            for (long cntk = 0; cntk < Nz; cntk++) 
            {
                tmpz[cntk] = fma(psi[cntk * Ny * Nx + cntj * Nx + cnti],
                                 psi[cntk * Ny * Nx + cntj * Nx + cnti], 0.0);
            }
            double n_xy = simpint(dz, tmpz.raw(), Nz);
            // Write x,y, n(x,y) to file
            fwrite(&x[cnti], sizeof(double), 1, file);
            fwrite(&y[cntj], sizeof(double), 1, file);
            fwrite(&n_xy, sizeof(double), 1, file);
        }
    }
}

/**
 * @brief Function to output the integrated density in x and z direction n(x,z)
 * @param psi: Host: 3D psi array
 * @param x: Host: x array
 * @param z: Host: z array
 * @param tmpx: Host: temporary array for x direction
 * @param file: File pointer to the output file
 */
void outdenxz(double *psi, MultiArray<double> &x, MultiArray<double> &z, MultiArray<double> &tmpx,
              FILE *file) 
{
    for (long cnti = 0; cnti < Nx; cnti++) 
    {
        for (long cntk = 0; cntk < Nz; cntk++) 
        {
            for (long cntj = 0; cntj < Ny; cntj++) 
            {
                tmpx[cntj] = fma(psi[cntk * Ny * Nx + cntj * Nx + cnti],
                                 psi[cntk * Ny * Nx + cntj * Nx + cnti], 0.0);
            }
            double n_xz = simpint(dy, tmpx.raw(), Ny);
            // Write x,z, n(x,z) to file
            fwrite(&x[cnti], sizeof(double), 1, file);
            fwrite(&z[cntk], sizeof(double), 1, file);
            fwrite(&n_xz, sizeof(double), 1, file);
        }
    }
}

/**
 * @brief Function to output the integrated density in y and z direction n(y,z)
 * @param psi: Host: 3D psi array
 * @param y: Host: y array
 * @param z: Host: z array
 * @param tmpx: Host: temporary array for x direction
 * @param file: File pointer to the output file
 */
void outdenyz(double *psi, MultiArray<double> &y, MultiArray<double> &z, MultiArray<double> &tmpx,
              FILE *file) 
{
    for (long cntj = 0; cntj < Ny; cntj++) 
    {
        for (long cntk = 0; cntk < Nz; cntk++) 
        {
            for (long cnti = 0; cnti < Nx; cnti++) 
            {
                tmpx[cnti] = fma(psi[cntk * Ny * Nx + cntj * Nx + cnti],
                                 psi[cntk * Ny * Nx + cntj * Nx + cnti], 0.0);
            }
            double n_yz = simpint(dx, tmpx.raw(), Nx);
            // Write y,z, n(y,z) to file
            fwrite(&y[cntj], sizeof(double), 1, file);
            fwrite(&z[cntk], sizeof(double), 1, file);
            fwrite(&n_yz, sizeof(double), 1, file);
        }
    }
}

/**
 * @brief Function to output the squared wave function in x and y direction psi2(x,y) for z = 0
 * @param psi: Host: 3D psi array
 * @param x: Host: x array
 * @param y: Host: y array
 * @param file: File pointer to the output file
 */
void outpsi2xy(double *psi, MultiArray<double> &x, MultiArray<double> &y, FILE *file) 
{
    for (long cnti = 0; cnti < Nx; cnti++) 
    {
        for (long cntj = 0; cntj < Ny; cntj++) 
        {
            double psi2_xy = fma(psi[Nz2 * Ny * Nx + cntj * Nx + cnti],
                                 psi[Nz2 * Ny * Nx + cntj * Nx + cnti], 0.0);
            fwrite(&x[cnti], sizeof(double), 1, file);
            fwrite(&y[cntj], sizeof(double), 1, file);
            fwrite(&psi2_xy, sizeof(double), 1, file);
        }
    }
}

/**
 * @brief Function to output the squared wave function in x and z direction psi2(x,z) for y = 0
 * @param psi: Host: 3D psi array
 * @param x: Host: x array
 * @param z: Host: z array
 * @param file: File pointer to the output file
 */
void outpsi2xz(double *psi, MultiArray<double> &x, MultiArray<double> &z, FILE *file) {
    for (long cnti = 0; cnti < Nx; cnti++) 
    {
        for (long cntk = 0; cntk < Nz; cntk++) 
        {
            double psi2_xz = fma(psi[cntk * Ny * Nx + Ny2 * Nx + cnti],
                                 psi[cntk * Ny * Nx + Ny2 * Nx + cnti], 0.0);
            fwrite(&x[cnti], sizeof(double), 1, file);
            fwrite(&z[cntk], sizeof(double), 1, file);
            fwrite(&psi2_xz, sizeof(double), 1, file);
        }
    }
}

/**
 * @brief Function to output the squared wave function in y and z direction psi2(y,z) for x = 0
 * @param psi: Host: 3D psi array
 * @param y: Host: y array
 * @param z: Host: z array
 * @param file: File pointer to the output file
 */
void outpsi2yz(double *psi, MultiArray<double> &y, MultiArray<double> &z, FILE *file) {
    for (long cntj = 0; cntj < Ny; cntj++) 
    {
        for (long cntk = 0; cntk < Nz; cntk++) 
        {
            double psi2_yz = fma(psi[cntk * Ny * Nx + cntj * Nx + Nx2],
                                 psi[cntk * Ny * Nx + cntj * Nx + Nx2], 0.0);
            fwrite(&y[cntj], sizeof(double), 1, file);
            fwrite(&z[cntk], sizeof(double), 1, file);
            fwrite(&psi2_yz, sizeof(double), 1, file);
        }
    }
}
