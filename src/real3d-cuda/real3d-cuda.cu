#include "real3d-cuda.cuh"
#include <cuComplex.h>

int main(int argc, char **argv) {
    if ((argc != 3) || (strcmp(*(argv + 1), "-i") != 0)) {
        std::fprintf(stderr, "Usage: %s -i <input parameter file> \n", *argv);
        exit(EXIT_FAILURE);
    }

    if (!cfg_init(argv[2])) {
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

    gd *= MS;
    edd = (4. * pi / 3.) * gd / g;
    Nad = Na;
    if (fabs(edd) < 1e-10) {
        q5 = 1.;
    } else {
        if (fabs(edd - 1.) < 1e-10) {
            q5 = 3. * sqrt(3.) / 2.;
        } else {
            std::complex<double> edd_c(edd, 0.0);

            std::complex<double> sqrt_edd = std::sqrt(edd_c);
            std::complex<double> sqrt_1_2edd = std::sqrt(1.0 + 2.0 * edd_c);
            std::complex<double> log_term =
                std::log(1.0 - edd_c) - 2.0 * std::log(-std::sqrt(3.0) * sqrt_edd + sqrt_1_2edd);

            std::complex<double> term1 = 6.0 * sqrt_1_2edd * (11.0 + edd * (4.0 + 9.0 * edd));
            std::complex<double> term2 =
                (5.0 * std::sqrt(3.0) * std::pow(-1.0 + edd_c, 3.0) * log_term) / sqrt_edd;

            q5 = ((term1 - term2) / 96.0).real();
        }
    }
    q5 *= QF;
    h2 = 32. * sqrt(pi) * pow(as * BOHR_RADIUS / aho, 2.5) * pow(Nad, 1.5) * (4. * q5) / 3.;
    h2 = par * h2;

    Nx2 = Nx / 2;
    Ny2 = Ny / 2;
    Nz2 = Nz / 2;

    // Copy constants Nx, Ny, Nz (number of grid points), dt (time step) to device
    cudaMemcpyToSymbol(d_Nx, &Nx, sizeof(long));
    cudaMemcpyToSymbol(d_Ny, &Ny, sizeof(long));
    cudaMemcpyToSymbol(d_Nz, &Nz, sizeof(long));
    cudaMemcpyToSymbol(d_dt, &dt, sizeof(double));

    // Initialize squared grid spacings
    dx2 = dx * dx;
    dy2 = dy * dy;
    dz2 = dz * dz;

    // Allocate memory for psi (pinned memory) and squared wave function (psi2)
    cuDoubleComplex *psi;
    cudaMallocHost(&psi, Nz * Ny * Nx * sizeof(cuDoubleComplex));
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
    MultiArray<cuDoubleComplex> calphax(Nx - 1), cgammax(Nx - 1);
    MultiArray<cuDoubleComplex> calphay(Ny - 1), cgammay(Ny - 1);
    MultiArray<cuDoubleComplex> calphaz(Nz - 1), cgammaz(Nz - 1);
    cuDoubleComplex Ax0, Ay0, Az0, Ax0r, Ay0r, Az0r, Ax, Ay, Az;

    // Initialize temporary arrays for density
    MultiArray<double> tmpx(Nx), tmpy(Ny), tmpz(Nz);

    // Setup Simpson3DTiledIntegrator for integration
    long TILE_SIZE = Nz;
    Simpson3DTiledIntegrator integ(Nx, Ny, TILE_SIZE);

    // Allocation of crank-nicolson coefficients on device
    CudaArray3D<cuDoubleComplex> d_calphax(Nx - 1);
    CudaArray3D<cuDoubleComplex> d_cgammax(Nx - 1);
    CudaArray3D<cuDoubleComplex> d_calphay(Ny - 1);
    CudaArray3D<cuDoubleComplex> d_cgammay(Ny - 1);
    CudaArray3D<cuDoubleComplex> d_calphaz(Nz - 1);
    CudaArray3D<cuDoubleComplex> d_cgammaz(Nz - 1);

    // Allocate memory for psi on device
    CudaArray3D<cuDoubleComplex> d_psi(Nx, Ny, Nz, false);

    // Allocate memory for work array on device
    CudaArray3D<double> d_work_array(Nx, Ny, Nz, true);
    CudaArray3D<cuDoubleComplex> d_work_array_complex(Nx, Ny, Nz, false);

    // Allocate memory for trap potential (d_pot) and dipole potential (d_potdd)
    CudaArray3D<double> d_pot(Nx, Ny, Nz, false);
    CudaArray3D<double> d_potdd(Nx, Ny, Nz, false);
    // Reuse d_work_array as the real buffer for psidd2 to reduce memory

    // FFT arrays
    cufftDoubleComplex *d_psi2_fft;
    cudaMalloc(&d_psi2_fft, Nz * Ny * (Nx2 + 1) * sizeof(cufftDoubleComplex));

    // Create plan for FFT of 3D array with explicit work area management
    cufftHandle forward_plan, backward_plan;
    size_t forward_worksize, backward_worksize;
    void *forward_work = nullptr;
    void *backward_work = nullptr;

    // Create forward plan
    cufftResult res = cufftCreate(&forward_plan);
    if (res != CUFFT_SUCCESS) {
        std::cerr << "CUFFT error: Forward plan creation failed" << std::endl;
        return -1;
    }

    res = cufftMakePlan3d(forward_plan, Nz, Ny, Nx, CUFFT_D2Z, &forward_worksize);
    if (res != CUFFT_SUCCESS) {
        std::cerr << "CUFFT error: Forward plan setup failed" << std::endl;
        cufftDestroy(forward_plan);
        return -1;
    }

    // Defer work area allocation until both plan sizes are known; will share one buffer

    // Create backward plan
    res = cufftCreate(&backward_plan);
    if (res != CUFFT_SUCCESS) {
        std::cerr << "CUFFT error: Backward plan creation failed" << std::endl;
        if (forward_work)
            cudaFree(forward_work);
        cufftDestroy(forward_plan);
        return -1;
    }

    res = cufftMakePlan3d(backward_plan, Nz, Ny, Nx, CUFFT_Z2D, &backward_worksize);
    if (res != CUFFT_SUCCESS) {
        std::cerr << "CUFFT error: Backward plan setup failed" << std::endl;
        if (forward_work)
            cudaFree(forward_work);
        cufftDestroy(forward_plan);
        cufftDestroy(backward_plan);
        return -1;
    }

    // Allocate a single shared work area for both forward and backward plans
    {
        size_t shared_worksize =
            (forward_worksize > backward_worksize) ? forward_worksize : backward_worksize;
        if (shared_worksize > 0) {
            cudaMalloc(&forward_work, shared_worksize);
            cufftSetWorkArea(forward_plan, forward_work);
            cufftSetWorkArea(backward_plan, forward_work);
        }
    }

    // Allocate pinned memory for RMS results
    double *h_rms_pinned;
    cudaHostAlloc(&h_rms_pinned, 3 * sizeof(double), cudaHostAllocDefault);

    // Initialize RMS output file that will store root mean square values <r>, <x>, <y>, <z>
    if (rmsout != NULL) {
        sprintf(filename, "%s.txt", rmsout);
        filerms = fopen(filename, "w");
    } else
        filerms = NULL;

    // Initialize chemical potential output file that will store chemical potential values, total
    // chemical pot., kinetic, trap, contact, dipole and quantum fluctuation terms
    if (muoutput != NULL) {
        sprintf(filename, "%s.txt", muoutput);
        filemu = fopen(filename, "w");
    } else
        filemu = NULL;

    // Initialize psi function
    initpsi((double *)psi, x2, y2, z2, x, y, z);

    // Read psi function from file that was saved from imaginary time propagation
    read_psi_from_file_complex(psi, input, Nx, Ny, Nz);

    // Initialize trap potential and dipole potential
    initpot(pot, x2, y2, z2);
    initpotdd(potdd, kx, ky, kz, kx2, ky2, kz2);

    // Generate coefficients
    gencoef(calphax, cgammax, calphay, cgammay, calphaz, cgammaz, Ax0, Ay0, Az0, Ax0r, Ay0r, Az0r,
            Ax, Ay, Az);

    minusAx = make_cuDoubleComplex(0., -Ax.y);
    minusAy = make_cuDoubleComplex(0., -Ay.y);
    minusAz = make_cuDoubleComplex(0., -Az.y);
    cudaMemcpyToSymbol(d_minusAx, &minusAx, sizeof(cuDoubleComplex));
    cudaMemcpyToSymbol(d_minusAy, &minusAy, sizeof(cuDoubleComplex));
    cudaMemcpyToSymbol(d_minusAz, &minusAz, sizeof(cuDoubleComplex));

    // Copy coefficients to device
    d_calphax.copyFromHost(calphax.raw());
    d_cgammax.copyFromHost(cgammax.raw());
    d_calphay.copyFromHost(calphay.raw());
    d_cgammay.copyFromHost(cgammay.raw());
    d_calphaz.copyFromHost(calphaz.raw());
    d_cgammaz.copyFromHost(cgammaz.raw());

    // Copy psi data to device
    d_psi.copyFromHost(psi);

    // Copy trap potential and dipole potential to device
    d_pot.copyFromHost(pot.raw());
    d_potdd.copyFromHost(potdd.raw());

    if (rmsout != NULL) {
        rms_output(filerms);
    }
    if (muoutput != NULL) {
        mu_output(filemu);
    }

    // Compute wave function norm
    calcnorm(d_psi, d_work_array, norm, integ);

    // Compute RMS values
    compute_rms_values(d_psi, d_work_array, integ, h_rms_pinned);
    if (rmsout != NULL) {
        double rms_r = sqrt(h_rms_pinned[0] * h_rms_pinned[0] + h_rms_pinned[1] * h_rms_pinned[1] +
                            h_rms_pinned[2] * h_rms_pinned[2]);
        std::fprintf(filerms, "%-9d %-19.10le %-19.16le %-19.16le %-19.16le\n", 0, rms_r,
                     h_rms_pinned[0], h_rms_pinned[1], h_rms_pinned[2]);
        fflush(filerms);
    }

    // Compute chemical potential terms
    if (muoutput != NULL) {
        calcmuen(muen, d_psi, d_work_array, d_pot, d_work_array, d_potdd, d_psi2_fft, forward_plan,
                 backward_plan, integ, g, gd, h2);
        std::fprintf(filemu, "%-9d %-19.16le %-19.16le %-19.16le %-19.16le %-19.16le %-19.16le\n",
                     0, muen[0] + muen[1] + muen[2] + muen[3], muen[3], muen[1], muen[0], muen[2],
                     muen[4]);
        fflush(filemu);
        mutotold = muen[0] + muen[1] + muen[2] + muen[3];
    }

    if (Niterout != NULL) {
        char itername[10];
        sprintf(itername, "-%06d-", 0);
        if (outflags & DEN_X) {
            // Open binary file for writing
            sprintf(filename, "%s%s1d_x.bin", Niterout, itername);
            file = fopen(filename, "wb");
            if (file == NULL) {
                std::fprintf(stderr, "Failed to open file %s\n", filename);
                exit(EXIT_FAILURE);
            }
            outdenx(psi, x, tmpy, tmpz, file);
            fclose(file);
        }
        if (outflags & DEN_Y) {
            sprintf(filename, "%s%s1d_y.bin", Niterout, itername);
            file = fopen(filename, "wb");
            if (file == NULL) {
                std::fprintf(stderr, "Failed to open file %s\n", filename);
                exit(EXIT_FAILURE);
            }
            outdeny(psi, y, tmpx, tmpz, file);
            fclose(file);
        }
        if (outflags & DEN_Z) {
            sprintf(filename, "%s%s1d_z.bin", Niterout, itername);
            file = fopen(filename, "wb");
            if (file == NULL) {
                std::fprintf(stderr, "Failed to open file %s\n", filename);
                exit(EXIT_FAILURE);
            }
            outdenz(psi, z, tmpx, tmpy, file);
            fclose(file);
        }
        if (outflags & DEN_XY) {
            // Open binary file for writing
            sprintf(filename, "%s%s2d_xy.bin", Niterout, itername);
            file = fopen(filename, "wb");
            if (file == NULL) {
                std::fprintf(stderr, "Failed to open file %s\n", filename);
                exit(EXIT_FAILURE);
            }
            outdenxy(psi, x, y, tmpz, file);
            fclose(file);
        }
        if (outflags & DEN_XZ) {
            // Open binary file for writing
            sprintf(filename, "%s%s2d_xz.bin", Niterout, itername);
            file = fopen(filename, "wb");
            if (file == NULL) {
                std::fprintf(stderr, "Failed to open file %s\n", filename);
                exit(EXIT_FAILURE);
            }
            outdenxz(psi, x, z, tmpy, file);
            fclose(file);
        }
        if (outflags & DEN_YZ) {
            // Open binary file for writing
            sprintf(filename, "%s%s2d_yz.bin", Niterout, itername);
            file = fopen(filename, "wb");
            if (file == NULL) {
                std::fprintf(stderr, "Failed to open file %s\n", filename);
                exit(EXIT_FAILURE);
            }
            outdenyz(psi, y, z, tmpx, file);
            fclose(file);
        }
        if (outflags & DEN_XY0) {
            sprintf(filename, "%s%s3d_xy0.bin", Niterout, itername);
            file = fopen(filename, "wb");
            if (file == NULL) {
                std::fprintf(stderr, "Failed to open file %s\n", filename);
                exit(EXIT_FAILURE);
            }
            outpsi2xy(psi, x, y, file);
            fclose(file);
        }
        if (outflags & DEN_X0Z) {
            sprintf(filename, "%s%s3d_x0z.bin", Niterout, itername);
            file = fopen(filename, "wb");
            if (file == NULL) {
                std::fprintf(stderr, "Failed to open file %s\n", filename);
                exit(EXIT_FAILURE);
            }
            outpsi2xz(psi, x, z, file);
            fclose(file);
        }
        if (outflags & DEN_0YZ) {
            sprintf(filename, "%s%s3d_0yz.bin", Niterout, itername);
            file = fopen(filename, "wb");
            if (file == NULL) {
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
    auto start = std::chrono::high_resolution_clock::now();
    for (long snap = 1; snap <= Nsnap; snap++) {
        for (long j = 0; j < nsteps; j++) {
            calc_psid2_potdd(forward_plan, backward_plan, d_psi.raw(), d_work_array.raw(),
                             d_psi2_fft, d_potdd.raw());
            calcnu(d_psi, d_work_array, d_pot, g, gd, h2);
            calclux(d_psi, d_work_array_complex.raw(), d_calphax, d_cgammax, Ax0r, Ax);
            calcluy(d_psi, d_work_array_complex.raw(), d_calphay, d_cgammay, Ay0r, Ay);
            calcluz(d_psi, d_work_array_complex.raw(), d_calphaz, d_cgammaz, Az0r, Az);
            calcnorm(d_psi, d_work_array, norm, integ);
        }

        compute_rms_values(d_psi, d_work_array, integ, h_rms_pinned);

        if (rmsout != NULL) {
            double rms_r =
                sqrt(h_rms_pinned[0] * h_rms_pinned[0] + h_rms_pinned[1] * h_rms_pinned[1] +
                     h_rms_pinned[2] * h_rms_pinned[2]);
            std::fprintf(filerms, "%-9li %-19.10le %-19.16le %-19.16le %-19.16le\n", snap, rms_r,
                         h_rms_pinned[0], h_rms_pinned[1], h_rms_pinned[2]);
            fflush(filerms);
        }

        calcmuen(muen, d_psi, d_work_array, d_pot, d_work_array, d_potdd, d_psi2_fft, forward_plan,
                 backward_plan, integ, g, gd, h2);
        if (muoutput != NULL) {
            std::fprintf(
                filemu, "%-9li %-19.16le %-19.16le %-19.16le %-19.16le %-19.16le %-19.16le\n", snap,
                muen[0] + muen[1] + muen[2] + muen[3], muen[3], muen[1], muen[0], muen[2], muen[4]);
            fflush(filemu);
        }
        if (Niterout != NULL) {
            // Move d_psi to host, host is pinned memory
            cudaMemcpy(psi, d_psi.data(), Nx * Ny * Nz * sizeof(double), cudaMemcpyDeviceToHost);
            char itername[32]; // Increased buffer size to prevent overflow
            sprintf(itername, "-%06li-", snap);
            if (outflags & DEN_X) {
                sprintf(filename, "%s%s1d_x.bin", Niterout, itername);
                file = fopen(filename, "wb");
                if (file == NULL) {
                    std::fprintf(stderr, "Failed to open file %s\n", filename);
                    exit(EXIT_FAILURE);
                }
                outdenx(psi, x, tmpy, tmpz, file);
                fclose(file);
            }
            if (outflags & DEN_Y) {
                sprintf(filename, "%s%s1d_y.bin", Niterout, itername);
                file = fopen(filename, "wb");
                if (file == NULL) {
                    std::fprintf(stderr, "Failed to open file %s\n", filename);
                    exit(EXIT_FAILURE);
                }
                outdeny(psi, y, tmpx, tmpz, file);
                fclose(file);
            }
            if (outflags & DEN_Z) {
                sprintf(filename, "%s%s1d_z.bin", Niterout, itername);
                file = fopen(filename, "wb");
                if (file == NULL) {
                    std::fprintf(stderr, "Failed to open file %s\n", filename);
                    exit(EXIT_FAILURE);
                }
                outdenz(psi, z, tmpx, tmpy, file);
                fclose(file);
            }
            if (outflags & DEN_XY) {
                sprintf(filename, "%s%s2d_xy.bin", Niterout, itername);
                file = fopen(filename, "wb");
                if (file == NULL) {
                    std::fprintf(stderr, "Failed to open file %s\n", filename);
                    exit(EXIT_FAILURE);
                }
                outdenxy(psi, x, y, tmpz, file);
                fclose(file);
            }
            if (outflags & DEN_XZ) {
                sprintf(filename, "%s%s2d_xz.bin", Niterout, itername);
                file = fopen(filename, "wb");
                if (file == NULL) {
                    std::fprintf(stderr, "Failed to open file %s\n", filename);
                    exit(EXIT_FAILURE);
                }
                outdenxz(psi, x, z, tmpy, file);
                fclose(file);
            }
            if (outflags & DEN_YZ) {
                sprintf(filename, "%s%s2d_yz.bin", Niterout, itername);
                file = fopen(filename, "wb");
                if (file == NULL) {
                    std::fprintf(stderr, "Failed to open file %s\n", filename);
                    exit(EXIT_FAILURE);
                }
                outdenyz(psi, y, z, tmpx, file);
                fclose(file);
            }
            if (outflags & DEN_XY0) {
                sprintf(filename, "%s%s3d_xy0.bin", Niterout, itername);
                file = fopen(filename, "wb");
                if (file == NULL) {
                    std::fprintf(stderr, "Failed to open file %s\n", filename);
                    exit(EXIT_FAILURE);
                }
                outpsi2xy(psi, x, y, file);
                fclose(file);
            }
            if (outflags & DEN_X0Z) {
                sprintf(filename, "%s%s3d_x0z.bin", Niterout, itername);
                file = fopen(filename, "wb");
                if (file == NULL) {
                    std::fprintf(stderr, "Failed to open file %s\n", filename);
                    exit(EXIT_FAILURE);
                }
                outpsi2xz(psi, x, z, file);
                fclose(file);
            }
            if (outflags & DEN_0YZ) {
                sprintf(filename, "%s%s3d_0yz.bin", Niterout, itername);
                file = fopen(filename, "wb");
                if (file == NULL) {
                    std::fprintf(stderr, "Failed to open file %s\n", filename);
                    exit(EXIT_FAILURE);
                }
                outpsi2yz(psi, y, z, file);
                fclose(file);
            }
        }
        mutotnew = muen[0] + muen[1] + muen[2] + muen[3] + muen[4];
        if (fabs((mutotold - mutotnew) / mutotnew) < murel)
            break;
        mutotold = mutotnew;
        if (mutotnew > muend)
            break;
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    if (rmsout != NULL) {
        // std::fprintf(filerms,
        // "-------------------------------------------------------------------\n\n");
        // std::fprintf(filerms, "Total time on GPU: %f seconds\n", duration.count());
        std::fprintf(filerms,
                     "-------------------------------------------------------------------\n\n");
        fclose(filerms);
    }
    if (muoutput != NULL) {
        std::fprintf(filemu, "---------------------------------------------------------------------"
                             "------------\n\n");
        std::fprintf(filemu, "Total time on GPU: %f seconds\n", duration.count());
        std::fprintf(filemu, "---------------------------------------------------------------------"
                             "------------\n\n");
        fclose(filemu);
    }

    // Cleanup pinned memory
    cudaFreeHost(h_rms_pinned);
    cudaFreeHost(psi);

    // Cleanup FFT plans and work areas
    cudaFree(d_psi2_fft);
    if (forward_work)
        cudaFree(forward_work);
    if (backward_work)
        cudaFree(backward_work);
    cufftDestroy(forward_plan);
    cufftDestroy(backward_plan);
    return 0;
}

/**
 * @brief Reading input parameters from the configuration file.
 */
void readpar(void) {
    const char *cfg_tmp;

    if ((cfg_tmp = cfg_read("OPTION")) == NULL) {
        std::fprintf(stderr, "OPTION is not defined in the configuration file\n");
        exit(EXIT_FAILURE);
    }
    opt = atol(cfg_tmp);
    if ((cfg_tmp = cfg_read("OPTION_MICROWAVE_SHIELDING")) == NULL) {
        std::fprintf(stderr,
                     "OPTION_MICROWAVE_SHIELDING is not defined in the configuration file\n");
        exit(EXIT_FAILURE);
    }
    optms = atol(cfg_tmp);
    if (optms == 0) {
        MS = 1;
    } else {
        MS = -1;
    }

    if ((cfg_tmp = cfg_read("NATOMS")) == NULL) {
        std::fprintf(stderr, "NATOMS is not defined in the configuration file.\n");
        exit(EXIT_FAILURE);
    }
    Na = atof(cfg_tmp);

    if ((cfg_tmp = cfg_read("AHO")) == NULL) {
        std::fprintf(stderr, "AHO is not defined in the configuration file.\n");
        exit(EXIT_FAILURE);
    }
    aho = atof(cfg_tmp);

    if ((cfg_tmp = cfg_read("G")) == NULL) {
        if ((cfg_tmp = cfg_read("AS")) == NULL) {
            std::fprintf(stderr, "AS is not defined in the configuration file.\n");
            exit(EXIT_FAILURE);
        }
        as = atof(cfg_tmp);

        g = 4. * pi * as * Na * BOHR_RADIUS / aho;
    } else {
        g = atof(cfg_tmp);
    }

    if ((cfg_tmp = cfg_read("GDD")) == NULL) {
        if ((cfg_tmp = cfg_read("ADD")) == NULL) {
            std::fprintf(stderr, "ADD is not defined in the configuration file.\n");
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
        std::fprintf(stderr, "NX is not defined in the configuration file.\n");
        exit(EXIT_FAILURE);
    }
    Nx = atol(cfg_tmp);

    if ((cfg_tmp = cfg_read("NY")) == NULL) {
        std::fprintf(stderr, "NY is not defined in the configuration file.\n");
        exit(EXIT_FAILURE);
    }
    Ny = atol(cfg_tmp);

    if ((cfg_tmp = cfg_read("NZ")) == NULL) {
        std::fprintf(stderr, "Nz is not defined in the configuration file.\n");
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
        std::fprintf(stderr, "MUREL is not defined in the configuration file.\n");
        exit(EXIT_FAILURE);
    }
    murel = atof(cfg_tmp);

    if ((cfg_tmp = cfg_read("MUEND")) == NULL) {
        std::fprintf(stderr, "MUEND is not defined in the configuration file.\n");
        exit(EXIT_FAILURE);
    }
    muend = atof(cfg_tmp);

    if ((cfg_tmp = cfg_read("GAMMA")) == NULL) {
        std::fprintf(stderr, "GAMMA is not defined in the configuration file.\n");
        exit(EXIT_FAILURE);
    }
    vgamma = atof(cfg_tmp);

    if ((cfg_tmp = cfg_read("NU")) == NULL) {
        std::fprintf(stderr, "NU is not defined in the configuration file.\n");
        exit(EXIT_FAILURE);
    }
    vnu = atof(cfg_tmp);

    if ((cfg_tmp = cfg_read("LAMBDA")) == NULL) {
        std::fprintf(stderr, "LAMBDA is not defined in the configuration file.\n");
        exit(EXIT_FAILURE);
    }
    vlambda = atof(cfg_tmp);

    if ((cfg_tmp = cfg_read("NITER")) == NULL) {
        std::fprintf(stderr, "NITER is not defined in the configuration file.\n");
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
            std::fprintf(stderr, "OUTFLAGS is not defined in the configuration file.\n");
            exit(EXIT_FAILURE);
        }
        outflags = atoi(cfg_tmp);
    } else
        outflags = 0;

    return;
}

/**
 * @brief Function to compute RMS values on device
 * @param d_psi: Device: 3D psi array
 * @param d_work_array:  Work array
 * @param d_x2: x2 array
 * @param d_y2: y2 array
 * @param d_z2: z2 array
 * @param integ: Simpson3DTiledIntegrator
 * @param h_rms_pinned: Output RMS values in pinned memory [rms_x, rms_y, rms_z]
 */
void compute_rms_values(
    const CudaArray3D<cuDoubleComplex> &d_psi, // Device: 3D psi array
    CudaArray3D<double> &d_work_array, Simpson3DTiledIntegrator &integ,
    double *h_rms_pinned) // Output RMS values in pinned memory [rms_x, rms_y, rms_z]clean
{
    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error after memcpy: %s\n", cudaGetErrorString(err));
    }

    // Configure kernel launch parameters
    dim3 blockSize(8, 8, 4); // Adjust based on your GPU
    dim3 gridSize((Nx + blockSize.x - 1) / blockSize.x, (Ny + blockSize.y - 1) / blockSize.y,
                  (Nz + blockSize.z - 1) / blockSize.z);

    // Compute x^2 * psi^2
    compute_single_weighted_psi_squared<<<gridSize, blockSize>>>(d_psi.raw(), d_work_array.raw(),
                                                                 0, // 0 for x direction
                                                                 dx);

    // cudaDeviceSynchronize();
    double x2_integral = integ.integrateDevice(dx, dy, dz, d_work_array.raw(), Nx, Ny, Nz);

    // Compute y^2 * psi^2 (reuse d_work_array)
    compute_single_weighted_psi_squared<<<gridSize, blockSize>>>(d_psi.raw(), d_work_array.raw(),
                                                                 1, // 1 for y direction
                                                                 dy);
    // cudaDeviceSynchronize();
    double y2_integral = integ.integrateDevice(dx, dy, dz, d_work_array.raw(), Nx, Ny, Nz);

    // Compute z^2 * psi^2 (reuse d_work_array)
    compute_single_weighted_psi_squared<<<gridSize, blockSize>>>(d_psi.raw(), d_work_array.raw(),
                                                                 2, // 2 for z direction
                                                                 dz);
    // cudaDeviceSynchronize();
    double z2_integral = integ.integrateDevice(dx, dy, dz, d_work_array.raw(), Nx, Ny, Nz);

    // Calculate RMS values and store in pinned memory
    h_rms_pinned[0] = sqrt(x2_integral); // rms_x
    h_rms_pinned[1] = sqrt(y2_integral); // rms_y
    h_rms_pinned[2] = sqrt(z2_integral); // rms_z
}

/**
 * @brief Kernel to compute single weighted psi squared
 * @param psi: Device: 3D psi array
 * @param coord_squared: x2, y2, or z2 array
 * @param result: Result array
 * @param direction: Direction (0=x, 1=y, 2=z)
 */
__global__ void compute_single_weighted_psi_squared(const cuDoubleComplex *__restrict__ psi,
                                                    double *result,
                                                    int direction, // 0=x, 1=y, 2=z
                                                    const double discretiz) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;

    if (idx >= d_Nx || idy >= d_Ny || idz >= d_Nz)
        return;

    int linear_idx = idz * d_Ny * d_Nx + idy * d_Nx + idx;
    cuDoubleComplex psi_val = __ldg(&psi[linear_idx]); // Read-only cache for psi
    double psi_squared = psi_val.x * psi_val.x + psi_val.y * psi_val.y;

    double weight = 0.0;
    if (direction == 0) {
        // x = idx * scale + (-0.5 * d_Nx * scale)
        double offset = (-0.5 * static_cast<double>(d_Nx)) * discretiz;
        double x = fma(static_cast<double>(idx), discretiz, offset);
        weight = fma(x, x, 0.0);
    } else if (direction == 1) {
        // y = idy * scale + (-0.5 * d_Ny * scale)
        double offset = (-0.5 * static_cast<double>(d_Ny)) * discretiz;
        double y = fma(static_cast<double>(idy), discretiz, offset);
        weight = fma(y, y, 0.0);
    } else if (direction == 2) {
        // z = idz * scale + (-0.5 * d_Nz * scale)
        double offset = (-0.5 * static_cast<double>(d_Nz)) * discretiz;
        double z = fma(static_cast<double>(idz), discretiz, offset);
        weight = fma(z, z, 0.0);
    }

    result[linear_idx] = fma(weight, psi_squared, 0.0);
}

/**
 * @brief Function to compute squared wave function on device
 * @param d_psi: Device: 3D psi array
 * @param d_psi2: Device: 3D psi2 array
 */
void calc_d_psi2(const cuDoubleComplex *d_psi, double *d_psi2) {
    dim3 threadsPerBlock(8, 8, 8);
    dim3 numBlocks((Nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (Ny + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (Nz + threadsPerBlock.z - 1) / threadsPerBlock.z);
    compute_d_psi2<<<numBlocks, threadsPerBlock>>>(d_psi, d_psi2);
    return;
}

/**
 * @brief Kernel to compute squared wave function
 * @param d_psi: Device: 3D psi array
 * @param d_psi2: Device: 3D psi2 array
 */
__global__ void compute_d_psi2(const cuDoubleComplex *__restrict__ d_psi,
                               double *__restrict__ d_psi2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;

    if (idx >= d_Nx || idy >= d_Ny || idz >= d_Nz)
        return;

    int linear_idx = idz * d_Ny * d_Nx + idy * d_Nx + idx;

    cuDoubleComplex psi_val = __ldg(&d_psi[linear_idx]);

    d_psi2[linear_idx] = psi_val.x * psi_val.x + psi_val.y * psi_val.y;
    ;
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
             MultiArray<double> &x, MultiArray<double> &y, MultiArray<double> &z) {
    long cnti, cntj, cntk;
    double cpsi;
    double tmp;
    cpsi = sqrt(2. * pi * sqrt(2. * pi) * sx * sy * sz);

#pragma omp parallel for private(cnti)
    for (cnti = 0; cnti < Nx; cnti++) {
        x[cnti] = (cnti - Nx2) * dx;
        x2[cnti] = x[cnti] * x[cnti];
    }

#pragma omp parallel for private(cntj)
    for (cntj = 0; cntj < Ny; cntj++) {
        y[cntj] = (cntj - Ny2) * dy;
        y2[cntj] = y[cntj] * y[cntj];
    }

#pragma omp parallel for private(cntk)
    for (cntk = 0; cntk < Nz; cntk++) {
        z[cntk] = (cntk - Nz2) * dz;
        z2[cntk] = z[cntk] * z[cntk];
    }

#pragma omp parallel for private(cnti, cntj, cntk, tmp)
    for (cntk = 0; cntk < Nz; cntk++) {
        for (cntj = 0; cntj < Ny; cntj++) {
            for (cnti = 0; cnti < Nx; cnti++) {
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
             MultiArray<double> &z2) {
    long cnti, cntj, cntk;
    double vgamma2 = vgamma * vgamma;
    double vnu2 = vnu * vnu;
    double vlambda2 = vlambda * vlambda;
#pragma omp parallel for private(cnti, cntj, cntk)
    for (cntk = 0; cntk < Nz; cntk++) {
        for (cntj = 0; cntj < Ny; cntj++) {
            for (cnti = 0; cnti < Nx; cnti++) {
                pot(cntk, cntj, cnti) =
                    0.5 * par * (vgamma2 * x2[cnti] + vnu2 * y2[cntj] + vlambda2 * z2[cntk]);
                ;
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
               MultiArray<double> &kz2) {
    long cnti, cntj, cntk;
    double dkx, dky, dkz, xk, tmp;

    dkx = 2. * pi / (Nx * dx);
    dky = 2. * pi / (Ny * dy);
    dkz = 2. * pi / (Nz * dz);

    for (cnti = 0; cnti < Nx2; cnti++)
        kx[cnti] = cnti * dkx;
    for (cnti = 0; cnti < Nx2; cnti++)
        kx[cnti + Nx2] = (cnti - Nx2) * dkx;
    for (cntj = 0; cntj < Ny2; cntj++)
        ky[cntj] = cntj * dky;
    for (cntj = 0; cntj < Ny2; cntj++)
        ky[cntj + Ny2] = (cntj - Ny2) * dky;
    for (cntk = 0; cntk < Nz2; cntk++)
        kz[cntk] = cntk * dkz;
    for (cntk = 0; cntk < Nz2; cntk++)
        kz[cntk + Nz2] = (cntk - Nz2) * dkz;

    for (cnti = 0; cnti < Nx; cnti++)
        kx2[cnti] = kx[cnti] * kx[cnti];
    for (cntj = 0; cntj < Ny; cntj++)
        ky2[cntj] = ky[cntj] * ky[cntj];
    for (cntk = 0; cntk < Nz; cntk++)
        kz2[cntk] = kz[cntk] * kz[cntk];

#pragma omp parallel for private(cnti, cntj, cntk, tmp, xk)
    for (cntk = 0; cntk < Nz; cntk++) {
        for (cntj = 0; cntj < Ny; cntj++) {
            for (cnti = 0; cnti < Nx; cnti++) {
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
void calcnorm(CudaArray3D<cuDoubleComplex> &d_psi, CudaArray3D<double> &d_psi2, double &norm,
              Simpson3DTiledIntegrator &integ) {
    calc_d_psi2(d_psi.raw(), d_psi2.raw());
    double raw_norm = integ.integrateDevice(dx, dy, dz, d_psi2.raw(), Nx, Ny, Nz);
    norm = 1.0 / sqrt(raw_norm);

    Nad *= raw_norm;
    g *= raw_norm;
    gd *= raw_norm;
    h2 *= pow(raw_norm, 1.5);

    // Apply normalization
    dim3 threadsPerBlock(8, 8, 8);
    dim3 numBlocks((Nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (Ny + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (Nz + threadsPerBlock.z - 1) / threadsPerBlock.z);

    multiply_by_norm<<<numBlocks, threadsPerBlock>>>(d_psi.raw(), norm);
    // cudaDeviceSynchronize(); // Ensure completion
}

/**
 * @brief Kernel to multiply the wave function by the norm on device
 * @param d_psi: Device: 3D psi array
 * @param norm: Wave function norm
 */
__global__ void multiply_by_norm(cuDoubleComplex *__restrict__ d_psi, const double norm) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;

    if (idx >= d_Nx || idy >= d_Ny || idz >= d_Nz)
        return;

    long linear_idx = idz * d_Ny * d_Nx + idy * d_Nx + idx;
    d_psi[linear_idx] =
        make_cuDoubleComplex(d_psi[linear_idx].x * norm, d_psi[linear_idx].y * norm);
}

/**
 * @brief Kernel to compute the squared wave function multiplied by the dipolar potential
 * @param d_psi2_fft: Device: 3D psi2 array in FFT format
 * @param potdd: Device: 3D dipolar potential array
 */
__global__ void compute_psid2_potdd(cufftDoubleComplex *d_psi2_fft,
                                    const double *__restrict__ potdd) {
    int grid_stride_z = gridDim.z * blockDim.z;
    int grid_stride_y = gridDim.y * blockDim.y;
    int grid_stride_x = gridDim.x * blockDim.x;
    int tdz = blockIdx.z * blockDim.z + threadIdx.z;
    int tdy = blockIdx.y * blockDim.y + threadIdx.y;
    int tdx = blockIdx.x * blockDim.x + threadIdx.x;

    // Grid-stride loop implementation
    for (int idz = tdz; idz < d_Nz; idz += grid_stride_z) {
        for (int idy = tdy; idy < d_Ny; idy += grid_stride_y) {
            for (int idx = tdx; idx < d_Nx / 2 + 1; idx += grid_stride_x) {

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
 * @param psidd2: Device: 3D psi2 array multiplied by the dipolar potential
 */
__global__ void calcpsidd2_boundaries(double *psidd2) {
    long cnti, cntj, cntk;

    int grid_stride_y = gridDim.y * blockDim.y;
    int grid_stride_x = gridDim.x * blockDim.x;

    int tdy = blockIdx.y * blockDim.y + threadIdx.y;
    int tdx = blockIdx.x * blockDim.x + threadIdx.x;
    // Memory layout: fastest changing = x, slowest = z
    // Index calculation: idx = k * (d_Nx * d_Ny) + j * d_Nx + i

    // First loop: Copy from first x-slice to last x-slice
    // (j=0 to j=d_Ny-1, k=0 to k=d_Nz-1)
    for (cntj = tdy; cntj < d_Ny; cntj += grid_stride_y) {
        for (cntk = tdx; cntk < d_Nz; cntk += grid_stride_x) {
            long first_idx = cntk * (d_Nx * d_Ny) + cntj * d_Nx + 0; // i=0 (first x-slice)
            long last_idx =
                cntk * (d_Nx * d_Ny) + cntj * d_Nx + (d_Nx - 1); // i=d_Nx-1 (last x-slice)

            psidd2[last_idx] = psidd2[first_idx];
        }
    }

    // Second loop: Copy from first y-slice to last y-slice
    // (i=0 to i=d_Nx-1, k=0 to k=d_Nz-1)
    for (cnti = tdy; cnti < d_Nx; cnti += grid_stride_y) {
        for (cntk = tdx; cntk < d_Nz; cntk += grid_stride_x) {
            long first_idx = cntk * (d_Nx * d_Ny) + 0 * d_Nx + cnti; // j=0 (first y-slice)
            long last_idx =
                cntk * (d_Nx * d_Ny) + (d_Ny - 1) * d_Nx + cnti; // j=d_Ny-1 (last y-slice)

            psidd2[last_idx] = psidd2[first_idx];
        }
    }

    // Third loop: Copy from first z-slice to last z-slice
    // (i=0 to i=d_Nx-1, j=0 to j=d_Ny-1)
    for (cnti = tdy; cnti < d_Nx; cnti += grid_stride_y) {
        for (cntj = tdx; cntj < d_Ny; cntj += grid_stride_x) {
            long first_idx = 0 * (d_Nx * d_Ny) + cntj * d_Nx + cnti; // k=0 (first z-slice)
            long last_idx =
                (d_Nz - 1) * (d_Nx * d_Ny) + cntj * d_Nx + cnti; // k=d_Nz-1 (last z-slice)

            psidd2[last_idx] = psidd2[first_idx];
        }
    }
}

/**
 * @brief Function to compute the FFT of the squared wave function multiplied by the dipolar
 * potential
 * @param forward_plan: Forward FFT plan
 * @param backward_plan: Backward FFT plan
 * @param d_psi: Device: 3D psi array
 * @param d_psi2_real: Device: 3D psi2 array
 * @param d_psi2_fft: Device: 3D psi2 array in FFT format
 * @param potdd: Device: 3D dipolar potential array
 */
void calc_psid2_potdd(cufftHandle forward_plan, cufftHandle backward_plan, cuDoubleComplex *d_psi,
                      double *d_psi2_real, cufftDoubleComplex *d_psi2_fft, const double *potdd) {
    calc_d_psi2(d_psi, d_psi2_real);
    cufftExecD2Z(forward_plan, (cufftDoubleReal *)d_psi2_real, d_psi2_fft);

    dim3 threadsPerBlock(8, 8, 8);
    dim3 numBlocks((Nx / 2 + 1 + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (Ny + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (Nz + threadsPerBlock.z - 1) / threadsPerBlock.z);
    compute_psid2_potdd<<<numBlocks, threadsPerBlock>>>(d_psi2_fft, potdd);

    cufftExecZ2D(backward_plan, d_psi2_fft, (cufftDoubleReal *)d_psi2_real);
    calcpsidd2_boundaries<<<numBlocks, threadsPerBlock>>>(d_psi2_real);
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
void gencoef(MultiArray<cuDoubleComplex> &calphax, MultiArray<cuDoubleComplex> &cgammax,
             MultiArray<cuDoubleComplex> &calphay, MultiArray<cuDoubleComplex> &cgammay,
             MultiArray<cuDoubleComplex> &calphaz, MultiArray<cuDoubleComplex> &cgammaz,
             cuDoubleComplex &Ax0, cuDoubleComplex &Ay0, cuDoubleComplex &Az0,
             cuDoubleComplex &Ax0r, cuDoubleComplex &Ay0r, cuDoubleComplex &Az0r,
             cuDoubleComplex &Ax, cuDoubleComplex &Ay, cuDoubleComplex &Az) {
    long cnti;
    cuDoubleComplex minus1;

    Ax0 = make_cuDoubleComplex(1., dt / dx2 / (3. - par));
    Ay0 = make_cuDoubleComplex(1., dt / dy2 / (3. - par));
    Az0 = make_cuDoubleComplex(1., dt / dz2 / (3. - par));

    Ax0r = make_cuDoubleComplex(1., -dt / dx2 / (3. - par));
    Ay0r = make_cuDoubleComplex(1., -dt / dy2 / (3. - par));
    Az0r = make_cuDoubleComplex(1., -dt / dz2 / (3. - par));

    Ax = make_cuDoubleComplex(0., -0.5 * dt / dx2 / (3. - par));
    Ay = make_cuDoubleComplex(0., -0.5 * dt / dy2 / (3. - par));
    Az = make_cuDoubleComplex(0., -0.5 * dt / dz2 / (3. - par));

    minusAx = make_cuDoubleComplex(0., -Ax.y);
    minusAy = make_cuDoubleComplex(0., -Ay.y);
    minusAz = make_cuDoubleComplex(0., -Az.y);

    minus1 = make_cuDoubleComplex(-1., 0.);

    calphax[Nx - 2] = make_cuDoubleComplex(0., 0.);
    cgammax[Nx - 2] = cuCdiv(minus1, Ax0);
    for (cnti = Nx - 2; cnti > 0; cnti--) {
        calphax[cnti - 1] = cuCmul(Ax, cgammax[cnti]);
        cgammax[cnti - 1] = cuCdiv(minus1, cuCadd(Ax0, cuCmul(Ax, calphax[cnti - 1])));
    }

    calphay[Ny - 2] = make_cuDoubleComplex(0., 0.);
    cgammay[Ny - 2] = cuCdiv(minus1, Ay0);
    for (cnti = Ny - 2; cnti > 0; cnti--) {
        calphay[cnti - 1] = cuCmul(Ay, cgammay[cnti]);
        cgammay[cnti - 1] = cuCdiv(minus1, cuCadd(Ay0, cuCmul(Ay, calphay[cnti - 1])));
    }

    calphaz[Nz - 2] = make_cuDoubleComplex(0., 0.);
    cgammaz[Nz - 2] = cuCdiv(minus1, Az0);
    for (cnti = Nz - 2; cnti > 0; cnti--) {
        calphaz[cnti - 1] = cuCmul(Az, cgammaz[cnti]);
        cgammaz[cnti - 1] = cuCdiv(minus1, cuCadd(Az0, cuCmul(Az, calphaz[cnti - 1])));
    }

    return;
}

/**
 * @brief Function to compute the time propagation with respect to H1 (part of the Hamiltonian
 * without spatial derivatives).
 * @param d_psi: Device: 3D psi array
 * @param d_psi2: Device: 3D psi2 array
 * @param d_pot: Device: 3D trap potential array
 * @param g: Host to Device: g coefficient for contact interaction term
 * @param gd: Host to Device: gd coefficient for dipolar interaction term
 * @param h2: Host to Device: h2 coefficient for quantum fluctuation term
 */
void calcnu(CudaArray3D<cuDoubleComplex> &d_psi, CudaArray3D<double> &d_psi2,
            CudaArray3D<double> &d_pot, double g, double gd, double h2) {
    // calc_d_psi2(d_psi, d_psi2);

    // Precompute ratio_gd on host (constant for all threads)
    double ratio_gd = gd / ((double)(Nx * Ny * Nz));

    dim3 threadsPerBlock(8, 8, 8);
    dim3 numBlocks((Nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (Ny + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (Nz + threadsPerBlock.z - 1) / threadsPerBlock.z);
    calcnu_kernel<<<numBlocks, threadsPerBlock>>>(d_psi.raw(), d_psi2.raw(), d_pot.raw(), g,
                                                  ratio_gd, h2);
    // cudaDeviceSynchronize();
    return;
}

/**
 * @brief Kernel to compute the time propagation with respect to H1 (part of the Hamiltonian without
 *    spatial derivatives).
 * @param d_psi: Device: 3D psi array
 * @param d_psi2: Device: 3D psi2 array
 * @param d_pot: Device: 3D trap potential array
 * @param g: Host: g coefficient for contact interaction term
 * @param ratio_gd: Precomputed ratio gd/(Nx*Ny*Nz) for dipolar interaction term
 * @param h2: Host: h2 coefficient for quantum fluctuation term
 */
__global__ void calcnu_kernel(cuDoubleComplex *__restrict__ d_psi, double *__restrict__ d_psi2,
                              const double *__restrict__ d_pot, const double g,
                              const double ratio_gd, const double h2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;

    if (idx >= d_Nx || idy >= d_Ny || idz >= d_Nz)
        return;

    int linear_idx = idz * d_Ny * d_Nx + idy * d_Nx + idx;
    double pot_val = __ldg(&d_pot[linear_idx]);
    cuDoubleComplex psi_val = d_psi[linear_idx];
    double psi2dd = __ldg(&d_psi2[linear_idx]) * ratio_gd;

    // I'm using cuCabs() for |psi|
    double psi_abs = cuCabs(psi_val);              // |psi|
    double psi_val2 = fma(psi_abs, psi_abs, 0.0);  // |psi|^2
    double psi_val3 = fma(psi_val2, psi_abs, 0.0); // |psi|^3

    // all coefficients
    double temp1 = fma(psi_val2, g, pot_val);
    double temp2 = fma(psi_val3, h2, psi2dd);
    double sum = temp1 + temp2;
    double tmp = fma(-d_dt, sum, 0.0);

    // compute e^{-i*tmp} = cos(tmp) - i sin(tmp)
    double s, c;
    sincos(tmp, &s, &c);

    // multiply psi by e^{-i tmp} using cuCmul (complex * complex)
    d_psi[linear_idx] = cuCmul(psi_val, make_cuDoubleComplex(c, s));
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
void calclux(CudaArray3D<cuDoubleComplex> &d_psi, cuDoubleComplex *d_cbeta,
             CudaArray3D<cuDoubleComplex> &d_calphax, CudaArray3D<cuDoubleComplex> &d_cgammax,
             cuDoubleComplex Ax0r, cuDoubleComplex Ax) {

    // Match mapping used in imag3d: threadIdx.x -> y, threadIdx.y -> z
    dim3 threadsPerBlock(32, 8);
    dim3 numBlocks((Ny + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (Nz + threadsPerBlock.y - 1) / threadsPerBlock.y);

    calclux_kernel<<<numBlocks, threadsPerBlock>>>(d_psi.raw(), d_cbeta, d_calphax.raw(),
                                                   d_cgammax.raw(), Ax0r, Ax);

    // cudaDeviceSynchronize();
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
__global__ void calclux_kernel(cuDoubleComplex *__restrict__ psi,
                               cuDoubleComplex *__restrict__ cbeta,
                               const cuDoubleComplex *__restrict__ calphax,
                               const cuDoubleComplex *__restrict__ cgammax,
                               const cuDoubleComplex Ax0r, const cuDoubleComplex Ax) {

    // Map y to threadIdx.x, z to threadIdx.y (as in imag3d)
    int cntj = blockIdx.x * blockDim.x + threadIdx.x;
    int cntk = blockIdx.y * blockDim.y + threadIdx.y;

    if (cntj >= d_Ny || cntk >= d_Nz)
        return;

    // Base offset for this (j,k) y-z position
    const long base_offset = cntk * d_Ny * d_Nx + cntj * d_Nx;

    // Forward elimination: fill cbeta array using a rolling window in x
    // Boundary condition: cbeta[nx-2] = psi[nx-1]
    long idx_i = base_offset + (d_Nx - 2);
    long idx_ip1 = base_offset + (d_Nx - 1);
    cbeta[idx_i] = psi[idx_ip1];

    cuDoubleComplex psi_ip1 = __ldg(&psi[idx_ip1]);
    cuDoubleComplex psi_i = __ldg(&psi[idx_i]);
    for (int cnti = d_Nx - 2; cnti > 0; cnti--) {
        long idx_im1 = idx_i - 1;
        cuDoubleComplex psi_im1 = __ldg(&psi[idx_im1]);

        cuDoubleComplex c = cuCadd(cuCadd(cuCmul(d_minusAx, psi_ip1), cuCmul(Ax0r, psi_i)),
                                   cuCmul(d_minusAx, psi_im1));
        cbeta[idx_im1] = cuCmul(__ldg(&cgammax[cnti]), cuCsub(cuCmul(Ax, cbeta[idx_i]), c));

        // Roll window down in x
        psi_ip1 = psi_i;
        psi_i = psi_im1;
        idx_i = idx_im1;
    }

    // Boundary condition
    psi[base_offset + 0] = make_cuDoubleComplex(0.0, 0.0);

    // Back substitution: update psi values
    for (int cnti = 0; cnti < d_Nx - 2; cnti++) {
        psi[base_offset + cnti + 1] =
            cuCfma(__ldg(&calphax[cnti]), psi[base_offset + cnti], cbeta[base_offset + cnti]);
    }

    // Boundary condition
    psi[base_offset + d_Nx - 1] = make_cuDoubleComplex(0.0, 0.0);
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
void calcluy(CudaArray3D<cuDoubleComplex> &d_psi, cuDoubleComplex *d_cbeta,
             CudaArray3D<cuDoubleComplex> &d_calphay, CudaArray3D<cuDoubleComplex> &d_cgammay,
             cuDoubleComplex Ay0r, cuDoubleComplex Ay) {

    // Match imag3d layout: x as fastest varying index in blocks
    dim3 threadsPerBlock(32, 8);
    dim3 numBlocks((Nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (Nz + threadsPerBlock.y - 1) / threadsPerBlock.y);

    calcluy_kernel<<<numBlocks, threadsPerBlock>>>(d_psi.raw(), d_cbeta, d_calphay.raw(),
                                                   d_cgammay.raw(), Ay0r, Ay);

    // cudaDeviceSynchronize();
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
__global__ void calcluy_kernel(cuDoubleComplex *__restrict__ psi,
                               cuDoubleComplex *__restrict__ cbeta,
                               const cuDoubleComplex *__restrict__ calphay,
                               const cuDoubleComplex *__restrict__ cgammay,
                               const cuDoubleComplex Ay0r, const cuDoubleComplex Ay) {

    // Map x to threadIdx.x, z to threadIdx.y (as in imag3d)
    int cnti = blockIdx.x * blockDim.x + threadIdx.x;
    int cntk = blockIdx.y * blockDim.y + threadIdx.y;

    if (cnti >= d_Nx || cntk >= d_Nz)
        return;

    // Base offset for this (i,k) x-z position
    const long base_offset = cntk * d_Ny * d_Nx + cnti;

    // Forward elimination: fill cbeta array using a rolling window in y
    // Boundary condition: cbeta[ny-2] = psi[ny-1]
    long idx_j = base_offset + (d_Ny - 2) * d_Nx;
    long idx_jp1 = base_offset + (d_Ny - 1) * d_Nx;
    cbeta[idx_j] = psi[idx_jp1];

    cuDoubleComplex psi_jp1 = __ldg(&psi[idx_jp1]);
    cuDoubleComplex psi_j = __ldg(&psi[idx_j]);
    for (int cntj = d_Ny - 2; cntj > 0; cntj--) {
        long idx_jm1 = idx_j - d_Nx;
        cuDoubleComplex psi_jm1 = __ldg(&psi[idx_jm1]);

        cuDoubleComplex c = cuCadd(cuCadd(cuCmul(d_minusAy, psi_jp1), cuCmul(Ay0r, psi_j)),
                                   cuCmul(d_minusAy, psi_jm1));
        cbeta[idx_jm1] = cuCmul(__ldg(&cgammay[cntj]), cuCsub(cuCmul(Ay, cbeta[idx_j]), c));

        // Roll window down in y
        psi_jp1 = psi_j;
        psi_j = psi_jm1;
        idx_j = idx_jm1;
    }

    // Boundary condition
    psi[base_offset + 0 * d_Nx] = make_cuDoubleComplex(0.0, 0.0);

    // Back substitution: update psi values
    for (int cntj = 0; cntj < d_Ny - 2; cntj++) {
        psi[base_offset + (cntj + 1) * d_Nx] =
            cuCfma(__ldg(&calphay[cntj]), psi[base_offset + cntj * d_Nx],
                   cbeta[base_offset + cntj * d_Nx]);
    }

    // Boundary condition
    psi[base_offset + (d_Ny - 1) * d_Nx] = make_cuDoubleComplex(0.0, 0.0);
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
void calcluz(CudaArray3D<cuDoubleComplex> &d_psi, cuDoubleComplex *d_cbeta,
             CudaArray3D<cuDoubleComplex> &d_calphaz, CudaArray3D<cuDoubleComplex> &d_cgammaz,
             cuDoubleComplex Az0r, cuDoubleComplex Az) {

    // Match imag3d layout: x as fastest varying in blocks over x-y plane
    dim3 threadsPerBlock(32, 8);
    dim3 numBlocks((Nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (Ny + threadsPerBlock.y - 1) / threadsPerBlock.y);

    calcluz_kernel<<<numBlocks, threadsPerBlock>>>(d_psi.raw(), d_cbeta, d_calphaz.raw(),
                                                   d_cgammaz.raw(), Az0r, Az);

    // cudaDeviceSynchronize();
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
__global__ void calcluz_kernel(cuDoubleComplex *__restrict__ psi,
                               cuDoubleComplex *__restrict__ cbeta,
                               const cuDoubleComplex *__restrict__ calphaz,
                               const cuDoubleComplex *__restrict__ cgammaz,
                               const cuDoubleComplex Az0r, const cuDoubleComplex Az) {

    // Map x to threadIdx.x, y to threadIdx.y (as in imag3d)
    int cnti = blockIdx.x * blockDim.x + threadIdx.x;
    int cntj = blockIdx.y * blockDim.y + threadIdx.y;

    if (cnti >= d_Nx || cntj >= d_Ny)
        return;

    // Base offset for this (i,j) x-y position - points to z=0
    const long base_offset = cntj * d_Nx + cnti;
    const long stride = d_Ny * d_Nx; // stride to move in z-direction

    // Forward elimination: fill cbeta array using a rolling window in z
    // Boundary condition: cbeta[nz-2] = psi[nz-1]
    long idx_k = base_offset + (d_Nz - 2) * stride;
    long idx_kp1 = base_offset + (d_Nz - 1) * stride;
    cbeta[idx_k] = psi[idx_kp1];

    cuDoubleComplex psi_kp1 = __ldg(&psi[idx_kp1]);
    cuDoubleComplex psi_k = __ldg(&psi[idx_k]);
    for (int cntk = d_Nz - 2; cntk > 0; cntk--) {
        long idx_km1 = idx_k - stride;
        cuDoubleComplex psi_km1 = __ldg(&psi[idx_km1]);

        cuDoubleComplex c = cuCadd(cuCadd(cuCmul(d_minusAz, psi_kp1), cuCmul(Az0r, psi_k)),
                                   cuCmul(d_minusAz, psi_km1));
        cbeta[idx_km1] = cuCmul(__ldg(&cgammaz[cntk]), cuCsub(cuCmul(Az, cbeta[idx_k]), c));

        // Roll window down in z
        psi_kp1 = psi_k;
        psi_k = psi_km1;
        idx_k = idx_km1;
    }

    // Boundary condition
    psi[base_offset + 0 * stride] = make_cuDoubleComplex(0.0, 0.0);

    // Back substitution: update psi values
    for (int cntk = 0; cntk < d_Nz - 2; cntk++) {
        psi[base_offset + (cntk + 1) * stride] =
            cuCfma(__ldg(&calphaz[cntk]), psi[base_offset + cntk * stride],
                   cbeta[base_offset + cntk * stride]);
    }

    // Boundary condition
    psi[base_offset + (d_Nz - 1) * stride] = make_cuDoubleComplex(0.0, 0.0);
}

/**
 * @brief Function to compute the chemical potential of the system
 * @param muen: Host: 3D muen array that stores the chemical potential of the system and its
 * contributions
 * @param d_psi: Device: 3D psi array
 * @param d_psi2: Device: 3D psi2 array
 * @param d_pot: Device: 3D trap potential array
 * @param d_psi2dd: Device: 3D psi2dd array
 * @param d_potdd: Device: 3D dipolar potential array
 * @param d_psi2_fft: Device: 3D psi2_fft array
 * @param forward_plan: FFT forward plan
 * @param backward_plan: FFT backward plan
 * @param integ: Simpson3DTiledIntegrator
 * @param g: Host to Device: g coefficient for contact interaction term
 * @param gd: Host to Device: gd coefficient for dipolar interaction term
 * @param h2: Host to Device: h2 coefficient for quantum fluctuation term
 */
void calcmuen(MultiArray<double> &muen, CudaArray3D<cuDoubleComplex> &d_psi,
              CudaArray3D<double> &d_psi2, CudaArray3D<double> &d_pot,
              CudaArray3D<double> &d_psi2dd, CudaArray3D<double> &d_potdd,
              cufftDoubleComplex *d_psi2_fft, cufftHandle forward_plan, cufftHandle backward_plan,
              Simpson3DTiledIntegrator &integ, const double g, const double gd, const double h2) {

    // Precompute constants
    const double inv_NxNyNz = 1.0 / ((double)Nx * Ny * Nz);
    const double half_g = 0.5 * g;
    const double half_gd = 0.5 * gd;
    const double half_h2 = 0.5 * h2;

    dim3 threadsPerBlock(8, 8, 8);
    dim3 numBlocks((Nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (Ny + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (Nz + threadsPerBlock.z - 1) / threadsPerBlock.z);

    // Step 1: Contact energy - Calculate ψ² and 0.5 * g * ψ⁴
    calcmuen_fused_contact<<<numBlocks, threadsPerBlock>>>(d_psi.raw(), d_psi2.raw(), half_g);
    muen[0] = integ.integrateDevice(dx, dy, dz, d_psi2.raw(), Nx, Ny, Nz);

    // Step 2: Potential energy - Calculate ψ² and 0.5 * ψ² * V
    calcmuen_fused_potential<<<numBlocks, threadsPerBlock>>>(d_psi.raw(), d_psi2.raw(),
                                                             d_pot.raw());
    muen[1] = integ.integrateDevice(dx, dy, dz, d_psi2.raw(), Nx, Ny, Nz);

    // Step 3: Dipolar energy - requires FFT computation first
    calc_psid2_potdd(forward_plan, backward_plan, d_psi.raw(), d_psi2dd.raw(), d_psi2_fft,
                     d_potdd.raw());
    calcmuen_fused_dipolar<<<numBlocks, threadsPerBlock>>>(d_psi.raw(), d_psi2.raw(),
                                                           d_psi2dd.raw(), half_gd);
    muen[2] = integ.integrateDevice(dx, dy, dz, d_psi2.raw(), Nx, Ny, Nz) * inv_NxNyNz;

    // Step 4: Kinetic energy - calculate gradients and kinetic energy density directly
    calcmuen_kin(d_psi, d_psi2, par);
    muen[3] = integ.integrateDevice(dx, dy, dz, d_psi2.raw(), Nx, Ny, Nz);

    // Step 5: H2 energy - calculate quantum fluctuation energy density
    calcmuen_fused_h2<<<numBlocks, threadsPerBlock>>>(d_psi.raw(), d_psi2.raw(), half_h2);
    muen[4] = integ.integrateDevice(dx, dy, dz, d_psi2.raw(), Nx, Ny, Nz);

    return;
}
/**
 * @brief Kernel to calculate contact energy term
 * @param d_psi: Device: 3D psi array
 * @param d_result: Device: 3D result array
 * @param half_g: Precomputed 0.5 * g coefficient for contact interaction term
 */
__global__ void calcmuen_fused_contact(const cuDoubleComplex *__restrict__ d_psi,
                                       double *__restrict__ d_result, double half_g) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;

    if (idx >= d_Nx || idy >= d_Ny || idz >= d_Nz)
        return;

    int linear_idx = idz * d_Ny * d_Nx + idy * d_Nx + idx;
    cuDoubleComplex psi_val = __ldg(&d_psi[linear_idx]);
    double psi2_val = psi_val.x * psi_val.x + psi_val.y * psi_val.y;
    double psi4_val = psi2_val * psi2_val;
    d_result[linear_idx] = psi4_val * half_g;
}

/**
 * @brief Kernel to calculate trap potential energy term
 * @param d_psi: Device: 3D psi array
 * @param d_result: Device: 3D result array
 * @param d_pot: Device: 3D trap potential array
 */
__global__ void calcmuen_fused_potential(const cuDoubleComplex *__restrict__ d_psi,
                                         double *__restrict__ d_result,
                                         const double *__restrict__ d_pot) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;

    if (idx >= d_Nx || idy >= d_Ny || idz >= d_Nz)
        return;

    int linear_idx = idz * d_Ny * d_Nx + idy * d_Nx + idx;
    cuDoubleComplex psi_val = __ldg(&d_psi[linear_idx]); // Read-only cache for psi
    double psi2_val = cuCabs(psi_val) * cuCabs(psi_val);
    d_result[linear_idx] =
        0.5 * psi2_val * __ldg(&d_pot[linear_idx]); // Read-only cache for potential
}

/**
 * @brief Kernel to calculate dipolar energy term
 * @param d_psi: Device: 3D psi array
 * @param d_result: Device: 3D result array
 * @param d_psidd2: Device: 3D dipolar psi squared array
 * @param half_gd: Precomputed 0.5 * gd coefficient for dipolar interaction term
 */
__global__ void calcmuen_fused_dipolar(const cuDoubleComplex *__restrict__ d_psi,
                                       double *__restrict__ d_result,
                                       const double *__restrict__ d_psidd2, const double half_gd) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;

    if (idx >= d_Nx || idy >= d_Ny || idz >= d_Nz)
        return;

    int linear_idx = idz * d_Ny * d_Nx + idy * d_Nx + idx;
    cuDoubleComplex psi_val = __ldg(&d_psi[linear_idx]); // Read-only cache for psi
    double psi2_val = cuCabs(psi_val) * cuCabs(psi_val);
    double psidd2_val = __ldg(&d_psidd2[linear_idx]); // Read-only cache for dipolar psi squared
    d_result[linear_idx] = psi2_val * psidd2_val * half_gd;
}

/**
 * @brief Kernel to calculate quantum fluctuation energy term
 * @param d_psi: Device: 3D psi array
 * @param d_result: Device: 3D result array
 * @param half_h2: Precomputed 0.5 * h2 coefficient for quantum fluctuation term
 */
__global__ void calcmuen_fused_h2(const cuDoubleComplex *__restrict__ d_psi,
                                  double *__restrict__ d_result, const double half_h2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;

    if (idx >= d_Nx || idy >= d_Ny || idz >= d_Nz)
        return;

    int linear_idx = idz * d_Ny * d_Nx + idy * d_Nx + idx;
    cuDoubleComplex psi_val = __ldg(&d_psi[linear_idx]); // Read-only cache for psi
    double psi_val2 = cuCabs(psi_val) * cuCabs(psi_val);
    double psi_val3 = psi_val2 * cuCabs(psi_val);
    double psi5_val = psi_val3 * psi_val2;
    d_result[linear_idx] = psi5_val * half_h2;
}

/**
 * @brief Function to calculate kinetic energy term
 * @param d_psi: Device: 3D psi array
 * @param d_work_array: Device: 3D work array
 * @param par: Host to Device: par coefficient either 1 or 2, defined in Input file
 */
void calcmuen_kin(CudaArray3D<cuDoubleComplex> &d_psi, CudaArray3D<double> &d_work_array, int par) {
    diff_complex(dx, dy, dz, d_psi.raw(), d_work_array.raw(), Nx, Ny, Nz, par);
}

/**
 * @brief Function to output the rms values
 * @param filerms: File pointer to the rms output file
 */
void rms_output(FILE *filerms) {
    std::fprintf(filerms, "\n**********************************************\n");
    if (cfg_read("G") != NULL) {
        std::fprintf(filerms, "Contact: G = %.6le, G * par = %.6le\n", g / par, g);
    } else {
        std::fprintf(filerms,
                     "Contact: Natoms = %.11le, as = %.6le * a0, G = %.6le, G * par = %.6le\n", Nad,
                     as, g / par, g);
    }
    if (optms == 0) {
        std::fprintf(filerms, "Regular ");
    } else {
        std::fprintf(filerms, "Microwave-shielded ");
    }
    if (cfg_read("GDD") != NULL) {
        std::fprintf(filerms, "DDI: GD = %.6le, GD * par = %.6le, edd = %.6le\n", gd / par, gd,
                     edd);
    } else {
        std::fprintf(filerms, "DDI: add = %.6le * a0, GD = %.6le, GD * par = %.6le, edd = %.6le\n",
                     add, gd / par, gd, edd);
    }
    std::fprintf(filerms, "     Dipolar cutoff Scut = %.6le,\n\n", cutoff);
    if (QF == 1) {
        std::fprintf(filerms, "QF = 1: h2 = %.6le,        q5 = %.6le\n\n", h2, q5);
    } else
        std::fprintf(filerms, "QF = 0\n\n");
    std::fprintf(filerms, "Trap parameters:\nGAMMA = %.6le, NU = %.6le, LAMBDA = %.6le\n", vgamma,
                 vnu, vlambda);
    std::fprintf(filerms, "Space discretization: NX = %li, NY = %li, NZ = %li\n", Nx, Ny, Nz);
    std::fprintf(filerms,
                 "                      DX = %.16le, DY = %.16le, DZ = %.16le, mx = %.2le, my = "
                 "%.2le, mz = %.2le\n",
                 dx, dy, dz, mx, my, mz);
    if (cfg_read("AHO") != NULL)
        std::fprintf(filerms, "      Unit of length: aho = %.6le m\n", aho);
    std::fprintf(filerms, "\nTime discretization: NITER = %li (NSNAP = %li)\n", Niter, Nsnap);
    std::fprintf(filerms, "                     DT = %.6le, mt = %.2le\n\n", dt, mt);
    if (input != NULL) {
        std::fprintf(filerms, "file %s\n", input);
    } else {
        std::fprintf(filerms, "Gaussian\n               SX = %.6le, SY = %.6le, SZ = %.6le\n", sx,
                     sy, sz);
    }
    std::fprintf(filerms, "MUREL = %.6le, MUEND=%.6le\n\n", murel, muend);
    std::fprintf(filerms, "-------------------------------------------------------------------\n");
    std::fprintf(filerms, "Snap      <r>            <x>            <y>            <z>\n");
    std::fprintf(filerms, "-------------------------------------------------------------------\n");
    fflush(filerms);
}

/**
 * @brief Function to output the chemical potential values
 * @param filemu: File pointer to the chemical potential output file
 */
void mu_output(FILE *filemu) {
    std::fprintf(filemu, "\n**********************************************\n");
    if (cfg_read("G") != NULL) {
        std::fprintf(filemu, "Contact: G = %.6le, G * par = %.6le\n", g / par, g);
    } else {
        std::fprintf(filemu,
                     "Contact: Natoms = %.11le, as = %.6le * a0, G = %.6le, G * par = %.6le\n", Nad,
                     as, g / par, g);
    }
    if (optms == 0) {
        std::fprintf(filemu, "Regular ");
    } else {
        std::fprintf(filemu, "Microwave-shielded ");
    }
    if (cfg_read("GDD") != NULL) {
        std::fprintf(filemu, "DDI: GD = %.6le, GD * par = %.6le, edd = %.6le\n", gd / par, gd, edd);
    } else {
        std::fprintf(filemu, "DDI: add = %.6le * a0, GD = %.6le, GD * par = %.6le, edd = %.6le\n",
                     add, gd / par, gd, edd);
    }
    std::fprintf(filemu, "     Dipolar cutoff Scut = %.6le,\n\n", cutoff);
    if (QF == 1) {
        std::fprintf(filemu, "QF = 1: h2 = %.6le,         q5 = %.6le\n\n", h2, q5);
    } else
        std::fprintf(filemu, "QF = 0\n\n");
    std::fprintf(filemu, "Trap parameters:\nGAMMA = %.6le, NU = %.6le, LAMBDA = %.6le\n", vgamma,
                 vnu, vlambda);
    std::fprintf(filemu, "Space discretization: NX = %li, NY = %li, NZ = %li\n", Nx, Ny, Nz);
    std::fprintf(filemu,
                 "                      DX = %.16le, DY = %.16le, DZ = %.16le, mx = %.2le, my = "
                 "%.2le, mz = %.2le\n",
                 dx, dy, dz, mx, my, mz);
    if (cfg_read("AHO") != NULL)
        std::fprintf(filemu, "      Unit of length: aho = %.6le m\n", aho);
    std::fprintf(filemu, "\nTime discretization: NITER = %li (NSNAP = %li)\n", Niter, Nsnap);
    std::fprintf(filemu, "                     DT = %.6le, mt = %.2le\n\n", dt, mt);
    if (input != NULL) {
        std::fprintf(filemu, "file %s\n", input);
    } else {
        std::fprintf(filemu, "Gaussian\n               SX = %.6le, SY = %.6le, SZ = %.6le\n", sx,
                     sy, sz);
    }
    std::fprintf(filemu, "MUREL = %.6le, MUEND=%.6le\n\n", murel, muend);
    std::fprintf(filemu, "-------------------------------------------------------------------------"
                         "----------------------\n");
    std::fprintf(filemu, "Snap      mu           Kin             Pot            Contact            "
                         "DDI            QF\n");
    std::fprintf(filemu, "-------------------------------------------------------------------------"
                         "----------------------\n");
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
void save_psi_from_gpu(cuDoubleComplex *psi, cuDoubleComplex *d_psi, const char *filename, long Nx,
                       long Ny, long Nz) {
    size_t total_size = Nx * Ny * Nz;

    cudaMemcpy(psi, d_psi, total_size * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        return;
    }

    FILE *file = fopen(filename, "wb");
    if (file == NULL) {
        std::fprintf(stderr, "Failed to open file %s\n", filename);
        return;
    }

    size_t written = fwrite(psi, sizeof(cuDoubleComplex), total_size, file);
    if (written != total_size) {
        std::fprintf(stderr, "Failed to write all data: wrote %zu of %zu complex elements\n",
                     written, total_size);
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
void read_psi_from_file_complex(cuDoubleComplex *psi, const char *filename, long Nx, long Ny,
                                long Nz) {
    size_t total_size = Nx * Ny * Nz;

    // Open file
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        std::fprintf(stderr, "Failed to open file %s\n", filename);
        return;
    }

    // Read and convert real data to complex
    double real_value;
    for (size_t i = 0; i < total_size; i++) {
        if (fread(&real_value, sizeof(double), 1, file) != 1) {
            std::fprintf(stderr, "Failed to read element %zu\n", i);
            fclose(file);
            return;
        }

        psi[i] = make_cuDoubleComplex(real_value, 0.0); // real part only
                                                        // psi[i].x = real_value;
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
void outdenx(cuDoubleComplex *psi, MultiArray<double> &x, MultiArray<double> &tmpy,
             MultiArray<double> &tmpz, FILE *file) {
    for (long cnti = 0; cnti < Nx; cnti++) {
        // For each x position
        for (long cntj = 0; cntj < Ny; cntj++) {
            // Compute |psi|^2
            for (long cntk = 0; cntk < Nz; cntk++) {
                tmpz[cntk] = fma(cuCabs(psi[cntk * Ny * Nx + cntj * Nx + cnti]),
                                 cuCabs(psi[cntk * Ny * Nx + cntj * Nx + cnti]), 0.0);
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
void outdeny(cuDoubleComplex *psi, MultiArray<double> &y, MultiArray<double> &tmpx,
             MultiArray<double> &tmpz, FILE *file) {
    for (long cntj = 0; cntj < Ny; cntj++) {
        // For each y position
        for (long cnti = 0; cnti < Nx; cnti++) {
            // Compute |psi|^2
            for (long cntk = 0; cntk < Nz; cntk++) {
                tmpz[cntk] = fma(cuCabs(psi[cntk * Ny * Nx + cntj * Nx + cnti]),
                                 cuCabs(psi[cntk * Ny * Nx + cntj * Nx + cnti]), 0.0);
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
void outdenz(cuDoubleComplex *psi, MultiArray<double> &z, MultiArray<double> &tmpx,
             MultiArray<double> &tmpy, FILE *file) {
    for (long cntk = 0; cntk < Nz; cntk++) {
        // For each z position
        for (long cntj = 0; cntj < Ny; cntj++) {
            // Compute |psi|^2
            for (long cnti = 0; cnti < Nx; cnti++) {
                tmpy[cntj] = fma(cuCabs(psi[cntk * Ny * Nx + cntj * Nx + cnti]),
                                 cuCabs(psi[cntk * Ny * Nx + cntj * Nx + cnti]), 0.0);
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
void outdenxy(cuDoubleComplex *psi, MultiArray<double> &x, MultiArray<double> &y,
              MultiArray<double> &tmpz, FILE *file) {
    for (long cnti = 0; cnti < Nx; cnti++) {
        for (long cntj = 0; cntj < Ny; cntj++) {
            for (long cntk = 0; cntk < Nz; cntk++) {
                tmpz[cntk] = fma(cuCabs(psi[cntk * Ny * Nx + cntj * Nx + cnti]),
                                 cuCabs(psi[cntk * Ny * Nx + cntj * Nx + cnti]), 0.0);
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
 * @param tmpx: Host: temporary array for y direction
 * @param file: File pointer to the output file
 */
void outdenxz(cuDoubleComplex *psi, MultiArray<double> &x, MultiArray<double> &z,
              MultiArray<double> &tmpx, FILE *file) {
    for (long cnti = 0; cnti < Nx; cnti++) {
        for (long cntk = 0; cntk < Nz; cntk++) {
            for (long cntj = 0; cntj < Ny; cntj++) {
                tmpx[cntj] = fma(cuCabs(psi[cntk * Ny * Nx + cntj * Nx + cnti]),
                                 cuCabs(psi[cntk * Ny * Nx + cntj * Nx + cnti]), 0.0);
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
void outdenyz(cuDoubleComplex *psi, MultiArray<double> &y, MultiArray<double> &z,
              MultiArray<double> &tmpx, FILE *file) {
    for (long cntj = 0; cntj < Ny; cntj++) {
        for (long cntk = 0; cntk < Nz; cntk++) {
            for (long cnti = 0; cnti < Nx; cnti++) {
                tmpx[cnti] = fma(cuCabs(psi[cntk * Ny * Nx + cntj * Nx + cnti]),
                                 cuCabs(psi[cntk * Ny * Nx + cntj * Nx + cnti]), 0.0);
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
void outpsi2xy(cuDoubleComplex *psi, MultiArray<double> &x, MultiArray<double> &y, FILE *file) {
    for (long cnti = 0; cnti < Nx; cnti++) {
        for (long cntj = 0; cntj < Ny; cntj++) {
            double psi2_xy = fma(cuCabs(psi[Nz2 * Ny * Nx + cntj * Nx + cnti]),
                                 cuCabs(psi[Nz2 * Ny * Nx + cntj * Nx + cnti]), 0.0);
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
void outpsi2xz(cuDoubleComplex *psi, MultiArray<double> &x, MultiArray<double> &z, FILE *file) {
    for (long cnti = 0; cnti < Nx; cnti++) {
        for (long cntk = 0; cntk < Nz; cntk++) {
            double psi2_xz = fma(cuCabs(psi[cntk * Ny * Nx + Ny2 * Nx + cnti]),
                                 cuCabs(psi[cntk * Ny * Nx + Ny2 * Nx + cnti]), 0.0);
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
void outpsi2yz(cuDoubleComplex *psi, MultiArray<double> &y, MultiArray<double> &z, FILE *file) {
    for (long cntj = 0; cntj < Ny; cntj++) {
        for (long cntk = 0; cntk < Nz; cntk++) {
            double psi2_yz = fma(cuCabs(psi[cntk * Ny * Nx + cntj * Nx + Nx2]),
                                 cuCabs(psi[cntk * Ny * Nx + cntj * Nx + Nx2]), 0.0);
            fwrite(&y[cntj], sizeof(double), 1, file);
            fwrite(&z[cntk], sizeof(double), 1, file);
            fwrite(&psi2_yz, sizeof(double), 1, file);
        }
    }
}