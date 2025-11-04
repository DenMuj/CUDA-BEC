## CUDA-BEC: 3D Gross-Pitaevskii (GPE) solver with contact and dipolar interactions

Programs for solving the time-dependent nonlinear Gross-Pitaevskii equation (GPE) with contact and dipolar interactions in 3D, in an anisotropic harmonic trap. The code supports both imaginary-time (ground-state) and real-time (dynamics) propagation. Quantum droplets are supported via an additional quantum fluctuation (LHY) term.

### Features
- Imaginary-time propagation (`imag3d-cuda`) for ground-state, with optional LHY term
- Real-time propagation (`real3d-cuda`) for dynamics, with optional LHY term
- CUDA acceleration with cuFFT; OpenMP on the host side
- Configurable input files for BEC or droplet scenarios
- Optional writing of RMS, chemical potential and energies (kinetic, trap potential, contact, dipolar, quantum fluctuations), density profiles, and final wavefunction

### Requirements
- CUDA Toolkit (e.g., 12.x). Default path expected: `/usr/local/cuda` (compiler at `/usr/local/cuda/bin/nvcc`, headers in `/usr/local/cuda/include`, libs in `/usr/local/cuda/lib64`).
- A C++ compiler with OpenMP support (e.g., `gcc` + `-fopenmp`).

Optional:
- Doxygen (to regenerate `docs/`).

### Build
Use the provided `Makefile` on Linux.

Basic builds (from the repository root):

```bash
make            # builds both imag3d-cuda and real3d-cuda
make imag3d-cuda
make real3d-cuda
```

Override paths if needed (no spaces around `=`):

```bash
make CUDA_HOME=/your/path/to/cuda
make OMP_HOME=/your/path/to/openmp
```

### Run
Input files live in `input/` and include BEC and droplet presets for both imaginary and real time:
- `input/imag3d-input-bec`, `input/imag3d-input-droplet`
- `input/real3d-input-bec`, `input/real3d-input-droplet`

Examples:

```bash
./imag3d-cuda -i input/imag3d-input-bec
./real3d-cuda  -i input/real3d-input-bec
```

Output files and their names are controlled by keys in the input files, e.g.:
- `MUOUTPUT` (e.g., `imag3d-mu` or `real3d-mu`)
- `RMSOUT` (e.g., `imag3d-rms` or `real3d-rms`)
- `NITEROUT` for density snapshots during iterations (e.g., `*-den-niter`)
- `FINALPSI` for the final wavefunction (e.g., `imag3d-finalpsi`), which can be reused as input in real-time runs (`INPUT`, `INPUT_TYPE`)

Whether a given file is written depends on whether its corresponding key is set in the input file and, for density profiles, which `OUTFLAGS` you enable.

### Repository structure
- `src/`
  - `imag3d-cuda/` – imaginary-time solver and headers
  - `real3d-cuda/` – real-time solver and headers
  - `utils/` – CUDA arrays, vector arrays, Simpson integration, spatial derivatives, config reader
- `input/` – example input files for BEC and droplet (imaginary and real time)
- `output/` – example outputs for the provided inputs (RMS, MU, density profiles, final wavefunction)
- `docs/` – Doxygen-generated documentation
- `Makefile` – builds `imag3d-cuda` and `real3d-cuda`

### Documentation
Generated API/structure docs live in `docs/` (built with Doxygen). Regenerate with your local Doxygen setup if you modify headers.

### Tips & troubleshooting
- If CUDA is not in `/usr/local/cuda`, pass `CUDA_HOME=/path/to/cuda` to `make`.
- If OpenMP headers/libs are not in standard paths, pass `OMP_HOME=/path/to/openmp`.
- The Makefile auto-detects your GPU architecture via `nvidia-smi`; if detection fails, it defaults to `sm_75`.
- Make sure to run `make` from the repository root (where `src/` and `Makefile` reside).

### Citation and license
If you use this code in academic work, please cite the appropriate publications. Add license information here if applicable (e.g., MIT, BSD, GPL).
