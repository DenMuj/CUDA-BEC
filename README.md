Programs designed to solve the time-dependent nonlinear partial differential Gross-Pitaevskii equation (GPE) with with contact and dipolar interactions in 3 spatial dimensions with a harmonic anisotropic trap. With the discovery of quantum dipolar droplets it was shown that the GPE can be used to describe them, but only if an additional term, quantum fluctuation term, is added which we have done.

***Structure of the DBEC-GP-CUDA directory:***

The directory consists of several folders and files.  
Folders:  
*src* - main folder that holds all the codes. Has main codes for imaginary time propagation (imag3d-cuda folder), real time propagation (real3d-cuda folder) and utils folder
which has class definitions for cuda arrays, vector arrays, simpson's integration, functions for spatial derivatives and functions for reading input parameters.  
*input* - folder that holds input files for programs. Has one input file for BEC in imaginary and real time propagation (imag3d-input-bec and real3d-input-bec respectively)
and one for a droplet in imaginary and real time propagation (imag3d-input-droplet and real3d-input-droplet respectively)  
*output* - folder that holdes output files as an example for the provided input files. Has 2 folders, one for BEC output and one for droplet output. Inside of each folders
are root mean square values for imaginary (imag3d-rms.txt) and real (real3d-rms.txt) time propagation with corresponding files for energies (imag3d-mu.txt and real3d-mu.txt)
and a binary file for the final wave function that is written from imaginary time propagation.  
*docs* - generated file using Doxygen that showes structure of codes.  
Files:  
*Makefile* - file used for compiling imag3d-cuda and real3d-cuda programs.  

***Compiling the programs***

We have provided a Makefile for compiling programs in LINUX. In order for the programs to compile, the following libraries need to be installed:  
1) CUDA Toolkit (for example CUDA Toolkit 12.0). The default path the program expect is ```/usr/local/cuda``` with compiler ```/usr/local/cuda/bin/nvcc```, headers
```/usr/local/cuda/include``` and libraries ```/usr/local/cuda/lib64```.  
2) OpenMP library (for example OpenMP 4.5). In our programs we have have used ```gcc``` compiler and Makefile assumes that OpenMP libraries and headers are in standard
system path.

Commands for compiling the program are (make sure you use these commands in the main directorium 
where both src, utils and Makefile are located):

```make```
which compiles both imag3d-cuda and real3d-cuda.  

```make imag3d-cuda``` or ```make real3d-cuda```  to compile just code for imaginary or real time propagation respectively.

If cuda is installed somwhere else, you can use that path with ```make CUDA_HOME = /your-path/cuda```. If OpenMP is not in standard path, use ```make OMP_HOME= /your-path/to-omp```.

***Running the programs***

Once the programs have been compiled we can run them. Input files necessary for the programs to run are located in ```./input/``` and they are ```imag3d-input-bec``` (for imaginary time propagation), ```real3d-input-bec``` (for real time propagation) for BEC and ```imag3d-input-droplet``` (for imaginary time propagation) and ```real3d-input-droplet``` (for real time propagation) for a droplet. The programs are then run as:  
```./imag3d-cuda -i input/imag3d-cuda-bec``` for imaginary time propagation (of a BEC for example) and  
```./real3d-cuda -i input/real3d-cuda-bec``` for real time propagation (of a BEC for example).  
While the program is running it can write to files ```imag3d-rms.txt```, ```imag3d-mu.txt``` for imaginary and ```real3d-rms.txt``` and ```real3d-mu.txt``` for real time propagation. Whether or not to write those file is determined by corresponding variables in input files. Also optionally is to write final wave function during imaginary time propagation (```imag3d-final.bin```) or density profiles (```imag3d-den-nitter*```).
