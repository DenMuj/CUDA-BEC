/**
 * @file cuda_error_check.cuh
 * @brief CUDA error checking macros with descriptive messages
 * 
 * This header provides efficient CUDA error checking that:
 * - Provides descriptive error messages with location info
 * - Uses async checks by default (no performance impact in main loop)
 * - Optionally enables full synchronization in debug mode
 * 
 * Usage:
 *   CUDA_CHECK(cudaMalloc(...));           // Check API calls
 *   CUDA_CHECK_KERNEL("kernelName");       // Check after kernel launch (async)
 *   CUDA_SYNC_CHECK("location");           // Force sync and check (debug only)
 *   CUFFT_CHECK(cufftExec...);             // Check cuFFT calls
 */

#ifndef CUDA_ERROR_CHECK_CUH
#define CUDA_ERROR_CHECK_CUH

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cufft.h>

// Enable full synchronization checks in debug mode
// Uncomment the following line for debugging kernel errors:
// #define CUDA_DEBUG_SYNC

/**
 * @brief Check CUDA API call and exit with descriptive message on error
 * @param call The CUDA API call to check
 * 
 * Use for: cudaMalloc, cudaMemcpy, cudaMemcpyToSymbol, cudaHostAlloc, etc.
 * Cost: Minimal - just checks return value, no sync
 */
#define CUDA_CHECK(call)                                                                           \
    do {                                                                                           \
        cudaError_t err = (call);                                                                  \
        if (err != cudaSuccess) {                                                                  \
            std::fprintf(stderr, "CUDA Error at %s:%d in %s:\n  %s\n  Error: %s\n", __FILE__,      \
                         __LINE__, __func__, #call, cudaGetErrorString(err));                      \
            std::exit(EXIT_FAILURE);                                                               \
        }                                                                                          \
    } while (0)

/**
 * @brief Check for errors after kernel launch (async - no performance impact)
 * @param kernel_name Descriptive name of the kernel for error messages
 * 
 * Use after: kernel<<<...>>>(...);
 * Cost:no synchronization
 * Note: May not catch the actual kernel error, but catches launch config errors
 */
#define CUDA_CHECK_KERNEL(kernel_name)                                                             \
    do {                                                                                           \
        cudaError_t err = cudaGetLastError();                                                      \
        if (err != cudaSuccess) {                                                                  \
            std::fprintf(stderr, "CUDA Kernel Launch Error at %s:%d:\n  Kernel: %s\n  Error: %s\n",\
                         __FILE__, __LINE__, kernel_name, cudaGetErrorString(err));                \
            std::exit(EXIT_FAILURE);                                                               \
        }                                                                                          \
    } while (0)

/**
 * @brief Synchronize and check for kernel execution errors (debug only)
 * @param location Descriptive location for error messages
 * 
 * Use: After kernel launches when debugging
 * Cost: forces GPU synchronization
 */
#ifdef CUDA_DEBUG_SYNC
#define CUDA_SYNC_CHECK(location)                                                                  \
    do {                                                                                           \
        cudaError_t err = cudaDeviceSynchronize();                                                 \
        if (err != cudaSuccess) {                                                                  \
            std::fprintf(stderr, "CUDA Sync Error at %s:%d [%s]:\n  Error: %s\n", __FILE__,        \
                         __LINE__, location, cudaGetErrorString(err));                             \
            std::exit(EXIT_FAILURE);                                                               \
        }                                                                                          \
    } while (0)
#else
#define CUDA_SYNC_CHECK(location) ((void)0)
#endif

/**
 * @brief Check cuFFT API call and exit with descriptive message on error
 * @param call The cuFFT API call to check
 * 
 * Use for: cufftPlan, cufftExec, cufftSetWorkArea, etc.
 */
#define CUFFT_CHECK(call)                                                                          \
    do {                                                                                           \
        cufftResult err = (call);                                                                  \
        if (err != CUFFT_SUCCESS) {                                                                \
            std::fprintf(stderr, "cuFFT Error at %s:%d in %s:\n  %s\n  Error code: %d\n", __FILE__,\
                         __LINE__, __func__, #call, (int)err);                                     \
            std::exit(EXIT_FAILURE);                                                               \
        }                                                                                          \
    } while (0)

/**
 * @brief Get cuFFT error string
 * @param error cuFFT error code
 * @return Human-readable error string
 */
inline const char* cufftGetErrorString(cufftResult error) {
    switch (error) {
        case CUFFT_SUCCESS:        return "CUFFT_SUCCESS";
        case CUFFT_INVALID_PLAN:   return "CUFFT_INVALID_PLAN";
        case CUFFT_ALLOC_FAILED:   return "CUFFT_ALLOC_FAILED";
        case CUFFT_INVALID_TYPE:   return "CUFFT_INVALID_TYPE";
        case CUFFT_INVALID_VALUE:  return "CUFFT_INVALID_VALUE";
        case CUFFT_INTERNAL_ERROR: return "CUFFT_INTERNAL_ERROR";
        case CUFFT_EXEC_FAILED:    return "CUFFT_EXEC_FAILED";
        case CUFFT_SETUP_FAILED:   return "CUFFT_SETUP_FAILED";
        case CUFFT_INVALID_SIZE:   return "CUFFT_INVALID_SIZE";
        case CUFFT_UNALIGNED_DATA: return "CUFFT_UNALIGNED_DATA";
        default:                   return "UNKNOWN_CUFFT_ERROR";
    }
}

#endif // CUDA_ERROR_CHECK_CUH

