#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include <chrono>
#include <vector>

// Original kernel from user
__global__ void diff_kernel_original(double h, double *f, double *df, long N) {
    long index = threadIdx.x + blockDim.x * blockIdx.x;

    if (index >= N) {
        return;
    }
    // Precompute reciprocal to avoid division
    const double inv_12h = 1.0 / (12.0 * h);
    const double inv_2h = 1.0 / (2.0 * h);

    if (index == 0) {
       df[index] = 0.;
    }

    if (index == 1) {
       df[index] = (f[2] - f[0]) * inv_2h;
    }

    if (index > 1 && index < N - 2) {
       df[index] = (f[index - 2] - 8. * f[index - 1] + 8. * f[index + 1] - f[index + 2]) * inv_12h;
    }

    if (index == N - 2) {
       df[index] = (f[N - 1] - f[N - 3]) * inv_2h;
    }

    if (index == N - 1) {
       df[index] = 0.;
    }
}

// Optimized kernel with reduced warp divergence
// __global__ void diff_kernel_optimized(double h, const double* __restrict__ f, 
//                                       double* __restrict__ df, long N) {
//     long index = threadIdx.x + blockDim.x * blockIdx.x;
    
//     if (index >= N) return;
    
//     // Precompute reciprocals to avoid division
//     const double inv_12h = 1.0 / (12.0 * h);
//     const double inv_2h = 1.0 / (2.0 * h);
    
//     double result=0.0;
    
//      // Use predicated execution instead of branches
//      bool is_boundary = (index == 0) || (index == N - 1);
//      bool is_near_boundary = (index == 1) || (index == N - 2);
     
//      if (!is_boundary) {
//          if (is_near_boundary) {
//              // Second-order accurate for near-boundary points
//              int offset = (index == 1) ? 0 : N - 3;
//              int sign = (index == 1) ? 1 : -1;
//              result = sign * (f[offset + 2] - f[offset]) * inv_2h;
//          } else {
//              // Fourth-order accurate Richardson extrapolation
//              result = (f[index - 2] - 8.0 * f[index - 1] + 
//                       8.0 * f[index + 1] - f[index + 2]) * inv_12h;
//          }
//      }
     
//      df[index] = result;
// }

// Kenel for 2nd order derivative
__global__ void diff_kernel_optimized(double h, const double* __restrict__ f, 
    double* __restrict__ df, long N) {
long index = threadIdx.x + blockDim.x * blockIdx.x;

if (index >= N) return;
// Compute all cases without branching where possible
double result = 0.0;
// Precompute reciprocal to avoid division
const double inv_12h = 1.0 / (12.0 * h);
const double inv_2h = 1.0 / (2.0 * h);

// Use predicated execution instead of branches
bool is_boundary = (index == 0) || (index == N - 1);
bool is_near_boundary = (index == 1) || (index == N - 2);

if (!is_boundary) {
    if (is_near_boundary) {
        // Second-order accurate for near-boundary points
        int offset = (index == 1) ? 0 : N - 3;
        int sign = (index == 1) ? 1 : -1;
        result = sign * (f[offset + 2] - f[offset]) * inv_2h;
    } else {
        // Fourth-order accurate Richardson extrapolation
        result = (f[index - 2] - 8.0 * f[index - 1] + 
                 8.0 * f[index + 1] - f[index + 2]) * inv_12h;
    }
}

df[index] = result;
}

// Function to initialize test data: f(x) = x^2 * sin(x)
void initializeFunction(double* h_f, int N, double h) {
    for (int i = 0; i < N; i++) {
        double x = i * h;
        h_f[i] = x * x * sin(x);
    }
}

// Analytical derivative: f'(x) = 2x*sin(x) + x^2*cos(x)
double analyticalDerivative(double x) {
    return 2.0 * x * sin(x) + x * x * cos(x);
}

// Function to check accuracy
void checkAccuracy(double* h_df, int N, double h) {
    double max_error = 0.0;
    double avg_error = 0.0;
    int count = 0;
    
    for (int i = 2; i < N - 2; i++) {  // Check interior points
        double x = i * h;
        double analytical = analyticalDerivative(x);
        double numerical = h_df[i];
        double error = fabs(analytical - numerical);
        max_error = fmax(max_error, error);
        avg_error += error;
        count++;
    }
    
    if (count > 0) {
        avg_error /= count;
        std::cout << "  Max error: " << max_error << std::endl;
        std::cout << "  Avg error: " << avg_error << std::endl;
    }
}

// Timer class for benchmarking
class CudaTimer {
private:
    cudaEvent_t start, stop;
    
public:
    CudaTimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }
    
    ~CudaTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    
    void startTimer() {
        cudaEventRecord(start);
    }
    
    float stopTimer() {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        return milliseconds;
    }
};

int main() {
    // Parameters
    const long N = 128;
    const double h = 0.1;
    const int num_iterations = 10000;  // Run multiple times for better timing
    
    std::cout << "Richardson Extrapolation Kernel Benchmark" << std::endl;
    std::cout << "=========================================" << std::endl;
    std::cout << "N = " << N << ", h = " << h << std::endl;
    std::cout << "Function: f(x) = x^2 * sin(x)" << std::endl;
    std::cout << "Iterations for timing: " << num_iterations << std::endl << std::endl;
    
    // Allocate host memory
    double* h_f = new double[N];
    double* h_df_original = new double[N];
    double* h_df_optimized = new double[N];
    
    // Initialize function values
    initializeFunction(h_f, N, h);
    
    // Allocate device memory
    double *d_f, *d_df_original, *d_df_optimized;
    cudaMalloc(&d_f, N * sizeof(double));
    cudaMalloc(&d_df_original, N * sizeof(double));
    cudaMalloc(&d_df_optimized, N * sizeof(double));
    
    // Copy data to device
    cudaMemcpy(d_f, h_f, N * sizeof(double), cudaMemcpyHostToDevice);
    
    // Kernel configuration
    int threadsPerBlock = 128;
    int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    std::cout << "Kernel configuration: " << numBlocks << " blocks x " << threadsPerBlock << " threads" << std::endl << std::endl;
    
    CudaTimer timer;
    
    // Warm-up runs
    for (int i = 0; i < 100; i++) {
        diff_kernel_original<<<numBlocks, threadsPerBlock>>>(h, d_f, d_df_original, N);
        diff_kernel_optimized<<<numBlocks, threadsPerBlock>>>(h, d_f, d_df_optimized, N);
    }
    cudaDeviceSynchronize();
    
    // Benchmark original kernel
    std::cout << "Original Kernel Performance:" << std::endl;
    timer.startTimer();
    for (int i = 0; i < num_iterations; i++) {
        diff_kernel_original<<<numBlocks, threadsPerBlock>>>(h, d_f, d_df_original, N);
    }
    float time_original = timer.stopTimer();
    std::cout << "  Total time: " << time_original << " ms" << std::endl;
    std::cout << "  Time per iteration: " << time_original / num_iterations << " ms" << std::endl;
    
    // Copy results back for accuracy check
    cudaMemcpy(h_df_original, d_df_original, N * sizeof(double), cudaMemcpyDeviceToHost);
    checkAccuracy(h_df_original, N, h);
    std::cout << std::endl;
    
    // Benchmark optimized kernel
    std::cout << "Optimized Kernel Performance:" << std::endl;
    timer.startTimer();
    for (int i = 0; i < num_iterations; i++) {
        diff_kernel_optimized<<<numBlocks, threadsPerBlock>>>(h, d_f, d_df_optimized, N);
    }
    float time_optimized = timer.stopTimer();
    std::cout << "  Total time: " << time_optimized << " ms" << std::endl;
    std::cout << "  Time per iteration: " << time_optimized / num_iterations << " ms" << std::endl;
    
    // Copy results back for accuracy check
    cudaMemcpy(h_df_optimized, d_df_optimized, N * sizeof(double), cudaMemcpyDeviceToHost);
    checkAccuracy(h_df_optimized, N, h);
    std::cout << std::endl;
    
    // Performance comparison
    std::cout << "Performance Summary:" << std::endl;
    std::cout << "====================" << std::endl;
    float speedup = time_original / time_optimized;
    std::cout << "Speedup: " << speedup << "x" << std::endl;
    std::cout << "Performance improvement: " << ((speedup - 1.0) * 100) << "%" << std::endl;
    
    // Verify both kernels produce same results
    // std::cout << std::endl << "Result Comparison:" << std::endl;
    // bool results_match = true;
    // for (int i = 0; i < N; i++) {
    //     if (fabs(h_df_original[i] - h_df_optimized[i]) > 1e-10) {
    //         results_match = false;
    //         std::cout << "  Mismatch at index " << i << ": " 
    //                   << h_df_original[i] << " vs " << h_df_optimized[i] << std::endl;
    //     }
    // }
    // if (results_match) {
    //     std::cout << "  Both kernels produce identical results ✓" << std::endl;
    // }
    
    // Print sample results
    std::cout << std::endl << "Sample derivative values (first 10 points):" << std::endl;
    std::cout << "  i\tx\t\tAnalytical\tNumerical\tError" << std::endl;
    for (int i = 0; i < 10 && i < N; i++) {
        double x = i * h;
        double analytical = analyticalDerivative(x);
        std::cout << "  " << i << "\t" << x << "\t\t" 
                  << analytical << "\t" << h_df_optimized[i] << "\t" 
                  << fabs(analytical - h_df_optimized[i]) << std::endl;
    }

    
    // Cleanup
    delete[] h_f;
    delete[] h_df_original;
    delete[] h_df_optimized;
    cudaFree(d_f);
    cudaFree(d_df_original);
    cudaFree(d_df_optimized);
    
    return 0;
}