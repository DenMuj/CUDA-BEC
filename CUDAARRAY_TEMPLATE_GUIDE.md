# CudaArray3D Template Guide

## Overview

Your `CudaArray3D` class has been successfully converted to a template-based implementation, similar to your `MultiArray` class. This allows you to specify different data types when creating CUDA arrays.

## What Changed

### Before (Original)
```cpp
CudaArray3D array(nx, ny, nz);  // Always used double
```

### After (Templated)
```cpp
CudaArray3D<double> array(nx, ny, nz);           // Explicit double
CudaArray3D<cuDoubleComplex> array(nx, ny, nz);  // Complex double
CudaArray3D<float> array(nx, ny, nz);            // Single precision
```

## Supported Data Types

The template supports the following data types with explicit instantiations:

### Real Number Types
- `double` - Double precision real numbers
- `float` - Single precision real numbers

### Complex Number Types
- `cuDoubleComplex` - CUDA double precision complex numbers
- `cuFloatComplex` - CUDA single precision complex numbers

## Usage Examples

### 1. Basic Usage with Different Types

```cpp
#include "src/utils/CudaArray.h"

// Double precision array (most common)
CudaArray3D<double> psi_real(256, 256, 256);

// Complex array for wave functions
CudaArray3D<cuDoubleComplex> psi_complex(256, 256, 256);

// Single precision for memory savings
CudaArray3D<float> temp_array(128, 128, 128);
```

### 2. Working with Complex Numbers

```cpp
#include <cuComplex.h>

// Create complex array
CudaArray3D<cuDoubleComplex> complex_array(nx, ny, nz);

// All existing methods work the same way
complex_array.memset(0);  // Zero initialize

// Copy data
cuDoubleComplex* host_data = new cuDoubleComplex[nx * ny * nz];
// ... fill host_data ...
complex_array.copyFromHost(host_data);

// Get device pointer for kernels
cuDoubleComplex* d_ptr = complex_array.raw();
```

### 3. Backward Compatibility

For existing code, you can use the convenience alias:

```cpp
// This still works and defaults to double
CudaArray3D_double array(nx, ny, nz);  // Same as CudaArray3D<double>
```

## Method Signatures

All methods now work with the template type `T`:

```cpp
template<typename T>
class CudaArray3D {
public:
    // Constructors
    CudaArray3D(size_t nx, size_t ny, size_t nz, bool use_pitched = true);
    explicit CudaArray3D(size_t n);  // 1D array
    
    // Data transfer
    void copyFromHost(const T* h_data);
    void copyToHost(T* h_data) const;
    void copyFromHostAsync(const T* h_data, cudaStream_t stream = 0);
    void copyToHostAsync(T* h_data, cudaStream_t stream = 0) const;
    
    // Device pointers
    T* raw();
    const T* raw() const;
    T* data();  // Alias for raw()
    const T* data() const;
    
    // Element access
    T* getElementPtr(size_t ix, size_t iy, size_t iz);
    
    // Memory layout (now uses sizeof(T))
    size_t getPitchElements() const;  // Uses sizeof(T)
    size_t getPitchBytes() const;     // Uses sizeof(T)
    size_t getSlicePitch() const;     // Uses sizeof(T)
    
    // ... other methods unchanged
};
```

## Migration Guide

### For New Code
Always use explicit template parameters:
```cpp
CudaArray3D<double> array(nx, ny, nz);           // Recommended
CudaArray3D<cuDoubleComplex> complex_array(nx, ny, nz);  // For complex data
```

### For Existing Code
Your existing code should continue to work if you:

1. **Option 1: Use the alias (minimal changes)**
   ```cpp
   // Change this:
   CudaArray3D array(nx, ny, nz);
   
   // To this:
   CudaArray3D_double array(nx, ny, nz);
   ```

2. **Option 2: Use explicit templates (recommended)**
   ```cpp
   // Change this:
   CudaArray3D array(nx, ny, nz);
   
   // To this:
   CudaArray3D<double> array(nx, ny, nz);
   ```

## Integration with Your Project

### Real3D and Imag3D CUDA Code
Your existing kernels will work unchanged:

```cpp
// In your kernel launches
CudaArray3D<double> d_psi(nx, ny, nz);
CudaArray3D<cuDoubleComplex> d_psi_complex(nx, ny, nz);

// Kernel calls remain the same
my_kernel<<<blocks, threads>>>(d_psi.raw(), d_psi_complex.raw(), nx, ny, nz);
```

### cuFFT Integration
Complex arrays work seamlessly with cuFFT:

```cpp
CudaArray3D<cuDoubleComplex> fft_array(nx, ny, nz);

// Use with cuFFT
cufftExecZ2Z(plan, 
    fft_array.raw(),     // input
    fft_array.raw(),     // output
    CUFFT_FORWARD);
```

## Performance Considerations

- **No Performance Impact**: Template instantiation happens at compile time
- **Memory Layout**: All optimizations (pitched memory, alignment) are preserved
- **Type Safety**: Compile-time type checking prevents mixing incompatible types

## Compilation Notes

- The template is explicitly instantiated for common types in `CudaArray.cpp`
- If you need additional types, add them to the explicit instantiation list
- CUDA headers (`cuComplex.h`, `cufft.h`) are automatically included

## Common Patterns

### Pattern 1: Real and Complex Arrays
```cpp
// Real part
CudaArray3D<double> psi_real(nx, ny, nz);

// Complex representation
CudaArray3D<cuDoubleComplex> psi_complex(nx, ny, nz);

// Convert real to complex (in kernel)
convert_real_to_complex<<<blocks, threads>>>(
    psi_real.raw(), psi_complex.raw(), nx, ny, nz);
```

### Pattern 2: Mixed Precision
```cpp
// High precision computation
CudaArray3D<double> precise_array(nx, ny, nz);

// Memory-efficient storage
CudaArray3D<float> storage_array(nx, ny, nz);

// Convert precision as needed
convert_precision<<<blocks, threads>>>(
    precise_array.raw(), storage_array.raw(), nx, ny, nz);
```

## Troubleshooting

### Compilation Errors
1. **Template instantiation errors**: Make sure you're using one of the explicitly instantiated types
2. **CUDA type errors**: Include `<cuComplex.h>` and `<cufft.h>` as needed
3. **Linking errors**: Ensure the explicit instantiations in `CudaArray.cpp` match your usage

### Runtime Issues
1. **Size mismatches**: Remember that `sizeof(T)` varies by type
2. **Alignment**: Complex types have different alignment requirements
3. **Memory**: Complex types use twice the memory of real types

## Summary

The templated `CudaArray3D` provides:
- ✅ Type safety at compile time
- ✅ Support for multiple data types (double, float, cuDoubleComplex, etc.)
- ✅ Backward compatibility with existing code
- ✅ Same performance and memory optimizations
- ✅ Seamless integration with CUDA and cuFFT

Your code can now handle different numerical types efficiently while maintaining the same easy-to-use interface!
