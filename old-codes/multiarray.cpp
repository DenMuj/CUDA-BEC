#include "multiarray.h"
#include <complex>

// Template implementations
// Default constructor
template<typename T>
MultiArray<T>::MultiArray() : dim1(0), dim2(0), dim3(0) {}

// Constructor for 1D
template<typename T>
MultiArray<T>::MultiArray(size_t n1)
    : data(n1), dim1(n1), dim2(1), dim3(1) {}

// Constructor for 2D
template<typename T>
MultiArray<T>::MultiArray(size_t n1, size_t n2)
    : data(n1 * n2), dim1(n1), dim2(n2), dim3(1) {}

// Constructor for 3D
template<typename T>
MultiArray<T>::MultiArray(size_t n1, size_t n2, size_t n3)
    : data(n1 * n2 * n3), dim1(n1), dim2(n2), dim3(n3) {}

// Resize methods
template<typename T>
void MultiArray<T>::resize(size_t n1) {
    data.resize(n1);
    dim1 = n1; dim2 = 1; dim3 = 1;
}

template<typename T>
void MultiArray<T>::resize(size_t n1, size_t n2) {
    data.resize(n1 * n2);
    dim1 = n1; dim2 = n2; dim3 = 1;
}

template<typename T>
void MultiArray<T>::resize(size_t n1, size_t n2, size_t n3) {
    data.resize(n1 * n2 * n3);
    dim1 = n1; dim2 = n2; dim3 = n3;
}

// === 1D access ===
template<typename T>
T& MultiArray<T>::at(size_t i) {
    if (i >= dim1) throw std::out_of_range("1D index out of range");
    return data[i];
}

template<typename T>
T& MultiArray<T>::operator[](size_t i) { 
    return data[i]; 
}

template<typename T>
const T& MultiArray<T>::operator[](size_t i) const { 
    return data[i]; 
}

// === 2D access ===
template<typename T>
T& MultiArray<T>::at(size_t i, size_t j) {
    if (i >= dim1 || j >= dim2)
        throw std::out_of_range("2D index out of range");
    return data[i * dim2 + j];
}

template<typename T>
T& MultiArray<T>::operator()(size_t i, size_t j) { 
    return data[i * dim2 + j]; 
}

template<typename T>
const T& MultiArray<T>::operator()(size_t i, size_t j) const { 
    return data[i * dim2 + j]; 
}

// === 3D access ===
template<typename T>
T& MultiArray<T>::at(size_t i, size_t j, size_t k) {
    if (i >= dim1 || j >= dim2 || k >= dim3)
        throw std::out_of_range("3D index out of range");
    return data[(i * dim2 + j) * dim3 + k];
}

template<typename T>
T& MultiArray<T>::operator()(size_t i, size_t j, size_t k) {
    return data[(i * dim2 + j) * dim3 + k];
}

template<typename T>
const T& MultiArray<T>::operator()(size_t i, size_t j, size_t k) const {
    return data[(i * dim2 + j) * dim3 + k];
}

// Get raw pointer if needed for legacy C code or CUDA kernels
template<typename T>
T* MultiArray<T>::raw() { 
    return data.data(); 
}

template<typename T>
const T* MultiArray<T>::raw() const { 
    return data.data(); 
}

// Utility functions
template<typename T>
size_t MultiArray<T>::size() const {
    return data.size();
}

// Explicit template instantiations for common types
// Add any types you plan to use here
template class MultiArray<double>;
template class MultiArray<float>;
template class MultiArray<int>;
template class MultiArray<std::complex<double>>;
template class MultiArray<std::complex<float>>;

//If using CUDA, uncomment these:
#include <cufft.h>
#include <cuComplex.h>
template class MultiArray<cuDoubleComplex>; 
template class MultiArray<cufftComplex>;