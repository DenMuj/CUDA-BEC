/**
 * @file multiarray.cpp
 * @brief Implementation of MultiArray class for 1D, 2D, and 3D arrays
 * 
 * This file contains the implementation of all MultiArray member functions,
 * including constructors, memory management, and access operations.
 */


#include "multiarray.h"
#include <complex>

/**
 * @brief Default constructor for MultiArray
 * @tparam T Data type
 */
template<typename T>
MultiArray<T>::MultiArray() : dim1(0), dim2(0), dim3(0) {}

/**
 * @brief Constructor for 1D MultiArray
 * @tparam T Data type
 * @param n1 Size of the 1D array
 */
template<typename T>
MultiArray<T>::MultiArray(size_t n1)
    : data(n1), dim1(n1), dim2(1), dim3(1) {}

/**
 * @brief Constructor for 2D MultiArray
 * @tparam T Data type
 * @param n1 Size of the 1D array
 * @param n2 Size of the 2D array
 */
template<typename T>
MultiArray<T>::MultiArray(size_t n1, size_t n2)
    : data(n1 * n2), dim1(n1), dim2(n2), dim3(1) {}

/**
 * @brief Constructor for 3D MultiArray
 * @tparam T Data type
 * @param n1 Size of the 1D array
 * @param n2 Size of the 2D array
 * @param n3 Size of the 3D array
 */
template<typename T>
MultiArray<T>::MultiArray(size_t n1, size_t n2, size_t n3)
    : data(n1 * n2 * n3), dim1(n1), dim2(n2), dim3(n3) {}

/**
 * @brief Resize the 1D MultiArray
 * @tparam T Data type
 * @param n1 Size of the 1D array
 */
template<typename T>
void MultiArray<T>::resize(size_t n1) {
    data.resize(n1);
    dim1 = n1; dim2 = 1; dim3 = 1;
}

/**
 * @brief Resize the 2D MultiArray
 * @tparam T Data type
 * @param n1 Size of the 1D array
 * @param n2 Size of the 2D array
 */
template<typename T>
void MultiArray<T>::resize(size_t n1, size_t n2) {
    data.resize(n1 * n2);
    dim1 = n1; dim2 = n2; dim3 = 1;
}

/**
 * @brief Resize the 3D MultiArray
 * @tparam T Data type
 * @param n1 Size of the 1D array
 * @param n2 Size of the 2D array
 * @param n3 Size of the 3D array
 */
template<typename T>
void MultiArray<T>::resize(size_t n1, size_t n2, size_t n3) {
    data.resize(n1 * n2 * n3);
    dim1 = n1; dim2 = n2; dim3 = n3;
}

/**
 * @brief Access the 1D MultiArray with bounds checking
 * @tparam T Data type
 * @param i Index of the 1D array
 */
template<typename T>
T& MultiArray<T>::at(size_t i) {
    if (i >= dim1) throw std::out_of_range("1D index out of range");
    return data[i];
}

/**
 * @brief Access the 1DMultiArray
 * @tparam T Data type
 * @param i Index of the 1D array
 */
template<typename T>
T& MultiArray<T>::operator[](size_t i) { 
    return data[i]; 
}
template<typename T>
const T& MultiArray<T>::operator[](size_t i) const { 
    return data[i]; 
}

/**
 * @brief Access the 2D MultiArray with bounds checking
 * @tparam T Data type
 * @param i Index of the 1D array
 * @param j Index of the 2D array
 */
template<typename T>
T& MultiArray<T>::at(size_t i, size_t j) {
    if (i >= dim1 || j >= dim2)
        throw std::out_of_range("2D index out of range");
    return data[i * dim2 + j];
}

/**
 * @brief Access the 2D MultiArray
 * @tparam T Data type
 * @param i Index of the 1D array
 * @param j Index of the 2D array
 */
template<typename T>
T& MultiArray<T>::operator()(size_t i, size_t j) { 
    return data[i * dim2 + j]; 
}
template<typename T>
const T& MultiArray<T>::operator()(size_t i, size_t j) const { 
    return data[i * dim2 + j]; 
}

/**
 * @brief Access the 3D MultiArray with bounds checking
 * @tparam T Data type
 * @param i Index of the 1D array
 * @param j Index of the 2D array
 * @param k Index of the 3D array
 */
template<typename T>
T& MultiArray<T>::at(size_t i, size_t j, size_t k) {
    if (i >= dim1 || j >= dim2 || k >= dim3)
        throw std::out_of_range("3D index out of range");
    return data[(i * dim2 + j) * dim3 + k];
}

/**
 * @brief Access the 3D MultiArray
 * @tparam T Data type
 * @param i Index of the 1D array
 * @param j Index of the 2D array
 * @param k Index of the 3D array
 */
template<typename T>
T& MultiArray<T>::operator()(size_t i, size_t j, size_t k) {
    return data[(i * dim2 + j) * dim3 + k];
}
template<typename T>
const T& MultiArray<T>::operator()(size_t i, size_t j, size_t k) const {
    return data[(i * dim2 + j) * dim3 + k];
}

/**
 * @brief Get raw pointer if needed
 * @tparam T Data type
 */
template<typename T>
T* MultiArray<T>::raw() { 
    return data.data(); 
}
template<typename T>
const T* MultiArray<T>::raw() const { 
    return data.data(); 
}

/**
 * @brief Get the size of the MultiArray
 * @tparam T Data type
 */
template<typename T>
size_t MultiArray<T>::size() const {
    return data.size();
}

/**
 * @brief Explicit template instantiations for common types
 * @tparam T Data type
 */
template class MultiArray<double>;
template class MultiArray<float>;
template class MultiArray<int>;
template class MultiArray<std::complex<double>>;
template class MultiArray<std::complex<float>>;

/**
 * @brief Explicit template instantiations for CUDA types
 * @tparam T Data type
 */
#include <cufft.h>
#include <cuComplex.h>
template class MultiArray<cuDoubleComplex>; 
template class MultiArray<cufftComplex>;