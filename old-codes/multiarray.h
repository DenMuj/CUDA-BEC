#ifndef MULTIARRAY_H
#define MULTIARRAY_H

#include <vector>
#include <stdexcept>

template<typename T>
class MultiArray {
private:
    std::vector<T> data;
    size_t dim1, dim2, dim3; // sizes (dim2, dim3 = 1 if unused)

public:
    // Constructors
    MultiArray();
    MultiArray(size_t n1);
    MultiArray(size_t n1, size_t n2);
    MultiArray(size_t n1, size_t n2, size_t n3);

    // Resize methods
    void resize(size_t n1);
    void resize(size_t n1, size_t n2);
    void resize(size_t n1, size_t n2, size_t n3);
    
    // === 1D access ===
    T& at(size_t i);
    T& operator[](size_t i);
    const T& operator[](size_t i) const;
    
    // === 2D access ===
    T& at(size_t i, size_t j);
    T& operator()(size_t i, size_t j);
    const T& operator()(size_t i, size_t j) const;
    
    // === 3D access ===
    T& at(size_t i, size_t j, size_t k);
    T& operator()(size_t i, size_t j, size_t k);
    const T& operator()(size_t i, size_t j, size_t k) const;
    
    // Get raw pointer if needed for CUDA kernels
    T* raw();
    const T* raw() const;
    
    // Utility functions
    size_t size() const;
    size_t getDim1() const { return dim1; }
    size_t getDim2() const { return dim2; }
    size_t getDim3() const { return dim3; }
};

#endif // MULTIARRAY_H