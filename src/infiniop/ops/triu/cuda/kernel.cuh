#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace op::triu::cuda {

// CUDA kernel for triu operation
template<typename T>
__global__ void triu_kernel(
    T *output,
    const T *input,
    size_t rows,
    size_t cols,
    int diagonal) {
    
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_elements = rows * cols;
    
    if (idx < total_elements) {
        size_t i = idx / cols;  // row index
        size_t j = idx % cols;  // col index
        
        // Keep upper triangular part (including diagonal offset)
        if (static_cast<int>(j) >= static_cast<int>(i) + diagonal) {
            output[idx] = input[idx];
        } else {
            output[idx] = T{};  // Zero out lower triangular part
        }
    }
}

// CUDA kernel for inplace triu operation
template<typename T>
__global__ void triu_kernel_inplace(
    T *input_output,
    size_t rows,
    size_t cols,
    int diagonal) {
    
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_elements = rows * cols;
    
    if (idx < total_elements) {
        size_t i = idx / cols;  // row index
        size_t j = idx % cols;  // col index
        
        // Zero out lower triangular part
        if (static_cast<int>(j) < static_cast<int>(i) + diagonal) {
            input_output[idx] = T{};
        }
    }
}

} // namespace op::triu::cuda