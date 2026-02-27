#pragma once
#include <cuda_runtime.h>
#include <type_traits>
#include <cmath>
#include "../../../reduce/cuda/reduce.cuh"

namespace op::cuda {

template <typename T>
__global__ void gaussian_nll_loss_kernel(
    T *output,
    const T *input,
    const T *target,
    const T *var,
    size_t n,
    T eps_val,
    int full) {

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    T diff = input[idx] - target[idx];
    T var_val = var[idx] + eps_val;
    T loss = T(0.5) * (log(var_val) + (diff * diff) / var_val);
    if (full) {
        T log_2pi = T(0.9189385332046727);  // log(2*pi) / 2
        loss += log_2pi;
    }
    output[idx] = loss;
}

template <typename T, typename Tcompute>
__global__ void gaussian_nll_loss_reduce_kernel(
    T *output,
    const T *input,
    const T *target,
    const T *var,
    size_t n,
    Tcompute eps_val,
    int full) {

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    Tcompute sum = 0;

    Tcompute log_2pi = full ? Tcompute(0.9189385332046727) : Tcompute(0.0);

    for (size_t i = idx; i < n; i += blockDim.x * gridDim.x) {
        Tcompute diff = static_cast<Tcompute>(input[i]) - static_cast<Tcompute>(target[i]);
        Tcompute var_val = static_cast<Tcompute>(var[i]) + eps_val;
        Tcompute loss = Tcompute(0.5) * (log(var_val) + (diff * diff) / var_val) + log_2pi;
        sum += loss;
    }

    using BlockReduce = cub::BlockReduce<Tcompute, 256>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    Tcompute block_sum = BlockReduce(temp_storage).Sum(sum);

    if (threadIdx.x == 0) {
        if (blockIdx.x == 0) {
            *output = static_cast<T>(block_sum);
        } else {
            atomicAdd(reinterpret_cast<Tcompute *>(output), block_sum);
        }
    }
}

} // namespace op::cuda
