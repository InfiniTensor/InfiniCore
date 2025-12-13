#pragma once
#include <cuda_runtime.h>
#include <type_traits>
#include "../../../reduce/cuda/reduce.cuh"

namespace op::cuda {

template <typename T>
__global__ void hinge_embedding_loss_kernel(
    T *output,
    const T *input,
    const T *target,
    size_t n,
    T margin_val) {

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    T t = target[idx];
    T in = input[idx];
    T loss;
    if (t > 0) {
        loss = max(static_cast<T>(0.0), margin_val - in);
    } else {
        loss = max(static_cast<T>(0.0), in);
    }
    output[idx] = loss;
}

template <typename T, typename Tcompute>
__global__ void hinge_embedding_loss_reduce_kernel(
    T *output,
    const T *input,
    const T *target,
    size_t n,
    Tcompute margin_val) {

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    Tcompute sum = 0;

    for (size_t i = idx; i < n; i += blockDim.x * gridDim.x) {
        Tcompute t = static_cast<Tcompute>(target[i]);
        Tcompute in = static_cast<Tcompute>(input[i]);
        if (t > 0) {
            sum += max(static_cast<Tcompute>(0.0), margin_val - in);
        } else {
            sum += max(static_cast<Tcompute>(0.0), in);
        }
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
