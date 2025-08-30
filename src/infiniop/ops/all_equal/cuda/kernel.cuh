#ifndef __ALL_EQUAL_CUDA_H__
#define __ALL_EQUAL_CUDA_H__
#include "../../../elementwise/nvidia/elementwise_nvidia.cuh"
#include <cstddef>
#include <cstdint>

template <unsigned int BLOCK_SIZE, typename Tdata>
__global__ void compareKernel(size_t input_numel, size_t ndim, const bool *__restrict__ input_contiguous, const bool *__restrict__ input_broadcasted, const size_t *__restrict__ input_shapes, const ptrdiff_t *__restrict__ output_strides, const ptrdiff_t *__restrict__ input_strides, const void *const *inputs, uint8_t *flags) {
    const Tdata *const a = reinterpret_cast<const Tdata *const *>(inputs)[0];
    const Tdata *const b = reinterpret_cast<const Tdata *const *>(inputs)[1];

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < input_numel) {
        op::elementwise::nvidia::InputIndexer indexer{idx, ndim, input_contiguous, input_broadcasted, input_shapes, input_strides, output_strides};
        size_t idx_a = indexer(0);
        size_t idx_b = indexer(1);
        flags[idx] = (a[idx_a] != b[idx_b]) ? 1 : 0;
    }
}

template <unsigned int BLOCK_SIZE>
__global__ void countKernel(uint8_t *flags, unsigned int *count, int input_numel) {
    __shared__ unsigned int s_data[BLOCK_SIZE];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    s_data[tid] = (idx < input_numel) ? flags[idx] : 0;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_data[tid] += s_data[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(count, s_data[0]);
    }
}

#endif // __ALL_EQUAL_CUDA_H__
