#ifndef __SUM_CUDA_H__
#define __SUM_CUDA_H__
#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
// const SumInfo *info,
// 规约到标量的情况
template<size_t BLOCK_SIZE, typename T>
INFINIOP_CUDA_KERNEL sumAllKernel(
    T *output,
    const T *input,
    size_t input_size,
    size_t permuted_input_shape_size,
    size_t *permuted_input_shape,
    ptrdiff_t *permuted_input_strides){
    __shared__ T s_data[BLOCK_SIZE];
    size_t tid = threadIdx.x;
    size_t idx = tid + blockIdx.x * blockDim.x;
    if(idx < input_size){
        size_t input_offset = device::nvidia::indexToOffset(idx, permuted_input_shape_size, permuted_input_shape, permuted_input_strides);
        s_data[tid] = input[input_offset];
    } else {
        s_data[tid] = 0.;
    }
    for(size_t s = blockDim.x / 2; s > 0; s >>=1){
        __syncthreads();
        if(tid < s){
            s_data[tid] += s_data[tid + s];
        }
    }

    if(tid == 0){
        atomicAdd(output, s_data[0]);
    }
}

// 规约到非标量的情况, 假设output是[output_size, reduce_num]这种结构，暂时一个thread负责一个[1, reduce_num]的块 后续可以优化为一个block负责一个[1, reduce_num]的块
template<size_t BLOCK_SIZE, typename T>
INFINIOP_CUDA_KERNEL sumKernel(
    T *output,
    const T *input,
    // size_t reduce_dim_size,
    size_t permuted_input_shape_size,
    size_t output_shape_size,
    // size_t input_size,
    size_t output_size,
    size_t reduce_num,
    size_t *permuted_input_shape,
    size_t *output_shape,
    ptrdiff_t *permuted_input_strides,
    ptrdiff_t *output_strides){
    size_t tid = threadIdx.x;
    size_t idx = tid + blockIdx.x * blockDim.x;
    if(idx >= output_size) return;
    size_t output_index = device::nvidia::indexToOffset(idx, output_shape_size, output_shape, output_strides);
    T tempSum = 0.;
    for(size_t i = 0; i < reduce_num; i++){
        size_t input_offset = device::nvidia::indexToOffset(i + idx * reduce_num, permuted_input_shape_size, permuted_input_shape, permuted_input_strides);
        tempSum += input[input_offset];
    }
    output[output_index] = tempSum;
}

#endif // __SUM_CUDA_H__
