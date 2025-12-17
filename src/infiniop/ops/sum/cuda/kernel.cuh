#ifndef __SUM_CUDA_H__
#define __SUM_CUDA_H__

// todo把具体的include 的相关代码放到对应的平台下的文件夹中

// const SumInfo *info,
// 规约到标量的情况

__forceinline__ __device__ __host__ size_t
indexToOffset(
    size_t flat_index,
    size_t ndim,
    const size_t *shape,
    const ptrdiff_t *strides) {
    size_t res = 0;
    for (size_t i = ndim; i-- > 0;) {
        res += (flat_index % shape[i]) * strides[i];
        flat_index /= shape[i];
    }
    return res;
}
// todo 去除 indexToOffset 和 common 中 device::nvidia::indexToOffset 重名的影响

// BLOCK_SIZE = 256, GRID_SIZE = (input_size + BLOCK_SIZE - 1) / BLOCK_SIZE
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
        size_t input_offset = indexToOffset(idx, permuted_input_shape_size, permuted_input_shape, permuted_input_strides);
        s_data[tid] = input[input_offset];
    } else {
        s_data[tid] = T(0);
    }
    __syncthreads();
    for(size_t s = blockDim.x / 2; s > 0; s >>=1){
        if(tid < s){
            s_data[tid] += s_data[tid + s];
        }
        __syncthreads();
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
    size_t output_index = indexToOffset(idx, output_shape_size, output_shape, output_strides);
    T tempSum = T(0);
    for(size_t i = 0; i < reduce_num; i++){
        size_t input_offset = indexToOffset(i + idx * reduce_num, permuted_input_shape_size, permuted_input_shape, permuted_input_strides);
        tempSum += input[input_offset];
    }
    output[output_index] = tempSum;
}

#endif // __SUM_CUDA_H__
