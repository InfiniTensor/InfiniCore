#ifndef __ALL_CUDA_H__
#define __ALL_CUDA_H__

// todo把具体的include 的相关代码放到对应的平台下的文件夹中

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

template<size_t BLOCK_SIZE, typename Tdata>
__global__ void allReduceOneKernel(
    bool *output,
    const Tdata *input,
    size_t input_size,
    size_t permuted_input_shape_size,
    size_t *permuted_input_shape,
    ptrdiff_t *permuted_input_strides){
    __shared__ bool s_data[BLOCK_SIZE];
    size_t tid = threadIdx.x;
    size_t idx = tid + blockIdx.x * blockDim.x;
    if(idx < input_size){
        size_t input_offset = indexToOffset(idx, permuted_input_shape_size, permuted_input_shape, permuted_input_strides);
        s_data[tid] = static_cast<bool>(input[input_offset]);
    } else {
        s_data[tid] = true;
    }
    __syncthreads();
    for(size_t s = blockDim.x / 2; s > 0; s >>=1){
        if(tid < s){
            // error: no viable overloaded '+='
            s_data[tid] = s_data[tid] && s_data[tid + s];
        }
        __syncthreads();
    }
    if(tid == 0){
        output[blockIdx.x] = s_data[0];
    }
    __syncthreads();
    if(blockIdx.x == 0){
        bool final_result = true;
        for(size_t i = 0; i < gridDim.x; i+=blockDim.x){
            final_result = final_result && output[i];
        }
        s_data[tid] = final_result;
        __syncthreads();

        for(size_t s = blockDim.x / 2; s > 0; s >>=1){
            if(tid < s){
                s_data[tid] = s_data[tid] && s_data[tid + s];
            }
            __syncthreads();
        }
        if(tid == 0){
            output[0] = s_data[0];
        }
    }
}

// 规约到非标量的情况, 假设output是[output_size, reduce_num]这种结构，暂时一个thread负责一个[1, reduce_num]的块 后续可以优化为一个block负责一个[1, reduce_num]的块
template<size_t BLOCK_SIZE, typename Tdata>
__global__ void allKernel(
    bool *output,
    const Tdata *input,
    size_t permuted_input_shape_size,
    size_t output_shape_size,
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
    bool tempRes = true;
    for(size_t i = 0; i < reduce_num; i++){
        size_t input_offset = indexToOffset(i + idx * reduce_num, permuted_input_shape_size, permuted_input_shape, permuted_input_strides);
        tempRes = tempRes && static_cast<bool>(input[input_offset]);
    }
        output[output_index] = tempRes;
    }

#endif // __ALL_CUDA_H__
