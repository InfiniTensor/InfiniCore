#ifndef __INFINIOP_INDEX_COPY_INPLACE_CUDA_KERNEL_CUH__
#define __INFINIOP_INDEX_COPY_INPLACE_CUDA_KERNEL_CUH__

#include "../../../devices/nvidia/nvidia_common.cuh"

template <typename Tdata>
__device__ void indexCopyInplaceKernelBlock(
    const Tdata *input_data,
    Tdata *output_data,
    const int64_t *__restrict__ index_data,
    int dim,//用扁平化参数
    int num_dims,
    size_t index_size,
    const size_t *output_shape,
    const ptrdiff_t *output_strides,
    const ptrdiff_t *input_strides) {//计算过程中不变的元数据都通过info结构体传递

    int64_t slice_idx = blockIdx.x;//每个block负责一个slice
    int64_t output_slice_offset = 0;
    int64_t input_slice_offset = 0;

    //块内地址计算地址偏移
    if(num_dims > 0){
        int64_t temp_slice_idx = slice_idx;
        for(ptrdiff_t i = num_dims - 1; i >= 0; --i){
            if(i == dim) continue;
            size_t current_dim_idx = temp_slice_idx % output_shape[i];
            temp_slice_idx /= output_shape[i];
            output_slice_offset += current_dim_idx * output_strides[i];
            input_slice_offset += current_dim_idx * input_strides[i];
        }
    }

    Tdata *output_slice_ptr = output_data + output_slice_offset;
    const Tdata *input_slice_ptr = input_data + input_slice_offset;
    //块内线程复制
    for(size_t i = threadIdx.x; i < index_size; i += blockDim.x){
        int64_t target_idx = index_data[i];

        if(num_dims == 0){//0维张量
            if(target_idx == 0){*output_slice_ptr = *input_slice_ptr;}
        }else{
            if(target_idx >= 0 && static_cast<size_t>(target_idx) < output_shape[dim]){
                output_slice_ptr[target_idx * output_strides[dim]] =
                    input_slice_ptr[i * input_strides[dim]];
                
            }
        }
    }
}

#endif
