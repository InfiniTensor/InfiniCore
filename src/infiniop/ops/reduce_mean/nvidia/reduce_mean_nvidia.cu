#include "../../../devices/nvidia/nvidia_common.cuh"
#include "reduce_mean_nvidia.cuh"

#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include <cub/block/block_reduce.cuh>

#include "../../../reduce/cuda/reduce.cuh"
#include "../../../../utils.h"
#include "infinicore.h"

template <unsigned int BLOCK_SIZE, typename Tdata>
INFINIOP_CUDA_KERNEL reduceMeanKernel(
    Tdata *__restrict__ output,
    const Tdata *__restrict__ input,
    size_t num_reductions,
    size_t reduce_size,
    ptrdiff_t reduce_stride,
    const size_t *input_shape,
    const ptrdiff_t *input_strides,
    const ptrdiff_t *output_strides,
    size_t ndim,
    size_t reduce_dim) {
    
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_reductions) {
        // Calculate the input and output offsets for this reduction
        size_t input_offset = 0;
        size_t output_offset = 0;
        size_t temp_idx = idx;
        
        for (size_t i = 0; i < ndim; i++) {
            if (i != reduce_dim) {
                size_t coord = temp_idx % input_shape[i];
                temp_idx /= input_shape[i];
                input_offset += coord * input_strides[i];
                output_offset += coord * output_strides[i];
            }
        }
        
        // Find the starting position for this reduction
        const Tdata *input_ptr = input + input_offset;
        
        // Calculate the sum
        if constexpr (std::is_same_v<Tdata, fp16_t>) {
            __half sum = __float2half(0.0f);
            for (size_t i = 0; i < reduce_size; i++) {
                __half val = *reinterpret_cast<const __half*>(&input_ptr[i * reduce_stride]);
                sum = __hadd(sum, val);
            }
            __half mean = __hdiv(sum, __float2half(static_cast<float>(reduce_size)));
            output[output_offset] = *reinterpret_cast<const Tdata*>(&mean);
        } else if constexpr (std::is_same_v<Tdata, bf16_t>) {
            float sum = 0.0f;
            for (size_t i = 0; i < reduce_size; i++) {
                __nv_bfloat16 val = *reinterpret_cast<const __nv_bfloat16*>(&input_ptr[i * reduce_stride]);
                sum += __bfloat162float(val);
            }
            float mean = sum / static_cast<float>(reduce_size);
            __nv_bfloat16 bf16_mean = __float2bfloat16(mean);
            output[output_offset] = *reinterpret_cast<const Tdata*>(&bf16_mean);
        } else {
            float sum = 0.0f;
            for (size_t i = 0; i < reduce_size; i++) {
                sum += input_ptr[i * reduce_stride];
            }
            output[output_offset] = sum / static_cast<float>(reduce_size);
        }
    }
}

template <typename T>
infiniStatus_t launchReduceMeanKernel(
    const op::reduce_mean::nvidia::ReduceMeanInfo &info,
    void *output,
    const void *input,
    cudaStream_t stream) {
    
    const int BLOCK_SIZE = 256;
    
    // Calculate number of reductions needed
    size_t num_reductions = info.output_size;
    size_t reduce_size = info.input_shape[info.reduce_dim];
    ptrdiff_t reduce_stride = info.input_strides[info.reduce_dim];
    
    // Allocate device memory for shape and strides
    size_t *d_input_shape;
    ptrdiff_t *d_input_strides, *d_output_strides;
    
    cudaMalloc(&d_input_shape, info.ndim * sizeof(size_t));
    cudaMalloc(&d_input_strides, info.ndim * sizeof(ptrdiff_t));
    cudaMalloc(&d_output_strides, info.ndim * sizeof(ptrdiff_t));
    
    cudaMemcpy(d_input_shape, info.input_shape.data(), info.ndim * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_strides, info.input_strides.data(), info.ndim * sizeof(ptrdiff_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output_strides, info.output_strides.data(), info.ndim * sizeof(ptrdiff_t), cudaMemcpyHostToDevice);
    
    // Calculate grid size
    size_t grid_size = (num_reductions + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Launch kernel
    reduceMeanKernel<BLOCK_SIZE, T><<<grid_size, BLOCK_SIZE, 0, stream>>>(
        static_cast<T*>(output),
        static_cast<const T*>(input),
        num_reductions,
        reduce_size,
        reduce_stride,
        d_input_shape,
        d_input_strides,
        d_output_strides,
        info.ndim,
        info.reduce_dim
    );
    
    // Clean up device memory
    cudaFree(d_input_shape);
    cudaFree(d_input_strides);
    cudaFree(d_output_strides);
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return INFINI_STATUS_INTERNAL_ERROR;
    }
    
    return INFINI_STATUS_SUCCESS;
}

namespace op::reduce_mean::nvidia {

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_desc,
    size_t dim) {
    
    // Validate inputs
    if (!input_desc || !output_desc) {
        return INFINI_STATUS_BAD_PARAM;
    }
    
    if (dim >= input_desc->ndim()) {
        return INFINI_STATUS_BAD_PARAM;
    }
    
    // Check dtype support
    if (input_desc->dtype() != INFINI_DTYPE_F32 && 
        input_desc->dtype() != INFINI_DTYPE_F16 && 
        input_desc->dtype() != INFINI_DTYPE_BF16) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    
    // Create info
    ReduceMeanInfo info;
    info.ndim = input_desc->ndim();
    info.reduce_dim = dim;
    info.dtype = input_desc->dtype();
    
    // Copy shapes and strides
    info.input_shape.resize(info.ndim);
    info.output_shape.resize(info.ndim);
    info.input_strides.resize(info.ndim);
    info.output_strides.resize(info.ndim);
    
    info.input_size = 1;
    info.output_size = 1;
    
    for (size_t i = 0; i < info.ndim; i++) {
        info.input_shape[i] = input_desc->shape()[i];
        info.input_strides[i] = input_desc->strides()[i];
        info.input_size *= input_desc->shape()[i];
        
        if (i == dim) {
            info.output_shape[i] = 1;
            info.output_strides[i] = output_desc->strides()[i];
        } else {
            info.output_shape[i] = input_desc->shape()[i];
            info.output_strides[i] = output_desc->strides()[i];
            info.output_size *= input_desc->shape()[i];
        }
    }
    
    *desc_ptr = new Descriptor(
        handle->device,
        handle->device_id,
        std::move(info),
        0
    );
    
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    void *stream) const {
    
    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    
    switch (_info.dtype) {
        case INFINI_DTYPE_F32:
            return ::launchReduceMeanKernel<float>(_info, output, input, cuda_stream);
        case INFINI_DTYPE_F16:
            return ::launchReduceMeanKernel<fp16_t>(_info, output, input, cuda_stream);
        case INFINI_DTYPE_BF16:
            return ::launchReduceMeanKernel<bf16_t>(_info, output, input, cuda_stream);
        default:
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace op::reduce_mean::nvidia