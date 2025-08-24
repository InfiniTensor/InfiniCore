#include "../../../devices/nvidia/nvidia_handle.cuh"
#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"

#include <cub/block/block_reduce.cuh>
#include "../../../reduce/cuda/reduce.cuh"
#include "../cuda/kernel.cuh"
#include "layer_norm_nvidia.cuh"
#include "../info.h"

namespace op::layer_norm::nvidia {

//  ---------------------- start: launchKernel: call kernel function of CUDA -----------------------
template <unsigned int BLOCK_SIZE, typename Tdata, typename Tcompute>
INFINIOP_CUDA_KERNEL launchKernel(
    Tdata * output,
    Tdata * input_standardization,
    Tdata * input_std_deviation,
    const Tdata * input,
    const Tdata * weight,
    const Tdata * bias,
    float eps,
    size_t normalized_size,
    const ptrdiff_t* output_strides,
    const ptrdiff_t* input_standardization_strides,
    const ptrdiff_t* input_std_deviation_strides,
    const ptrdiff_t* input_strides,
    ptrdiff_t weight_stride,
    ptrdiff_t bias_stride,
    bool bias_exist
) {
    layerNormKernel<BLOCK_SIZE, Tdata, Tcompute>(
        output,
        input_standardization,
        input_std_deviation,
        input,
        weight,
        bias,
        eps,
        normalized_size,
        output_strides,
        input_standardization_strides,
        input_std_deviation_strides,
        input_strides,
        weight_stride,
        bias_stride,
        bias_exist
    );
}
//  ----------------------- end: launchKernel: call kernel function of CUDA ------------------------

//  ----------------------------------- start: call launchKernel -----------------------------------
template<unsigned int BLOCK_SIZE, typename Tdata>
infiniStatus_t calculate_layer_norm(
    const LayerNormInfo &info,
    Tdata * output,
    Tdata * input_standardization,
    Tdata * input_std_deviation,
    const Tdata * input,
    const Tdata * weight,
    const Tdata * bias,
    cudaStream_t stream,
    void *workspace
) {
    size_t ndim = info.ndim;
    ptrdiff_t * input_strides_cuda = reinterpret_cast<ptrdiff_t*>(workspace);
    ptrdiff_t * output_strides_cuda = input_strides_cuda + ndim;
    ptrdiff_t * input_standardization_strides_cuda = output_strides_cuda + ndim;
    ptrdiff_t * input_std_deviation_strides_cuda = input_standardization_strides_cuda + ndim;

    CHECK_CUDA(cudaMemcpyAsync(input_strides_cuda, info.input_strides.data(), sizeof(ptrdiff_t) * ndim, cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(output_strides_cuda, info.output_strides.data(), sizeof(ptrdiff_t) * ndim, cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(input_standardization_strides_cuda, info.input_standardization_strides.data(), sizeof(ptrdiff_t) * (ndim - 1), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(input_std_deviation_strides_cuda, info.input_std_deviation_strides.data(), sizeof(ptrdiff_t) * (ndim - 1), cudaMemcpyHostToDevice, stream));

    launchKernel<1, Tdata, float><<<dim3(info.input_shape[0], info.input_shape[1]), 1, 0, stream>>>(
        output,
        input_standardization,
        input_std_deviation,
        input,
        weight,
        bias,
        info.eps,
        info.normalized_size,
        output_strides_cuda,
        input_standardization_strides_cuda,
        input_std_deviation_strides_cuda,
        input_strides_cuda,
        info.weight_strides[0],
        info.bias_exist ? info.bias_strides[0] : 0,
        info.bias_exist
    );
    return INFINI_STATUS_SUCCESS;
}
//  ------------------------------------ end: call launchKernel ------------------------------------


struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_standardization_desc,
    infiniopTensorDescriptor_t input_std_deviation_desc,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t weight_desc,
    infiniopTensorDescriptor_t bias_desc,
    float eps
) {
    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);
//  --------------------- start: check data type and calculate workspace size ----------------------
    auto dtype = output_desc->dtype();
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);
    size_t WorkSpaceSize = output_desc->ndim() * sizeof(size_t) * 5;
//  ---------------------- end: check data type and calculate workspace size -----------------------
    auto result = LayerNormInfo::createLayerNormInfo(
        output_desc,
        input_standardization_desc,
        input_std_deviation_desc,
        input_desc,
        weight_desc,
        bias_desc,
        eps
    );
    CHECK_RESULT(result);
    const LayerNormInfo &info = result.take();
    *desc_ptr = new Descriptor(
        dtype, std::move(info), WorkSpaceSize,
        new Opaque{handle->internal()},
        handle->device, handle->device_id
    );    
    return INFINI_STATUS_SUCCESS;
}


infiniStatus_t Descriptor::calculate(
    void * workspace,
    size_t workspace_size,
    void * output,
    void * input_standardization,
    void * input_std_deviation,
    const void * input,
    const void * weight,
    const void * bias,
    void *stream_
) const {
    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }
    cudaStream_t stream = (cudaStream_t)stream_;

    #define CALCULATE_LAYER_NORM(BLOCK_SIZE, TDATA) \
        calculate_layer_norm<BLOCK_SIZE, TDATA>(_info, (TDATA *)output, (TDATA *)input_standardization, (TDATA *)input_std_deviation, (const TDATA *)input, (const TDATA *)weight, (const TDATA *)bias, stream, workspace)
    #define CALCULATE_LAYER_NORM_WITH_BLOCK_SIZE(BLOCK_SIZE)          \
    {                                                                 \
        if (_info.dtype == INFINI_DTYPE_F16)                          \
            return CALCULATE_LAYER_NORM(BLOCK_SIZE, half);            \
        else if (_info.dtype == INFINI_DTYPE_F32)                     \
            return CALCULATE_LAYER_NORM(BLOCK_SIZE, float);           \
        else if (_info.dtype == INFINI_DTYPE_BF16)                    \
            return CALCULATE_LAYER_NORM(BLOCK_SIZE, __nv_bfloat16);   \
        else                                                          \
            return INFINI_STATUS_BAD_TENSOR_DTYPE;                    \
    }
    
    if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_1024)
        CALCULATE_LAYER_NORM_WITH_BLOCK_SIZE(CUDA_BLOCK_SIZE_1024)
    else if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_512)
        CALCULATE_LAYER_NORM_WITH_BLOCK_SIZE(CUDA_BLOCK_SIZE_512)
    else if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_4096)
        CALCULATE_LAYER_NORM_WITH_BLOCK_SIZE(CUDA_BLOCK_SIZE_4096)
    else
        return INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED;

    return INFINI_STATUS_SUCCESS;
}
} // namespace op::layer_norm::nvidia
