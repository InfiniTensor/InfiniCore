#include "timestep_embedding_nvidia.cuh"

#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "../../../tensor.h"

#include <cmath>
#include <cuda_runtime.h>

namespace {

template <typename T>
INFINIOP_CUDA_KERNEL timestepEmbeddingKernel(
    float *__restrict__ output,
    const T *__restrict__ timestep,
    size_t num_timesteps,
    size_t embedding_dim,
    float log_max_period) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t numel = num_timesteps * embedding_dim;
    if (index >= numel) {
        return;
    }

    const size_t half_dim = embedding_dim / 2;
    const size_t timestep_index = index / embedding_dim;
    const size_t output_dim = index % embedding_dim;
    const size_t frequency_index = output_dim % half_dim;
    const float frequency = expf(
        -log_max_period * static_cast<float>(frequency_index)
        / static_cast<float>(half_dim));
    const float angle = static_cast<float>(timestep[timestep_index]) * frequency;
    output[index] = output_dim < half_dim ? cosf(angle) : sinf(angle);
}

} // namespace

namespace op::timestep_embedding::nvidia {

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t timestep_desc) {
    CHECK_OR_RETURN(timestep_desc->shape().size() == 1,
                    INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(output_desc->shape().size() == 2,
                    INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(output_desc->shape()[0] == timestep_desc->shape()[0]
                        && output_desc->shape()[1] > 0
                        && output_desc->shape()[1] % 2 == 0,
                    INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(output_desc->dtype() == INFINI_DTYPE_F32,
                    INFINI_STATUS_BAD_TENSOR_DTYPE);
    CHECK_DTYPE(timestep_desc->dtype(),
                INFINI_DTYPE_F16, INFINI_DTYPE_BF16, INFINI_DTYPE_F32);
    CHECK_OR_RETURN(output_desc->isContiguous() && timestep_desc->isContiguous(),
                    INFINI_STATUS_BAD_TENSOR_STRIDES);

    *desc_ptr = new Descriptor(
        timestep_desc->shape()[0],
        output_desc->shape()[1],
        timestep_desc->dtype(),
        new Opaque{reinterpret_cast<device::nvidia::Handle *>(handle)->internal()},
        handle->device,
        handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *output,
    const void *timestep,
    float max_period,
    void *stream) const {
    if (max_period <= 0.0f) {
        return INFINI_STATUS_BAD_PARAM;
    }
    const size_t numel = _num_timesteps * _embedding_dim;
    if (numel == 0) {
        return INFINI_STATUS_SUCCESS;
    }

    constexpr size_t block_size = 256;
    const size_t grid_size = (numel + block_size - 1) / block_size;
    const float log_max_period = std::log(max_period);
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);

    switch (_input_dtype) {
    case INFINI_DTYPE_F16:
        timestepEmbeddingKernel<half><<<grid_size, block_size, 0, cuda_stream>>>(
            reinterpret_cast<float *>(output),
            reinterpret_cast<const half *>(timestep),
            _num_timesteps,
            _embedding_dim,
            log_max_period);
        break;
    case INFINI_DTYPE_BF16:
        timestepEmbeddingKernel<cuda_bfloat16><<<grid_size, block_size, 0, cuda_stream>>>(
            reinterpret_cast<float *>(output),
            reinterpret_cast<const cuda_bfloat16 *>(timestep),
            _num_timesteps,
            _embedding_dim,
            log_max_period);
        break;
    case INFINI_DTYPE_F32:
        timestepEmbeddingKernel<float><<<grid_size, block_size, 0, cuda_stream>>>(
            reinterpret_cast<float *>(output),
            reinterpret_cast<const float *>(timestep),
            _num_timesteps,
            _embedding_dim,
            log_max_period);
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return cudaGetLastError() == cudaSuccess
             ? INFINI_STATUS_SUCCESS
             : INFINI_STATUS_INTERNAL_ERROR;
}

} // namespace op::timestep_embedding::nvidia
