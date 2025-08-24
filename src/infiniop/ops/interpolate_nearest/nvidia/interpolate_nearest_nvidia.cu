#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "../cuda/kernel.cuh"
#include "interpolate_nearest_nvidia.cuh"
#include <cstddef>
#include <cstdint>
#include <cuda_bf16.h>

namespace op::interpolate_nearest::nvidia {

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;

    Opaque(std::shared_ptr<device::nvidia::Handle::Internal> internal_)
        : internal(internal_) {}
};

Descriptor::~Descriptor() { delete _opaque; }

infiniStatus_t Descriptor::create(infiniopHandle_t handle_,
                                  Descriptor **desc_ptr,
                                  infiniopTensorDescriptor_t output_desc,
                                  infiniopTensorDescriptor_t input_desc) {

    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);
    auto dtype = output_desc->dtype();

    // Check supported data types
    if (dtype != INFINI_DTYPE_F16 && dtype != INFINI_DTYPE_F32 && dtype != INFINI_DTYPE_BF16 && dtype != INFINI_DTYPE_I8) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    InterpolateNearestInfo info;
    CHECK_STATUS(InterpolateNearestInfo::create(&info, output_desc, input_desc));

    *desc_ptr = new Descriptor(dtype, info, 0, new Opaque{handle->internal()},
                               handle->device, handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(void *workspace, size_t workspace_size,
                                     void *output, const void *input,
                                     void *stream) const {

    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);

    size_t total_elements = calculate_total_elements(_info);

    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;

    switch (_dtype) {
    case INFINI_DTYPE_F32: {
        float *typed_output = reinterpret_cast<float *>(output);
        const float *typed_input = reinterpret_cast<const float *>(input);
        interpolate_nearest_kernel<float>
            <<<grid_size, block_size, 0, cuda_stream>>>(typed_output, typed_input,
                                                        _info);
    } break;

    case INFINI_DTYPE_F16: {
        half *typed_output = reinterpret_cast<half *>(output);
        const half *typed_input = reinterpret_cast<const half *>(input);
        interpolate_nearest_kernel<half><<<grid_size, block_size, 0, cuda_stream>>>(
            typed_output, typed_input, _info);
    } break;

    case INFINI_DTYPE_BF16: {
        auto typed_output = reinterpret_cast<__nv_bfloat16 *>(output);
        auto typed_input = reinterpret_cast<const __nv_bfloat16 *>(input);
        interpolate_nearest_kernel<__nv_bfloat16>
            <<<grid_size, block_size, 0, cuda_stream>>>(typed_output, typed_input,
                                                        _info);
    } break;

    case INFINI_DTYPE_I8: {
        auto typed_output = reinterpret_cast<int8_t *>(output);
        auto typed_input = reinterpret_cast<const int8_t *>(input);
        interpolate_nearest_kernel<int8_t>
            <<<grid_size, block_size, 0, cuda_stream>>>(typed_output, typed_input,
                                                        _info);
    } break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaStreamSynchronize(cuda_stream));
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::interpolate_nearest::nvidia
