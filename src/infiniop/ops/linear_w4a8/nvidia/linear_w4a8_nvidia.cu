#include "linear_w4a8_nvidia.cuh"

#include "../../../devices/nvidia/nvidia_handle.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "../../w4a8_common/cuda/w4a8_kernel.cuh"

namespace op::linear_w4a8::nvidia {
namespace {

template <typename T>
void launch(T *output,
            const T *input,
            const int8_t *packed_weight,
            const float *weight_scale,
            const T *bias,
            void *workspace,
            const LinearW4A8Info &info,
            cudaStream_t stream) {
    auto *quantized_input = reinterpret_cast<int8_t *>(workspace);
    auto *input_scale = reinterpret_cast<float *>(
        reinterpret_cast<uint8_t *>(workspace) + info.quantizedInputBytes());
    op::w4a8_common::cuda::launchLinear(
        output, input, packed_weight, weight_scale, bias,
        quantized_input, input_scale,
        info.M, info.N, info.K, info.alpha, stream);
}

} // namespace

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

Descriptor::~Descriptor() { delete _opaque; }

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t packed_weight_desc,
    infiniopTensorDescriptor_t weight_scale_desc,
    infiniopTensorDescriptor_t bias_desc,
    float alpha) {
    auto info = LinearW4A8Info::create(
        output_desc, input_desc, packed_weight_desc, weight_scale_desc,
        bias_desc, alpha);
    CHECK_RESULT(info);
    auto nvidia_handle = reinterpret_cast<device::nvidia::Handle *>(handle);
    *desc_ptr = new Descriptor(
        new Opaque{nvidia_handle->internal()}, info.take(),
        handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    const void *packed_weight,
    const void *weight_scale,
    const void *bias,
    void *stream) const {
    CHECK_OR_RETURN(workspace != nullptr && workspace_size >= workspaceSize(),
                    INFINI_STATUS_INSUFFICIENT_WORKSPACE);
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    const auto *packed_ptr = reinterpret_cast<const int8_t *>(packed_weight);
    const auto *scale_ptr = reinterpret_cast<const float *>(weight_scale);
    switch (_info.dtype) {
    case INFINI_DTYPE_F16:
        launch(reinterpret_cast<half *>(output),
               reinterpret_cast<const half *>(input), packed_ptr, scale_ptr,
               reinterpret_cast<const half *>(bias), workspace, _info, cuda_stream);
        return INFINI_STATUS_SUCCESS;
    case INFINI_DTYPE_BF16:
        launch(reinterpret_cast<__nv_bfloat16 *>(output),
               reinterpret_cast<const __nv_bfloat16 *>(input), packed_ptr, scale_ptr,
               reinterpret_cast<const __nv_bfloat16 *>(bias), workspace, _info, cuda_stream);
        return INFINI_STATUS_SUCCESS;
    case INFINI_DTYPE_F32:
        launch(reinterpret_cast<float *>(output),
               reinterpret_cast<const float *>(input), packed_ptr, scale_ptr,
               reinterpret_cast<const float *>(bias), workspace, _info, cuda_stream);
        return INFINI_STATUS_SUCCESS;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace op::linear_w4a8::nvidia
