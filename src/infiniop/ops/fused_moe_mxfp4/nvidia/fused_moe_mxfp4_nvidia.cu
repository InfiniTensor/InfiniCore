#include "fused_moe_mxfp4_nvidia.cuh"

#include "../../../devices/nvidia/nvidia_handle.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "../../mxfp4_common/cuda/fused_moe_mxfp4_kernel.cuh"

namespace op::fused_moe_mxfp4::nvidia {

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

Descriptor::~Descriptor() { delete _opaque; }

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t selected_experts_desc,
    infiniopTensorDescriptor_t routing_weights_desc,
    infiniopTensorDescriptor_t w13_packed_desc,
    infiniopTensorDescriptor_t w13_scale_desc,
    infiniopTensorDescriptor_t w2_packed_desc,
    infiniopTensorDescriptor_t w2_scale_desc,
    infiniopFusedMoeActivation_t activation) {
    auto info = FusedMoeMxfp4Info::create(
        output_desc, input_desc, selected_experts_desc, routing_weights_desc,
        w13_packed_desc, w13_scale_desc, w2_packed_desc, w2_scale_desc, activation);
    CHECK_RESULT(info);
    auto value = info.take();
    const size_t workspace_size = op::mxfp4_common::cuda::fusedMoeMxfp4WorkspaceSize(value);
    auto nvidia_handle = reinterpret_cast<device::nvidia::Handle *>(handle);
    *desc_ptr = new Descriptor(
        new Opaque{nvidia_handle->internal()}, value, workspace_size,
        handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    const void *selected_experts,
    const void *routing_weights,
    const void *w13_packed,
    const void *w13_scale,
    const void *w2_packed,
    const void *w2_scale,
    void *stream) const {
    CHECK_OR_RETURN(workspace != nullptr && workspace_size >= _workspace_size,
                    INFINI_STATUS_INSUFFICIENT_WORKSPACE);
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    const auto *ids = reinterpret_cast<const int32_t *>(selected_experts);
    const auto *weights = reinterpret_cast<const float *>(routing_weights);
    const auto *w13 = reinterpret_cast<const uint8_t *>(w13_packed);
    const auto *w13_s = reinterpret_cast<const uint8_t *>(w13_scale);
    const auto *w2 = reinterpret_cast<const uint8_t *>(w2_packed);
    const auto *w2_s = reinterpret_cast<const uint8_t *>(w2_scale);
    switch (_info.dtype) {
    case INFINI_DTYPE_F16:
        op::mxfp4_common::cuda::launchFusedMoeMxfp4(
            reinterpret_cast<half *>(output),
            reinterpret_cast<half *>(workspace),
            reinterpret_cast<const half *>(input),
            ids, weights, w13, w13_s, w2, w2_s, _info, cuda_stream);
        return INFINI_STATUS_SUCCESS;
    case INFINI_DTYPE_BF16:
        op::mxfp4_common::cuda::launchFusedMoeMxfp4(
            reinterpret_cast<__nv_bfloat16 *>(output),
            reinterpret_cast<__nv_bfloat16 *>(workspace),
            reinterpret_cast<const __nv_bfloat16 *>(input),
            ids, weights, w13, w13_s, w2, w2_s, _info, cuda_stream);
        return INFINI_STATUS_SUCCESS;
    case INFINI_DTYPE_F32:
        op::mxfp4_common::cuda::launchFusedMoeMxfp4(
            reinterpret_cast<float *>(output),
            reinterpret_cast<float *>(workspace),
            reinterpret_cast<const float *>(input),
            ids, weights, w13, w13_s, w2, w2_s, _info, cuda_stream);
        return INFINI_STATUS_SUCCESS;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace op::fused_moe_mxfp4::nvidia
