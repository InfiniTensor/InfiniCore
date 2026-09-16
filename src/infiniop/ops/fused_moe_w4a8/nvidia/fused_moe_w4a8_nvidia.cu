#include "fused_moe_w4a8_nvidia.cuh"

#include "../../../devices/nvidia/nvidia_handle.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "../../w4a8_common/cuda/w4a8_kernel.cuh"

namespace op::fused_moe_w4a8::nvidia {
namespace {

struct Workspace {
    int8_t *quantized_input;
    float *input_scale;
    void *activated;
    int8_t *quantized_activated;
    float *activated_scale;
};

Workspace splitWorkspace(void *workspace, const FusedMoeW4A8Info &info) {
    auto *base = reinterpret_cast<uint8_t *>(workspace);
    Workspace result{};
    result.quantized_input = reinterpret_cast<int8_t *>(base);
    base += info.quantizedInputBytes();
    result.input_scale = reinterpret_cast<float *>(base);
    base += info.inputScaleBytes();
    result.activated = base;
    base += info.activatedBytes();
    result.quantized_activated = reinterpret_cast<int8_t *>(base);
    base += info.quantizedActivatedBytes();
    result.activated_scale = reinterpret_cast<float *>(base);
    return result;
}

template <typename T>
void launch(T *output,
            const T *input,
            const int32_t *selected_experts,
            const float *routing_weights,
            const int8_t *w13_packed,
            const float *w13_scale,
            const int8_t *w2_packed,
            const float *w2_scale,
            void *workspace,
            const FusedMoeW4A8Info &info,
            cudaStream_t stream) {
    auto buffers = splitWorkspace(workspace, info);
    op::w4a8_common::cuda::launchFusedMoe(
        output, input, selected_experts, routing_weights,
        w13_packed, w13_scale, w2_packed, w2_scale,
        buffers.quantized_input, buffers.input_scale,
        reinterpret_cast<T *>(buffers.activated),
        buffers.quantized_activated, buffers.activated_scale,
        info.num_tokens, info.topk, info.num_experts,
        info.hidden_size, info.intermediate_size,
        info.activation, stream);
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
    infiniopTensorDescriptor_t selected_experts_desc,
    infiniopTensorDescriptor_t routing_weights_desc,
    infiniopTensorDescriptor_t w13_packed_desc,
    infiniopTensorDescriptor_t w13_scale_desc,
    infiniopTensorDescriptor_t w2_packed_desc,
    infiniopTensorDescriptor_t w2_scale_desc,
    infiniopFusedMoeActivation_t activation,
    bool weights_are_aiter_shuffled) {
    CHECK_OR_RETURN(!weights_are_aiter_shuffled, INFINI_STATUS_BAD_PARAM);
    auto info = FusedMoeW4A8Info::create(
        output_desc, input_desc, selected_experts_desc, routing_weights_desc,
        w13_packed_desc, w13_scale_desc, w2_packed_desc, w2_scale_desc,
        activation, weights_are_aiter_shuffled);
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
    const void *selected_experts,
    const void *routing_weights,
    const void *w13_packed,
    const void *w13_scale,
    const void *w2_packed,
    const void *w2_scale,
    void *stream) const {
    CHECK_OR_RETURN(workspace != nullptr && workspace_size >= workspaceSize(),
                    INFINI_STATUS_INSUFFICIENT_WORKSPACE);
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    const auto *ids = reinterpret_cast<const int32_t *>(selected_experts);
    const auto *route_weights = reinterpret_cast<const float *>(routing_weights);
    const auto *w13 = reinterpret_cast<const int8_t *>(w13_packed);
    const auto *w13_s = reinterpret_cast<const float *>(w13_scale);
    const auto *w2 = reinterpret_cast<const int8_t *>(w2_packed);
    const auto *w2_s = reinterpret_cast<const float *>(w2_scale);
    switch (_info.dtype) {
    case INFINI_DTYPE_F16:
        launch(reinterpret_cast<half *>(output),
               reinterpret_cast<const half *>(input), ids, route_weights,
               w13, w13_s, w2, w2_s, workspace, _info, cuda_stream);
        return INFINI_STATUS_SUCCESS;
    case INFINI_DTYPE_BF16:
        launch(reinterpret_cast<__nv_bfloat16 *>(output),
               reinterpret_cast<const __nv_bfloat16 *>(input), ids, route_weights,
               w13, w13_s, w2, w2_s, workspace, _info, cuda_stream);
        return INFINI_STATUS_SUCCESS;
    case INFINI_DTYPE_F32:
        launch(reinterpret_cast<float *>(output),
               reinterpret_cast<const float *>(input), ids, route_weights,
               w13, w13_s, w2, w2_s, workspace, _info, cuda_stream);
        return INFINI_STATUS_SUCCESS;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace op::fused_moe_w4a8::nvidia
