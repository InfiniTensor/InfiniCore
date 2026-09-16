// SPDX-License-Identifier: MIT
// The optimized GEMM kernels are adapted from ROCm/AITER.

#include "fused_moe_w4a8_hygon.cuh"

#include "../../../devices/nvidia/nvidia_handle.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "../../w4a8_common/cuda/w4a8_kernel.cuh"
#include <cstdint>
#include <cstring>

namespace op::fused_moe_w4a8::hygon {
infiniStatus_t launchAiterFusedMoe(
    void *output,
    const void *input,
    const int32_t *selected_experts,
    const float *routing_weights,
    const int8_t *w13_packed,
    const float *w13_scale,
    const int8_t *w2_packed,
    const float *w2_scale,
    void *workspace,
    const FusedMoeW4A8Info &info,
    void *stream);

namespace {

struct Workspace {
    int8_t *quantized_input;
    float *input_scale;
    void *activated;
    int8_t *quantized_activated;
    float *activated_scale;
    void *gemm1_output;
    void *gemm2_output;
    int32_t *sorted_token_ids;
    int32_t *expert_ids;
    int32_t *counts;
    int32_t *offsets;
    int32_t *cursors;
    int32_t *num_tokens_post_padded;
};

Workspace splitPortableWorkspace(void *workspace, const FusedMoeW4A8Info &info) {
    auto *base = static_cast<uint8_t *>(workspace);
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
void launchPortable(
    T *output,
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
    auto buffers = splitPortableWorkspace(workspace, info);
    op::w4a8_common::cuda::launchFusedMoe(
        output, input, selected_experts, routing_weights,
        w13_packed, w13_scale, w2_packed, w2_scale,
        buffers.quantized_input, buffers.input_scale,
        reinterpret_cast<T *>(buffers.activated),
        buffers.quantized_activated, buffers.activated_scale,
        info.num_tokens, info.topk, info.num_experts,
        info.hidden_size, info.intermediate_size, info.activation, stream);
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
    auto info = FusedMoeW4A8Info::create(
        output_desc, input_desc, selected_experts_desc, routing_weights_desc,
        w13_packed_desc, w13_scale_desc, w2_packed_desc, w2_scale_desc,
        activation, weights_are_aiter_shuffled);
    CHECK_RESULT(info);
    if (weights_are_aiter_shuffled) {
        CHECK_OR_RETURN(input_desc->dtype() == INFINI_DTYPE_F16
                            || input_desc->dtype() == INFINI_DTYPE_BF16,
                        INFINI_STATUS_BAD_TENSOR_DTYPE);
    }
    auto hygon_handle = reinterpret_cast<device::nvidia::Handle *>(handle);
    *desc_ptr = new Descriptor(
        new Opaque{hygon_handle->internal()}, info.take(),
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
    void *stream_) const {
    CHECK_OR_RETURN(workspace != nullptr && workspace_size >= workspaceSize(),
                    INFINI_STATUS_INSUFFICIENT_WORKSPACE);
    auto stream = reinterpret_cast<cudaStream_t>(stream_);
    const auto *ids = static_cast<const int32_t *>(selected_experts);
    const auto *route_weights = static_cast<const float *>(routing_weights);
    const auto *w13 = static_cast<const int8_t *>(w13_packed);
    const auto *w13_s = static_cast<const float *>(w13_scale);
    const auto *w2 = static_cast<const int8_t *>(w2_packed);
    const auto *w2_s = static_cast<const float *>(w2_scale);

    if (_info.dtype == INFINI_DTYPE_F16) {
        if (_info.weights_are_aiter_shuffled) {
            return launchAiterFusedMoe(
                output, input, ids, route_weights, w13, w13_s, w2, w2_s,
                workspace, _info, stream_);
        }
        launchPortable(
            static_cast<half *>(output), static_cast<const half *>(input),
            ids, route_weights, w13, w13_s, w2, w2_s,
            workspace, _info, stream);
        return INFINI_STATUS_SUCCESS;
    }
    if (_info.dtype == INFINI_DTYPE_BF16) {
        if (_info.weights_are_aiter_shuffled) {
            return launchAiterFusedMoe(
                output, input, ids, route_weights, w13, w13_s, w2, w2_s,
                workspace, _info, stream_);
        }
        launchPortable(
            static_cast<__nv_bfloat16 *>(output),
            static_cast<const __nv_bfloat16 *>(input), ids, route_weights,
            w13, w13_s, w2, w2_s, workspace, _info, stream);
        return INFINI_STATUS_SUCCESS;
    }
    if (_info.dtype == INFINI_DTYPE_F32
        && !_info.weights_are_aiter_shuffled) {
        launchPortable(
            static_cast<float *>(output), static_cast<const float *>(input),
            ids, route_weights, w13, w13_s, w2, w2_s,
            workspace, _info, stream);
        return INFINI_STATUS_SUCCESS;
    }
    return INFINI_STATUS_BAD_TENSOR_DTYPE;
}

} // namespace op::fused_moe_w4a8::hygon
