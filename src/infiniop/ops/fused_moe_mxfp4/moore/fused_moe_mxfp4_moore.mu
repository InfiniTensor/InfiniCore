#include "../../../devices/moore/moore_common.h"
#include "../../../devices/moore/moore_kernel_common.h"
#include "fused_moe_mxfp4_moore.h"

#include "../../mxfp4_common/cuda/fused_moe_mxfp4_kernel.cuh"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace op::fused_moe_mxfp4::moore {
namespace {

template <typename T>
infiniStatus_t launch(
    const FusedMoeMxfp4Info &info,
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    const int32_t *selected_experts,
    const float *routing_weights,
    const uint8_t *w13_packed,
    const uint8_t *w13_scale,
    const uint8_t *w2_packed,
    const uint8_t *w2_scale,
    musaStream_t stream) {
    if (workspace_size < op::mxfp4_common::cuda::fusedMoeMxfp4WorkspaceSize(info)) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }
    if (workspace == nullptr) {
        return INFINI_STATUS_NULL_POINTER;
    }

    op::mxfp4_common::cuda::launchFusedMoeMxfp4(
        static_cast<T *>(output),
        static_cast<T *>(workspace),
        static_cast<const T *>(input),
        selected_experts, routing_weights, w13_packed, w13_scale, w2_packed, w2_scale,
        info, stream);
    CHECK_MOORE(musaGetLastError());
    return INFINI_STATUS_SUCCESS;
}

} // namespace

struct Descriptor::Opaque {
    std::shared_ptr<device::moore::Handle::Internal> internal;
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
    *desc_ptr = new Descriptor(
        new Opaque{reinterpret_cast<device::moore::Handle *>(handle)->internal()},
        value, op::mxfp4_common::cuda::fusedMoeMxfp4WorkspaceSize(value),
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
    if (output == nullptr || input == nullptr || selected_experts == nullptr
        || routing_weights == nullptr || w13_packed == nullptr || w13_scale == nullptr
        || w2_packed == nullptr || w2_scale == nullptr) {
        return INFINI_STATUS_NULL_POINTER;
    }

    auto musa_stream = reinterpret_cast<musaStream_t>(stream);
    const auto *ids = static_cast<const int32_t *>(selected_experts);
    const auto *weights = static_cast<const float *>(routing_weights);
    const auto *w13 = static_cast<const uint8_t *>(w13_packed);
    const auto *w13_s = static_cast<const uint8_t *>(w13_scale);
    const auto *w2 = static_cast<const uint8_t *>(w2_packed);
    const auto *w2_s = static_cast<const uint8_t *>(w2_scale);
    switch (_info.dtype) {
    case INFINI_DTYPE_F16:
        return launch<half>(_info, workspace, workspace_size, output, input,
                            ids, weights, w13, w13_s, w2, w2_s, musa_stream);
    case INFINI_DTYPE_BF16:
        return launch<cuda_bfloat16>(_info, workspace, workspace_size, output, input,
                                     ids, weights, w13, w13_s, w2, w2_s, musa_stream);
    case INFINI_DTYPE_F32:
        return launch<float>(_info, workspace, workspace_size, output, input,
                             ids, weights, w13, w13_s, w2, w2_s, musa_stream);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace op::fused_moe_mxfp4::moore
