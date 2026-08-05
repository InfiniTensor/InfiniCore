#include "../../../devices/moore/moore_common.h"
#include "../../../devices/moore/moore_kernel_common.h"
#include "fused_moe_moore.h"

#include "../cuda/kernel.cuh"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace op::fused_moe::moore {

namespace {

constexpr size_t ALIGN_BYTES = 256;

size_t alignUp(size_t x, size_t align = ALIGN_BYTES) {
    return (x + align - 1) / align * align;
}

size_t dtypeSize(infiniDtype_t dtype) {
    switch (dtype) {
    case INFINI_DTYPE_F16:
    case INFINI_DTYPE_BF16:
        return 2;
    case INFINI_DTYPE_F32:
        return 4;
    default:
        return 0;
    }
}

size_t workspaceBytes(const FusedMoeInfo &info) {
    const size_t elem_size = dtypeSize(info.dtype);
    const size_t route_count = info.N * info.topk;
    size_t size = 0;
    size += alignUp(route_count * info.w1_cols * elem_size);
    size += alignUp(route_count * info.inter_size * elem_size);
    size += alignUp(info.N * info.hidden_size * sizeof(float));
    return size;
}

template <typename T>
infiniStatus_t launchFusedMoe(
    const FusedMoeInfo &info,
    void *workspace,
    size_t workspace_size,
    void *out,
    const void *input,
    const void *token_selected_experts,
    const void *token_final_scales,
    const void *w1,
    const void *w2,
    const void *b1,
    const void *b2,
    musaStream_t stream) {
    if (workspace_size < workspaceBytes(info)) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }
    if (workspace == nullptr && workspaceBytes(info) != 0) {
        return INFINI_STATUS_NULL_POINTER;
    }

    const size_t route_count = info.N * info.topk;
    const size_t out_count = info.N * info.hidden_size;
    const int threads = 256;

    auto *base = reinterpret_cast<std::byte *>(workspace);
    T *w1_out = reinterpret_cast<T *>(base);
    base += alignUp(route_count * info.w1_cols * sizeof(T));
    T *activated = reinterpret_cast<T *>(base);
    base += alignUp(route_count * info.inter_size * sizeof(T));
    float *out_accum = reinterpret_cast<float *>(base);

    CHECK_MOORE(musaMemsetAsync(out_accum, 0, out_count * sizeof(float), stream));

    size_t w1_total = route_count * info.w1_cols;
    int w1_blocks = static_cast<int>((w1_total + threads - 1) / threads);
    fusedMoeW1Kernel<T><<<w1_blocks, threads, 0, stream>>>(
        w1_out,
        static_cast<const T *>(input),
        static_cast<const int32_t *>(token_selected_experts),
        static_cast<const T *>(w1),
        static_cast<const T *>(b1),
        route_count,
        info.hidden_size,
        info.topk,
        info.w1_cols,
        info.num_experts);
    CHECK_MOORE(musaGetLastError());

    size_t act_total = route_count * info.inter_size;
    int act_blocks = static_cast<int>((act_total + threads - 1) / threads);
    fusedMoeActivationKernel<T><<<act_blocks, threads, 0, stream>>>(
        activated,
        w1_out,
        route_count,
        info.inter_size,
        info.w1_cols,
        static_cast<int>(info.activation));
    CHECK_MOORE(musaGetLastError());

    size_t w2_total = route_count * info.hidden_size;
    int w2_blocks = static_cast<int>((w2_total + threads - 1) / threads);
    fusedMoeW2ScatterKernel<T><<<w2_blocks, threads, 0, stream>>>(
        out_accum,
        activated,
        static_cast<const int32_t *>(token_selected_experts),
        static_cast<const float *>(token_final_scales),
        static_cast<const T *>(w2),
        static_cast<const T *>(b2),
        route_count,
        info.hidden_size,
        info.inter_size,
        info.topk,
        info.num_experts);
    CHECK_MOORE(musaGetLastError());

    int cast_blocks = static_cast<int>((out_count + threads - 1) / threads);
    fusedMoeCastKernel<T><<<cast_blocks, threads, 0, stream>>>(
        static_cast<T *>(out), out_accum, out_count);
    CHECK_MOORE(musaGetLastError());

    return INFINI_STATUS_SUCCESS;
}

} // namespace

struct Descriptor::Opaque {
    std::shared_ptr<device::moore::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t token_selected_experts_desc,
    infiniopTensorDescriptor_t token_final_scales_desc,
    infiniopTensorDescriptor_t w1_desc,
    infiniopTensorDescriptor_t w2_desc,
    infiniopTensorDescriptor_t b1_desc,
    infiniopTensorDescriptor_t b2_desc,
    infiniopFusedMoeActivation_t activation) {
    auto info = FusedMoeInfo::create(out_desc, input_desc, token_selected_experts_desc,
                                     token_final_scales_desc, w1_desc, w2_desc,
                                     b1_desc, b2_desc, activation);
    CHECK_RESULT(info);
    auto taken = info.take();
    *desc_ptr = new Descriptor(
        new Opaque{reinterpret_cast<device::moore::Handle *>(handle)->internal()},
        taken, workspaceBytes(taken), handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size,
    void *out,
    const void *input,
    const void *token_selected_experts,
    const void *token_final_scales,
    const void *w1,
    const void *w2,
    const void *b1,
    const void *b2,
    void *stream_) const {
    if (out == nullptr || input == nullptr || token_selected_experts == nullptr || token_final_scales == nullptr || w1 == nullptr || w2 == nullptr || (_info.has_b1 && b1 == nullptr) || (_info.has_b2 && b2 == nullptr)) {
        return INFINI_STATUS_NULL_POINTER;
    }

    musaStream_t stream = (musaStream_t)stream_;
    if (_info.dtype == INFINI_DTYPE_F16) {
        return launchFusedMoe<half>(_info, workspace, workspace_size, out, input,
                                    token_selected_experts, token_final_scales,
                                    w1, w2, b1, b2, stream);
    } else if (_info.dtype == INFINI_DTYPE_BF16) {
        return launchFusedMoe<cuda_bfloat16>(_info, workspace, workspace_size, out, input,
                                             token_selected_experts, token_final_scales,
                                             w1, w2, b1, b2, stream);
    } else if (_info.dtype == INFINI_DTYPE_F32) {
        return launchFusedMoe<float>(_info, workspace, workspace_size, out, input,
                                     token_selected_experts, token_final_scales,
                                     w1, w2, b1, b2, stream);
    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace op::fused_moe::moore
