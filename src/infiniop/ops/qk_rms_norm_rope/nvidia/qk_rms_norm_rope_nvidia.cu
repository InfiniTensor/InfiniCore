#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "../../../tensor.h"
#include "qk_rms_norm_rope_nvidia.cuh"
#include <cmath>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

template <int WARPS>
__global__ void fused_qk_norm_rope(
    __nv_bfloat16 *qout, __nv_bfloat16 *kout, const __nv_bfloat16 *qin, const __nv_bfloat16 *kin,
    const __nv_bfloat16 *qw, const __nv_bfloat16 *kw,
    const int64_t *pos, const __nv_bfloat16 *sine, const __nv_bfloat16 *cosine,
    op::qk_rms_norm_rope::nvidia::Info info) {
    const int lane = threadIdx.x % 32;
    const int row = blockIdx.x * WARPS + threadIdx.x / 32;
    if (row >= info.tokens * (info.q_heads + info.k_heads)) {
        return;
    }
    const int heads = info.q_heads + info.k_heads;
    const int token = row / heads, head = row % heads;
    const auto *x = head < info.q_heads ? qin + token * info.q_stride + head * 128 : kin + token * info.k_stride + (head - info.q_heads) * 128;
    const auto *w = head < info.q_heads ? qw : kw;
    auto *out = (head < info.q_heads ? qout + token * info.qo_stride + head * 128
                                     : kout + token * info.ko_stride + (head - info.q_heads) * 128);
    float a[2], b[2];
    float sum = 0;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int j = lane + i * 32;
        a[i] = __bfloat162float(x[j]);
        b[i] = __bfloat162float(x[j + 64]);
        sum += a[i] * a[i] + b[i] * b[i];
    }
    for (int delta = 16; delta; delta /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, delta, 32);
    }
    sum = __shfl_sync(0xffffffff, sum, 0, 32);
    float inv = rsqrtf(sum * (1.f / 128.f) + info.epsilon);
    const int64_t position = pos[token * info.pos_stride];
    if (position < 0 || position >= static_cast<int64_t>(info.max_positions)) {
        return;
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int j = lane + i * 32;
        float u = __bfloat162float(__float2bfloat16(a[i] * __bfloat162float(w[j]) * inv));
        float v = __bfloat162float(__float2bfloat16(b[i] * __bfloat162float(w[j + 64]) * inv));
        float sn = __bfloat162float(sine[position * info.sin_stride + j]);
        float cs = __bfloat162float(cosine[position * info.cos_stride + j]);
        float uc = __bfloat162float(__float2bfloat16(u * cs));
        float vs = __bfloat162float(__float2bfloat16(v * sn));
        float us = __bfloat162float(__float2bfloat16(u * sn));
        float vc = __bfloat162float(__float2bfloat16(v * cs));
        out[j] = __float2bfloat16(uc - vs);
        out[j + 64] = __float2bfloat16(us + vc);
    }
}

namespace op::qk_rms_norm_rope::nvidia {
infiniStatus_t Descriptor::create(infiniopHandle_t handle, Descriptor **desc_ptr, infiniopTensorDescriptor_t q_out,
                                  infiniopTensorDescriptor_t k_out,
                                  infiniopTensorDescriptor_t q,
                                  infiniopTensorDescriptor_t k,
                                  infiniopTensorDescriptor_t q_weight,
                                  infiniopTensorDescriptor_t k_weight,
                                  infiniopTensorDescriptor_t cos,
                                  infiniopTensorDescriptor_t sin,
                                  infiniopTensorDescriptor_t positions, float epsilon) {
    for (auto t : {q_out, k_out, q, k, q_weight, k_weight, cos, sin, positions}) {
        CHECK_OR_RETURN(t != nullptr, INFINI_STATUS_NULL_POINTER);
    }
    for (auto t : {q_out, k_out, q, k, q_weight, k_weight, cos, sin}) {
        CHECK_OR_RETURN(t->dtype() == INFINI_DTYPE_BF16, INFINI_STATUS_BAD_TENSOR_DTYPE);
    }
    CHECK_OR_RETURN(positions->dtype() == INFINI_DTYPE_I64, INFINI_STATUS_BAD_TENSOR_DTYPE);
    CHECK_OR_RETURN(q->ndim() == 2 && k->ndim() == 2 && q_out->ndim() == 2 && k_out->ndim() == 2, INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(q->dim(0) > 0 && q->dim(0) == k->dim(0) && q_out->shape() == q->shape() && k_out->shape() == k->shape(), INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(q->dim(1) > 0 && k->dim(1) > 0 && q->dim(1) % 128 == 0 && k->dim(1) % 128 == 0, INFINI_STATUS_BAD_TENSOR_SHAPE);
    for (auto t : {q, k, q_out, k_out}) {
        CHECK_OR_RETURN(t->stride(1) == 1 && t->stride(0) >= static_cast<ptrdiff_t>(t->dim(1)), INFINI_STATUS_BAD_TENSOR_STRIDES);
    }
    for (auto t : {q_weight, k_weight}) {
        CHECK_OR_RETURN(t->ndim() == 1 && t->dim(0) == 128, INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(t->stride(0) == 1, INFINI_STATUS_BAD_TENSOR_STRIDES);
    }
    CHECK_OR_RETURN(cos->ndim() == 2 && sin->ndim() == 2 && cos->shape() == sin->shape() && cos->dim(0) > 0 && cos->dim(1) == 64, INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(cos->stride(1) == 1 && sin->stride(1) == 1 && cos->stride(0) >= 64 && sin->stride(0) >= 64, INFINI_STATUS_BAD_TENSOR_STRIDES);
    CHECK_OR_RETURN(positions->ndim() == 1 && positions->dim(0) == q->dim(0), INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(positions->stride(0) > 0, INFINI_STATUS_BAD_TENSOR_STRIDES);
    CHECK_OR_RETURN(std::isfinite(epsilon) && epsilon > 0, INFINI_STATUS_BAD_PARAM);
    Info info{q->dim(0), q->dim(1) / 128, k->dim(1) / 128, cos->dim(0), q->stride(0), k->stride(0), q_out->stride(0), k_out->stride(0), cos->stride(0), sin->stride(0), positions->stride(0), epsilon};
    *desc_ptr = new Descriptor(info, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}
infiniStatus_t Descriptor::calculate(void *workspace, size_t workspace_size, void *q_out,
                                     void *k_out,
                                     const void *q,
                                     const void *k,
                                     const void *q_weight,
                                     const void *k_weight,
                                     const void *cos,
                                     const void *sin,
                                     const void *positions, void *stream) const {
    (void)workspace;
    (void)workspace_size;
    const size_t rows = info_.tokens * (info_.q_heads + info_.k_heads);
    fused_qk_norm_rope<8><<<(rows + 7) / 8, 256, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<__nv_bfloat16 *>(q_out), static_cast<__nv_bfloat16 *>(k_out),
        static_cast<const __nv_bfloat16 *>(q), static_cast<const __nv_bfloat16 *>(k),
        static_cast<const __nv_bfloat16 *>(q_weight), static_cast<const __nv_bfloat16 *>(k_weight),
        static_cast<const int64_t *>(positions), static_cast<const __nv_bfloat16 *>(sin), static_cast<const __nv_bfloat16 *>(cos), info_);
    CHECK_CUDA(cudaGetLastError());
    return INFINI_STATUS_SUCCESS;
}
} // namespace op::qk_rms_norm_rope::nvidia
