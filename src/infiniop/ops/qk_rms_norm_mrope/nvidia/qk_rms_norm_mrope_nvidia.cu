#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "../../../tensor.h"
#include "qk_rms_norm_mrope_nvidia.cuh"
#include <cmath>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

template <int WARPS>
__global__ void fused_qk_norm_mrope(
    __nv_bfloat16 *qout, __nv_bfloat16 *kout, const __nv_bfloat16 *qin, const __nv_bfloat16 *kin,
    const __nv_bfloat16 *qw, const __nv_bfloat16 *kw,
    const int64_t *pos, const __nv_bfloat16 *sine, const __nv_bfloat16 *cosine,
    op::qk_rms_norm_mrope::nvidia::Info info) {
    const int lane = threadIdx.x % 32;
    const int row = blockIdx.x * WARPS + threadIdx.x / 32;
    const int heads = info.q_heads + info.k_heads;
    if (row >= info.tokens * heads) {
        return;
    }
    const int token = row / heads, head = row % heads;
    const bool isq = head < info.q_heads;
    const auto *x = isq ? qin + token * info.qs + head * info.qh
                        : kin + token * info.ks + (head - info.q_heads) * info.kh;
    const auto *w = isq ? qw : kw;
    auto *out = isq ? qout + token * info.qos + head * info.qoh
                    : kout + token * info.kos + (head - info.q_heads) * info.koh;
    float a[8], sum = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        a[i] = __bfloat162float(x[lane + i * 32]);
        sum += a[i] * a[i];
    }
    for (int d = 16; d; d /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, d, 32);
    }
    const float inv = rsqrtf(__shfl_sync(0xffffffff, sum, 0, 32) * (1.f / 256.f) + info.epsilon);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        a[i] = __bfloat162float(__float2bfloat16_rn(a[i] * __bfloat162float(w[lane + i * 32]) * inv));
    }
    int axis;
    if (info.interleaved) {
        axis = lane % 3 == 1 && lane < info.section_h * 3 ? 1
                                                          : (lane % 3 == 2 && lane < info.section_w * 3 ? 2 : 0);
    } else {
        axis = lane < info.section_t ? 0 : (lane < info.section_t + info.section_h ? 1 : 2);
    }
    const int64_t raw_pos = pos[axis * info.pa + token * info.pt];
    const int64_t p = raw_pos >= 0 && raw_pos < static_cast<int64_t>(info.max_positions) ? raw_pos : 0;
    const float cs = __bfloat162float(cosine[p * info.cs + lane]);
    const float sn = __bfloat162float(sine[p * info.ss + lane]);
    out[lane] = __float2bfloat16_rn(a[0] * cs - a[1] * sn);
    out[lane + 32] = __float2bfloat16_rn(a[1] * cs + a[0] * sn);
#pragma unroll
    for (int i = 2; i < 8; ++i) {
        out[lane + i * 32] = __float2bfloat16_rn(a[i]);
    }
}

namespace op::qk_rms_norm_mrope::nvidia {
infiniStatus_t Descriptor::create(infiniopHandle_t handle, Descriptor **desc_ptr,
                                  infiniopTensorDescriptor_t q_out, infiniopTensorDescriptor_t k_out,
                                  infiniopTensorDescriptor_t q, infiniopTensorDescriptor_t k,
                                  infiniopTensorDescriptor_t q_weight, infiniopTensorDescriptor_t k_weight,
                                  infiniopTensorDescriptor_t cos, infiniopTensorDescriptor_t sin,
                                  infiniopTensorDescriptor_t positions, float epsilon,
                                  int section_t, int section_h, int section_w, bool interleaved) {
    for (auto t : {q_out, k_out, q, k, q_weight, k_weight, cos, sin, positions}) {
        CHECK_OR_RETURN(t != nullptr, INFINI_STATUS_NULL_POINTER);
    }
    for (auto t : {q_out, k_out, q, k, q_weight, k_weight, cos, sin}) {
        CHECK_OR_RETURN(t->dtype() == INFINI_DTYPE_BF16, INFINI_STATUS_BAD_TENSOR_DTYPE);
    }
    CHECK_OR_RETURN(positions->dtype() == INFINI_DTYPE_I64, INFINI_STATUS_BAD_TENSOR_DTYPE);
    CHECK_OR_RETURN(q->ndim() == 3 && k->ndim() == 3 && q_out->ndim() == 3 && k_out->ndim() == 3, INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(q->dim(0) > 0 && q->dim(1) > 0 && k->dim(1) > 0 && q->dim(0) == k->dim(0)
                        && q->dim(2) == 256 && k->dim(2) == 256 && q_out->shape() == q->shape() && k_out->shape() == k->shape(),
                    INFINI_STATUS_BAD_TENSOR_SHAPE);
    for (auto t : {q, k, q_out, k_out}) {
        CHECK_OR_RETURN(t->stride(2) == 1 && t->stride(1) >= 256 && t->stride(0) >= t->stride(1) * static_cast<ptrdiff_t>(t->dim(1)), INFINI_STATUS_BAD_TENSOR_STRIDES);
    }
    for (auto t : {q_weight, k_weight}) {
        CHECK_OR_RETURN(t->ndim() == 1 && t->dim(0) == 256, INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(t->stride(0) == 1, INFINI_STATUS_BAD_TENSOR_STRIDES);
    }
    CHECK_OR_RETURN(cos->ndim() == 2 && sin->ndim() == 2 && cos->shape() == sin->shape() && cos->dim(0) > 0 && cos->dim(1) == 32, INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(cos->stride(1) == 1 && sin->stride(1) == 1 && cos->stride(0) >= 32 && sin->stride(0) >= 32, INFINI_STATUS_BAD_TENSOR_STRIDES);
    const bool axes = positions->ndim() == 2;
    CHECK_OR_RETURN((axes && positions->dim(0) == 3 && positions->dim(1) == q->dim(0))
                        || (positions->ndim() == 1 && positions->dim(0) == q->dim(0)),
                    INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(positions->stride(axes ? 1 : 0) > 0 && (!axes || positions->stride(0) >= positions->stride(1) * static_cast<ptrdiff_t>(q->dim(0))), INFINI_STATUS_BAD_TENSOR_STRIDES);
    CHECK_OR_RETURN(std::isfinite(epsilon) && epsilon > 0 && section_t > 0 && section_h > 0 && section_w > 0 && section_t + section_h + section_w == 32, INFINI_STATUS_BAD_PARAM);
    Info info{q->dim(0), q->dim(1), k->dim(1), cos->dim(0),
              q->stride(0), q->stride(1), k->stride(0), k->stride(1), q_out->stride(0), q_out->stride(1), k_out->stride(0), k_out->stride(1),
              cos->stride(0), sin->stride(0), axes ? positions->stride(0) : 0, positions->stride(axes ? 1 : 0),
              epsilon, section_t, section_h, section_w, interleaved};
    *desc_ptr = new Descriptor(info, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}
infiniStatus_t Descriptor::calculate(void *workspace, size_t workspace_size, void *q_out, void *k_out,
                                     const void *q, const void *k, const void *q_weight, const void *k_weight,
                                     const void *cos, const void *sin, const void *positions, void *stream) const {
    (void)workspace;
    (void)workspace_size;
    const size_t rows = info_.tokens * (info_.q_heads + info_.k_heads);
    fused_qk_norm_mrope<8><<<(rows + 7) / 8, 256, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<__nv_bfloat16 *>(q_out), static_cast<__nv_bfloat16 *>(k_out),
        static_cast<const __nv_bfloat16 *>(q), static_cast<const __nv_bfloat16 *>(k),
        static_cast<const __nv_bfloat16 *>(q_weight), static_cast<const __nv_bfloat16 *>(k_weight),
        static_cast<const int64_t *>(positions), static_cast<const __nv_bfloat16 *>(sin), static_cast<const __nv_bfloat16 *>(cos), info_);
    CHECK_CUDA(cudaGetLastError());
    return INFINI_STATUS_SUCCESS;
}
} // namespace op::qk_rms_norm_mrope::nvidia
