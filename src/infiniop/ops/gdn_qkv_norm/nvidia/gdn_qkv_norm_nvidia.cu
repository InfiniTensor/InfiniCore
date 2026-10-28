#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../tensor.h"
#include "gdn_qkv_norm_nvidia.cuh"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
namespace op::gdn_qkv_norm::nvidia {
namespace {
__global__ void gdnQKVNormKernel(__nv_bfloat16 *qo, __nv_bfloat16 *ko, __nv_bfloat16 *vo,
                                 const __nv_bfloat16 *q, const __nv_bfloat16 *k, const __nv_bfloat16 *v, Info info) {
    const size_t work = static_cast<size_t>(blockIdx.x) * 8 + threadIdx.x / 32;
    if (work >= info.tokens * 32) {
        return;
    }
    const size_t token = work / 32, head = work % 32;
    const int lane = threadIdx.x % 32;
    if (head < 16) {
        float qv[4], kv[4];
        float qs = 0.f, ks = 0.f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int d = lane + 32 * i;
            qv[i] = __bfloat162float(q[token * info.qs + head * info.qh + d]);
            kv[i] = __bfloat162float(k[token * info.ks + head * info.kh + d]);
            qs += qv[i] * qv[i];
            ks += kv[i] * kv[i];
        }
#pragma unroll
        for (int d = 16; d > 0; d /= 2) {
            qs += __shfl_down_sync(0xffffffff, qs, d, 32);
            ks += __shfl_down_sync(0xffffffff, ks, d, 32);
        }
        const float qn = rsqrtf(__shfl_sync(0xffffffff, qs, 0, 32) + 1e-6f);
        const float kn = rsqrtf(__shfl_sync(0xffffffff, ks, 0, 32) + 1e-6f);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const size_t out = (token * 16 + head) * 128 + lane + 32 * i;
            qo[out] = __float2bfloat16(qv[i] * qn);
            ko[out] = __float2bfloat16(kv[i] * kn);
        }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const size_t d = lane + 32 * i;
        vo[(token * 32 + head) * 128 + d] = v[token * info.vs + head * info.vh + d];
    }
}
} // namespace
infiniStatus_t Descriptor::create(infiniopHandle_t handle, Descriptor **desc_ptr,
                                  infiniopTensorDescriptor_t qo, infiniopTensorDescriptor_t ko,
                                  infiniopTensorDescriptor_t vo, infiniopTensorDescriptor_t q,
                                  infiniopTensorDescriptor_t k, infiniopTensorDescriptor_t v) {
    if (!handle || !desc_ptr || !qo || !ko || !vo || !q || !k || !v) {
        return INFINI_STATUS_NULL_POINTER;
    }
    if (q->ndim() != 4 || q->shape()[0] != 1 || q->shape()[1] == 0 || q->shape()[2] != 16 || q->shape()[3] != 128
        || k->shape() != q->shape() || v->shape() != std::vector<size_t>{1, q->shape()[1], 32, 128}
        || qo->shape() != q->shape() || ko->shape() != q->shape() || vo->shape() != v->shape()) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }
    for (auto *d : {qo, ko, vo, q, k, v}) {
        if (d->dtype() != INFINI_DTYPE_BF16) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        if (d->strides()[3] != 1) {
            return INFINI_STATUS_BAD_TENSOR_STRIDES;
        }
        for (auto s : d->strides()) {
            if (s <= 0) {
                return INFINI_STATUS_BAD_TENSOR_STRIDES;
            }
        }
    }
    for (auto *d : {qo, ko, vo}) {
        if (!d->isContiguous()) {
            return INFINI_STATUS_BAD_TENSOR_STRIDES;
        }
    }
    Info info{q->shape()[1], q->strides()[1], q->strides()[2], k->strides()[1], k->strides()[2], v->strides()[1], v->strides()[2]};
    *desc_ptr = new Descriptor(info, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}
infiniStatus_t Descriptor::calculate(void *workspace, size_t workspace_size, void *qo, void *ko, void *vo,
                                     const void *q, const void *k, const void *v, void *stream) const {
    (void)workspace;
    (void)workspace_size;
    gdnQKVNormKernel<<<(info_.tokens * 32 + 7) / 8, 256, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<__nv_bfloat16 *>(qo), static_cast<__nv_bfloat16 *>(ko), static_cast<__nv_bfloat16 *>(vo),
        static_cast<const __nv_bfloat16 *>(q), static_cast<const __nv_bfloat16 *>(k), static_cast<const __nv_bfloat16 *>(v), info_);
    return cudaGetLastError() == cudaSuccess ? INFINI_STATUS_SUCCESS : INFINI_STATUS_INTERNAL_ERROR;
}
} // namespace op::gdn_qkv_norm::nvidia
