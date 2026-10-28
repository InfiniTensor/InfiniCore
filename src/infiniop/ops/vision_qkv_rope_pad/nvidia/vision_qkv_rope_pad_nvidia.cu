#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../tensor.h"
#include "vision_qkv_rope_pad_nvidia.cuh"
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace op::vision_qkv_rope_pad::nvidia {
namespace {
// Fuse strided QKV extraction, 2-D NeoX rotation and zero head padding.
// The original attention scale remains 1/sqrt(72), not 1/sqrt(128).
__global__ void visionQKVRoPEPadKernel(__nv_bfloat16 *qo, __nv_bfloat16 *ko, __nv_bfloat16 *vo,
                                       const __nv_bfloat16 *qkv, const __nv_bfloat16 *cos,
                                       const __nv_bfloat16 *sin, const int64_t *rows,
                                       const int64_t *cols, Info info) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= info.tokens * 16 * 128) {
        return;
    }
    const size_t d = idx % 128;
    const size_t head = (idx / 128) % 16;
    const size_t token = idx / (128 * 16);
    const ptrdiff_t qoff = token * info.qos + head * info.qoh + d;
    const ptrdiff_t koff = token * info.kos + head * info.koh + d;
    const ptrdiff_t voff = token * info.vos + head * info.voh + d;
    if (d >= 72) {
        qo[qoff] = __float2bfloat16(0.f);
        ko[koff] = __float2bfloat16(0.f);
        vo[voff] = __float2bfloat16(0.f);
        return;
    }
    const ptrdiff_t base = token * info.qs + head * info.qh;
    const size_t pair = d % 36;
    const int64_t raw = pair < 18 ? rows[token * info.rs] : cols[token * info.ps];
    const size_t pos = raw >= 0 && static_cast<size_t>(raw) < info.max_positions ? static_cast<size_t>(raw) : 0;
    const float c = __bfloat162float(cos[pos * info.cs + pair]);
    const float s = __bfloat162float(sin[pos * info.ss + pair]);
    const float q0 = __bfloat162float(qkv[base + pair]);
    const float q1 = __bfloat162float(qkv[base + pair + 36]);
    const float k0 = __bfloat162float(qkv[base + info.qp + pair]);
    const float k1 = __bfloat162float(qkv[base + info.qp + pair + 36]);
    qo[qoff] = __float2bfloat16(d < 36 ? q0 * c - q1 * s : q1 * c + q0 * s);
    ko[koff] = __float2bfloat16(d < 36 ? k0 * c - k1 * s : k1 * c + k0 * s);
    vo[voff] = qkv[base + 2 * info.qp + d];
}
} // namespace

infiniStatus_t Descriptor::create(infiniopHandle_t handle, Descriptor **desc_ptr,
                                  infiniopTensorDescriptor_t q_out, infiniopTensorDescriptor_t k_out,
                                  infiniopTensorDescriptor_t v_out, infiniopTensorDescriptor_t qkv,
                                  infiniopTensorDescriptor_t cos, infiniopTensorDescriptor_t sin,
                                  infiniopTensorDescriptor_t rows, infiniopTensorDescriptor_t cols) {
    if (!handle || !desc_ptr || !q_out || !k_out || !v_out || !qkv || !cos || !sin || !rows || !cols) {
        return INFINI_STATUS_NULL_POINTER;
    }
    for (auto *d : {q_out, k_out, v_out, qkv, cos, sin}) {
        if (d->dtype() != INFINI_DTYPE_BF16) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        for (auto s : d->strides()) {
            if (s <= 0) {
                return INFINI_STATUS_BAD_TENSOR_STRIDES;
            }
        }
    }
    if (rows->dtype() != INFINI_DTYPE_I64 || cols->dtype() != INFINI_DTYPE_I64) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    if (qkv->ndim() != 4 || qkv->shape()[0] == 0 || qkv->shape()[1] != 3
        || qkv->shape()[2] != 16 || qkv->shape()[3] != 72) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }
    const size_t tokens = qkv->shape()[0];
    const std::vector<size_t> shape{tokens, 16, 128};
    for (auto *d : {q_out, k_out, v_out}) {
        if (d->shape() != shape) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        if (d->strides()[2] != 1) {
            return INFINI_STATUS_BAD_TENSOR_STRIDES;
        }
    }
    if (cos->ndim() != 2 || cos->shape()[0] == 0 || cos->shape()[1] < 36 || sin->shape() != cos->shape()
        || rows->shape() != std::vector<size_t>{tokens} || cols->shape() != rows->shape()) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }
    if (qkv->strides()[3] != 1 || cos->strides()[1] != 1 || sin->strides()[1] != 1
        || rows->strides()[0] <= 0 || cols->strides()[0] <= 0) {
        return INFINI_STATUS_BAD_TENSOR_STRIDES;
    }
    Info info{tokens, cos->shape()[0], qkv->strides()[0], qkv->strides()[1], qkv->strides()[2],
              cos->strides()[0], sin->strides()[0], rows->strides()[0], cols->strides()[0],
              q_out->strides()[0], q_out->strides()[1], k_out->strides()[0], k_out->strides()[1],
              v_out->strides()[0], v_out->strides()[1]};
    *desc_ptr = new Descriptor(info, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(void *workspace, size_t workspace_size,
                                     void *q_out, void *k_out, void *v_out, const void *qkv,
                                     const void *cos, const void *sin, const void *rows,
                                     const void *cols, void *stream) const {
    (void)workspace;
    (void)workspace_size;
    constexpr unsigned threads = 256;
    const size_t count = info_.tokens * 16 * 128;
    visionQKVRoPEPadKernel<<<(count + threads - 1) / threads, threads, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<__nv_bfloat16 *>(q_out), static_cast<__nv_bfloat16 *>(k_out), static_cast<__nv_bfloat16 *>(v_out),
        static_cast<const __nv_bfloat16 *>(qkv), static_cast<const __nv_bfloat16 *>(cos), static_cast<const __nv_bfloat16 *>(sin),
        static_cast<const int64_t *>(rows), static_cast<const int64_t *>(cols), info_);
    return cudaGetLastError() == cudaSuccess ? INFINI_STATUS_SUCCESS : INFINI_STATUS_INTERNAL_ERROR;
}
} // namespace op::vision_qkv_rope_pad::nvidia
