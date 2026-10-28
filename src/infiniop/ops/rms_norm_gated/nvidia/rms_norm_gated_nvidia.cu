#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "../../../tensor.h"
#include "rms_norm_gated_nvidia.cuh"
#include <cmath>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>
template <int W>
__global__ void gated_norm(__nv_bfloat16 *out, const __nv_bfloat16 *x, const __nv_bfloat16 *z, const __nv_bfloat16 *weight, int tokens, int heads, ptrdiff_t xs, ptrdiff_t zs, ptrdiff_t ys, float eps) {
    int row = blockIdx.x * W + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (row >= tokens * heads) {
        return;
    }
    int token = row / heads, head = row % heads;
    auto in = x + token * xs + head * 128;
    auto gate = z + token * zs + head * 128;
    float a[4], sum = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        a[i] = __bfloat162float(in[lane + i * 32]);
        sum += a[i] * a[i];
    }
    for (int d = 16; d; d /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, d, 32);
    }
    float inv = rsqrtf(__shfl_sync(0xffffffff, sum, 0, 32) / 128.f + eps);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        int j = lane + i * 32;
        float up = __bfloat162float(__float2bfloat16_rn(a[i] * __bfloat162float(weight[j]) * inv));
        float g = __bfloat162float(gate[j]);
        float sig = __bfloat162float(__float2bfloat16_rn(__frcp_rn(__fadd_rn(1.f, __expf(-g)))));
        out[token * ys + head * 128 + j] = __float2bfloat16_rn(__fmul_rn(__fmul_rn(g, sig), up));
    }
}

namespace op::rms_norm_gated::nvidia {
infiniStatus_t Descriptor::create(infiniopHandle_t handle, Descriptor **desc_ptr, infiniopTensorDescriptor_t out, infiniopTensorDescriptor_t x, infiniopTensorDescriptor_t gate, infiniopTensorDescriptor_t weight, float epsilon) {
    for (auto t : {out, x, gate, weight}) {
        CHECK_OR_RETURN(t != nullptr, INFINI_STATUS_NULL_POINTER);
        CHECK_OR_RETURN(t->dtype() == INFINI_DTYPE_BF16, INFINI_STATUS_BAD_TENSOR_DTYPE);
    }
    CHECK_OR_RETURN(x->ndim() == 2 && out->ndim() == 2 && gate->ndim() == 2, INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(x->shape() == out->shape() && x->shape() == gate->shape() && x->dim(0) > 0 && x->dim(1) > 0 && x->dim(1) % 128 == 0, INFINI_STATUS_BAD_TENSOR_SHAPE);
    for (auto t : {out, x, gate}) {
        CHECK_OR_RETURN(t->stride(1) == 1 && t->stride(0) >= static_cast<ptrdiff_t>(t->dim(1)), INFINI_STATUS_BAD_TENSOR_STRIDES);
    }
    CHECK_OR_RETURN(weight->ndim() == 1 && weight->dim(0) == 128, INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(weight->stride(0) == 1, INFINI_STATUS_BAD_TENSOR_STRIDES);
    CHECK_OR_RETURN(std::isfinite(epsilon) && epsilon > 0, INFINI_STATUS_BAD_PARAM);
    *desc_ptr = new Descriptor({x->dim(0), x->dim(1) / 128, x->stride(0), gate->stride(0), out->stride(0), epsilon}, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}
infiniStatus_t Descriptor::calculate(void *workspace, size_t workspace_size, void *out, const void *x, const void *gate, const void *weight, void *stream) const {
    (void)workspace;
    (void)workspace_size;
    gated_norm<8><<<(info_.tokens * info_.heads + 7) / 8, 256, 0, (cudaStream_t)stream>>>((__nv_bfloat16 *)out, (const __nv_bfloat16 *)x, (const __nv_bfloat16 *)gate, (const __nv_bfloat16 *)weight, info_.tokens, info_.heads, info_.xs, info_.zs, info_.ys, info_.epsilon);
    CHECK_CUDA(cudaGetLastError());
    return INFINI_STATUS_SUCCESS;
}
} // namespace op::rms_norm_gated::nvidia
