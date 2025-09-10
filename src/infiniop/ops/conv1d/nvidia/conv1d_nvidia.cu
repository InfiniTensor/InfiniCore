#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_handle.cuh"
#include "conv1d_nvidia.cuh"
#include "infiniop/ops/conv1d.h"
#include "../../../tensor.h"

#include <cuda_fp16.h>
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#include <cuda_bf16.h>
#endif
#endif

#define DESTROY_CUDNN_DESCRIPTOR(ptr, destroy) \
    do { if (ptr) { destroy(ptr); ptr = nullptr; } } while (0)

namespace op::conv1d::nvidia {

template <typename T, typename AccT = float>
__global__ void conv1d_manual_kernel(
    T * __restrict__ y,           // [B, C, L]
    const T * __restrict__ x,     // [B, C, L_padded]
    const T * __restrict__ w,     // [C, 1, K]
    size_t B, size_t C, size_t L, size_t L_padded, size_t K)
{
    size_t b = blockIdx.x;
    size_t c = blockIdx.y;
    size_t l = threadIdx.x;

    if (b >= B || c >= C || l >= L) return;

    AccT acc = (AccT)0;
    for (size_t k = 0; k < K; ++k) {
        acc += (AccT)x[b * C * L_padded + c * L_padded + l + k] * (AccT)w[c * K + k];
    }

    y[b * C * L + c * L + l] = (T)acc;
}

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
    size_t cudnn_workspace_size = 0;

#ifdef ENABLE_CUDNN_API
    cudnnTensorDescriptor_t x_desc = nullptr;
    cudnnTensorDescriptor_t y_desc = nullptr;
    cudnnFilterDescriptor_t w_desc = nullptr;
    cudnnConvolutionDescriptor_t conv_desc = nullptr;
    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
#endif

    Opaque(std::shared_ptr<device::nvidia::Handle::Internal> p) : internal(std::move(p)) {}

    ~Opaque() {
#ifdef ENABLE_CUDNN_API
        DESTROY_CUDNN_DESCRIPTOR(x_desc, cudnnDestroyTensorDescriptor);
        DESTROY_CUDNN_DESCRIPTOR(y_desc, cudnnDestroyTensorDescriptor);
        DESTROY_CUDNN_DESCRIPTOR(w_desc, cudnnDestroyFilterDescriptor);
        DESTROY_CUDNN_DESCRIPTOR(conv_desc, cudnnDestroyConvolutionDescriptor);
#endif
    }

#ifdef ENABLE_CUDNN_API
    // L_in is the padded length, L_out is the original length
    infiniStatus_t init(size_t B, size_t C, size_t L_in, size_t L_out, size_t K, infiniDtype_t dtype) {
        return INFINI_STATUS_SUCCESS;
    }
#endif
};

Descriptor::~Descriptor() { if (_opaque) delete _opaque; }

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t w_desc,
    size_t K)
{
#ifndef ENABLE_CUDNN_API
    (void)handle_; (void)desc_ptr; (void)y_desc; (void)x_desc; (void)w_desc; (void)K;
    return INFINI_STATUS_NOT_IMPLEMENTED;
#else
    auto h = reinterpret_cast<device::nvidia::Handle*>(handle_);
    auto dtype = y_desc->dtype();
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);

    if (x_desc->ndim() < 4 || y_desc->ndim() < 4) return INFINI_STATUS_BAD_TENSOR_SHAPE;
    size_t B = x_desc->shape()[0];
    size_t C = x_desc->shape()[1];
    size_t L_in = x_desc->shape()[3]; // Padded length from x_desc
    size_t L_out = y_desc->shape()[3]; // Original length from y_desc

    if (y_desc->shape()[0] != B || y_desc->shape()[1] != C)
        return INFINI_STATUS_BAD_TENSOR_SHAPE;

    if (!(w_desc->ndim() == 3 || w_desc->ndim() == 4)) return INFINI_STATUS_BAD_TENSOR_SHAPE;
    if (w_desc->ndim() == 3) {
        if (w_desc->shape()[0] != C || w_desc->shape()[1] != 1 || w_desc->shape()[2] != K)
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
    } else { // 4D
        if (w_desc->shape()[0] != C || w_desc->shape()[1] != 1 || w_desc->shape()[2] != 1 || w_desc->shape()[3] != K)
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    auto opaque = new Opaque(h->internal());
    auto status = opaque->init(B, C, L_in, L_out, K, dtype);
    if (status != INFINI_STATUS_SUCCESS) { delete opaque; return status; }

    *desc_ptr = new Descriptor(dtype, B, C, L_out, K, L_in, opaque->cudnn_workspace_size, opaque, h->device, h->device_id);
    return INFINI_STATUS_SUCCESS;
#endif
}

infiniStatus_t Descriptor::fn(
    void *workspace, size_t workspace_size,
    void *y, const void *x, const void *w,
    void *stream) const
{
#ifndef ENABLE_CUDNN_API
    (void)workspace; (void)workspace_size; (void)y; (void)x; (void)w; (void)stream;
    return INFINI_STATUS_NOT_IMPLEMENTED;
#else
    (void)workspace_size; // validated by caller
    auto st = reinterpret_cast<cudaStream_t>(stream);

    size_t B = _B, C = _C, L = _L, K = _K;
    size_t L_padded = _L_padded;
    dim3 blk(L, 1, 1), grd(B, C, 1);

    if (this->dtype() == INFINI_DTYPE_F32) {
        conv1d_manual_kernel<float><<<grd, blk, 0, st>>>(
            (float *)y, (const float *)x, (const float *)w, B, C, L, L_padded, K);
    } else if (this->dtype() == INFINI_DTYPE_F16) {
        conv1d_manual_kernel<__half><<<grd, blk, 0, st>>>(
            (__half *)y, (const __half *)x, (const __half *)w, B, C, L, L_padded, K);
    } else if (this->dtype() == INFINI_DTYPE_BF16) {
    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
        conv1d_manual_kernel<__nv_bfloat16><<<grd, blk, 0, st>>>(
            (__nv_bfloat16 *)y, (const __nv_bfloat16 *)x, (const __nv_bfloat16 *)w, B, C, L, L_padded, K);
    #else
        return INFINI_STATUS_NOT_IMPLEMENTED;
    #endif
    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
#endif
}

template <typename T, typename AccT = float>
__global__ void conv1d_update_kernel(
    T * __restrict__ y,           // [B*C]
    const T * __restrict__ x_now, // [B*C]
    const T * __restrict__ w,     // [C*K]
    T * __restrict__ state,       // [B*C*(K-1)]
    size_t B, size_t C, size_t K)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * C) return;
    size_t b = idx / C;
    size_t c = idx % C;

    const T *wc = w + c * K;
    T *sc = state + (b * C + c) * (K - 1);
    T x = x_now[b * C + c];

    AccT acc = (AccT)0;
    #pragma unroll
    for (int k = 0; k < (int)K - 1; ++k) acc += (AccT)wc[k] * (AccT)sc[k];
    acc += (AccT)wc[K - 1] * (AccT)x;

    y[b * C + c] = (T)acc;

    for (int k = 0; k < (int)K - 2; ++k) sc[k] = sc[k + 1];
    if (K >= 2) sc[K - 2] = x;
}

infiniStatus_t Descriptor::update(void *params_void) const {
    auto *p = reinterpret_cast<infiniopConv1dUpdateParams_t *>(params_void);
    auto st = reinterpret_cast<cudaStream_t>(p->stream);
    size_t N = p->B * p->C;
    dim3 blk(256), grd((N + blk.x - 1) / blk.x);

    if (p->dtype == INFINI_DTYPE_F32) {
        conv1d_update_kernel<float><<<grd, blk, 0, st>>>(
            (float *)p->y, (const float *)p->x_now, (const float *)p->w,
            (float *)p->conv_state, p->B, p->C, p->K);
    } else if (p->dtype == INFINI_DTYPE_F16) {
        conv1d_update_kernel<__half, float><<<grd, blk, 0, st>>>(
            (__half *)p->y, (const __half *)p->x_now, (const __half *)p->w,
            (__half *)p->conv_state, p->B, p->C, p->K);
    } else if (p->dtype == INFINI_DTYPE_BF16) {
    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
        conv1d_update_kernel<__nv_bfloat16, float><<<grd, blk, 0, st>>>(
            (__nv_bfloat16 *)p->y, (const __nv_bfloat16 *)p->x_now, (const __nv_bfloat16 *)p->w,
            (__nv_bfloat16 *)p->conv_state, p->B, p->C, p->K);
    #else
        return INFINI_STATUS_NOT_IMPLEMENTED;
    #endif
    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::conv1d::nvidia
