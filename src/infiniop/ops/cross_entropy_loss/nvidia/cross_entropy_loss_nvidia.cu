#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <limits.h>
#include <math_constants.h>
#include <memory>
#include <numeric>
#include <stdio.h>
#include <vector>

#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_handle.cuh"
#include "cross_entropy_loss_nvidia.cuh"

namespace op::cross_entropy_loss::nvidia {
namespace cuda {

__device__ __forceinline__ float to_float(float v) { return v; }
__device__ __forceinline__ float to_float(double v) { return (float)v; }
__device__ __forceinline__ float to_float(half v) { return __half2float(v); }
__device__ __forceinline__ float to_float(__nv_bfloat16 v) {
    return __bfloat162float(v);
}

template <typename T_in, typename T_out = float>
__global__ void
softmaxCrossEntropy_per_sample(T_out *__restrict__ loss,
                               const T_in *__restrict__ logits,
                               const int64_t *__restrict__ target, int N, int C,
                               long long inner_size, int64_t ignore_index) {
    long long total = (long long)N * inner_size;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }

    int n = (int)(idx / inner_size);
    int inr = (int)(idx % inner_size);

    int64_t t = target[(long long)n * inner_size + inr];
    if (ignore_index != LLONG_MIN && t == ignore_index) {
        loss[idx] = (T_out)0;
        return;
    }
    if (t < 0 || t >= C) {
        loss[idx] = (T_out)0;
        return;
    }

    const long long base = ((long long)n * C * inner_size) + inr;

    // 数值稳定 LSE：lse = log(sum exp(x - m)) + m
    float m = -CUDART_INF_F;
    for (int c = 0; c < C; ++c) {
        m = fmaxf(m, to_float(logits[base + (long long)c * inner_size]));
    }

    float sum_exp = 0.f;
    for (int c = 0; c < C; ++c) {
        sum_exp += expf(to_float(logits[base + (long long)c * inner_size]) - m);
    }

    float lse = logf(sum_exp) + m;
    float logit_t = to_float(logits[base + (long long)(int)t * inner_size]);
    loss[idx] = (T_out)(lse - logit_t);
}

} // namespace cuda

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
    std::vector<size_t> logits_shape;
    Opaque(std::shared_ptr<device::nvidia::Handle::Internal> p) : internal(p) {}
    ~Opaque() = default;
};

Descriptor::~Descriptor() {
    if (_opaque) {
        delete _opaque;
    }
}

infiniStatus_t Descriptor::create(infiniopHandle_t handle_,
                                  Descriptor **desc_ptr,
                                  infiniopTensorDescriptor_t /*loss_desc*/,
                                  infiniopTensorDescriptor_t logits_desc,
                                  infiniopTensorDescriptor_t /*target_desc*/) {
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API)
    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);
    auto dtype = logits_desc->dtype();
    CHECK_DTYPE(dtype, INFINI_DTYPE_F32, INFINI_DTYPE_F16, INFINI_DTYPE_BF16);

    const auto &orig = logits_desc->shape();
    auto opaque = new Opaque(handle->internal());

    if (orig.size() == 1) {
        opaque->logits_shape = {1, orig[0]};
    } else {
        opaque->logits_shape = orig;
    }

    const auto &s = opaque->logits_shape;
    long long N = (long long)s[0];
    long long inner = 1;
    for (size_t i = 2; i < s.size(); ++i) {
        inner *= (long long)s[i];
    }

    size_t workspace_size = (size_t)(N * inner) * sizeof(float);
    *desc_ptr = new Descriptor(dtype, workspace_size, opaque, handle->device,
                               handle->device_id);
    return INFINI_STATUS_SUCCESS;
#else
    return INFINI_STATUS_NOT_IMPLEMENTED;
#endif
}

infiniStatus_t Descriptor::calculate(void *workspace, size_t workspace_size,
                                     void *loss, const void *logits,
                                     const void *target, void *stream) const {
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API)
    const auto &s = _opaque->logits_shape;
    int N = (int)s[0];
    int C = (int)s[1];
    long long inner = 1;
    for (size_t i = 2; i < s.size(); ++i) {
        inner *= (long long)s[i];
    }
    long long total = (long long)N * inner;

    size_t need_ws = (size_t)total * sizeof(float);
    if (workspace_size < need_ws) {
        return INFINI_STATUS_INTERNAL_ERROR;
    }
    float *per_sample = reinterpret_cast<float *>(workspace);

    const int64_t *tgt_i64 = reinterpret_cast<const int64_t *>(target);
    const int64_t ignore_index = -100;

    // 1) 写 per-sample loss -> workspace（float）
    dim3 block(256);
    dim3 grid((total + block.x - 1) / block.x);
    cudaStream_t st = (cudaStream_t)stream;

    if (_dtype == INFINI_DTYPE_F32) {
        cuda::softmaxCrossEntropy_per_sample<float, float><<<grid, block, 0, st>>>(
            per_sample, (const float *)logits, tgt_i64, N, C, inner, ignore_index);
    } else if (_dtype == INFINI_DTYPE_F16) {
        cuda::softmaxCrossEntropy_per_sample<half, float><<<grid, block, 0, st>>>(
            per_sample, (const half *)logits, tgt_i64, N, C, inner, ignore_index);
    } else if (_dtype == INFINI_DTYPE_BF16) {
        cuda::softmaxCrossEntropy_per_sample<__nv_bfloat16, float>
            <<<grid, block, 0, st>>>(per_sample, (const __nv_bfloat16 *)logits,
                                     tgt_i64, N, C, inner, ignore_index);
    }
    {
        auto err = cudaGetLastError();
        if (err != cudaSuccess) {
            return INFINI_STATUS_INTERNAL_ERROR;
        }
    }

    // 2) host 侧 mean（仅统计 target != ignore_index）
    std::vector<float> h_loss((size_t)total);
    std::vector<int64_t> h_tgt((size_t)total);
    if (cudaMemcpyAsync(h_loss.data(), per_sample, need_ws,
                        cudaMemcpyDeviceToHost, st)
        != cudaSuccess) {
        return INFINI_STATUS_INTERNAL_ERROR;
    }
    if (cudaMemcpyAsync(h_tgt.data(), tgt_i64, (size_t)total * sizeof(int64_t),
                        cudaMemcpyDeviceToHost, st)
        != cudaSuccess) {
        return INFINI_STATUS_INTERNAL_ERROR;
    }
    if (cudaStreamSynchronize(st) != cudaSuccess) {
        return INFINI_STATUS_INTERNAL_ERROR;
    }

    double acc = 0.0;
    long long cnt = 0;
    for (long long i = 0; i < total; ++i) {
        if (h_tgt[i] != ignore_index) {
            acc += (double)h_loss[i];
            ++cnt;
        }
    }
    double mean = (cnt > 0) ? (acc / (double)cnt) : 0.0;

    // 3) 把标量 mean 写回 device 的 loss 指针（按输入 dtype 写 1 个元素）
    if (_dtype == INFINI_DTYPE_F32) {
        float v = (float)mean;
        if (cudaMemcpyAsync(loss, &v, sizeof(float), cudaMemcpyHostToDevice, st) != cudaSuccess) {
            return INFINI_STATUS_INTERNAL_ERROR;
        }
    } else if (_dtype == INFINI_DTYPE_F16) {
        half v = __float2half((float)mean);
        if (cudaMemcpyAsync(loss, &v, sizeof(half), cudaMemcpyHostToDevice, st) != cudaSuccess) {
            return INFINI_STATUS_INTERNAL_ERROR;
        }
    } else if (_dtype == INFINI_DTYPE_BF16) {
        __nv_bfloat16 v = __float2bfloat16((float)mean);
        if (cudaMemcpyAsync(loss, &v, sizeof(__nv_bfloat16), cudaMemcpyHostToDevice,
                            st)
            != cudaSuccess) {
            return INFINI_STATUS_INTERNAL_ERROR;
        }
    }
    if (cudaStreamSynchronize(st) != cudaSuccess) {
        return INFINI_STATUS_INTERNAL_ERROR;
    }

    return INFINI_STATUS_SUCCESS;
#else
    return INFINI_STATUS_NOT_IMPLEMENTED;
#endif
}
} // namespace op::cross_entropy_loss::nvidia
