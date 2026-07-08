#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_HYGON_API) || defined(ENABLE_ALI_API)

#include "deepseek_v4_mhc_nvidia.cuh"

#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <memory>
#include <type_traits>

namespace op::deepseek_v4_mhc::nvidia {

struct ParamsDescriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

struct PreCollapseDescriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

struct ScaleMixesDescriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

ParamsDescriptor::~ParamsDescriptor() {
    delete _opaque;
}

PreCollapseDescriptor::~PreCollapseDescriptor() {
    delete _opaque;
}

ScaleMixesDescriptor::~ScaleMixesDescriptor() {
    delete _opaque;
}

namespace {

template <typename T>
__device__ float to_float(T value) {
    if constexpr (std::is_same_v<T, half>) {
        return __half2float(value);
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        return __bfloat162float(value);
    } else {
        return static_cast<float>(value);
    }
}

template <typename T>
__device__ T from_float(float value) {
    if constexpr (std::is_same_v<T, half>) {
        return __float2half(value);
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        return __float2bfloat16(value);
    } else {
        return static_cast<T>(value);
    }
}

__device__ float sigmoid_stable(float value) {
    if (value >= 0.0f) {
        const float z = expf(-value);
        return 1.0f / (1.0f + z);
    }
    const float z = expf(value);
    return z / (1.0f + z);
}

template <typename T>
__global__ void mhc_params_kernel(
    T *pre,
    T *post,
    T *comb,
    const T *mixes,
    const float *base,
    const float *scale,
    size_t token_count,
    size_t hc_mult,
    size_t mix_hc,
    size_t sinkhorn_iters,
    float eps) {
    const size_t token = blockIdx.x;
    if (token >= token_count || threadIdx.x != 0) {
        return;
    }

    float row[16];
    float comb_raw[256];
    const T *mix = mixes + token * mix_hc;
    const size_t pre_offset = token * hc_mult;

    for (size_t i = 0; i < hc_mult; ++i) {
        const float pre_value = sigmoid_stable(scale[0] * to_float<T>(mix[i]) + base[i]) + eps;
        const float post_value = 2.0f * sigmoid_stable(scale[1] * to_float<T>(mix[hc_mult + i]) + base[hc_mult + i]);
        pre[pre_offset + i] = from_float<T>(pre_value);
        post[pre_offset + i] = from_float<T>(post_value);
    }

    for (size_t i = 0; i < hc_mult; ++i) {
        float max_value = -CUDART_INF_F;
        for (size_t j = 0; j < hc_mult; ++j) {
            const size_t idx = 2 * hc_mult + i * hc_mult + j;
            row[j] = scale[2] * to_float<T>(mix[idx]) + base[idx];
            max_value = fmaxf(max_value, row[j]);
        }
        float sum = 0.0f;
        for (size_t j = 0; j < hc_mult; ++j) {
            row[j] = expf(row[j] - max_value);
            sum += row[j];
        }
        for (size_t j = 0; j < hc_mult; ++j) {
            comb_raw[i * hc_mult + j] = row[j] / sum + eps;
        }
    }

    for (size_t j = 0; j < hc_mult; ++j) {
        float col_sum = eps;
        for (size_t i = 0; i < hc_mult; ++i) {
            col_sum += comb_raw[i * hc_mult + j];
        }
        for (size_t i = 0; i < hc_mult; ++i) {
            comb_raw[i * hc_mult + j] /= col_sum;
        }
    }
    for (size_t iter = 1; iter < sinkhorn_iters; ++iter) {
        for (size_t i = 0; i < hc_mult; ++i) {
            float row_sum = eps;
            for (size_t j = 0; j < hc_mult; ++j) {
                row_sum += comb_raw[i * hc_mult + j];
            }
            for (size_t j = 0; j < hc_mult; ++j) {
                comb_raw[i * hc_mult + j] /= row_sum;
            }
        }
        for (size_t j = 0; j < hc_mult; ++j) {
            float col_sum = eps;
            for (size_t i = 0; i < hc_mult; ++i) {
                col_sum += comb_raw[i * hc_mult + j];
            }
            for (size_t i = 0; i < hc_mult; ++i) {
                comb_raw[i * hc_mult + j] /= col_sum;
            }
        }
    }

    T *comb_out = comb + token * hc_mult * hc_mult;
    for (size_t i = 0; i < hc_mult * hc_mult; ++i) {
        comb_out[i] = from_float<T>(comb_raw[i]);
    }
}


template <typename OutT, typename MixT>
__global__ void mhc_pre_collapse_kernel_hc4(
    OutT *__restrict__ collapsed,
    MixT *__restrict__ post,
    MixT *__restrict__ comb,
    const OutT *__restrict__ x,
    const MixT *__restrict__ mixes,
    const float *__restrict__ base,
    const float *__restrict__ scale,
    size_t token_count,
    size_t hidden_size,
    size_t sinkhorn_iters,
    float eps) {
    const size_t d = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t token = blockIdx.y;
    if (token >= token_count) {
        return;
    }

    const MixT *mix = mixes + token * 24;
    float pre[4];
    if (d == 0) {
        float comb_raw[16];
        for (size_t i = 0; i < 4; ++i) {
            pre[i] = sigmoid_stable(scale[0] * to_float<MixT>(mix[i]) + base[i]) + eps;
            const float post_value = 2.0f * sigmoid_stable(scale[1] * to_float<MixT>(mix[4 + i]) + base[4 + i]);
            post[token * 4 + i] = from_float<MixT>(post_value);
        }

        for (size_t i = 0; i < 4; ++i) {
            float row[4];
            float max_value = -CUDART_INF_F;
            for (size_t j = 0; j < 4; ++j) {
                const size_t idx = 8 + i * 4 + j;
                row[j] = scale[2] * to_float<MixT>(mix[idx]) + base[idx];
                max_value = fmaxf(max_value, row[j]);
            }
            float sum = 0.0f;
            for (size_t j = 0; j < 4; ++j) {
                row[j] = expf(row[j] - max_value);
                sum += row[j];
            }
            for (size_t j = 0; j < 4; ++j) {
                comb_raw[i * 4 + j] = row[j] / sum + eps;
            }
        }

        for (size_t j = 0; j < 4; ++j) {
            float col_sum = eps;
            for (size_t i = 0; i < 4; ++i) {
                col_sum += comb_raw[i * 4 + j];
            }
            for (size_t i = 0; i < 4; ++i) {
                comb_raw[i * 4 + j] /= col_sum;
            }
        }
        for (size_t iter = 1; iter < sinkhorn_iters; ++iter) {
            for (size_t i = 0; i < 4; ++i) {
                float row_sum = eps;
                for (size_t j = 0; j < 4; ++j) {
                    row_sum += comb_raw[i * 4 + j];
                }
                for (size_t j = 0; j < 4; ++j) {
                    comb_raw[i * 4 + j] /= row_sum;
                }
            }
            for (size_t j = 0; j < 4; ++j) {
                float col_sum = eps;
                for (size_t i = 0; i < 4; ++i) {
                    col_sum += comb_raw[i * 4 + j];
                }
                for (size_t i = 0; i < 4; ++i) {
                    comb_raw[i * 4 + j] /= col_sum;
                }
            }
        }

        MixT *comb_out = comb + token * 16;
        for (size_t i = 0; i < 16; ++i) {
            comb_out[i] = from_float<MixT>(comb_raw[i]);
        }
    }

    if (d < hidden_size) {
        pre[0] = sigmoid_stable(scale[0] * to_float<MixT>(mix[0]) + base[0]) + eps;
        pre[1] = sigmoid_stable(scale[0] * to_float<MixT>(mix[1]) + base[1]) + eps;
        pre[2] = sigmoid_stable(scale[0] * to_float<MixT>(mix[2]) + base[2]) + eps;
        pre[3] = sigmoid_stable(scale[0] * to_float<MixT>(mix[3]) + base[3]) + eps;
        const MixT pre0 = from_float<MixT>(pre[0]);
        const MixT pre1 = from_float<MixT>(pre[1]);
        const MixT pre2 = from_float<MixT>(pre[2]);
        const MixT pre3 = from_float<MixT>(pre[3]);
        const OutT *x_token = x + token * 4 * hidden_size + d;
        float acc = to_float<MixT>(pre0) * to_float<OutT>(x_token[0]);
        acc += to_float<MixT>(pre1) * to_float<OutT>(x_token[hidden_size]);
        acc += to_float<MixT>(pre2) * to_float<OutT>(x_token[2 * hidden_size]);
        acc += to_float<MixT>(pre3) * to_float<OutT>(x_token[3 * hidden_size]);
        collapsed[token * hidden_size + d] = from_float<OutT>(acc);
    }
}

template <typename OutT, typename MixT>
__global__ void mhc_pre_collapse_kernel(
    OutT *__restrict__ collapsed,
    MixT *__restrict__ post,
    MixT *__restrict__ comb,
    const OutT *__restrict__ x,
    const MixT *__restrict__ mixes,
    const float *__restrict__ base,
    const float *__restrict__ scale,
    size_t token_count,
    size_t hc_mult,
    size_t hidden_size,
    size_t mix_hc,
    size_t sinkhorn_iters,
    float eps) {
    const size_t d = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t token = blockIdx.y;
    if (token >= token_count) {
        return;
    }

    const MixT *mix = mixes + token * mix_hc;
    if (d == 0) {
        float row[16];
        float comb_raw[256];
        for (size_t i = 0; i < hc_mult; ++i) {
            const float post_value = 2.0f * sigmoid_stable(scale[1] * to_float<MixT>(mix[hc_mult + i]) + base[hc_mult + i]);
            post[token * hc_mult + i] = from_float<MixT>(post_value);
        }

        for (size_t i = 0; i < hc_mult; ++i) {
            float max_value = -CUDART_INF_F;
            for (size_t j = 0; j < hc_mult; ++j) {
                const size_t idx = 2 * hc_mult + i * hc_mult + j;
                row[j] = scale[2] * to_float<MixT>(mix[idx]) + base[idx];
                max_value = fmaxf(max_value, row[j]);
            }
            float sum = 0.0f;
            for (size_t j = 0; j < hc_mult; ++j) {
                row[j] = expf(row[j] - max_value);
                sum += row[j];
            }
            for (size_t j = 0; j < hc_mult; ++j) {
                comb_raw[i * hc_mult + j] = row[j] / sum + eps;
            }
        }

        for (size_t j = 0; j < hc_mult; ++j) {
            float col_sum = eps;
            for (size_t i = 0; i < hc_mult; ++i) {
                col_sum += comb_raw[i * hc_mult + j];
            }
            for (size_t i = 0; i < hc_mult; ++i) {
                comb_raw[i * hc_mult + j] /= col_sum;
            }
        }
        for (size_t iter = 1; iter < sinkhorn_iters; ++iter) {
            for (size_t i = 0; i < hc_mult; ++i) {
                float row_sum = eps;
                for (size_t j = 0; j < hc_mult; ++j) {
                    row_sum += comb_raw[i * hc_mult + j];
                }
                for (size_t j = 0; j < hc_mult; ++j) {
                    comb_raw[i * hc_mult + j] /= row_sum;
                }
            }
            for (size_t j = 0; j < hc_mult; ++j) {
                float col_sum = eps;
                for (size_t i = 0; i < hc_mult; ++i) {
                    col_sum += comb_raw[i * hc_mult + j];
                }
                for (size_t i = 0; i < hc_mult; ++i) {
                    comb_raw[i * hc_mult + j] /= col_sum;
                }
            }
        }

        MixT *comb_out = comb + token * hc_mult * hc_mult;
        for (size_t i = 0; i < hc_mult * hc_mult; ++i) {
            comb_out[i] = from_float<MixT>(comb_raw[i]);
        }
    }

    if (d < hidden_size) {
        float acc = 0.0f;
        const OutT *x_token = x + token * hc_mult * hidden_size + d;
        for (size_t i = 0; i < hc_mult; ++i) {
            const float pre = sigmoid_stable(scale[0] * to_float<MixT>(mix[i]) + base[i]) + eps;
            const MixT pre_quant = from_float<MixT>(pre);
            acc += to_float<MixT>(pre_quant) * to_float<OutT>(x_token[i * hidden_size]);
        }
        collapsed[token * hidden_size + d] = from_float<OutT>(acc);
    }
}

template <typename OutT, typename MixT>
infiniStatus_t launch_pre_collapse_typed(
    const DeepseekV4MHCPreCollapseInfo &info,
    void *collapsed,
    void *post,
    void *comb,
    const void *x,
    const void *mixes,
    const void *base,
    const void *scale,
    cudaStream_t stream) {
    constexpr int threads = 256;
    const dim3 block(threads);
    const dim3 grid((info.hidden_size + threads - 1) / threads, info.token_count);
    if (info.hc_mult == 4) {
        mhc_pre_collapse_kernel_hc4<OutT, MixT><<<grid, block, 0, stream>>>(
            reinterpret_cast<OutT *>(collapsed),
            reinterpret_cast<MixT *>(post),
            reinterpret_cast<MixT *>(comb),
            reinterpret_cast<const OutT *>(x),
            reinterpret_cast<const MixT *>(mixes),
            reinterpret_cast<const float *>(base),
            reinterpret_cast<const float *>(scale),
            info.token_count,
            info.hidden_size,
            info.sinkhorn_iters,
            info.epsilon);
    } else {
        mhc_pre_collapse_kernel<OutT, MixT><<<grid, block, 0, stream>>>(
            reinterpret_cast<OutT *>(collapsed),
            reinterpret_cast<MixT *>(post),
            reinterpret_cast<MixT *>(comb),
            reinterpret_cast<const OutT *>(x),
            reinterpret_cast<const MixT *>(mixes),
            reinterpret_cast<const float *>(base),
            reinterpret_cast<const float *>(scale),
            info.token_count,
            info.hc_mult,
            info.hidden_size,
            info.mix_hc,
            info.sinkhorn_iters,
            info.epsilon);
    }
    CHECK_CUDA(cudaGetLastError());
    return INFINI_STATUS_SUCCESS;
}

template <typename OutT>
infiniStatus_t launch_pre_collapse_by_mix_dtype(
    const DeepseekV4MHCPreCollapseInfo &info,
    void *collapsed,
    void *post,
    void *comb,
    const void *x,
    const void *mixes,
    const void *base,
    const void *scale,
    cudaStream_t stream) {
    if (info.coeff_dtype == INFINI_DTYPE_F16) {
        return launch_pre_collapse_typed<OutT, half>(info, collapsed, post, comb, x, mixes, base, scale, stream);
    }
    if (info.coeff_dtype == INFINI_DTYPE_BF16) {
        return launch_pre_collapse_typed<OutT, __nv_bfloat16>(info, collapsed, post, comb, x, mixes, base, scale, stream);
    }
    if (info.coeff_dtype == INFINI_DTYPE_F32) {
        return launch_pre_collapse_typed<OutT, float>(info, collapsed, post, comb, x, mixes, base, scale, stream);
    }
    return INFINI_STATUS_BAD_TENSOR_DTYPE;
}


template <typename XType, typename MixT>
__global__ void mhc_scale_mixes_kernel(
    MixT *__restrict__ scaled,
    const XType *__restrict__ x,
    const MixT *__restrict__ raw_mixes,
    size_t token_count,
    size_t flat_dim,
    size_t mix_hc,
    float eps) {
    const size_t token = blockIdx.x;
    if (token >= token_count) {
        return;
    }

    extern __shared__ float sdata[];
    float sum = 0.0f;
    const XType *x_token = x + token * flat_dim;
    for (size_t i = threadIdx.x; i < flat_dim; i += blockDim.x) {
        const float v = to_float<XType>(x_token[i]);
        sum += v * v;
    }
    sdata[threadIdx.x] = sum;
    __syncthreads();

    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        }
        __syncthreads();
    }

    const float rsqrt_value = rsqrtf(sdata[0] / static_cast<float>(flat_dim) + eps);
    const MixT *raw = raw_mixes + token * mix_hc;
    MixT *out = scaled + token * mix_hc;
    for (size_t i = threadIdx.x; i < mix_hc; i += blockDim.x) {
        const float value = to_float<MixT>(raw[i]) * rsqrt_value;
        out[i] = from_float<MixT>(value);
    }
}

template <typename XType, typename MixT>
infiniStatus_t launch_scale_mixes_typed(
    const DeepseekV4MHCScaleMixesInfo &info,
    void *scaled,
    const void *x,
    const void *raw_mixes,
    cudaStream_t stream) {
    constexpr int threads = 256;
    mhc_scale_mixes_kernel<XType, MixT><<<info.token_count, threads, threads * sizeof(float), stream>>>(
        reinterpret_cast<MixT *>(scaled),
        reinterpret_cast<const XType *>(x),
        reinterpret_cast<const MixT *>(raw_mixes),
        info.token_count,
        info.flat_dim,
        info.mix_hc,
        info.epsilon);
    CHECK_CUDA(cudaGetLastError());
    return INFINI_STATUS_SUCCESS;
}

template <typename XType>
infiniStatus_t launch_scale_mixes_by_mix_dtype(
    const DeepseekV4MHCScaleMixesInfo &info,
    void *scaled,
    const void *x,
    const void *raw_mixes,
    cudaStream_t stream) {
    if (info.mixes_dtype == INFINI_DTYPE_F16) {
        return launch_scale_mixes_typed<XType, half>(info, scaled, x, raw_mixes, stream);
    }
    if (info.mixes_dtype == INFINI_DTYPE_BF16) {
        return launch_scale_mixes_typed<XType, __nv_bfloat16>(info, scaled, x, raw_mixes, stream);
    }
    if (info.mixes_dtype == INFINI_DTYPE_F32) {
        return launch_scale_mixes_typed<XType, float>(info, scaled, x, raw_mixes, stream);
    }
    return INFINI_STATUS_BAD_TENSOR_DTYPE;
}

template <typename T>
infiniStatus_t launch_typed(
    const DeepseekV4MHCParamsInfo &info,
    void *pre,
    void *post,
    void *comb,
    const void *mixes,
    const void *base,
    const void *scale,
    cudaStream_t stream) {
    mhc_params_kernel<T><<<info.token_count, 1, 0, stream>>>(
        reinterpret_cast<T *>(pre),
        reinterpret_cast<T *>(post),
        reinterpret_cast<T *>(comb),
        reinterpret_cast<const T *>(mixes),
        reinterpret_cast<const float *>(base),
        reinterpret_cast<const float *>(scale),
        info.token_count,
        info.hc_mult,
        info.mix_hc,
        info.sinkhorn_iters,
        info.epsilon);
    CHECK_CUDA(cudaGetLastError());
    return INFINI_STATUS_SUCCESS;
}

} // namespace

infiniStatus_t ParamsDescriptor::create(
    infiniopHandle_t handle,
    ParamsDescriptor **desc_ptr,
    infiniopTensorDescriptor_t pre_desc,
    infiniopTensorDescriptor_t post_desc,
    infiniopTensorDescriptor_t comb_desc,
    infiniopTensorDescriptor_t mixes_desc,
    infiniopTensorDescriptor_t base_desc,
    infiniopTensorDescriptor_t scale_desc,
    size_t sinkhorn_iters,
    float epsilon) {
    auto result = DeepseekV4MHCParamsInfo::create(pre_desc, post_desc, comb_desc, mixes_desc, base_desc, scale_desc, sinkhorn_iters, epsilon);
    CHECK_RESULT(result);
    *desc_ptr = new ParamsDescriptor(
        new Opaque{reinterpret_cast<device::nvidia::Handle *>(handle)->internal()},
        result.take(),
        0,
        handle->device,
        handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t ParamsDescriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *pre,
    void *post,
    void *comb,
    const void *mixes,
    const void *base,
    const void *scale,
    void *stream) const {
    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    if (_info.dtype == INFINI_DTYPE_F16) {
        return launch_typed<half>(_info, pre, post, comb, mixes, base, scale, cuda_stream);
    }
    if (_info.dtype == INFINI_DTYPE_BF16) {
        return launch_typed<__nv_bfloat16>(_info, pre, post, comb, mixes, base, scale, cuda_stream);
    }
    if (_info.dtype == INFINI_DTYPE_F32) {
        return launch_typed<float>(_info, pre, post, comb, mixes, base, scale, cuda_stream);
    }
    return INFINI_STATUS_BAD_TENSOR_DTYPE;
}


infiniStatus_t PreCollapseDescriptor::create(
    infiniopHandle_t handle,
    PreCollapseDescriptor **desc_ptr,
    infiniopTensorDescriptor_t collapsed_desc,
    infiniopTensorDescriptor_t post_desc,
    infiniopTensorDescriptor_t comb_desc,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t mixes_desc,
    infiniopTensorDescriptor_t base_desc,
    infiniopTensorDescriptor_t scale_desc,
    size_t sinkhorn_iters,
    float epsilon) {
    auto result = DeepseekV4MHCPreCollapseInfo::create(
        collapsed_desc, post_desc, comb_desc, x_desc, mixes_desc, base_desc, scale_desc, sinkhorn_iters, epsilon);
    CHECK_RESULT(result);
    *desc_ptr = new PreCollapseDescriptor(
        new Opaque{reinterpret_cast<device::nvidia::Handle *>(handle)->internal()},
        result.take(),
        0,
        handle->device,
        handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t PreCollapseDescriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *collapsed,
    void *post,
    void *comb,
    const void *x,
    const void *mixes,
    const void *base,
    const void *scale,
    void *stream) const {
    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    if (_info.dtype == INFINI_DTYPE_F16) {
        return launch_pre_collapse_by_mix_dtype<half>(_info, collapsed, post, comb, x, mixes, base, scale, cuda_stream);
    }
    if (_info.dtype == INFINI_DTYPE_BF16) {
        return launch_pre_collapse_by_mix_dtype<__nv_bfloat16>(_info, collapsed, post, comb, x, mixes, base, scale, cuda_stream);
    }
    if (_info.dtype == INFINI_DTYPE_F32) {
        return launch_pre_collapse_by_mix_dtype<float>(_info, collapsed, post, comb, x, mixes, base, scale, cuda_stream);
    }
    return INFINI_STATUS_BAD_TENSOR_DTYPE;
}


infiniStatus_t ScaleMixesDescriptor::create(
    infiniopHandle_t handle,
    ScaleMixesDescriptor **desc_ptr,
    infiniopTensorDescriptor_t scaled_desc,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t raw_mixes_desc,
    float epsilon) {
    auto result = DeepseekV4MHCScaleMixesInfo::create(scaled_desc, x_desc, raw_mixes_desc, epsilon);
    CHECK_RESULT(result);
    *desc_ptr = new ScaleMixesDescriptor(
        new Opaque{reinterpret_cast<device::nvidia::Handle *>(handle)->internal()},
        result.take(),
        0,
        handle->device,
        handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t ScaleMixesDescriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *scaled,
    const void *x,
    const void *raw_mixes,
    void *stream) const {
    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    if (_info.dtype == INFINI_DTYPE_F16) {
        return launch_scale_mixes_by_mix_dtype<half>(_info, scaled, x, raw_mixes, cuda_stream);
    }
    if (_info.dtype == INFINI_DTYPE_BF16) {
        return launch_scale_mixes_by_mix_dtype<__nv_bfloat16>(_info, scaled, x, raw_mixes, cuda_stream);
    }
    if (_info.dtype == INFINI_DTYPE_F32) {
        return launch_scale_mixes_by_mix_dtype<float>(_info, scaled, x, raw_mixes, cuda_stream);
    }
    return INFINI_STATUS_BAD_TENSOR_DTYPE;
}

} // namespace op::deepseek_v4_mhc::nvidia

#endif
