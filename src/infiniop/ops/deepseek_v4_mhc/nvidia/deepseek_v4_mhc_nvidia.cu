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

ParamsDescriptor::~ParamsDescriptor() {
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

} // namespace op::deepseek_v4_mhc::nvidia

#endif
