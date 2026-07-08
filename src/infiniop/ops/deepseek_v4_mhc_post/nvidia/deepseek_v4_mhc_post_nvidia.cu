#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_HYGON_API) || defined(ENABLE_ALI_API)

#include "deepseek_v4_mhc_post_nvidia.cuh"

#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <memory>
#include <type_traits>

namespace op::deepseek_v4_mhc_post::nvidia {

struct PostDescriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

PostDescriptor::~PostDescriptor() {
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

template <typename OutT, typename CoeffT>
__global__ void mhc_post_kernel_hc4(
    OutT *__restrict__ y,
    const OutT *__restrict__ new_x,
    const OutT *__restrict__ residual,
    const CoeffT *__restrict__ post,
    const CoeffT *__restrict__ comb,
    size_t token_count,
    size_t hidden_size) {
    const size_t d = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t token = blockIdx.y;
    const size_t stream = blockIdx.z;
    if (token >= token_count || d >= hidden_size || stream >= 4) {
        return;
    }

    const size_t residual_offset = token * 4 * hidden_size + d;
    const float r0 = to_float<OutT>(residual[residual_offset]);
    const float r1 = to_float<OutT>(residual[residual_offset + hidden_size]);
    const float r2 = to_float<OutT>(residual[residual_offset + 2 * hidden_size]);
    const float r3 = to_float<OutT>(residual[residual_offset + 3 * hidden_size]);
    const float x_value = to_float<OutT>(new_x[token * hidden_size + d]);

    const CoeffT *post_token = post + token * 4;
    const CoeffT *comb_row = comb + token * 16 + stream * 4;
    float acc = to_float<CoeffT>(post_token[stream]) * x_value;
    acc += to_float<CoeffT>(comb_row[0]) * r0;
    acc += to_float<CoeffT>(comb_row[1]) * r1;
    acc += to_float<CoeffT>(comb_row[2]) * r2;
    acc += to_float<CoeffT>(comb_row[3]) * r3;

    y[token * 4 * hidden_size + stream * hidden_size + d] = from_float<OutT>(acc);
}

template <typename OutT, typename CoeffT>
__global__ void mhc_post_kernel(
    OutT *__restrict__ y,
    const OutT *__restrict__ new_x,
    const OutT *__restrict__ residual,
    const CoeffT *__restrict__ post,
    const CoeffT *__restrict__ comb,
    size_t token_count,
    size_t hc_mult,
    size_t hidden_size) {
    const size_t d = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t token = blockIdx.y;
    if (token >= token_count || d >= hidden_size) {
        return;
    }

    float residual_values[16];
    const size_t limited_hc = hc_mult <= 16 ? hc_mult : 16;
    const OutT *residual_token = residual + token * hc_mult * hidden_size;
    for (size_t j = 0; j < limited_hc; ++j) {
        residual_values[j] = to_float<OutT>(residual_token[j * hidden_size + d]);
    }

    const float x_value = to_float<OutT>(new_x[token * hidden_size + d]);
    const CoeffT *post_token = post + token * hc_mult;
    const CoeffT *comb_token = comb + token * hc_mult * hc_mult;
    OutT *y_token = y + token * hc_mult * hidden_size;

    for (size_t i = 0; i < limited_hc; ++i) {
        float acc = to_float<CoeffT>(post_token[i]) * x_value;
        const CoeffT *comb_row = comb_token + i * hc_mult;
        for (size_t j = 0; j < limited_hc; ++j) {
            acc += to_float<CoeffT>(comb_row[j]) * residual_values[j];
        }
        y_token[i * hidden_size + d] = from_float<OutT>(acc);
    }

    for (size_t i = limited_hc; i < hc_mult; ++i) {
        float acc = to_float<CoeffT>(post_token[i]) * x_value;
        const CoeffT *comb_row = comb_token + i * hc_mult;
        for (size_t j = 0; j < hc_mult; ++j) {
            acc += to_float<CoeffT>(comb_row[j])
                 * to_float<OutT>(residual_token[j * hidden_size + d]);
        }
        y_token[i * hidden_size + d] = from_float<OutT>(acc);
    }
}

template <typename OutT, typename CoeffT>
infiniStatus_t launch_typed(
    const DeepseekV4MHCPostInfo &info,
    void *y,
    const void *new_x,
    const void *residual,
    const void *post,
    const void *comb,
    cudaStream_t stream) {
    constexpr int threads = 256;
    const dim3 block(threads);
    const dim3 grid((info.hidden_size + threads - 1) / threads, info.token_count);
    if (info.hc_mult == 4) {
        const dim3 grid_hc4((info.hidden_size + threads - 1) / threads, info.token_count, 4);
        mhc_post_kernel_hc4<OutT, CoeffT><<<grid_hc4, block, 0, stream>>>(
            reinterpret_cast<OutT *>(y),
            reinterpret_cast<const OutT *>(new_x),
            reinterpret_cast<const OutT *>(residual),
            reinterpret_cast<const CoeffT *>(post),
            reinterpret_cast<const CoeffT *>(comb),
            info.token_count,
            info.hidden_size);
    } else {
        mhc_post_kernel<OutT, CoeffT><<<grid, block, 0, stream>>>(
            reinterpret_cast<OutT *>(y),
            reinterpret_cast<const OutT *>(new_x),
            reinterpret_cast<const OutT *>(residual),
            reinterpret_cast<const CoeffT *>(post),
            reinterpret_cast<const CoeffT *>(comb),
            info.token_count,
            info.hc_mult,
            info.hidden_size);
    }
    CHECK_CUDA(cudaGetLastError());
    return INFINI_STATUS_SUCCESS;
}

template <typename OutT>
infiniStatus_t launch_by_coeff_dtype(
    const DeepseekV4MHCPostInfo &info,
    void *y,
    const void *new_x,
    const void *residual,
    const void *post,
    const void *comb,
    cudaStream_t stream) {
    if (info.coeff_dtype == INFINI_DTYPE_F16) {
        return launch_typed<OutT, half>(info, y, new_x, residual, post, comb, stream);
    }
    if (info.coeff_dtype == INFINI_DTYPE_BF16) {
        return launch_typed<OutT, __nv_bfloat16>(info, y, new_x, residual, post, comb, stream);
    }
    if (info.coeff_dtype == INFINI_DTYPE_F32) {
        return launch_typed<OutT, float>(info, y, new_x, residual, post, comb, stream);
    }
    return INFINI_STATUS_BAD_TENSOR_DTYPE;
}

} // namespace

infiniStatus_t PostDescriptor::create(
    infiniopHandle_t handle,
    PostDescriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t new_x_desc,
    infiniopTensorDescriptor_t residual_desc,
    infiniopTensorDescriptor_t post_desc,
    infiniopTensorDescriptor_t comb_desc) {
    auto result = DeepseekV4MHCPostInfo::create(y_desc, new_x_desc, residual_desc, post_desc, comb_desc);
    CHECK_RESULT(result);
    *desc_ptr = new PostDescriptor(
        new Opaque{reinterpret_cast<device::nvidia::Handle *>(handle)->internal()},
        result.take(),
        0,
        handle->device,
        handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t PostDescriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *new_x,
    const void *residual,
    const void *post,
    const void *comb,
    void *stream) const {
    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    if (_info.dtype == INFINI_DTYPE_F16) {
        return launch_by_coeff_dtype<half>(_info, y, new_x, residual, post, comb, cuda_stream);
    }
    if (_info.dtype == INFINI_DTYPE_BF16) {
        return launch_by_coeff_dtype<__nv_bfloat16>(_info, y, new_x, residual, post, comb, cuda_stream);
    }
    if (_info.dtype == INFINI_DTYPE_F32) {
        return launch_by_coeff_dtype<float>(_info, y, new_x, residual, post, comb, cuda_stream);
    }
    return INFINI_STATUS_BAD_TENSOR_DTYPE;
}

} // namespace op::deepseek_v4_mhc_post::nvidia

#endif
