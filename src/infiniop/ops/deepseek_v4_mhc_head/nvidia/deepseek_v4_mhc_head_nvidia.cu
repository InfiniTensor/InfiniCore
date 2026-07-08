#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_HYGON_API) || defined(ENABLE_ALI_API)

#include "deepseek_v4_mhc_head_nvidia.cuh"

#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <memory>
#include <type_traits>

namespace op::deepseek_v4_mhc_head::nvidia {

struct HeadCollapseDescriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

HeadCollapseDescriptor::~HeadCollapseDescriptor() {
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

template <typename OutT, typename MixT>
__global__ void mhc_head_collapse_kernel(
    OutT *y,
    const OutT *x,
    const MixT *mixes,
    const float *base,
    const float *scale,
    size_t token_count,
    size_t hc_mult,
    size_t hidden_size,
    float eps) {
    const size_t d = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t token = blockIdx.y;
    if (token >= token_count || d >= hidden_size) {
        return;
    }

    float acc = 0.0f;
    const OutT *x_token = x + token * hc_mult * hidden_size;
    const MixT *mix_token = mixes + token * hc_mult;
    for (size_t h = 0; h < hc_mult; ++h) {
        const float pre = sigmoid_stable(to_float<MixT>(mix_token[h]) * scale[0] + base[h]) + eps;
        const OutT pre_quant = from_float<OutT>(pre);
        acc += to_float<OutT>(pre_quant) * to_float<OutT>(x_token[h * hidden_size + d]);
    }
    y[token * hidden_size + d] = from_float<OutT>(acc);
}

template <typename OutT, typename MixT>
infiniStatus_t launch_typed(
    const DeepseekV4MHCHeadCollapseInfo &info,
    void *y,
    const void *x,
    const void *mixes,
    const void *base,
    const void *scale,
    cudaStream_t stream) {
    constexpr int threads = 256;
    const dim3 block(threads);
    const dim3 grid((info.hidden_size + threads - 1) / threads, info.token_count);
    mhc_head_collapse_kernel<OutT, MixT><<<grid, block, 0, stream>>>(
        reinterpret_cast<OutT *>(y),
        reinterpret_cast<const OutT *>(x),
        reinterpret_cast<const MixT *>(mixes),
        reinterpret_cast<const float *>(base),
        reinterpret_cast<const float *>(scale),
        info.token_count,
        info.hc_mult,
        info.hidden_size,
        info.epsilon);
    CHECK_CUDA(cudaGetLastError());
    return INFINI_STATUS_SUCCESS;
}

template <typename OutT>
infiniStatus_t launch_by_mix_dtype(
    const DeepseekV4MHCHeadCollapseInfo &info,
    void *y,
    const void *x,
    const void *mixes,
    const void *base,
    const void *scale,
    cudaStream_t stream) {
    if (info.mixes_dtype == INFINI_DTYPE_F16) {
        return launch_typed<OutT, half>(info, y, x, mixes, base, scale, stream);
    }
    if (info.mixes_dtype == INFINI_DTYPE_BF16) {
        return launch_typed<OutT, __nv_bfloat16>(info, y, x, mixes, base, scale, stream);
    }
    if (info.mixes_dtype == INFINI_DTYPE_F32) {
        return launch_typed<OutT, float>(info, y, x, mixes, base, scale, stream);
    }
    return INFINI_STATUS_BAD_TENSOR_DTYPE;
}

} // namespace

infiniStatus_t HeadCollapseDescriptor::create(
    infiniopHandle_t handle,
    HeadCollapseDescriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t mixes_desc,
    infiniopTensorDescriptor_t base_desc,
    infiniopTensorDescriptor_t scale_desc,
    float epsilon) {
    auto result = DeepseekV4MHCHeadCollapseInfo::create(y_desc, x_desc, mixes_desc, base_desc, scale_desc, epsilon);
    CHECK_RESULT(result);
    *desc_ptr = new HeadCollapseDescriptor(
        new Opaque{reinterpret_cast<device::nvidia::Handle *>(handle)->internal()},
        result.take(),
        0,
        handle->device,
        handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t HeadCollapseDescriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *y,
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
        return launch_by_mix_dtype<half>(_info, y, x, mixes, base, scale, cuda_stream);
    }
    if (_info.dtype == INFINI_DTYPE_BF16) {
        return launch_by_mix_dtype<__nv_bfloat16>(_info, y, x, mixes, base, scale, cuda_stream);
    }
    if (_info.dtype == INFINI_DTYPE_F32) {
        return launch_by_mix_dtype<float>(_info, y, x, mixes, base, scale, cuda_stream);
    }
    return INFINI_STATUS_BAD_TENSOR_DTYPE;
}

} // namespace op::deepseek_v4_mhc_head::nvidia

#endif
