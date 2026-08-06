#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "../../../tensor.h"
#include "scaled_mm_w4a8_nvidia.cuh"

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {
constexpr size_t THREADS = 256;

__device__ __forceinline__ int unpack_signed_int4(uint8_t packed, bool high) {
    const int value = high ? (packed >> 4) : (packed & 0x0f);
    return value < 8 ? value : value - 16;
}

template <typename T>
__device__ __forceinline__ T from_float(float value);

template <>
__device__ __forceinline__ half from_float<half>(float value) {
    return __float2half_rn(value);
}

template <>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float value) {
    return __float2bfloat16(value);
}

template <typename T>
INFINIOP_CUDA_KERNEL scaledMmW4A8Kernel(
    T *__restrict__ out,
    const int8_t *__restrict__ a,
    const uint8_t *__restrict__ b,
    const float *__restrict__ a_scales,
    const float *__restrict__ b_scales,
    const T *__restrict__ bias,
    size_t m,
    size_t n,
    size_t k,
    bool trans_weight) {
    const size_t row = blockIdx.y;
    const size_t column = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m || column >= n) {
        return;
    }

    int32_t acc = 0;
    for (size_t inner = 0; inner < k; ++inner) {
        uint8_t packed;
        int weight;
        if (trans_weight) {
            packed = b[column * (k / 2) + inner / 2];
            weight = unpack_signed_int4(packed, (inner & 1) != 0);
        } else {
            packed = b[inner * (n / 2) + column / 2];
            weight = unpack_signed_int4(packed, (column & 1) != 0);
        }
        acc += static_cast<int32_t>(a[row * k + inner]) * weight;
    }
    float value = static_cast<float>(acc) * a_scales[row] * b_scales[column];
    if (bias != nullptr) {
        value += static_cast<float>(bias[column]);
    }
    out[row * n + column] = from_float<T>(value);
}

INFINIOP_CUDA_KERNEL prepareGlmAwqWeightKernel(
    uint8_t *__restrict__ qweight,
    const uint8_t *__restrict__ checkpoint_weight,
    size_t k, size_t n) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = k * (n / 2);
    if (index >= total) {
        return;
    }
    const size_t inner = index / (n / 2);
    const size_t output_pair = index % (n / 2);
    uint8_t packed = 0;
    for (size_t lane = 0; lane < 2; ++lane) {
        const size_t output = output_pair * 2 + lane;
        const size_t packed_inner = (inner / 32) * 16 + inner % 16;
        const uint8_t source = checkpoint_weight[output * (k / 2) + packed_inner];
        const uint8_t signed_nibble = ((inner % 32) >= 16 ? source >> 4 : source) & 0x0f;
        const uint8_t unsigned_nibble = (signed_nibble + 8) & 0x0f;
        packed |= unsigned_nibble << (lane * 4);
    }
    qweight[index] = packed;
}

INFINIOP_CUDA_KERNEL prepareGlmAwqScalesKernel(
    __nv_bfloat16 *__restrict__ scales,
    const float *__restrict__ channel_scales,
    size_t groups, size_t n) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = groups * n;
    if (index >= total) {
        return;
    }
    scales[index] = __float2bfloat16(channel_scales[index % n] * 18.0f);
}
} // namespace

namespace op::scaled_mm_w4a8::nvidia {

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc,
    infiniopTensorDescriptor_t a_scales_desc,
    infiniopTensorDescriptor_t b_scales_desc,
    infiniopTensorDescriptor_t bias_desc,
    bool trans_weight) {
    const auto out_shape = out_desc->shape();
    const auto a_shape = a_desc->shape();
    const auto b_shape = b_desc->shape();
    const auto as_shape = a_scales_desc->shape();
    const auto bs_shape = b_scales_desc->shape();
    CHECK_OR_RETURN(out_shape.size() == 2 && a_shape.size() == 2 && b_shape.size() == 2
                        && as_shape.size() == 2 && bs_shape.size() == 2,
                    INFINI_STATUS_BAD_TENSOR_SHAPE);
    const size_t m = a_shape[0];
    const size_t k = a_shape[1];
    const size_t n = trans_weight ? b_shape[0] : b_shape[1] * 2;
    CHECK_OR_RETURN((k % 2) == 0 && (n % 2) == 0
                        && out_shape[0] == m && out_shape[1] == n
                        && ((!trans_weight && b_shape[0] == k)
                            || (trans_weight && b_shape[1] * 2 == k))
                        && as_shape[0] == m && as_shape[1] == 1
                        && bs_shape[0] == n && bs_shape[1] == 1,
                    INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(out_desc->isContiguous() && a_desc->isContiguous()
                        && b_desc->isContiguous() && a_scales_desc->isContiguous()
                        && b_scales_desc->isContiguous()
                        && (bias_desc == nullptr || bias_desc->isContiguous()),
                    INFINI_STATUS_BAD_TENSOR_STRIDES);
    CHECK_OR_RETURN(a_desc->dtype() == INFINI_DTYPE_I8 && b_desc->dtype() == INFINI_DTYPE_I8
                        && a_scales_desc->dtype() == INFINI_DTYPE_F32
                        && b_scales_desc->dtype() == INFINI_DTYPE_F32
                        && (out_desc->dtype() == INFINI_DTYPE_F16 || out_desc->dtype() == INFINI_DTYPE_BF16)
                        && (bias_desc == nullptr || (bias_desc->dtype() == out_desc->dtype() && bias_desc->shape().size() == 1 && bias_desc->shape()[0] == n)),
                    INFINI_STATUS_BAD_TENSOR_DTYPE);
    *desc_ptr = new Descriptor(m, n, k, out_desc->dtype(), trans_weight,
                               bias_desc != nullptr, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *out, const void *a, const void *b, const void *a_scales,
    const void *b_scales, const void *bias, void *stream) const {
    const dim3 grid(static_cast<unsigned int>((_n + THREADS - 1) / THREADS),
                    static_cast<unsigned int>(_m));
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    if (_out_dtype == INFINI_DTYPE_F16) {
        scaledMmW4A8Kernel<half><<<grid, THREADS, 0, cuda_stream>>>(
            static_cast<half *>(out), static_cast<const int8_t *>(a),
            static_cast<const uint8_t *>(b), static_cast<const float *>(a_scales),
            static_cast<const float *>(b_scales), static_cast<const half *>(bias),
            _m, _n, _k, _trans_weight);
    } else {
        scaledMmW4A8Kernel<__nv_bfloat16><<<grid, THREADS, 0, cuda_stream>>>(
            static_cast<__nv_bfloat16 *>(out), static_cast<const int8_t *>(a),
            static_cast<const uint8_t *>(b), static_cast<const float *>(a_scales),
            static_cast<const float *>(b_scales), static_cast<const __nv_bfloat16 *>(bias),
            _m, _n, _k, _trans_weight);
    }
    return cudaGetLastError() == cudaSuccess ? INFINI_STATUS_SUCCESS : INFINI_STATUS_INTERNAL_ERROR;
}

infiniStatus_t prepareGlmW4A16Awq(
    infiniopTensorDescriptor_t qweight_desc,
    infiniopTensorDescriptor_t qzeros_desc,
    infiniopTensorDescriptor_t scales_desc,
    infiniopTensorDescriptor_t checkpoint_weight_desc,
    infiniopTensorDescriptor_t channel_scales_desc,
    void *qweight, void *qzeros, void *scales,
    const void *checkpoint_weight, const void *channel_scales, void *stream) {
    const auto checkpoint_shape = checkpoint_weight_desc->shape();
    const auto qweight_shape = qweight_desc->shape();
    const auto qzeros_shape = qzeros_desc->shape();
    const auto scales_shape = scales_desc->shape();
    const auto channel_shape = channel_scales_desc->shape();
    CHECK_OR_RETURN(checkpoint_shape.size() == 2 && qweight_shape.size() == 2
                        && qzeros_shape.size() == 2 && scales_shape.size() == 2,
                    INFINI_STATUS_BAD_TENSOR_SHAPE);
    const size_t n = checkpoint_shape[0];
    const size_t k = checkpoint_shape[1] * 2;
    CHECK_OR_RETURN((k % 256) == 0 && (n % 2) == 0
                        && qweight_shape[0] == k && qweight_shape[1] == n / 2
                        && qzeros_shape[0] == k / 64 && qzeros_shape[1] == n / 2
                        && scales_shape[0] == k / 64 && scales_shape[1] == n
                        && ((channel_shape.size() == 2
                             && channel_shape[0] == n && channel_shape[1] == 1)
                            || (channel_shape.size() == 2
                                && channel_shape[0] == 1 && channel_shape[1] == n)),
                    INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(qweight_desc->dtype() == INFINI_DTYPE_I8
                        && qzeros_desc->dtype() == INFINI_DTYPE_I8
                        && scales_desc->dtype() == INFINI_DTYPE_BF16
                        && checkpoint_weight_desc->dtype() == INFINI_DTYPE_I8
                        && channel_scales_desc->dtype() == INFINI_DTYPE_F32,
                    INFINI_STATUS_BAD_TENSOR_DTYPE);
    CHECK_OR_RETURN(qweight_desc->isContiguous() && qzeros_desc->isContiguous()
                        && scales_desc->isContiguous()
                        && checkpoint_weight_desc->isContiguous()
                        && channel_scales_desc->isContiguous(),
                    INFINI_STATUS_BAD_TENSOR_STRIDES);
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    const size_t weight_elements = k * (n / 2);
    prepareGlmAwqWeightKernel<<<(weight_elements + THREADS - 1) / THREADS,
                                THREADS, 0, cuda_stream>>>(
        static_cast<uint8_t *>(qweight),
        static_cast<const uint8_t *>(checkpoint_weight), k, n);
    const auto memset_status = cudaMemsetAsync(qzeros, 0x88, (k / 64) * (n / 2),
                                               cuda_stream);
    if (memset_status != cudaSuccess) {
        return INFINI_STATUS_INTERNAL_ERROR;
    }
    const size_t scale_elements = (k / 64) * n;
    prepareGlmAwqScalesKernel<<<(scale_elements + THREADS - 1) / THREADS,
                                THREADS, 0, cuda_stream>>>(
        static_cast<__nv_bfloat16 *>(scales),
        static_cast<const float *>(channel_scales), k / 64, n);
    return cudaGetLastError() == cudaSuccess ? INFINI_STATUS_SUCCESS : INFINI_STATUS_INTERNAL_ERROR;
}

} // namespace op::scaled_mm_w4a8::nvidia
