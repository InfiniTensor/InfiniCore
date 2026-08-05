#include "mxfp4_dequantize_nvidia.cuh"

#include "../../../devices/nvidia/nvidia_handle.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace op::mxfp4_dequantize::nvidia {
namespace {

__device__ __forceinline__ float decode_e2m1(uint8_t value) {
    float magnitude;
    switch (value & 0x7) {
    case 0:
        magnitude = 0.0f;
        break;
    case 1:
        magnitude = 0.5f;
        break;
    case 2:
        magnitude = 1.0f;
        break;
    case 3:
        magnitude = 1.5f;
        break;
    case 4:
        magnitude = 2.0f;
        break;
    case 5:
        magnitude = 3.0f;
        break;
    case 6:
        magnitude = 4.0f;
        break;
    default:
        magnitude = 6.0f;
        break;
    }
    return value & 0x8 ? -magnitude : magnitude;
}

template <typename T>
__device__ __forceinline__ T cast_output(float value);

template <>
__device__ __forceinline__ half cast_output(float value) {
    return __float2half_rn(value);
}

template <>
__device__ __forceinline__ __nv_bfloat16 cast_output(float value) {
    return __float2bfloat16_rn(value);
}

template <>
__device__ __forceinline__ float cast_output(float value) {
    return value;
}

template <typename T>
INFINIOP_CUDA_KERNEL dequantize_kernel(
    T *out,
    const uint8_t *packed,
    const uint8_t *scales,
    size_t packed_numel,
    size_t packed_width,
    size_t scales_width) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= packed_numel) {
        return;
    }

    const size_t row = index / packed_width;
    const size_t packed_col = index - row * packed_width;
    const int exponent = static_cast<int>(scales[row * scales_width + packed_col / 16]) - 127;
    const uint8_t byte = packed[index];
    const size_t out_index = index * 2;
    out[out_index] = cast_output<T>(ldexpf(decode_e2m1(byte & 0xf), exponent));
    out[out_index + 1] = cast_output<T>(ldexpf(decode_e2m1(byte >> 4), exponent));
}

template <typename T>
void launch(T *out,
            const uint8_t *packed,
            const uint8_t *scales,
            const Mxfp4DequantizeInfo &info,
            cudaStream_t stream) {
    constexpr size_t block_size = 256;
    const size_t grid_size = (info.packed_numel + block_size - 1) / block_size;
    dequantize_kernel<<<grid_size, block_size, 0, stream>>>(
        out, packed, scales, info.packed_numel,
        info.logical_width / 2, info.logical_width / 32);
}

} // namespace

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

Descriptor::~Descriptor() { delete _opaque; }

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t packed_desc,
    infiniopTensorDescriptor_t scales_desc) {
    auto info = Mxfp4DequantizeInfo::create(out_desc, packed_desc, scales_desc);
    CHECK_RESULT(info);
    auto nvidia_handle = reinterpret_cast<device::nvidia::Handle *>(handle);
    *desc_ptr = new Descriptor(
        new Opaque{nvidia_handle->internal()}, info.take(), handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *, size_t, void *out,
    const void *packed, const void *scales, void *stream) const {
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    auto packed_ptr = reinterpret_cast<const uint8_t *>(packed);
    auto scales_ptr = reinterpret_cast<const uint8_t *>(scales);
    switch (_info.output_dtype) {
    case INFINI_DTYPE_F16:
        launch(reinterpret_cast<half *>(out), packed_ptr, scales_ptr, _info, cuda_stream);
        return INFINI_STATUS_SUCCESS;
    case INFINI_DTYPE_BF16:
        launch(reinterpret_cast<__nv_bfloat16 *>(out), packed_ptr, scales_ptr, _info, cuda_stream);
        return INFINI_STATUS_SUCCESS;
    case INFINI_DTYPE_F32:
        launch(reinterpret_cast<float *>(out), packed_ptr, scales_ptr, _info, cuda_stream);
        return INFINI_STATUS_SUCCESS;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace op::mxfp4_dequantize::nvidia
