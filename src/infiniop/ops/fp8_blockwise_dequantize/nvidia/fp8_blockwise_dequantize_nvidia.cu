#include "fp8_blockwise_dequantize_nvidia.cuh"

#include "../../../devices/nvidia/nvidia_handle.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace op::fp8_blockwise_dequantize::nvidia {
namespace {

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
    const uint8_t *q,
    const float *scales,
    size_t numel,
    size_t cols,
    size_t block_rows,
    size_t block_cols,
    size_t scales_cols) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numel) {
        return;
    }

    const size_t row = index / cols;
    const size_t col = index - row * cols;
    const size_t scale_index = (row / block_rows) * scales_cols + col / block_cols;
    out[index] = cast_output<T>(infiniopFp8E4m3Decode(q[index]) * scales[scale_index]);
}

template <typename T>
void launch(T *out,
            const uint8_t *q,
            const float *scales,
            const Fp8BlockwiseDequantizeInfo &info,
            cudaStream_t stream) {
    constexpr size_t block_size = 256;
    const size_t numel = info.rows * info.cols;
    const size_t grid_size = (numel + block_size - 1) / block_size;
    dequantize_kernel<<<grid_size, block_size, 0, stream>>>(
        out, q, scales, numel, info.cols,
        info.block_rows, info.block_cols, info.scales_cols);
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
    infiniopTensorDescriptor_t q_desc,
    infiniopTensorDescriptor_t scales_desc) {
    auto info = Fp8BlockwiseDequantizeInfo::create(out_desc, q_desc, scales_desc);
    CHECK_RESULT(info);
    auto nvidia_handle = reinterpret_cast<device::nvidia::Handle *>(handle);
    *desc_ptr = new Descriptor(
        new Opaque{nvidia_handle->internal()}, info.take(), handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *, size_t, void *out,
    const void *q, const void *scales, void *stream) const {
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    auto q_ptr = reinterpret_cast<const uint8_t *>(q);
    auto scales_ptr = reinterpret_cast<const float *>(scales);
    switch (_info.output_dtype) {
    case INFINI_DTYPE_F16:
        launch(reinterpret_cast<half *>(out), q_ptr, scales_ptr, _info, cuda_stream);
        return INFINI_STATUS_SUCCESS;
    case INFINI_DTYPE_BF16:
        launch(reinterpret_cast<__nv_bfloat16 *>(out), q_ptr, scales_ptr, _info, cuda_stream);
        return INFINI_STATUS_SUCCESS;
    case INFINI_DTYPE_F32:
        launch(reinterpret_cast<float *>(out), q_ptr, scales_ptr, _info, cuda_stream);
        return INFINI_STATUS_SUCCESS;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace op::fp8_blockwise_dequantize::nvidia
