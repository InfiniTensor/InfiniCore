#include "silu_and_mul_nvidia.cuh"

#include "../../../devices/nvidia/nvidia_handle.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <memory>
#include <type_traits>

namespace op::silu_and_mul::nvidia {

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

namespace {

template <typename T>
__device__ float to_float(T v) {
    if constexpr (std::is_same_v<T, half>) {
        return __half2float(v);
    } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
        return __bfloat162float(v);
    } else {
        return static_cast<float>(v);
    }
}

template <typename T>
__device__ T from_float(float v) {
    if constexpr (std::is_same_v<T, half>) {
        return __float2half(v);
    } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
        return __float2bfloat16(v);
    } else {
        return static_cast<T>(v);
    }
}

template <typename T>
INFINIOP_CUDA_KERNEL siluAndMulKernel(T *y, const T *x, size_t n, size_t hidden) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (; idx < n; idx += stride) {
        size_t row = idx / hidden;
        size_t col = idx - row * hidden;
        const T *row_x = x + row * hidden * 2;
        float gate = to_float(row_x[col]);
        float up = to_float(row_x[col + hidden]);
        float silu = gate / (1.0f + expf(-gate));
        y[idx] = from_float<T>(silu * up);
    }
}

template <typename T>
infiniStatus_t launch(const SiluAndMulInfo &info, void *y, const void *x, void *stream) {
    size_t n = info.batch_size * info.out_hidden_dim;
    if (n == 0) {
        return INFINI_STATUS_SUCCESS;
    }
    constexpr int block = 256;
    int grid = static_cast<int>((n + block - 1) / block);
    grid = grid > 65535 ? 65535 : grid;
    siluAndMulKernel<T><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
        reinterpret_cast<T *>(y), reinterpret_cast<const T *>(x), n, info.out_hidden_dim);
    CHECK_CUDA(cudaGetLastError());
    return INFINI_STATUS_SUCCESS;
}

} // namespace

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc) {

    if (!desc_ptr) {
        return INFINI_STATUS_BAD_PARAM;
    }

    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);
    auto dtype = y_desc->dtype();
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);
    if (x_desc->dtype() != dtype) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    auto result = SiluAndMulInfo::create(y_desc, x_desc);
    CHECK_RESULT(result);

    *desc_ptr = new Descriptor(
        new Opaque{handle->internal()},
        result.take(),
        0,
        handle->device,
        handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size,
    void *y,
    const void *x,
    void *stream) const {
    switch (_info.dtype) {
    case INFINI_DTYPE_F16:
        return launch<half>(_info, y, x, stream);
    case INFINI_DTYPE_F32:
        return launch<float>(_info, y, x, stream);
    case INFINI_DTYPE_BF16:
        return launch<cuda_bfloat16>(_info, y, x, stream);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace op::silu_and_mul::nvidia
