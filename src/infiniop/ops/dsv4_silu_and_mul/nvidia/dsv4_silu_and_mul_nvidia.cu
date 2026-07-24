#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "dsv4_silu_and_mul_nvidia.cuh"

namespace {
constexpr int BLOCK = 256;

template <typename T>
__device__ float toFloat(T v) {
    return static_cast<float>(v);
}

template <typename T>
__device__ T fromFloat(float v) {
    return static_cast<T>(v);
}

template <typename T>
__global__ void fallbackKernel(const T *gate, const T *up, T *y, size_t total) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (; idx < total; idx += stride) {
        float g = toFloat(gate[idx]);
        float u = toFloat(up[idx]);
        y[idx] = fromFloat<T>((1.0f / (1.0f + expf(-g))) * g * u);
    }
}

#if defined(ENABLE_HYGON_API)
template <int BS, int EPT>
__global__ void hygonBf16Kernel(const __nv_bfloat16 *gate, const __nv_bfloat16 *up, __nv_bfloat16 *y, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    int tid = threadIdx.x;
    int off = row * cols;
#pragma unroll
    for (int i = 0; i < EPT; ++i) {
        int idx = tid + i * BS;
        if (idx < cols) {
            float g = static_cast<float>(gate[off + idx]);
            float u = static_cast<float>(up[off + idx]);
            y[off + idx] = static_cast<__nv_bfloat16>((1.0f / (1.0f + expf(-g))) * g * u);
        }
    }
}
#endif

template <typename T>
infiniStatus_t launchFallback(const op::dsv4_silu_and_mul::Info &info, void *y, const void *gate, const void *up, cudaStream_t stream) {
    size_t total = info.rows * info.cols;
    size_t blocks = (total + BLOCK - 1) / BLOCK;
    int grid = static_cast<int>(blocks < 4096 ? blocks : 4096);
    fallbackKernel<T><<<grid, BLOCK, 0, stream>>>(static_cast<const T *>(gate), static_cast<const T *>(up), static_cast<T *>(y), total);
    return INFINI_STATUS_SUCCESS;
}
} // namespace

namespace op::dsv4_silu_and_mul::nvidia {

infiniStatus_t Descriptor::create(infiniopHandle_t handle, Descriptor **desc_ptr, infiniopTensorDescriptor_t y_desc, infiniopTensorDescriptor_t gate_desc, infiniopTensorDescriptor_t up_desc) {
    Info info;
    CHECK_STATUS(createInfo(&info, y_desc, gate_desc, up_desc));
    *desc_ptr = new Descriptor(info, 0, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(void *, size_t, void *y, const void *gate, const void *up, void *stream) const {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    switch (_info.dtype) {
    case INFINI_DTYPE_F16:
        return launchFallback<half>(_info, y, gate, up, s);
    case INFINI_DTYPE_BF16:
#if defined(ENABLE_HYGON_API)
        hygonBf16Kernel<256, 8><<<_info.rows, 256, 0, s>>>(static_cast<const __nv_bfloat16 *>(gate), static_cast<const __nv_bfloat16 *>(up), static_cast<__nv_bfloat16 *>(y), static_cast<int>(_info.rows), static_cast<int>(_info.cols));
        return INFINI_STATUS_SUCCESS;
#else
        return launchFallback<__nv_bfloat16>(_info, y, gate, up, s);
#endif
    case INFINI_DTYPE_F32:
        return launchFallback<float>(_info, y, gate, up, s);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace op::dsv4_silu_and_mul::nvidia
