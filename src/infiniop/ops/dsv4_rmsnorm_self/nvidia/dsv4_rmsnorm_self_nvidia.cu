#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "dsv4_rmsnorm_self_nvidia.cuh"

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

__device__ __forceinline__ float warpSum(float v) {
#if defined(ENABLE_HYGON_API)
    for (int o = 32; o > 0; o >>= 1) {
        v += __shfl_xor(v, o, 64);
    }
#else
    for (int o = 16; o > 0; o >>= 1) {
        v += __shfl_xor_sync(0xffffffff, v, o, 32);
    }
#endif
    return v;
}

template <typename T>
__global__ void fallbackKernel(const T *x, T *y, size_t rows, size_t cols, float eps) {
    size_t row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    __shared__ float red[BLOCK];
    float local = 0.0f;
    size_t base = row * cols;
    for (size_t i = threadIdx.x; i < cols; i += blockDim.x) {
        float v = toFloat(x[base + i]);
        local += v * v;
    }
    red[threadIdx.x] = local;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            red[threadIdx.x] += red[threadIdx.x + stride];
        }
        __syncthreads();
    }
    float rrms = rsqrtf(red[0] / static_cast<float>(cols) + eps);
    for (size_t i = threadIdx.x; i < cols; i += blockDim.x) {
        y[base + i] = fromFloat<T>(toFloat(x[base + i]) * rrms);
    }
}

#if defined(ENABLE_HYGON_API)
template <int BS, int EPT>
__global__ void hygonBf16Kernel(const __nv_bfloat16 *x, __nv_bfloat16 *y, int rows, int cols, float eps) {
    int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    int tid = threadIdx.x;
    int off = row * cols;
    __shared__ float s_rrms;
    __shared__ float s_red[BS / 64 + 1];
    float vals[EPT];
    float ss = 0.0f;
#pragma unroll
    for (int i = 0; i < EPT; ++i) {
        int idx = tid + i * BS;
        if (idx < cols) {
            float v = static_cast<float>(x[off + idx]);
            vals[i] = v;
            ss += v * v;
        }
    }
    int lane = tid & 63;
    int wid = tid >> 6;
    ss = warpSum(ss);
    if (lane == 0) {
        s_red[wid] = ss;
    }
    __syncthreads();
    if (wid == 0) {
        ss = lane < (BS >> 6) ? s_red[lane] : 0.0f;
        ss = warpSum(ss);
        if (lane == 0) {
            s_rrms = rsqrtf(ss / static_cast<float>(cols) + eps);
        }
    }
    __syncthreads();
    float rrms = s_rrms;
#pragma unroll
    for (int i = 0; i < EPT; ++i) {
        int idx = tid + i * BS;
        if (idx < cols) {
            y[off + idx] = static_cast<__nv_bfloat16>(vals[i] * rrms);
        }
    }
}
#endif

template <typename T>
infiniStatus_t launchFallback(const op::dsv4_rmsnorm_self::Info &info, void *y, const void *x, float eps, cudaStream_t stream) {
    fallbackKernel<T><<<info.rows, BLOCK, 0, stream>>>(static_cast<const T *>(x), static_cast<T *>(y), info.rows, info.cols, eps);
    return INFINI_STATUS_SUCCESS;
}
} // namespace

namespace op::dsv4_rmsnorm_self::nvidia {

infiniStatus_t Descriptor::create(infiniopHandle_t handle, Descriptor **desc_ptr, infiniopTensorDescriptor_t y_desc, infiniopTensorDescriptor_t x_desc, float epsilon) {
    Info info;
    CHECK_STATUS(createInfo(&info, y_desc, x_desc));
    *desc_ptr = new Descriptor(info, epsilon, 0, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(void *, size_t, void *y, const void *x, void *stream) const {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    switch (_info.dtype) {
    case INFINI_DTYPE_F16:
        return launchFallback<half>(_info, y, x, _epsilon, s);
    case INFINI_DTYPE_BF16:
#if defined(ENABLE_HYGON_API)
        hygonBf16Kernel<256, 16><<<_info.rows, 256, 0, s>>>(static_cast<const __nv_bfloat16 *>(x), static_cast<__nv_bfloat16 *>(y), static_cast<int>(_info.rows), static_cast<int>(_info.cols), _epsilon);
        return INFINI_STATUS_SUCCESS;
#else
        return launchFallback<__nv_bfloat16>(_info, y, x, _epsilon, s);
#endif
    case INFINI_DTYPE_F32:
        return launchFallback<float>(_info, y, x, _epsilon, s);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace op::dsv4_rmsnorm_self::nvidia
