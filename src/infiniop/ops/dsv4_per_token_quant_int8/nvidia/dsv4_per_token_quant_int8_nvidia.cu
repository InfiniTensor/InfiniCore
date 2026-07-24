#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "dsv4_per_token_quant_int8_nvidia.cuh"

namespace {
constexpr int BLOCK = 256;

template <typename T>
__device__ float toFloat(T v) {
    return static_cast<float>(v);
}

__device__ __forceinline__ float warpMax(float v) {
#if defined(ENABLE_HYGON_API)
    for (int o = 32; o > 0; o >>= 1) {
        v = fmaxf(v, __shfl_xor(v, o, 64));
    }
#else
    for (int o = 16; o > 0; o >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o, 32));
    }
#endif
    return v;
}

__device__ __forceinline__ float roundHe(float v) {
#if defined(ENABLE_HYGON_API)
    return __builtin_rintf(v);
#else
    return nearbyintf(v);
#endif
}

template <typename T>
__global__ void fallbackKernel(const T *__restrict__ x, int8_t *__restrict__ q, float *__restrict__ scale, size_t rows, size_t cols) {
    size_t row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    __shared__ float red[BLOCK];
    float local = 0.0f;
    size_t base = row * cols;
    for (size_t i = threadIdx.x; i < cols; i += blockDim.x) {
        local = fmaxf(local, fabsf(toFloat(x[base + i])));
    }
    red[threadIdx.x] = local;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    float amax = fmaxf(red[0], 1e-10f);
    float inv = 127.0f / amax;
    if (threadIdx.x == 0) {
        scale[row] = amax / 127.0f;
    }
    for (size_t i = threadIdx.x; i < cols; i += blockDim.x) {
        int qi = static_cast<int>(nearbyintf(toFloat(x[base + i]) * inv));
        qi = max(-128, min(127, qi));
        q[base + i] = static_cast<int8_t>(qi);
    }
}

#if defined(ENABLE_HYGON_API)
template <int BS, int EPT>
__global__ void hygonBf16Kernel(const __nv_bfloat16 *__restrict__ x, int8_t *__restrict__ q, float *__restrict__ scale, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    int tid = threadIdx.x;
    int off = row * cols;
    __shared__ float s_inv;
    __shared__ float s_red[BS / 64 + 1];
    float vals[EPT];
    float local_max = 0.0f;
#pragma unroll
    for (int i = 0; i < EPT; ++i) {
        int idx = tid + i * BS;
        if (idx < cols) {
            float v = static_cast<float>(x[off + idx]);
            vals[i] = v;
            local_max = fmaxf(local_max, fabsf(v));
        }
    }
    int lane = tid & 63;
    int wid = tid >> 6;
    local_max = warpMax(local_max);
    if (lane == 0) {
        s_red[wid] = local_max;
    }
    __syncthreads();
    if (wid == 0) {
        local_max = lane < (BS >> 6) ? s_red[lane] : 0.0f;
        local_max = warpMax(local_max);
        if (lane == 0) {
            float amax = fmaxf(local_max, 1e-10f);
            s_inv = 127.0f / amax;
            scale[row] = amax / 127.0f;
        }
    }
    __syncthreads();
    float inv = s_inv;
#pragma unroll
    for (int i = 0; i < EPT; ++i) {
        int idx = tid + i * BS;
        if (idx < cols) {
            int qi = static_cast<int>(roundHe(vals[i] * inv));
            qi = max(-128, min(127, qi));
            q[off + idx] = static_cast<int8_t>(qi);
        }
    }
}
#endif

template <typename T>
infiniStatus_t launchFallback(const op::dsv4_per_token_quant_int8::Info &info, void *q, void *scale, const void *x, cudaStream_t stream) {
    fallbackKernel<T><<<info.rows, BLOCK, 0, stream>>>(static_cast<const T *>(x), static_cast<int8_t *>(q), static_cast<float *>(scale), info.rows, info.cols);
    return INFINI_STATUS_SUCCESS;
}
} // namespace

namespace op::dsv4_per_token_quant_int8::nvidia {

infiniStatus_t Descriptor::create(infiniopHandle_t handle, Descriptor **desc_ptr, infiniopTensorDescriptor_t q_desc, infiniopTensorDescriptor_t scale_desc, infiniopTensorDescriptor_t x_desc) {
    Info info;
    CHECK_STATUS(createInfo(&info, q_desc, scale_desc, x_desc));
    *desc_ptr = new Descriptor(info, 0, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(void *, size_t, void *q, void *scale, const void *x, void *stream) const {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    switch (_info.dtype) {
    case INFINI_DTYPE_F16:
        return launchFallback<half>(_info, q, scale, x, s);
    case INFINI_DTYPE_BF16:
#if defined(ENABLE_HYGON_API)
        hygonBf16Kernel<256, 16><<<_info.rows, 256, 0, s>>>(static_cast<const __nv_bfloat16 *>(x), static_cast<int8_t *>(q), static_cast<float *>(scale), static_cast<int>(_info.rows), static_cast<int>(_info.cols));
        return INFINI_STATUS_SUCCESS;
#else
        return launchFallback<__nv_bfloat16>(_info, q, scale, x, s);
#endif
    case INFINI_DTYPE_F32:
        return launchFallback<float>(_info, q, scale, x, s);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace op::dsv4_per_token_quant_int8::nvidia
