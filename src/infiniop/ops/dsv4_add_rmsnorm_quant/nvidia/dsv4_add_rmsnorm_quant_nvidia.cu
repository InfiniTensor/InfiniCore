
#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "dsv4_add_rmsnorm_quant_nvidia.cuh"
namespace {
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
__device__ float toFloat(T v) { return static_cast<float>(v); }
template <typename T>
__device__ T fromFloat(float v) { return static_cast<T>(v); }
template <int BS, int EPT, typename T>
__global__ void kernel(T *res, const T *x, const T *w, int8_t *q, float *s, int rows, int cols, float eps) {
    int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    int tid = threadIdx.x, off = row * cols;
    __shared__ float s_rrms, s_inv, s_red[BS / 64 + 1];
    float added[EPT], ss = 0.f;
#pragma unroll
    for (int i = 0; i < EPT; i++) {
        int idx = tid + i * BS;
        if (idx < cols) {
            float a = toFloat(res[off + idx]) + toFloat(x[off + idx]);
            res[off + idx] = fromFloat<T>(a);
            added[i] = a;
            ss += a * a;
        }
    }
    int lane = tid & 63, wid = tid >> 6;
    ss = warpSum(ss);
    if (lane == 0) {
        s_red[wid] = ss;
    }
    __syncthreads();
    if (wid == 0) {
        ss = (lane < (BS >> 6)) ? s_red[lane] : 0.f;
        ss = warpSum(ss);
        if (lane == 0) {
            s_rrms = rsqrtf(ss / (float)cols + eps);
        }
    }
    __syncthreads();
    float rrms = s_rrms, lmax = 0.f;
#pragma unroll
    for (int i = 0; i < EPT; i++) {
        int idx = tid + i * BS;
        if (idx < cols) {
            float nm = added[i] * rrms * toFloat(w[idx]);
            added[i] = nm;
            lmax = fmaxf(lmax, fabsf(nm));
        }
    }
    lmax = warpMax(lmax);
    if (lane == 0) {
        s_red[wid] = lmax;
    }
    __syncthreads();
    if (wid == 0) {
        lmax = (lane < (BS >> 6)) ? s_red[lane] : 0.f;
        lmax = warpMax(lmax);
        if (lane == 0) {
            float am = fmaxf(lmax, 1e-10f);
            s_inv = 127.f / am;
            s[row] = am / 127.f;
        }
    }
    __syncthreads();
    float inv = s_inv;
#pragma unroll
    for (int i = 0; i < EPT; i++) {
        int idx = tid + i * BS;
        if (idx < cols) {
            int qi = (int)roundHe(added[i] * inv);
            qi = max(-128, min(127, qi));
            q[off + idx] = (int8_t)qi;
        }
    }
}
template <typename T>
infiniStatus_t launch(const op::dsv4_add_rmsnorm_quant::Info &info, float eps, void *res, void *q, void *scale, const void *x, const void *weight, cudaStream_t st) {
    kernel<256, 16, T><<<info.rows, 256, 0, st>>>(static_cast<T *>(res), static_cast<const T *>(x), static_cast<const T *>(weight), static_cast<int8_t *>(q), static_cast<float *>(scale), (int)info.rows, (int)info.cols, eps);
    return INFINI_STATUS_SUCCESS;
}
} // namespace
namespace op::dsv4_add_rmsnorm_quant::nvidia {
infiniStatus_t Descriptor::create(infiniopHandle_t handle, Descriptor **desc_ptr, infiniopTensorDescriptor_t res_desc, infiniopTensorDescriptor_t q_desc, infiniopTensorDescriptor_t scale_desc, infiniopTensorDescriptor_t x_desc, infiniopTensorDescriptor_t weight_desc, float epsilon) {
    Info info;
    CHECK_STATUS(createInfo(&info, res_desc, q_desc, scale_desc, x_desc, weight_desc));
    *desc_ptr = new Descriptor(info, epsilon, 0, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}
infiniStatus_t Descriptor::calculate(void *, size_t, void *res, void *q, void *scale, const void *x, const void *weight, void *stream) const {
    cudaStream_t st = (cudaStream_t)stream;
    switch (_info.dtype) {
    case INFINI_DTYPE_BF16:
        return launch<__nv_bfloat16>(_info, _epsilon, res, q, scale, x, weight, st);
    case INFINI_DTYPE_F16:
        return launch<half>(_info, _epsilon, res, q, scale, x, weight, st);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}
} // namespace op::dsv4_add_rmsnorm_quant::nvidia
