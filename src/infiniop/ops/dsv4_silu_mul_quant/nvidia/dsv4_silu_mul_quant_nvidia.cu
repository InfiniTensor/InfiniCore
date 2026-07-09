
#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "dsv4_silu_mul_quant_nvidia.cuh"
namespace {
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
template <int BS, int EPT, typename T>
__global__ void kernel(const T *g, const T *u, int8_t *q, float *s, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    int tid = threadIdx.x, off = row * cols;
    __shared__ float s_inv, s_red[BS / 64 + 1];
    float vals[EPT], lmax = 0.f;
#pragma unroll
    for (int i = 0; i < EPT; i++) {
        int idx = tid + i * BS;
        if (idx < cols) {
            float gv = toFloat(g[off + idx]), uv = toFloat(u[off + idx]);
            float h = (1.f / (1.f + expf(-gv))) * gv * uv;
            vals[i] = h;
            lmax = fmaxf(lmax, fabsf(h));
        }
    }
    int lane = tid & 63, wid = tid >> 6;
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
            int qi = (int)roundHe(vals[i] * inv);
            qi = max(-128, min(127, qi));
            q[off + idx] = (int8_t)qi;
        }
    }
}
template <typename T>
infiniStatus_t launch(const op::dsv4_silu_mul_quant::Info &info, void *q, void *scale, const void *gate, const void *up, cudaStream_t st) {
    kernel<256, 8, T><<<info.rows, 256, 0, st>>>(static_cast<const T *>(gate), static_cast<const T *>(up), static_cast<int8_t *>(q), static_cast<float *>(scale), (int)info.rows, (int)info.cols);
    return INFINI_STATUS_SUCCESS;
}
} // namespace
namespace op::dsv4_silu_mul_quant::nvidia {
infiniStatus_t Descriptor::create(infiniopHandle_t handle, Descriptor **desc_ptr, infiniopTensorDescriptor_t q_desc, infiniopTensorDescriptor_t scale_desc, infiniopTensorDescriptor_t gate_desc, infiniopTensorDescriptor_t up_desc) {
    Info info;
    CHECK_STATUS(createInfo(&info, q_desc, scale_desc, gate_desc, up_desc));
    *desc_ptr = new Descriptor(info, 0, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}
infiniStatus_t Descriptor::calculate(void *, size_t, void *q, void *scale, const void *gate, const void *up, void *stream) const {
    cudaStream_t st = (cudaStream_t)stream;
    switch (_info.dtype) {
    case INFINI_DTYPE_BF16:
        return launch<__nv_bfloat16>(_info, q, scale, gate, up, st);
    case INFINI_DTYPE_F16:
        return launch<half>(_info, q, scale, gate, up, st);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}
} // namespace op::dsv4_silu_mul_quant::nvidia
