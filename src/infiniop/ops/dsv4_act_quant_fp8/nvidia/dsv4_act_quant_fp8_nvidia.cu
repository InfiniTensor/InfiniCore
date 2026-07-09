
#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "dsv4_act_quant_fp8_nvidia.cuh"
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
template <typename T>
__device__ float toFloat(T v) { return static_cast<float>(v); }
#if defined(ENABLE_HYGON_API)
using fp8_out_t = uint8_t;
__device__ __forceinline__ uint8_t f32ToE4m3fn(float v) {
    if (isnan(v)) {
        return 0x7f;
    }
    int sign = v < 0.0f;
    float a = fminf(fabsf(v), 448.0f);
    if (a == 0.0f) {
        return static_cast<uint8_t>(sign << 7);
    }
    int exp_field;
    int mant;
    if (a < 0.015625f) {
        exp_field = 0;
        mant = static_cast<int>(nearbyintf(a * 512.0f));
        mant = max(0, min(7, mant));
    } else {
        int exp_unbiased = static_cast<int>(floorf(log2f(a)));
        float base = exp2f(static_cast<float>(exp_unbiased));
        exp_field = exp_unbiased + 7;
        mant = static_cast<int>(nearbyintf((a / base - 1.0f) * 8.0f));
        if (mant == 8) {
            mant = 0;
            exp_field += 1;
        }
        if (exp_field >= 15) {
            exp_field = 15;
            mant = min(6, mant);
        }
    }
    return static_cast<uint8_t>((sign << 7) | (exp_field << 3) | mant);
}
__device__ __forceinline__ fp8_out_t toFp8(float v) { return f32ToE4m3fn(v); }
#else
using fp8_out_t = cuda_fp8_e4m3;
__device__ __forceinline__ fp8_out_t toFp8(float v) { return cuda_fp8_e4m3(v); }
#endif

template <int BS, typename T>
__global__ void kernel(const T *x, fp8_out_t *xq, float *scale, int rows, int cols, float fp8_max) {
    int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    int tid = threadIdx.x, off = row * cols;
    float lmax = 0.f;
    for (int i = tid; i < cols; i += BS) {
        lmax = fmaxf(lmax, fabsf(toFloat(x[off + i])));
    }
    lmax = warpMax(lmax);
#if defined(ENABLE_HYGON_API)
    constexpr int WARP = 64;
#else
    constexpr int WARP = 32;
#endif
    constexpr int NWARP = (BS + WARP - 1) / WARP;
    __shared__ float warp_max[NWARP];
    if ((tid & (WARP - 1)) == 0) {
        warp_max[tid / WARP] = lmax;
    }
    __syncthreads();
    if (tid < WARP) {
        lmax = tid < NWARP ? warp_max[tid] : 0.f;
        lmax = warpMax(lmax);
    }
    __shared__ float inv;
    if (tid == 0) {
        float am = fmaxf(lmax, 1e-12f);
        inv = fp8_max / am;
        scale[row] = am / fp8_max;
    }
    __syncthreads();
    float s = inv;
    for (int i = tid; i < cols; i += BS) {
        xq[off + i] = toFp8(toFloat(x[off + i]) * s);
    }
}
template <typename T>
infiniStatus_t launch(const op::dsv4_act_quant_fp8::Info &info, void *xq, void *scale, const void *x, cudaStream_t st) {
    kernel<128, T><<<info.rows, 128, 0, st>>>(static_cast<const T *>(x), static_cast<fp8_out_t *>(xq), static_cast<float *>(scale), (int)info.rows, (int)info.cols, info.fp8_max);
    return INFINI_STATUS_SUCCESS;
}
} // namespace
namespace op::dsv4_act_quant_fp8::nvidia {
infiniStatus_t Descriptor::create(infiniopHandle_t handle, Descriptor **desc_ptr, infiniopTensorDescriptor_t xq_desc, infiniopTensorDescriptor_t scale_desc, infiniopTensorDescriptor_t x_desc, float fp8_max) {
    Info info;
    CHECK_STATUS(createInfo(&info, xq_desc, scale_desc, x_desc, fp8_max));
    *desc_ptr = new Descriptor(info, 0, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}
infiniStatus_t Descriptor::calculate(void *, size_t, void *xq, void *scale, const void *x, void *stream) const {
    cudaStream_t st = (cudaStream_t)stream;
    switch (_info.dtype) {
    case INFINI_DTYPE_BF16:
        return launch<__nv_bfloat16>(_info, xq, scale, x, st);
    case INFINI_DTYPE_F16:
        return launch<half>(_info, xq, scale, x, st);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}
} // namespace op::dsv4_act_quant_fp8::nvidia
