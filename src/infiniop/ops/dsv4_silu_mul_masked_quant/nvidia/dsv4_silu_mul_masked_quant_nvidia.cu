#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "dsv4_silu_mul_masked_quant_nvidia.cuh"

namespace {
constexpr int BLOCK = 256;
constexpr int EPT = 8;

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

template <typename T, int BS, int ELEMS_PER_THREAD>
__global__ void siluMulMaskedQuantKernel(const T *__restrict__ gate,
                                         const T *__restrict__ up,
                                         const int *__restrict__ mask,
                                         int8_t *__restrict__ q,
                                         float *__restrict__ scale,
                                         int rows,
                                         int cols) {
    int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    if (mask != nullptr && mask[row] == 0) {
        return;
    }

    int tid = threadIdx.x;
    int off = row * cols;
    __shared__ float s_inv;
    __shared__ float s_red[BS / 64 + 1];
    float vals[ELEMS_PER_THREAD];
    float local_max = 0.0f;

#pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
        int idx = tid + i * BS;
        if (idx < cols) {
            float gv = toFloat(gate[off + idx]);
            float uv = toFloat(up[off + idx]);
            float h = (1.0f / (1.0f + expf(-gv))) * gv * uv;
            vals[i] = h;
            local_max = fmaxf(local_max, fabsf(h));
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
    for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
        int idx = tid + i * BS;
        if (idx < cols) {
            int qi = static_cast<int>(roundHe(vals[i] * inv));
            qi = max(-128, min(127, qi));
            q[off + idx] = static_cast<int8_t>(qi);
        }
    }
}

template <typename T>
infiniStatus_t launch(const op::dsv4_silu_mul_masked_quant::Info &info, void *q, void *scale, const void *gate, const void *up, const void *mask, cudaStream_t stream) {
    siluMulMaskedQuantKernel<T, BLOCK, EPT><<<static_cast<unsigned int>(info.rows), BLOCK, 0, stream>>>(
        static_cast<const T *>(gate),
        static_cast<const T *>(up),
        static_cast<const int *>(mask),
        static_cast<int8_t *>(q),
        static_cast<float *>(scale),
        static_cast<int>(info.rows),
        static_cast<int>(info.cols));
    return INFINI_STATUS_SUCCESS;
}
} // namespace

namespace op::dsv4_silu_mul_masked_quant::nvidia {

infiniStatus_t Descriptor::create(infiniopHandle_t handle, Descriptor **desc_ptr, infiniopTensorDescriptor_t q_desc, infiniopTensorDescriptor_t scale_desc, infiniopTensorDescriptor_t gate_desc, infiniopTensorDescriptor_t up_desc, infiniopTensorDescriptor_t mask_desc) {
    Info info;
    CHECK_STATUS(createInfo(&info, q_desc, scale_desc, gate_desc, up_desc, mask_desc));
    *desc_ptr = new Descriptor(info, 0, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(void *, size_t, void *q, void *scale, const void *gate, const void *up, const void *mask, void *stream) const {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    switch (_info.dtype) {
    case INFINI_DTYPE_F16:
        return launch<half>(_info, q, scale, gate, up, mask, s);
    case INFINI_DTYPE_BF16:
        return launch<__nv_bfloat16>(_info, q, scale, gate, up, mask, s);
    case INFINI_DTYPE_F32:
        return launch<float>(_info, q, scale, gate, up, mask, s);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace op::dsv4_silu_mul_masked_quant::nvidia
