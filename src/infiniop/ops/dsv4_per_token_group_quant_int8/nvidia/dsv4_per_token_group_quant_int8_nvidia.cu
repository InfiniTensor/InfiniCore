#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "dsv4_per_token_group_quant_int8_nvidia.cuh"

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

template <typename T, int BS>
__global__ void groupQuantKernel(const T *__restrict__ x, int8_t *__restrict__ q, float *__restrict__ scale, int rows, int cols, int group_size, int groups) {
    int row = blockIdx.x;
    int group = blockIdx.y;
    int g_start = group * group_size;
    int g_end = min(g_start + group_size, cols);
    if (row >= rows || group >= groups || g_start >= cols) {
        return;
    }

    int tid = threadIdx.x;
    float local_max = 0.0f;
    int base = row * cols + g_start;
    for (int i = tid; i < g_end - g_start; i += BS) {
        float v = toFloat(x[base + i]);
        local_max = fmaxf(local_max, fabsf(v));
    }

    int lane = tid & 63;
    int wid = tid >> 6;
    local_max = warpMax(local_max);

    __shared__ float wave_max[16];
    if (lane == 0) {
        wave_max[wid] = local_max;
    }
    __syncthreads();

    int num_waves = (BS + 63) / 64;
    float group_max = 0.0f;
    for (int w = 0; w < num_waves; ++w) {
        group_max = fmaxf(group_max, wave_max[w]);
    }

    __shared__ float scale_value;
    if (tid == 0) {
        float amax = fmaxf(group_max, 1e-10f);
        scale_value = amax / 127.0f;
        scale[row * groups + group] = scale_value;
    }
    __syncthreads();

    float s = scale_value;
    for (int i = tid; i < g_end - g_start; i += BS) {
        float qf = toFloat(x[base + i]) / s;
        qf = fmaxf(-128.0f, fminf(127.0f, qf));
        q[base + i] = static_cast<int8_t>(static_cast<int>(qf));
    }
}

template <typename T>
infiniStatus_t launch(const op::dsv4_per_token_group_quant_int8::Info &info, void *q, void *scale, const void *x, cudaStream_t stream) {
    dim3 grid(static_cast<unsigned int>(info.rows), static_cast<unsigned int>(info.groups), 1);
    groupQuantKernel<T, BLOCK><<<grid, BLOCK, 0, stream>>>(static_cast<const T *>(x), static_cast<int8_t *>(q), static_cast<float *>(scale), static_cast<int>(info.rows), static_cast<int>(info.cols), info.group_size, static_cast<int>(info.groups));
    return INFINI_STATUS_SUCCESS;
}
} // namespace

namespace op::dsv4_per_token_group_quant_int8::nvidia {

infiniStatus_t Descriptor::create(infiniopHandle_t handle, Descriptor **desc_ptr, infiniopTensorDescriptor_t q_desc, infiniopTensorDescriptor_t scale_desc, infiniopTensorDescriptor_t x_desc, int group_size) {
    Info info;
    CHECK_STATUS(createInfo(&info, q_desc, scale_desc, x_desc, group_size));
    *desc_ptr = new Descriptor(info, 0, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(void *, size_t, void *q, void *scale, const void *x, void *stream) const {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    switch (_info.dtype) {
    case INFINI_DTYPE_F16:
        return launch<half>(_info, q, scale, x, s);
    case INFINI_DTYPE_BF16:
        return launch<__nv_bfloat16>(_info, q, scale, x, s);
    case INFINI_DTYPE_F32:
        return launch<float>(_info, q, scale, x, s);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace op::dsv4_per_token_group_quant_int8::nvidia
