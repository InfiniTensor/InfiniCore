#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "rwkv5_wkv_nvidia.cuh"

namespace op::rwkv5_wkv::nvidia {

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

namespace {

template <typename T>
__device__ float to_float(T value) {
    return static_cast<float>(value);
}

template <>
__device__ float to_float<half>(half value) {
    return __half2float(value);
}

template <>
__device__ float to_float<__nv_bfloat16>(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

template <typename T>
__device__ T from_float(float value) {
    return static_cast<T>(value);
}

template <>
__device__ half from_float<half>(float value) {
    return __float2half_rn(value);
}

template <>
__device__ __nv_bfloat16 from_float<__nv_bfloat16>(float value) {
    return __float2bfloat16_rn(value);
}

template <typename T>
__global__ void rwkv5_wkv_kernel(
    T *out,
    const T *receptance,
    const T *key,
    const T *value,
    const T *time_decay,
    const T *time_faaaa,
    float *state,
    size_t batch,
    size_t seq_len,
    size_t hidden_size,
    size_t num_heads,
    size_t head_size) {

    const size_t linear = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = batch * num_heads * head_size;
    if (linear >= total) {
        return;
    }

    const size_t s_out = linear % head_size;
    const size_t tmp = linear / head_size;
    const size_t h = tmp % num_heads;
    const size_t b = tmp / num_heads;
    const size_t channel_base = h * head_size;
    const size_t state_col_base = ((b * num_heads + h) * head_size) * head_size + s_out;

    for (size_t t = 0; t < seq_len; ++t) {
        const size_t token_base = (b * seq_len + t) * hidden_size + channel_base;
        const float vj = to_float<T>(value[token_base + s_out]);
        float acc = 0.0f;
        for (size_t i = 0; i < head_size; ++i) {
            const float r = to_float<T>(receptance[token_base + i]);
            const float k = to_float<T>(key[token_base + i]);
            const float time_first = to_float<T>(time_faaaa[channel_base + i]);
            const float decay = __expf(-__expf(to_float<T>(time_decay[channel_base + i])));
            const size_t state_idx = state_col_base + i * head_size;
            const float att = k * vj;
            acc += r * (time_first * att + state[state_idx]);
            state[state_idx] = att + decay * state[state_idx];
        }
        out[token_base + s_out] = from_float<T>(acc);
    }
}

template <typename T>
infiniStatus_t launch_typed(
    const Rwkv5WkvInfo &info,
    void *out,
    const void *receptance,
    const void *key,
    const void *value,
    const void *time_decay,
    const void *time_faaaa,
    void *state,
    cudaStream_t stream) {

    constexpr int threads = 256;
    const size_t total = info.batch * info.num_heads * info.head_size;
    const dim3 blocks(static_cast<unsigned int>((total + threads - 1) / threads));
    rwkv5_wkv_kernel<T><<<blocks, threads, 0, stream>>>(
        reinterpret_cast<T *>(out),
        reinterpret_cast<const T *>(receptance),
        reinterpret_cast<const T *>(key),
        reinterpret_cast<const T *>(value),
        reinterpret_cast<const T *>(time_decay),
        reinterpret_cast<const T *>(time_faaaa),
        reinterpret_cast<float *>(state),
        info.batch,
        info.seq_len,
        info.hidden_size,
        info.num_heads,
        info.head_size);
    CHECK_CUDA(cudaGetLastError());
    return INFINI_STATUS_SUCCESS;
}

} // namespace

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t receptance_desc,
    infiniopTensorDescriptor_t key_desc,
    infiniopTensorDescriptor_t value_desc,
    infiniopTensorDescriptor_t time_decay_desc,
    infiniopTensorDescriptor_t time_faaaa_desc,
    infiniopTensorDescriptor_t state_desc) {

    auto result = Rwkv5WkvInfo::create(
        out_desc,
        receptance_desc,
        key_desc,
        value_desc,
        time_decay_desc,
        time_faaaa_desc,
        state_desc);
    CHECK_RESULT(result);
    auto info = result.take();

    *desc_ptr = new Descriptor(
        new Opaque{reinterpret_cast<device::nvidia::Handle *>(handle)->internal()},
        info,
        0,
        handle->device,
        handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *out,
    const void *receptance,
    const void *key,
    const void *value,
    const void *time_decay,
    const void *time_faaaa,
    void *state,
    void *stream_) const {

    (void)workspace;
    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }
    auto stream = reinterpret_cast<cudaStream_t>(stream_);
    if (_info.dtype == INFINI_DTYPE_F16) {
        return launch_typed<half>(_info, out, receptance, key, value, time_decay, time_faaaa, state, stream);
    }
    if (_info.dtype == INFINI_DTYPE_BF16) {
        return launch_typed<__nv_bfloat16>(_info, out, receptance, key, value, time_decay, time_faaaa, state, stream);
    }
    if (_info.dtype == INFINI_DTYPE_F32) {
        return launch_typed<float>(_info, out, receptance, key, value, time_decay, time_faaaa, state, stream);
    }
    return INFINI_STATUS_BAD_TENSOR_DTYPE;
}

} // namespace op::rwkv5_wkv::nvidia
