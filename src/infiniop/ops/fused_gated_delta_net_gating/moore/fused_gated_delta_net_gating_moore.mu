#include "../../../devices/moore/moore_common.h"
#include "../../../devices/moore/moore_handle.h"
#include "../../../devices/moore/moore_kernel_common.h"
#include "fused_gated_delta_net_gating_moore.h"

namespace op::fused_gated_delta_net_gating::moore {

struct Descriptor::Opaque {
    std::shared_ptr<device::moore::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t g_desc,
    infiniopTensorDescriptor_t beta_output_desc,
    infiniopTensorDescriptor_t A_log_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc,
    infiniopTensorDescriptor_t dt_bias_desc,
    float beta,
    float threshold) {

    auto result = FusedGatedDeltaNetGatingInfo::create(
        g_desc, beta_output_desc, A_log_desc, a_desc, b_desc, dt_bias_desc, beta, threshold);
    CHECK_RESULT(result);

    *desc_ptr = new Descriptor(
        new Opaque{reinterpret_cast<device::moore::Handle *>(handle)->internal()},
        result.take(),
        0,
        handle->device,
        handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

namespace {

template <typename T>
__device__ __forceinline__ float load_as_float(const T *ptr, ptrdiff_t offset) {
    return static_cast<float>(ptr[offset]);
}

template <>
__device__ __forceinline__ float load_as_float<half>(const half *ptr, ptrdiff_t offset) {
    return __half2float(ptr[offset]);
}

template <>
__device__ __forceinline__ float load_as_float<cuda_bfloat16>(const cuda_bfloat16 *ptr, ptrdiff_t offset) {
    return __bfloat162float(ptr[offset]);
}

__device__ __forceinline__ float sigmoidf_stable(float x) {
    if (x >= 0.0f) {
        float z = expf(-x);
        return 1.0f / (1.0f + z);
    }
    float z = expf(x);
    return z / (1.0f + z);
}

__device__ __forceinline__ float softplus_beta_threshold(float x, float beta, float threshold) {
    float bx = beta * x;
    return bx <= threshold ? log1pf(expf(bx)) / beta : x;
}

template <typename T>
__global__ void fused_gated_delta_net_gating_kernel(
    float *g,
    float *beta_output,
    const T *A_log,
    const T *a,
    const T *b,
    const T *dt_bias,
    size_t total,
    size_t seq_len,
    size_t hidden,
    ptrdiff_t g_s0,
    ptrdiff_t g_s1,
    ptrdiff_t g_s2,
    ptrdiff_t beta_s0,
    ptrdiff_t beta_s1,
    ptrdiff_t beta_s2,
    ptrdiff_t A_log_s0,
    ptrdiff_t a_s0,
    ptrdiff_t a_s1,
    ptrdiff_t a_s2,
    ptrdiff_t b_s0,
    ptrdiff_t b_s1,
    ptrdiff_t b_s2,
    ptrdiff_t dt_bias_s0,
    float beta,
    float threshold) {

    size_t linear = blockIdx.x * blockDim.x + threadIdx.x;
    if (linear >= total) {
        return;
    }

    size_t h = linear % hidden;
    size_t tmp = linear / hidden;
    size_t s = tmp % seq_len;
    size_t batch = tmp / seq_len;

    ptrdiff_t g_off = static_cast<ptrdiff_t>(batch) * g_s0 + static_cast<ptrdiff_t>(s) * g_s1 + static_cast<ptrdiff_t>(h) * g_s2;
    ptrdiff_t beta_off = static_cast<ptrdiff_t>(batch) * beta_s0 + static_cast<ptrdiff_t>(s) * beta_s1 + static_cast<ptrdiff_t>(h) * beta_s2;
    ptrdiff_t a_off = static_cast<ptrdiff_t>(batch) * a_s0 + static_cast<ptrdiff_t>(s) * a_s1 + static_cast<ptrdiff_t>(h) * a_s2;
    ptrdiff_t b_off = static_cast<ptrdiff_t>(batch) * b_s0 + static_cast<ptrdiff_t>(s) * b_s1 + static_cast<ptrdiff_t>(h) * b_s2;

    float x = load_as_float(a, a_off) + load_as_float(dt_bias, static_cast<ptrdiff_t>(h) * dt_bias_s0);
    g[g_off] = -expf(load_as_float(A_log, static_cast<ptrdiff_t>(h) * A_log_s0)) * softplus_beta_threshold(x, beta, threshold);
    beta_output[beta_off] = sigmoidf_stable(load_as_float(b, b_off));
}

template <typename T>
infiniStatus_t launch_kernel(const FusedGatedDeltaNetGatingInfo &info,
                             float *g,
                             float *beta_output,
                             const void *A_log,
                             const void *a,
                             const void *b,
                             const void *dt_bias,
                             musaStream_t stream) {
    size_t total = info.numel();
    if (total == 0) {
        return INFINI_STATUS_SUCCESS;
    }

    constexpr int block = 256;
    int grid = static_cast<int>((total + block - 1) / block);
    fused_gated_delta_net_gating_kernel<T><<<grid, block, 0, stream>>>(
        g,
        beta_output,
        static_cast<const T *>(A_log),
        static_cast<const T *>(a),
        static_cast<const T *>(b),
        static_cast<const T *>(dt_bias),
        total,
        info.seq_len,
        info.hidden,
        info.g_strides[0],
        info.g_strides[1],
        info.g_strides[2],
        info.beta_output_strides[0],
        info.beta_output_strides[1],
        info.beta_output_strides[2],
        info.A_log_strides[0],
        info.a_strides[0],
        info.a_strides[1],
        info.a_strides[2],
        info.b_strides[0],
        info.b_strides[1],
        info.b_strides[2],
        info.dt_bias_strides[0],
        info.beta,
        info.threshold);
    return INFINI_STATUS_SUCCESS;
}

} // namespace

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *g,
    void *beta_output,
    const void *A_log,
    const void *a,
    const void *b,
    const void *dt_bias,
    void *stream) const {
    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    musaStream_t cuda_stream = reinterpret_cast<musaStream_t>(stream);
    switch (_info.input_dtype) {
    case INFINI_DTYPE_F32:
        return launch_kernel<float>(_info, static_cast<float *>(g), static_cast<float *>(beta_output), A_log, a, b, dt_bias, cuda_stream);
    case INFINI_DTYPE_F16:
        return launch_kernel<half>(_info, static_cast<float *>(g), static_cast<float *>(beta_output), A_log, a, b, dt_bias, cuda_stream);
    case INFINI_DTYPE_BF16:
        return launch_kernel<cuda_bfloat16>(_info, static_cast<float *>(g), static_cast<float *>(beta_output), A_log, a, b, dt_bias, cuda_stream);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace op::fused_gated_delta_net_gating::moore
