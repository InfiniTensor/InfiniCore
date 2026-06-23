#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "chunk_gated_delta_rule_nvidia.cuh"

#include "../cuda/kernel.cuh"
#include <cuda_runtime.h>

template <typename Tdata, typename Tgate, typename Tcompute, size_t Dk, size_t Dv, size_t NUM_THREADS>
INFINIOP_CUDA_KERNEL chunkGatedDeltaRule(
    Tcompute *state_workspace,
    Tdata *out,
    Tdata *initial_state,
    Tdata *final_state,
    const Tdata *q,
    const Tdata *k,
    const Tdata *v,
    const Tgate *g,
    const Tgate *beta,
    const void *cu_seqlens,
    const void *initial_state_indices,
    const void *final_state_indices,
    bool cu_seqlens_i64,
    bool initial_state_indices_i64,
    bool final_state_indices_i64,
    bool use_qk_l2norm,
    bool has_cu_seqlens,
    bool indexed_state_pool,
    size_t T,
    size_t chunk_size,
    size_t pool_size,
    size_t Hk,
    size_t value_heads_per_key_head,
    ptrdiff_t out_s0,
    ptrdiff_t out_s1,
    ptrdiff_t out_s2,
    ptrdiff_t initial_s0,
    ptrdiff_t initial_s1,
    ptrdiff_t initial_s2,
    ptrdiff_t initial_s3,
    ptrdiff_t final_s0,
    ptrdiff_t final_s1,
    ptrdiff_t final_s2,
    ptrdiff_t final_s3,
    ptrdiff_t q_s0,
    ptrdiff_t q_s1,
    ptrdiff_t q_s2,
    ptrdiff_t k_s0,
    ptrdiff_t k_s1,
    ptrdiff_t k_s2,
    ptrdiff_t v_s0,
    ptrdiff_t v_s1,
    ptrdiff_t v_s2,
    ptrdiff_t g_s0,
    ptrdiff_t g_s1,
    ptrdiff_t g_s2,
    ptrdiff_t beta_s0,
    ptrdiff_t beta_s1,
    ptrdiff_t beta_s2) {
    chunkGatedDeltaRuleRecurrentKernel<Tdata, Tgate, Tcompute, Dk, Dv, NUM_THREADS>(
        state_workspace, out, initial_state, final_state, q, k, v, g, beta, cu_seqlens,
        initial_state_indices, final_state_indices, cu_seqlens_i64,
        initial_state_indices_i64, final_state_indices_i64, use_qk_l2norm,
        has_cu_seqlens, indexed_state_pool, T, chunk_size, pool_size, Hk, value_heads_per_key_head,
        out_s0, out_s1, out_s2, initial_s0, initial_s1, initial_s2, initial_s3,
        final_s0, final_s1, final_s2, final_s3, q_s0, q_s1, q_s2, k_s0, k_s1,
        k_s2, v_s0, v_s1, v_s2, g_s0, g_s1, g_s2, beta_s0, beta_s1, beta_s2);
}

template <typename Tdata, typename Tgate, typename Tcompute, size_t Dk, size_t Dv, size_t NUM_THREADS>
INFINIOP_CUDA_KERNEL chunkGatedDeltaRuleChunked(
    Tcompute *state_workspace,
    Tdata *out,
    Tdata *initial_state,
    Tdata *final_state,
    const Tdata *q,
    const Tdata *k,
    const Tdata *v,
    const Tgate *g,
    const Tgate *beta,
    const void *cu_seqlens,
    const void *initial_state_indices,
    const void *final_state_indices,
    bool cu_seqlens_i64,
    bool initial_state_indices_i64,
    bool final_state_indices_i64,
    bool use_qk_l2norm,
    bool has_cu_seqlens,
    bool indexed_state_pool,
    size_t T,
    size_t chunk_size,
    size_t pool_size,
    size_t Hk,
    size_t value_heads_per_key_head,
    ptrdiff_t out_s0,
    ptrdiff_t out_s1,
    ptrdiff_t out_s2,
    ptrdiff_t initial_s0,
    ptrdiff_t initial_s1,
    ptrdiff_t initial_s2,
    ptrdiff_t initial_s3,
    ptrdiff_t final_s0,
    ptrdiff_t final_s1,
    ptrdiff_t final_s2,
    ptrdiff_t final_s3,
    ptrdiff_t q_s0,
    ptrdiff_t q_s1,
    ptrdiff_t q_s2,
    ptrdiff_t k_s0,
    ptrdiff_t k_s1,
    ptrdiff_t k_s2,
    ptrdiff_t v_s0,
    ptrdiff_t v_s1,
    ptrdiff_t v_s2,
    ptrdiff_t g_s0,
    ptrdiff_t g_s1,
    ptrdiff_t g_s2,
    ptrdiff_t beta_s0,
    ptrdiff_t beta_s1,
    ptrdiff_t beta_s2) {
    chunkGatedDeltaRuleKernel<Tdata, Tgate, Tcompute, Dk, Dv, NUM_THREADS>(
        state_workspace, out, initial_state, final_state, q, k, v, g, beta, cu_seqlens,
        initial_state_indices, final_state_indices, cu_seqlens_i64,
        initial_state_indices_i64, final_state_indices_i64, use_qk_l2norm,
        has_cu_seqlens, indexed_state_pool, T, chunk_size, pool_size, Hk, value_heads_per_key_head,
        out_s0, out_s1, out_s2, initial_s0, initial_s1, initial_s2, initial_s3,
        final_s0, final_s1, final_s2, final_s3, q_s0, q_s1, q_s2, k_s0, k_s1,
        k_s2, v_s0, v_s1, v_s2, g_s0, g_s1, g_s2, beta_s0, beta_s1, beta_s2);
}

namespace op {
namespace chunk_gated_delta_rule {
namespace nvidia {

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t initial_state_desc,
    infiniopTensorDescriptor_t final_state_desc,
    infiniopTensorDescriptor_t q_desc,
    infiniopTensorDescriptor_t k_desc,
    infiniopTensorDescriptor_t v_desc,
    infiniopTensorDescriptor_t g_desc,
    infiniopTensorDescriptor_t beta_desc,
    infiniopTensorDescriptor_t cu_seqlens_desc,
    infiniopTensorDescriptor_t initial_state_indices_desc,
    infiniopTensorDescriptor_t final_state_indices_desc,
    bool use_qk_l2norm,
    size_t chunk_size) {
    auto info = ChunkGatedDeltaRuleInfo::create(
        out_desc, initial_state_desc, final_state_desc, q_desc, k_desc, v_desc,
        g_desc, beta_desc, cu_seqlens_desc, initial_state_indices_desc,
        final_state_indices_desc, use_qk_l2norm, chunk_size);
    CHECK_RESULT(info);

    auto info_value = info.take();
    // We always want to use fast path, slow path is kept as a ref
    const bool use_chunked_fallback = false;
    const size_t per_block_workspace = use_chunked_fallback
                                         ? info_value.Dk * info_value.Dv + info_value.chunk_size * info_value.Dk * 3 + info_value.chunk_size * info_value.Dv * 3 + info_value.chunk_size * info_value.chunk_size + info_value.chunk_size * 3
                                         : info_value.Dk * info_value.Dv;
    const size_t workspace_size = info_value.B * info_value.Hv * per_block_workspace * sizeof(float);

    *desc_ptr = new Descriptor(
        new Opaque{reinterpret_cast<device::nvidia::Handle *>(handle)->internal()},
        info_value, workspace_size, handle->device, handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

template <typename Tdata, size_t Dk, size_t Dv, size_t NUM_THREADS>
infiniStatus_t launchKernelWithGateDtype(
    void *workspace,
    void *out,
    void *initial_state,
    void *final_state,
    const void *q,
    const void *k,
    const void *v,
    const void *g,
    const void *beta,
    const void *cu_seqlens,
    const void *initial_state_indices,
    const void *final_state_indices,
    const ChunkGatedDeltaRuleInfo &info,
    cudaStream_t stream) {

#define LAUNCH_ARGS(TYPE)                                                                                                                       \
    static_cast<float *>(workspace), static_cast<Tdata *>(out), static_cast<Tdata *>(initial_state), static_cast<Tdata *>(final_state),         \
        static_cast<const Tdata *>(q), static_cast<const Tdata *>(k), static_cast<const Tdata *>(v),                                            \
        static_cast<const TYPE *>(g), static_cast<const TYPE *>(beta), cu_seqlens, initial_state_indices,                                       \
        final_state_indices, info.cu_seqlens_dtype == INFINI_DTYPE_I64,                                                                         \
        info.initial_state_indices_dtype == INFINI_DTYPE_I64, info.final_state_indices_dtype == INFINI_DTYPE_I64,                               \
        info.use_qk_l2norm, info.has_cu_seqlens, info.indexed_state_pool, info.T, info.chunk_size, info.pool_size, info.Hk,                     \
        info.value_heads_per_key_head, info.out_strides[0], info.out_strides[1], info.out_strides[2],                                           \
        info.initial_state_strides[0], info.initial_state_strides[1], info.initial_state_strides[2],                                            \
        info.initial_state_strides[3], info.final_state_strides.empty() ? 0 : info.final_state_strides[0],                                      \
        info.final_state_strides.empty() ? 0 : info.final_state_strides[1], info.final_state_strides.empty() ? 0 : info.final_state_strides[2], \
        info.final_state_strides.empty() ? 0 : info.final_state_strides[3], info.q_strides[0], info.q_strides[1],                               \
        info.q_strides[2], info.k_strides[0], info.k_strides[1], info.k_strides[2], info.v_strides[0],                                          \
        info.v_strides[1], info.v_strides[2], info.g_strides[0], info.g_strides[1], info.g_strides[2],                                          \
        info.beta_strides[0], info.beta_strides[1], info.beta_strides[2]

#define LAUNCH_GATE(TYPE)                                                                 \
    do {                                                                                  \
        if (false) {                                                                      \
            chunkGatedDeltaRuleChunked<Tdata, TYPE, float, Dk, Dv, NUM_THREADS>           \
                <<<dim3(uint32_t(info.B), uint32_t(info.Hv), 1), dim3(NUM_THREADS),       \
                   (Dk + Dk + NUM_THREADS) * sizeof(float), stream>>>(LAUNCH_ARGS(TYPE)); \
        } else {                                                                          \
            chunkGatedDeltaRule<Tdata, TYPE, float, Dk, Dv, NUM_THREADS>                  \
                <<<dim3(uint32_t(info.B), uint32_t(info.Hv), 1), dim3(NUM_THREADS),       \
                   (Dk + Dk + NUM_THREADS) * sizeof(float), stream>>>(LAUNCH_ARGS(TYPE)); \
        }                                                                                 \
    } while (0)

    if (info.gate_dtype == INFINI_DTYPE_F16) {
        LAUNCH_GATE(half);
    } else if (info.gate_dtype == INFINI_DTYPE_BF16) {
        LAUNCH_GATE(__nv_bfloat16);
    } else if (info.gate_dtype == INFINI_DTYPE_F32) {
        LAUNCH_GATE(float);
    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
#undef LAUNCH_GATE
#undef LAUNCH_ARGS

    return INFINI_STATUS_SUCCESS;
}

template <size_t Dk, size_t Dv, size_t NUM_THREADS>
infiniStatus_t launchKernel(
    void *workspace,
    void *out,
    void *initial_state,
    void *final_state,
    const void *q,
    const void *k,
    const void *v,
    const void *g,
    const void *beta,
    const void *cu_seqlens,
    const void *initial_state_indices,
    const void *final_state_indices,
    const ChunkGatedDeltaRuleInfo &info,
    cudaStream_t stream) {
    if (info.data_dtype == INFINI_DTYPE_F16) {
        return launchKernelWithGateDtype<half, Dk, Dv, NUM_THREADS>(
            workspace, out, initial_state, final_state, q, k, v, g, beta, cu_seqlens,
            initial_state_indices, final_state_indices, info, stream);
    }
    if (info.data_dtype == INFINI_DTYPE_BF16) {
        return launchKernelWithGateDtype<__nv_bfloat16, Dk, Dv, NUM_THREADS>(
            workspace, out, initial_state, final_state, q, k, v, g, beta, cu_seqlens,
            initial_state_indices, final_state_indices, info, stream);
    }
    if (info.data_dtype == INFINI_DTYPE_F32) {
        return launchKernelWithGateDtype<float, Dk, Dv, NUM_THREADS>(
            workspace, out, initial_state, final_state, q, k, v, g, beta, cu_seqlens,
            initial_state_indices, final_state_indices, info, stream);
    }
    return INFINI_STATUS_BAD_TENSOR_DTYPE;
}

infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size,
    void *out, void *initial_state, void *final_state,
    const void *q, const void *k, const void *v,
    const void *g, const void *beta, const void *cu_seqlens,
    const void *initial_state_indices, const void *final_state_indices,
    void *stream_) const {
    cudaStream_t stream = (cudaStream_t)stream_;
    if (workspace == nullptr || workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    if (_info.Dk == 128 && _info.Dv == 128) {
        if (_opaque->internal->maxThreadsPerBlock() >= 128) {
            return launchKernel<128, 128, 128>(
                workspace, out, initial_state, final_state, q, k, v, g, beta, cu_seqlens,
                initial_state_indices, final_state_indices, _info, stream);
        }
    } else if (_info.Dk == 64 && _info.Dv == 64) {
        if (_opaque->internal->maxThreadsPerBlock() >= 64) {
            return launchKernel<64, 64, 64>(
                workspace, out, initial_state, final_state, q, k, v, g, beta, cu_seqlens,
                initial_state_indices, final_state_indices, _info, stream);
        }
    }

    return INFINI_STATUS_BAD_TENSOR_SHAPE;
}

} // namespace nvidia
} // namespace chunk_gated_delta_rule
} // namespace op
