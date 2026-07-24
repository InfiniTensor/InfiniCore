#include "../../../devices/moore/moore_common.h"
#include "../../../devices/moore/moore_kernel_common.h"
#include "kimi_delta_attention_moore.h"

#include "../cuda/kernel.cuh"

namespace op::kimi_delta_attention::moore {

struct Descriptor::Opaque {
    std::shared_ptr<device::moore::Handle::Internal> internal;
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
    infiniopTensorDescriptor_t A_log_desc,
    infiniopTensorDescriptor_t dt_bias_desc,
    infiniopTensorDescriptor_t cu_seqlens_desc,
    infiniopTensorDescriptor_t initial_state_indices_desc,
    infiniopTensorDescriptor_t final_state_indices_desc,
    float scale,
    float lower_bound,
    bool use_qk_l2norm) {

    auto info = KimiDeltaAttentionInfo::create(
        out_desc,
        initial_state_desc,
        final_state_desc,
        q_desc,
        k_desc,
        v_desc,
        g_desc,
        beta_desc,
        A_log_desc,
        dt_bias_desc,
        cu_seqlens_desc,
        initial_state_indices_desc,
        final_state_indices_desc,
        scale,
        lower_bound,
        use_qk_l2norm);
    CHECK_RESULT(info);

    *desc_ptr = new Descriptor(
        new Opaque{reinterpret_cast<device::moore::Handle *>(handle)->internal()},
        info.take(),
        0,
        handle->device,
        handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

template <typename Tdata, typename Tgate>
static infiniStatus_t launch_fallback(const KimiDeltaAttentionInfo &info,
                                      void *out,
                                      void *initial_state,
                                      void *final_state,
                                      const void *q,
                                      const void *k,
                                      const void *v,
                                      const void *g,
                                      const void *beta,
                                      const void *A_log,
                                      const void *dt_bias,
                                      const void *cu_seqlens,
                                      const void *initial_state_indices,
                                      const void *final_state_indices,
                                      musaStream_t stream) {
    constexpr int threads = 256;
    dim3 grid(static_cast<uint32_t>(info.B), static_cast<uint32_t>(info.H), static_cast<uint32_t>(info.D));
    size_t shared = info.is_decode ? threads * sizeof(float) : (info.D * 3 + threads) * sizeof(float);

    // TODO(kimi_delta_attention): Dispatch MoonshotAI FlashKDA SM90 fast path for
    // BF16, D=128, contiguous tensors. Original source:
    // https://github.com/MoonshotAI/FlashKDA/tree/master/csrc
    if (info.is_decode) {
        kimiDeltaAttentionDecodeCudaKernel<Tdata, Tgate><<<grid, threads, shared, stream>>>(
            static_cast<Tdata *>(out),
            static_cast<Tdata *>(initial_state),
            static_cast<Tdata *>(final_state),
            static_cast<const Tdata *>(q),
            static_cast<const Tdata *>(k),
            static_cast<const Tdata *>(v),
            static_cast<const Tgate *>(g),
            static_cast<const Tgate *>(beta),
            static_cast<const float *>(A_log),
            static_cast<const float *>(dt_bias),
            cu_seqlens,
            initial_state_indices,
            final_state_indices,
            info.cu_seqlens_dtype == INFINI_DTYPE_I64,
            info.initial_state_indices_dtype == INFINI_DTYPE_I64,
            info.final_state_indices_dtype == INFINI_DTYPE_I64,
            info.use_qk_l2norm,
            info.has_cu_seqlens,
            info.indexed_state_pool,
            info.D,
            info.pool_size,
            info.scale,
            info.lower_bound,
            info.out_strides[0],
            info.out_strides[1],
            info.out_strides[2],
            info.initial_state_strides[0],
            info.initial_state_strides[1],
            info.initial_state_strides[2],
            info.initial_state_strides[3],
            info.final_state_strides.empty() ? 0 : info.final_state_strides[0],
            info.final_state_strides.empty() ? 0 : info.final_state_strides[1],
            info.final_state_strides.empty() ? 0 : info.final_state_strides[2],
            info.final_state_strides.empty() ? 0 : info.final_state_strides[3],
            info.q_strides[0],
            info.q_strides[1],
            info.q_strides[2],
            info.k_strides[0],
            info.k_strides[1],
            info.k_strides[2],
            info.v_strides[0],
            info.v_strides[1],
            info.v_strides[2],
            info.g_strides[0],
            info.g_strides[1],
            info.g_strides[2],
            info.beta_strides[0],
            info.beta_strides[1],
            info.beta_strides[2],
            info.A_log_strides[0],
            info.dt_bias_strides[0]);
    } else {
        kimiDeltaAttentionRecurrentCudaKernel<Tdata, Tgate><<<grid, threads, shared, stream>>>(
            static_cast<Tdata *>(out),
            static_cast<Tdata *>(initial_state),
            static_cast<Tdata *>(final_state),
            static_cast<const Tdata *>(q),
            static_cast<const Tdata *>(k),
            static_cast<const Tdata *>(v),
            static_cast<const Tgate *>(g),
            static_cast<const Tgate *>(beta),
            static_cast<const float *>(A_log),
            static_cast<const float *>(dt_bias),
            cu_seqlens,
            initial_state_indices,
            final_state_indices,
            info.cu_seqlens_dtype == INFINI_DTYPE_I64,
            info.initial_state_indices_dtype == INFINI_DTYPE_I64,
            info.final_state_indices_dtype == INFINI_DTYPE_I64,
            info.use_qk_l2norm,
            info.has_cu_seqlens,
            info.indexed_state_pool,
            info.T,
            info.D,
            info.pool_size,
            info.scale,
            info.lower_bound,
            info.out_strides[0],
            info.out_strides[1],
            info.out_strides[2],
            info.initial_state_strides[0],
            info.initial_state_strides[1],
            info.initial_state_strides[2],
            info.initial_state_strides[3],
            info.final_state_strides.empty() ? 0 : info.final_state_strides[0],
            info.final_state_strides.empty() ? 0 : info.final_state_strides[1],
            info.final_state_strides.empty() ? 0 : info.final_state_strides[2],
            info.final_state_strides.empty() ? 0 : info.final_state_strides[3],
            info.q_strides[0],
            info.q_strides[1],
            info.q_strides[2],
            info.k_strides[0],
            info.k_strides[1],
            info.k_strides[2],
            info.v_strides[0],
            info.v_strides[1],
            info.v_strides[2],
            info.g_strides[0],
            info.g_strides[1],
            info.g_strides[2],
            info.beta_strides[0],
            info.beta_strides[1],
            info.beta_strides[2],
            info.A_log_strides[0],
            info.dt_bias_strides[0]);
    }
    return INFINI_STATUS_SUCCESS;
}

template <typename Tdata>
static infiniStatus_t launch_for_gate(const KimiDeltaAttentionInfo &info,
                                      void *out,
                                      void *initial_state,
                                      void *final_state,
                                      const void *q,
                                      const void *k,
                                      const void *v,
                                      const void *g,
                                      const void *beta,
                                      const void *A_log,
                                      const void *dt_bias,
                                      const void *cu_seqlens,
                                      const void *initial_state_indices,
                                      const void *final_state_indices,
                                      musaStream_t stream) {
    switch (info.gate_dtype) {
    case INFINI_DTYPE_F16:
        return launch_fallback<Tdata, half>(info, out, initial_state, final_state, q, k, v, g, beta, A_log, dt_bias, cu_seqlens, initial_state_indices, final_state_indices, stream);
    case INFINI_DTYPE_BF16:
        return launch_fallback<Tdata, __nv_bfloat16>(info, out, initial_state, final_state, q, k, v, g, beta, A_log, dt_bias, cu_seqlens, initial_state_indices, final_state_indices, stream);
    case INFINI_DTYPE_F32:
        return launch_fallback<Tdata, float>(info, out, initial_state, final_state, q, k, v, g, beta, A_log, dt_bias, cu_seqlens, initial_state_indices, final_state_indices, stream);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *out,
    void *initial_state,
    void *final_state,
    const void *q,
    const void *k,
    const void *v,
    const void *g,
    const void *beta,
    const void *A_log,
    const void *dt_bias,
    const void *cu_seqlens,
    const void *initial_state_indices,
    const void *final_state_indices,
    void *stream_) const {
    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }
    if (_info.has_cu_seqlens && cu_seqlens == nullptr) {
        return INFINI_STATUS_NULL_POINTER;
    }
    if (_info.has_initial_state_indices && initial_state_indices == nullptr) {
        return INFINI_STATUS_NULL_POINTER;
    }
    if (_info.has_final_state_indices && final_state_indices == nullptr) {
        return INFINI_STATUS_NULL_POINTER;
    }
    if (!_info.has_final_state_indices && final_state == nullptr) {
        return INFINI_STATUS_NULL_POINTER;
    }

    musaStream_t stream = reinterpret_cast<musaStream_t>(stream_);
    switch (_info.data_dtype) {
    case INFINI_DTYPE_F16:
        return launch_for_gate<half>(_info, out, initial_state, final_state, q, k, v, g, beta, A_log, dt_bias, cu_seqlens, initial_state_indices, final_state_indices, stream);
    case INFINI_DTYPE_BF16:
        return launch_for_gate<__nv_bfloat16>(_info, out, initial_state, final_state, q, k, v, g, beta, A_log, dt_bias, cu_seqlens, initial_state_indices, final_state_indices, stream);
    case INFINI_DTYPE_F32:
        return launch_for_gate<float>(_info, out, initial_state, final_state, q, k, v, g, beta, A_log, dt_bias, cu_seqlens, initial_state_indices, final_state_indices, stream);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace op::kimi_delta_attention::moore
