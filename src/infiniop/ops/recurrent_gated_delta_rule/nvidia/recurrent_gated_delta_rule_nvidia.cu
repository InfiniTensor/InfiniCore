#include "../../../devices/nvidia/nvidia_common.cuh"
#include "recurrent_gated_delta_rule_nvidia.cuh"

#include "../../../devices/nvidia/nvidia_kernel_common.cuh"

#include "../cuda/kernel.cuh"
#include <cuda_runtime.h>

template <typename Tdata, typename Tgate, typename Tcompute, size_t Dk, size_t Dv, size_t WARPS_PER_BLOCK>
INFINIOP_CUDA_KERNEL recurrentGatedDeltaRuleIndexedPoolWarp(
    Tdata *out, Tdata *initial_state, Tdata *final_state,
    const Tdata *q, const Tdata *k, const Tdata *v,
    const Tgate *g, const Tgate *beta,
    const void *initial_state_indices,
    const void *final_state_indices,
    bool initial_state_indices_i64,
    bool final_state_indices_i64,
    bool use_qk_l2norm,
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
    recurrentGatedDeltaRuleIndexedPoolWarpKernel<Tdata, Tgate, Tcompute, Dk, Dv, WARPS_PER_BLOCK>(
        out, initial_state, final_state, q, k, v, g, beta,
        initial_state_indices, final_state_indices,
        initial_state_indices_i64, final_state_indices_i64,
        use_qk_l2norm,
        Hk, value_heads_per_key_head,
        out_s0, out_s1, out_s2,
        initial_s0, initial_s1, initial_s2, initial_s3,
        final_s0, final_s1, final_s2, final_s3,
        q_s0, q_s1, q_s2,
        k_s0, k_s1, k_s2,
        v_s0, v_s1, v_s2,
        g_s0, g_s1, g_s2,
        beta_s0, beta_s1, beta_s2);
}
namespace op {
namespace recurrent_gated_delta_rule {
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
    infiniopTensorDescriptor_t initial_state_indices_desc,
    infiniopTensorDescriptor_t final_state_indices_desc,
    bool use_qk_l2norm) {
    auto info = RecurrentGatedDeltaRuleInfo::create(
        out_desc, initial_state_desc, final_state_desc, q_desc, k_desc, v_desc,
        g_desc, beta_desc, initial_state_indices_desc, final_state_indices_desc,
        use_qk_l2norm);
    CHECK_RESULT(info);

    size_t workspace_size = 0;

    *desc_ptr = new Descriptor(
        new Opaque{reinterpret_cast<device::nvidia::Handle *>(handle)->internal()},
        info.take(), workspace_size, handle->device, handle->device_id);

    return infiniStatus_t::INFINI_STATUS_SUCCESS;
}

template <typename Tdata, typename Tgate, size_t Dk, size_t Dv, size_t WARPS_PER_BLOCK>
infiniStatus_t launchIndexedPoolWarpKernelTyped(
    const RecurrentGatedDeltaRuleInfo &_info,
    void *out, void *initial_state, void *final_state,
    const void *q, const void *k, const void *v,
    const void *g, const void *beta,
    const void *initial_state_indices,
    const void *final_state_indices,
    bool initial_state_indices_i64,
    bool final_state_indices_i64,
    cudaStream_t stream) {
    constexpr size_t NUM_THREADS = WARPS_PER_BLOCK * 32;
    dim3 grid(uint32_t(_info.B), uint32_t(_info.Hv), uint32_t((_info.Dv + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK));
    dim3 block(NUM_THREADS);
    size_t shared_mem_size = (Dk + Dk + NUM_THREADS) * sizeof(float);

    auto final_s0 = _info.final_state_strides.empty() ? 0 : _info.final_state_strides[0];
    auto final_s1 = _info.final_state_strides.empty() ? 0 : _info.final_state_strides[1];
    auto final_s2 = _info.final_state_strides.empty() ? 0 : _info.final_state_strides[2];
    auto final_s3 = _info.final_state_strides.empty() ? 0 : _info.final_state_strides[3];

    recurrentGatedDeltaRuleIndexedPoolWarp<Tdata, Tgate, float, Dk, Dv, WARPS_PER_BLOCK>
        <<<grid, block, shared_mem_size, stream>>>(
            static_cast<Tdata *>(out),
            static_cast<Tdata *>(initial_state),
            static_cast<Tdata *>(final_state),
            static_cast<const Tdata *>(q),
            static_cast<const Tdata *>(k),
            static_cast<const Tdata *>(v),
            static_cast<const Tgate *>(g),
            static_cast<const Tgate *>(beta),
            initial_state_indices,
            final_state_indices,
            initial_state_indices_i64,
            final_state_indices_i64,
            _info.use_qk_l2norm,
            _info.Hk,
            _info.value_heads_per_key_head,
            _info.out_strides[0],
            _info.out_strides[1],
            _info.out_strides[2],
            _info.initial_state_strides[0],
            _info.initial_state_strides[1],
            _info.initial_state_strides[2],
            _info.initial_state_strides[3],
            final_s0,
            final_s1,
            final_s2,
            final_s3,
            _info.q_strides[0],
            _info.q_strides[1],
            _info.q_strides[2],
            _info.k_strides[0],
            _info.k_strides[1],
            _info.k_strides[2],
            _info.v_strides[0],
            _info.v_strides[1],
            _info.v_strides[2],
            _info.g_strides[0],
            _info.g_strides[1],
            _info.g_strides[2],
            _info.beta_strides[0],
            _info.beta_strides[1],
            _info.beta_strides[2]);
    return infiniStatus_t::INFINI_STATUS_SUCCESS;
}
template <typename Tdata, size_t Dk, size_t Dv, size_t WARPS_PER_BLOCK>
infiniStatus_t launchIndexedPoolWarpKernelForGate(
    const RecurrentGatedDeltaRuleInfo &_info,
    void *out, void *initial_state, void *final_state,
    const void *q, const void *k, const void *v,
    const void *g, const void *beta,
    const void *initial_state_indices,
    const void *final_state_indices,
    bool initial_state_indices_i64,
    bool final_state_indices_i64,
    cudaStream_t stream) {
    switch (_info.gate_dtype) {
    case INFINI_DTYPE_F16:
        return launchIndexedPoolWarpKernelTyped<Tdata, half, Dk, Dv, WARPS_PER_BLOCK>(
            _info, out, initial_state, final_state, q, k, v, g, beta,
            initial_state_indices, final_state_indices, initial_state_indices_i64, final_state_indices_i64, stream);
    case INFINI_DTYPE_BF16:
        return launchIndexedPoolWarpKernelTyped<Tdata, __nv_bfloat16, Dk, Dv, WARPS_PER_BLOCK>(
            _info, out, initial_state, final_state, q, k, v, g, beta,
            initial_state_indices, final_state_indices, initial_state_indices_i64, final_state_indices_i64, stream);
    case INFINI_DTYPE_F32:
        return launchIndexedPoolWarpKernelTyped<Tdata, float, Dk, Dv, WARPS_PER_BLOCK>(
            _info, out, initial_state, final_state, q, k, v, g, beta,
            initial_state_indices, final_state_indices, initial_state_indices_i64, final_state_indices_i64, stream);
    default:
        return infiniStatus_t::INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

template <size_t Dk, size_t Dv, size_t WARPS_PER_BLOCK>
infiniStatus_t launchIndexedPoolWarpKernel(
    const RecurrentGatedDeltaRuleInfo &_info,
    void *out, void *initial_state, void *final_state,
    const void *q, const void *k, const void *v,
    const void *g, const void *beta,
    const void *initial_state_indices,
    const void *final_state_indices,
    bool initial_state_indices_i64,
    bool final_state_indices_i64,
    cudaStream_t stream) {
    switch (_info.data_dtype) {
    case INFINI_DTYPE_F16:
        return launchIndexedPoolWarpKernelForGate<half, Dk, Dv, WARPS_PER_BLOCK>(
            _info, out, initial_state, final_state, q, k, v, g, beta,
            initial_state_indices, final_state_indices, initial_state_indices_i64, final_state_indices_i64, stream);
    case INFINI_DTYPE_BF16:
        return launchIndexedPoolWarpKernelForGate<__nv_bfloat16, Dk, Dv, WARPS_PER_BLOCK>(
            _info, out, initial_state, final_state, q, k, v, g, beta,
            initial_state_indices, final_state_indices, initial_state_indices_i64, final_state_indices_i64, stream);
    case INFINI_DTYPE_F32:
        return launchIndexedPoolWarpKernelForGate<float, Dk, Dv, WARPS_PER_BLOCK>(
            _info, out, initial_state, final_state, q, k, v, g, beta,
            initial_state_indices, final_state_indices, initial_state_indices_i64, final_state_indices_i64, stream);
    default:
        return infiniStatus_t::INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}
infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size,
    void *out, void *initial_state, void *final_state,
    const void *q, const void *k, const void *v,
    const void *g, const void *beta,
    const void *initial_state_indices,
    const void *final_state_indices,
    void *stream_) const {
    cudaStream_t stream = (cudaStream_t)stream_;

    if (_info.has_initial_state_indices && initial_state_indices == nullptr) {
        return INFINI_STATUS_NULL_POINTER;
    }
    if (_info.has_final_state_indices && final_state_indices == nullptr) {
        return INFINI_STATUS_NULL_POINTER;
    }
    if (!_info.has_final_state_indices && final_state == nullptr) {
        return INFINI_STATUS_NULL_POINTER;
    }

    bool initial_indices_i64 = _info.initial_state_indices_dtype == INFINI_DTYPE_I64;
    bool final_indices_i64 = _info.final_state_indices_dtype == INFINI_DTYPE_I64;

    if (_info.Dk == 128 && _info.Dv == 128) {
        if (_opaque->internal->maxThreadsPerBlock() >= 256) {
            return launchIndexedPoolWarpKernel<128, 128, 8>(
                _info, out, initial_state, final_state, q, k, v, g, beta,
                initial_state_indices, final_state_indices,
                initial_indices_i64, final_indices_i64, stream);
        }
    } else if (_info.Dk == 64 && _info.Dv == 64) {
        if (_opaque->internal->maxThreadsPerBlock() >= 256) {
            return launchIndexedPoolWarpKernel<64, 64, 8>(
                _info, out, initial_state, final_state, q, k, v, g, beta,
                initial_state_indices, final_state_indices,
                initial_indices_i64, final_indices_i64, stream);
        }
    }

    return infiniStatus_t::INFINI_STATUS_BAD_TENSOR_SHAPE;
}

} // namespace nvidia
} // namespace recurrent_gated_delta_rule
} // namespace op
