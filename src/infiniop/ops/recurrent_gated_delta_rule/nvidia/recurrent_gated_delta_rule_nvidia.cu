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
    bool indexed_state_pool,
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
    extern __shared__ char shared_memory[];
    recurrentGatedDeltaRuleIndexedPoolWarpKernel<Tdata, Tgate, Tcompute, Dk, Dv, WARPS_PER_BLOCK>(
        out, initial_state, final_state, q, k, v, g, beta,
        initial_state_indices, final_state_indices,
        initial_state_indices_i64, final_state_indices_i64,
        use_qk_l2norm,
        indexed_state_pool, pool_size,
        Hk, value_heads_per_key_head,
        out_s0, out_s1, out_s2,
        initial_s0, initial_s1, initial_s2, initial_s3,
        final_s0, final_s1, final_s2, final_s3,
        q_s0, q_s1, q_s2,
        k_s0, k_s1, k_s2,
        v_s0, v_s1, v_s2,
        g_s0, g_s1, g_s2,
        beta_s0, beta_s1, beta_s2,
        reinterpret_cast<Tcompute *>(shared_memory));
}
// Decode specialization: two blocks per head, eight value rows per warp.
template <typename Tdata, typename Tgate, typename Tcompute>
__global__ void recurrentGatedDeltaRuleHeadTileKernel(
    Tdata *out, Tdata *initial_state, const Tdata *q, const Tdata *k,
    const Tdata *v, const Tgate *g, const Tgate *beta,
    const void *initial_state_indices, const void *final_state_indices,
    bool initial_state_indices_i64, bool final_state_indices_i64,
    bool use_qk_l2norm, size_t pool_size,
    size_t value_heads_per_key_head, ptrdiff_t out_s0,
    ptrdiff_t out_s2, ptrdiff_t initial_s0, ptrdiff_t initial_s1,
    ptrdiff_t initial_s2, ptrdiff_t initial_s3, ptrdiff_t q_s0,
    ptrdiff_t q_s2, ptrdiff_t k_s0, ptrdiff_t k_s2,
    ptrdiff_t v_s0, ptrdiff_t v_s2, ptrdiff_t g_s0,
    ptrdiff_t g_s2, ptrdiff_t beta_s0, ptrdiff_t beta_s2) {
    constexpr int D = 128;
    constexpr int VALUES_PER_WARP = 8;
    constexpr int KEY_VALUES_PER_LANE = 4;
    constexpr int WARP_SIZE = INFINIOP_RECURRENT_DELTA_RULE_WARP_SIZE;
    constexpr int NUM_WARPS = 8;

    const int batch_idx = blockIdx.x;
    const int value_head_idx = blockIdx.y;
    const int key_head_idx = value_head_idx / static_cast<int>(value_heads_per_key_head);
    const int64_t read_slot = op::recurrent_gated_delta_rule::cuda::loadOptionalIndex(
        initial_state_indices, initial_state_indices_i64, batch_idx, batch_idx);
    const int64_t write_slot = op::recurrent_gated_delta_rule::cuda::loadOptionalIndex(
        final_state_indices, final_state_indices_i64, batch_idx, batch_idx);
    if (read_slot < 0 || write_slot < 0 || read_slot >= static_cast<int64_t>(pool_size) || write_slot >= static_cast<int64_t>(pool_size)) {
        if (threadIdx.x < D) {
            out[batch_idx * out_s0 + value_head_idx * out_s2 + threadIdx.x] = static_cast<Tdata>(0);
        }
        return;
    }

    const int value_group_offset = blockIdx.z * (NUM_WARPS * VALUES_PER_WARP);
    const int warp_idx = threadIdx.x / WARP_SIZE;
    const int lane_idx = threadIdx.x % WARP_SIZE;
    const ptrdiff_t q_base = static_cast<ptrdiff_t>(batch_idx) * q_s0 + static_cast<ptrdiff_t>(key_head_idx) * q_s2;
    const ptrdiff_t k_base = static_cast<ptrdiff_t>(batch_idx) * k_s0 + static_cast<ptrdiff_t>(key_head_idx) * k_s2;
    const ptrdiff_t v_base = static_cast<ptrdiff_t>(batch_idx) * v_s0 + static_cast<ptrdiff_t>(value_head_idx) * v_s2;
    const ptrdiff_t g_base = static_cast<ptrdiff_t>(batch_idx) * g_s0 + static_cast<ptrdiff_t>(value_head_idx) * g_s2;
    const ptrdiff_t beta_base = static_cast<ptrdiff_t>(batch_idx) * beta_s0 + static_cast<ptrdiff_t>(value_head_idx) * beta_s2;

    Tcompute q_local[KEY_VALUES_PER_LANE];
    Tcompute k_local[KEY_VALUES_PER_LANE];
    Tcompute q_sum = 0;
    Tcompute k_sum = 0;
#pragma unroll
    for (int i = 0; i < KEY_VALUES_PER_LANE; ++i) {
        const int key_dim_idx = lane_idx + i * WARP_SIZE;
        q_local[i] = static_cast<Tcompute>(
            op::recurrent_gated_delta_rule::cuda::loadAsFloat(q, q_base + key_dim_idx));
        k_local[i] = static_cast<Tcompute>(
            op::recurrent_gated_delta_rule::cuda::loadAsFloat(k, k_base + key_dim_idx));
        q_sum += q_local[i] * q_local[i];
        k_sum += k_local[i] * k_local[i];
    }
    q_sum = op::recurrent_gated_delta_rule::cuda::warpReduceSum(q_sum);
    k_sum = op::recurrent_gated_delta_rule::cuda::warpReduceSum(k_sum);

    const Tcompute q_norm = use_qk_l2norm ? rsqrtf(q_sum + static_cast<Tcompute>(1e-6))
                                          : static_cast<Tcompute>(1);
    const Tcompute k_norm = use_qk_l2norm ? rsqrtf(k_sum + static_cast<Tcompute>(1e-6))
                                          : static_cast<Tcompute>(1);
    const Tcompute query_scale = rsqrtf(static_cast<Tcompute>(D));
    Tcompute kq_memory = 0;
#pragma unroll
    for (int i = 0; i < KEY_VALUES_PER_LANE; ++i) {
        q_local[i] *= q_norm * query_scale;
        k_local[i] *= k_norm;
        kq_memory += k_local[i] * q_local[i];
    }
    kq_memory = op::recurrent_gated_delta_rule::cuda::warpReduceSum(kq_memory);

    const Tcompute decay = expf(__shfl_sync(
        INFINIOP_RECURRENT_DELTA_RULE_FULL_MASK,
        static_cast<Tcompute>(
            op::recurrent_gated_delta_rule::cuda::loadAsFloat(g, g_base)),
        0, WARP_SIZE));
    const Tcompute beta_value = __shfl_sync(
        INFINIOP_RECURRENT_DELTA_RULE_FULL_MASK,
        static_cast<Tcompute>(
            op::recurrent_gated_delta_rule::cuda::loadAsFloat(beta, beta_base)),
        0, WARP_SIZE);

    const ptrdiff_t initial_head_base = static_cast<ptrdiff_t>(read_slot) * initial_s0 + static_cast<ptrdiff_t>(value_head_idx) * initial_s1;
    const ptrdiff_t final_head_base = static_cast<ptrdiff_t>(write_slot) * initial_s0 + static_cast<ptrdiff_t>(value_head_idx) * initial_s1;
    const ptrdiff_t out_base = static_cast<ptrdiff_t>(batch_idx) * out_s0 + static_cast<ptrdiff_t>(value_head_idx) * out_s2;

#pragma unroll
    for (int local_v = 0; local_v < VALUES_PER_WARP; ++local_v) {
        const int value_dim_idx = value_group_offset + warp_idx * VALUES_PER_WARP + local_v;
        Tcompute state[KEY_VALUES_PER_LANE];
        Tcompute kv_memory = 0;
        Tcompute hq_memory = 0;
        const ptrdiff_t initial_row_base = initial_head_base + static_cast<ptrdiff_t>(value_dim_idx) * initial_s2;
#pragma unroll
        for (int i = 0; i < KEY_VALUES_PER_LANE; ++i) {
            const int key_dim_idx = lane_idx + i * WARP_SIZE;
            state[i] = static_cast<Tcompute>(
                op::recurrent_gated_delta_rule::cuda::loadAsFloat(
                    initial_state,
                    initial_row_base + static_cast<ptrdiff_t>(key_dim_idx) * initial_s3));
            kv_memory += state[i] * decay * k_local[i];
            hq_memory += state[i] * decay * q_local[i];
        }
        kv_memory = op::recurrent_gated_delta_rule::cuda::warpReduceSum(kv_memory);
        hq_memory = op::recurrent_gated_delta_rule::cuda::warpReduceSum(hq_memory);

        Tcompute delta = 0;
        if (lane_idx == 0) {
            const Tcompute v_value = static_cast<Tcompute>(
                op::recurrent_gated_delta_rule::cuda::loadAsFloat(
                    v, v_base + value_dim_idx));
            delta = (v_value - kv_memory) * beta_value;
            out[out_base + value_dim_idx] = static_cast<Tdata>(hq_memory + delta * kq_memory);
        }
        delta = __shfl_sync(
            INFINIOP_RECURRENT_DELTA_RULE_FULL_MASK, delta, 0, WARP_SIZE);

        const ptrdiff_t final_row_base = final_head_base + static_cast<ptrdiff_t>(value_dim_idx) * initial_s2;
#pragma unroll
        for (int i = 0; i < KEY_VALUES_PER_LANE; ++i) {
            const int key_dim_idx = lane_idx + i * WARP_SIZE;
            const Tcompute updated = state[i] * decay + k_local[i] * delta;
            initial_state
                [final_row_base + static_cast<ptrdiff_t>(key_dim_idx) * initial_s3]
                = static_cast<Tdata>(updated);
        }
    }
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
    constexpr size_t NUM_THREADS = WARPS_PER_BLOCK * INFINIOP_RECURRENT_DELTA_RULE_WARP_SIZE;
    dim3 grid(uint32_t(_info.B), uint32_t(_info.Hv), uint32_t((_info.Dv + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK));
    dim3 block(NUM_THREADS);
    size_t shared_mem_size = (Dk * 3 + NUM_THREADS + 1) * sizeof(float);

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
            _info.indexed_state_pool,
            _info.pool_size,
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
template <typename Tdata, typename Tgate>
infiniStatus_t launchHeadTileKernelTyped(
    const RecurrentGatedDeltaRuleInfo &_info, void *out, void *initial_state,
    const void *q, const void *k, const void *v, const void *g,
    const void *beta, const void *initial_state_indices,
    const void *final_state_indices, bool initial_state_indices_i64,
    bool final_state_indices_i64, cudaStream_t stream) {
    constexpr int THREADS_PER_BLOCK = 256;
    dim3 grid(uint32_t(_info.B), uint32_t(_info.Hv), 2);
    dim3 block(THREADS_PER_BLOCK);
    recurrentGatedDeltaRuleHeadTileKernel<Tdata, Tgate, float>
        <<<grid, block, 0, stream>>>(
            static_cast<Tdata *>(out),
            static_cast<Tdata *>(initial_state),
            static_cast<const Tdata *>(q), static_cast<const Tdata *>(k),
            static_cast<const Tdata *>(v), static_cast<const Tgate *>(g),
            static_cast<const Tgate *>(beta), initial_state_indices,
            final_state_indices, initial_state_indices_i64,
            final_state_indices_i64, _info.use_qk_l2norm, _info.pool_size,
            _info.value_heads_per_key_head, _info.out_strides[0],
            _info.out_strides[2],
            _info.initial_state_strides[0], _info.initial_state_strides[1],
            _info.initial_state_strides[2], _info.initial_state_strides[3],
            _info.q_strides[0], _info.q_strides[2],
            _info.k_strides[0], _info.k_strides[2],
            _info.v_strides[0], _info.v_strides[2],
            _info.g_strides[0], _info.g_strides[2],
            _info.beta_strides[0], _info.beta_strides[2]);
    return infiniStatus_t::INFINI_STATUS_SUCCESS;
}

template <typename Tdata>
infiniStatus_t launchHeadTileKernelForGate(
    const RecurrentGatedDeltaRuleInfo &_info, void *out, void *initial_state,
    const void *q, const void *k, const void *v, const void *g,
    const void *beta, const void *initial_state_indices,
    const void *final_state_indices, bool initial_state_indices_i64,
    bool final_state_indices_i64, cudaStream_t stream) {
    switch (_info.gate_dtype) {
    case INFINI_DTYPE_F16:
        return launchHeadTileKernelTyped<Tdata, half>(
            _info, out, initial_state, q, k, v, g, beta,
            initial_state_indices, final_state_indices, initial_state_indices_i64,
            final_state_indices_i64, stream);
    case INFINI_DTYPE_BF16:
        return launchHeadTileKernelTyped<Tdata, __nv_bfloat16>(
            _info, out, initial_state, q, k, v, g, beta,
            initial_state_indices, final_state_indices, initial_state_indices_i64,
            final_state_indices_i64, stream);
    case INFINI_DTYPE_F32:
        return launchHeadTileKernelTyped<Tdata, float>(
            _info, out, initial_state, q, k, v, g, beta,
            initial_state_indices, final_state_indices, initial_state_indices_i64,
            final_state_indices_i64, stream);
    default:
        return infiniStatus_t::INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

infiniStatus_t launchHeadTileKernel(
    const RecurrentGatedDeltaRuleInfo &_info, void *out, void *initial_state,
    const void *q, const void *k, const void *v, const void *g,
    const void *beta, const void *initial_state_indices,
    const void *final_state_indices, bool initial_state_indices_i64,
    bool final_state_indices_i64, cudaStream_t stream) {
    switch (_info.data_dtype) {
    case INFINI_DTYPE_F16:
        return launchHeadTileKernelForGate<half>(
            _info, out, initial_state, q, k, v, g, beta,
            initial_state_indices, final_state_indices, initial_state_indices_i64,
            final_state_indices_i64, stream);
    case INFINI_DTYPE_BF16:
        return launchHeadTileKernelForGate<__nv_bfloat16>(
            _info, out, initial_state, q, k, v, g, beta,
            initial_state_indices, final_state_indices, initial_state_indices_i64,
            final_state_indices_i64, stream);
    case INFINI_DTYPE_F32:
        return launchHeadTileKernelForGate<float>(
            _info, out, initial_state, q, k, v, g, beta,
            initial_state_indices, final_state_indices, initial_state_indices_i64,
            final_state_indices_i64, stream);
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

    constexpr size_t WARPS_PER_BLOCK = 256 / INFINIOP_RECURRENT_DELTA_RULE_WARP_SIZE;
    if (_info.Dk == 128 && _info.Dv == 128
        && _info.has_initial_state_indices && _info.has_final_state_indices
        && _info.Hk == 16 && _info.Hv == 32
        && _opaque->internal->maxThreadsPerBlock() >= 256) {
        return launchHeadTileKernel(
            _info, out, initial_state, q, k, v, g, beta,
            initial_state_indices, final_state_indices,
            initial_indices_i64, final_indices_i64, stream);
    }
    if (_info.Dk == 128 && _info.Dv == 128) {
        if (_opaque->internal->maxThreadsPerBlock() >= 256) {
            return launchIndexedPoolWarpKernel<128, 128, WARPS_PER_BLOCK>(
                _info, out, initial_state, final_state, q, k, v, g, beta,
                initial_state_indices, final_state_indices,
                initial_indices_i64, final_indices_i64, stream);
        }
    } else if (_info.Dk == 64 && _info.Dv == 64) {
        if (_opaque->internal->maxThreadsPerBlock() >= 256) {
            return launchIndexedPoolWarpKernel<64, 64, WARPS_PER_BLOCK>(
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
