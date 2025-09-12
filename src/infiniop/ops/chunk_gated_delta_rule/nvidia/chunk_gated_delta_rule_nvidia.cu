// chunk_gated_delta_rule_nvidia.cu

#include "../../../devices/nvidia/nvidia_common.cuh"
#include "chunk_gated_delta_rule_nvidia.cuh"

#include "../../../devices/nvidia/nvidia_kernel_common.cuh"

#include "../cuda/kernel.cuh"
#include <cuda_runtime.h>



// Kernel Launcher Wrapper
template <typename Tdata, typename Tcompute, size_t Dk, size_t Dv, size_t NUM_THREADS>
INFINIOP_CUDA_KERNEL chunkGatedDeltaRule(
    Tdata* out, Tdata* final_state,
    const Tdata* q, const Tdata* k, const Tdata* v,
    const Tdata* g, const Tdata* beta, const Tdata* initial_state,
    bool use_qk_l2norm, size_t chunk_size, size_t T
) {
    chunkGatedDeltaRuleKernel<Tdata, Tcompute, Dk, Dv, NUM_THREADS>(
        out, final_state, q, k, v, g, beta, initial_state, use_qk_l2norm, chunk_size, T
    );
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
    infiniopTensorDescriptor_t final_state_desc,
    infiniopTensorDescriptor_t q_desc,
    infiniopTensorDescriptor_t k_desc,
    infiniopTensorDescriptor_t v_desc,
    infiniopTensorDescriptor_t g_desc,
    infiniopTensorDescriptor_t beta_desc,
    const std::optional<infiniopTensorDescriptor_t>& initial_state_desc,
    bool use_qk_l2norm,
    size_t chunk_size
) {
    auto info = ChunkGatedDeltaRuleInfo::create(
        out_desc, final_state_desc, q_desc, k_desc, v_desc,
        g_desc, beta_desc, initial_state_desc, use_qk_l2norm, chunk_size);
    CHECK_RESULT(info);

    // Calculate workspace size if needed, here it's 0
    size_t workspace_size = 0;

    *desc_ptr = new Descriptor(
        new Opaque{reinterpret_cast<device::nvidia::Handle *>(handle)->internal()},
        info.take(), workspace_size, handle->device, handle->device_id);
    
    return infiniStatus_t::INFINI_STATUS_SUCCESS;
}

template <size_t Dk, size_t Dv, size_t NUM_THREADS>
infiniStatus_t launchKernel(
    void *out, void* final_state,
    const void *q, const void *k, const void *v,
    const void *g, const void* beta, const void* initial_state,
    bool use_qk_l2norm,
    infiniDtype_t dtype,
    size_t B, size_t H, size_t T, size_t chunk_size,
    cudaStream_t stream
) {
    dim3 grid(uint32_t(B), uint32_t(H), 1);
    dim3 block(NUM_THREADS);
    // Shared memory for local Q, K, and one reduction value
    // size_t shared_mem_size = (Dk + Dk + NUM_THREADS) * sizeof(float);

    using Tcompute = float;
    using BlockScan = cub::BlockScan<Tcompute, NUM_THREADS>;
    // using BlockReduce = cub::BlockReduce<Tcompute, NUM_THREADS>;

    // size_t shared_mem_size = (
    //     chunk_size * (3 * Dk + Dv + 3) + 
    //     chunk_size * chunk_size +
    //     Dk * Dv
    // ) * sizeof(Tcompute) + sizeof(typename BlockScan::TempStorage) + sizeof(typename BlockReduce::TempStorage);
    // size_t shared_mem_size = (
    //     // q_s, k_s, k_beta_s, k_cumdecay_s
    //     chunk_size * 4 * Dk +
    //     // v_s, value_prime_s, v_prime_s, attn_inter_s
    //     chunk_size * 4 * Dv +
    //     // g_s, beta_s, g_cumsum_s
    //     chunk_size * 3 +
    //     // attn_s (removed decay_mask_s)
    //     chunk_size * chunk_size +
    //     // inter_chunk_state_s
    //     Dk * Dv
    // ) * sizeof(Tcompute) + sizeof(typename BlockScan::TempStorage) + sizeof(typename BlockReduce::TempStorage);

    // size_t shared_mem_size = (
    //     // q_s, k_s, k_beta_s, k_cumdecay_s
    //     chunk_size * 4 * Dk +
    //     // v_s, value_prime_s, v_prime_s (v_new_s is still here from prev version)
    //     chunk_size * 4 * Dv +
    //     // g_s, beta_s, g_cumsum_s
    //     chunk_size * 3 +
    //     // attn_s
    //     chunk_size * chunk_size +
    //     // inter_chunk_state_s
    //     Dk * Dv
    // ) * sizeof(Tcompute) + sizeof(typename BlockScan::TempStorage) + sizeof(typename BlockReduce::TempStorage);
    size_t shared_mem_size = (
        // q_s, k_s, k_beta_s, k_cumdecay_s
        chunk_size * 4 * Dk +
        // v_s, value_prime_s, v_prime_s, attn_inter_s
        chunk_size * 4 * Dv +
        // g_s, beta_s, g_cumsum_s
        chunk_size * 3 +
        // attn_s
        chunk_size * chunk_size
        // NOTE: Dk * Dv term for inter_chunk_state_s has been removed.
    ) * sizeof(Tcompute) + sizeof(typename BlockScan::TempStorage);

    if (dtype == INFINI_DTYPE_F16) {
        chunkGatedDeltaRule<half, float, Dk, Dv, NUM_THREADS>
            <<<grid, block, shared_mem_size, stream>>>(
                (half*)out, (half*)final_state,
                (const half*)q, (const half*)k, (const half*)v,
                (const half*)g, (const half*)beta, (const half*)initial_state,
                use_qk_l2norm, chunk_size, T
            );
    } else if (dtype == INFINI_DTYPE_BF16) {
        chunkGatedDeltaRule<__nv_bfloat16, float, Dk, Dv, NUM_THREADS>
            <<<grid, block, shared_mem_size, stream>>>(
                (__nv_bfloat16*)out, (__nv_bfloat16*)final_state,
                (const __nv_bfloat16*)q, (const __nv_bfloat16*)k, (const __nv_bfloat16*)v,
                (const __nv_bfloat16*)g, (const __nv_bfloat16*)beta, (const __nv_bfloat16*)initial_state,
                use_qk_l2norm, chunk_size, T
            );
    } else if (dtype == INFINI_DTYPE_F32) {
        chunkGatedDeltaRule<float, float, Dk, Dv, NUM_THREADS>
            <<<grid, block, shared_mem_size, stream>>>(
                (float*)out, (float*)final_state,
                (const float*)q, (const float*)k, (const float*)v,
                (const float*)g, (const float*)beta, (const float*)initial_state,
                use_qk_l2norm, chunk_size, T
            );
    } else {
        return infiniStatus_t::INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    return infiniStatus_t::INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size,
    void *out, void* final_state,
    const void *q, const void *k, const void *v,
    const void *g, const void* beta, const void* initial_state,
    void *stream_
) const {
    cudaStream_t stream = (cudaStream_t)stream_;

    // Specialize for common shapes and thread counts
    if (_info.Dk == 128 && _info.Dv == 128) {
         if (_opaque->internal->maxThreadsPerBlock() >= 128) {
            return launchKernel<128, 128, 128>(
                out, final_state, q, k, v, g, beta, initial_state, _info.use_qk_l2norm,
                _info.dtype, _info.B, _info.H, _info.T, _info.chunk_size, stream);
        }
    } else if (_info.Dk == 64 && _info.Dv == 64) {
        if (_opaque->internal->maxThreadsPerBlock() >= 64) {
            return launchKernel<64, 64, 64>(
                out, final_state, q, k, v, g, beta, initial_state, _info.use_qk_l2norm,
                _info.dtype, _info.B, _info.H, _info.T, _info.chunk_size, stream);
        }
    }
    
    // Fallback or error for unsupported shapes
    // You can add more specializations for other shapes here.
    return infiniStatus_t::INFINI_STATUS_BAD_TENSOR_SHAPE;
}

} // namespace nvidia
} // namespace chunk_gated_delta_rule
} // namespace op