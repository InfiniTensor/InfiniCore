#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "../../../devices/nvidia/nvidia_common.cuh"
#include "paged_attention_nvidia.cuh"

namespace op::paged_attention::nvidia {

infiniStatus_t launch_decode_hd64_i64(
    void *workspace, size_t workspace_size,
    void *out, const void *q, const void *k_cache, const void *v_cache,
    infiniDtype_t dtype, const int64_t *block_tables, const int64_t *cache_lens, const float *alibi_slopes,
    size_t num_heads, size_t num_seqs, size_t num_kv_heads, float scale, size_t max_num_blocks_per_seq, size_t page_block_size,
    ptrdiff_t q_stride, ptrdiff_t k_batch_stride, ptrdiff_t k_row_stride, ptrdiff_t k_head_stride,
    ptrdiff_t v_batch_stride, ptrdiff_t v_row_stride, ptrdiff_t v_head_stride, ptrdiff_t o_stride,
    cudaStream_t stream);

infiniStatus_t launch_decode_hd64_i32(
    void *workspace, size_t workspace_size,
    void *out, const void *q, const void *k_cache, const void *v_cache,
    infiniDtype_t dtype, const int32_t *block_tables, const int32_t *cache_lens, const float *alibi_slopes,
    size_t num_heads, size_t num_seqs, size_t num_kv_heads, float scale, size_t max_num_blocks_per_seq, size_t page_block_size,
    ptrdiff_t q_stride, ptrdiff_t k_batch_stride, ptrdiff_t k_row_stride, ptrdiff_t k_head_stride,
    ptrdiff_t v_batch_stride, ptrdiff_t v_row_stride, ptrdiff_t v_head_stride, ptrdiff_t o_stride,
    cudaStream_t stream);

infiniStatus_t launch_decode_hd64_u32(
    void *workspace, size_t workspace_size,
    void *out, const void *q, const void *k_cache, const void *v_cache,
    infiniDtype_t dtype, const uint32_t *block_tables, const uint32_t *cache_lens, const float *alibi_slopes,
    size_t num_heads, size_t num_seqs, size_t num_kv_heads, float scale, size_t max_num_blocks_per_seq, size_t page_block_size,
    ptrdiff_t q_stride, ptrdiff_t k_batch_stride, ptrdiff_t k_row_stride, ptrdiff_t k_head_stride,
    ptrdiff_t v_batch_stride, ptrdiff_t v_row_stride, ptrdiff_t v_head_stride, ptrdiff_t o_stride,
    cudaStream_t stream);

infiniStatus_t launch_decode_hd128_i64(
    void *workspace, size_t workspace_size,
    void *out, const void *q, const void *k_cache, const void *v_cache,
    infiniDtype_t dtype, const int64_t *block_tables, const int64_t *cache_lens, const float *alibi_slopes,
    size_t num_heads, size_t num_seqs, size_t num_kv_heads, float scale, size_t max_num_blocks_per_seq, size_t page_block_size,
    ptrdiff_t q_stride, ptrdiff_t k_batch_stride, ptrdiff_t k_row_stride, ptrdiff_t k_head_stride,
    ptrdiff_t v_batch_stride, ptrdiff_t v_row_stride, ptrdiff_t v_head_stride, ptrdiff_t o_stride,
    cudaStream_t stream);

infiniStatus_t launch_decode_hd128_i32(
    void *workspace, size_t workspace_size,
    void *out, const void *q, const void *k_cache, const void *v_cache,
    infiniDtype_t dtype, const int32_t *block_tables, const int32_t *cache_lens, const float *alibi_slopes,
    size_t num_heads, size_t num_seqs, size_t num_kv_heads, float scale, size_t max_num_blocks_per_seq, size_t page_block_size,
    ptrdiff_t q_stride, ptrdiff_t k_batch_stride, ptrdiff_t k_row_stride, ptrdiff_t k_head_stride,
    ptrdiff_t v_batch_stride, ptrdiff_t v_row_stride, ptrdiff_t v_head_stride, ptrdiff_t o_stride,
    cudaStream_t stream);

infiniStatus_t launch_decode_hd128_u32(
    void *workspace, size_t workspace_size,
    void *out, const void *q, const void *k_cache, const void *v_cache,
    infiniDtype_t dtype, const uint32_t *block_tables, const uint32_t *cache_lens, const float *alibi_slopes,
    size_t num_heads, size_t num_seqs, size_t num_kv_heads, float scale, size_t max_num_blocks_per_seq, size_t page_block_size,
    ptrdiff_t q_stride, ptrdiff_t k_batch_stride, ptrdiff_t k_row_stride, ptrdiff_t k_head_stride,
    ptrdiff_t v_batch_stride, ptrdiff_t v_row_stride, ptrdiff_t v_head_stride, ptrdiff_t o_stride,
    cudaStream_t stream);

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
    infiniopTensorDescriptor_t q_desc,
    infiniopTensorDescriptor_t k_cache_desc,
    infiniopTensorDescriptor_t v_cache_desc,
    infiniopTensorDescriptor_t block_tables_desc,
    infiniopTensorDescriptor_t cache_lens_desc,
    const std::optional<infiniopTensorDescriptor_t> &alibi_slopes_desc,
    float scale) {

    auto info_res = PagedAttentionInfo::create(out_desc, q_desc, k_cache_desc, v_cache_desc, block_tables_desc, cache_lens_desc, alibi_slopes_desc, scale);
    CHECK_RESULT(info_res);
    auto info = info_res.take();
    // Reserve workspace for optional split-kv decode (partial acc + m/l).
    // Workspace is independent of runtime env toggles; kernels will clamp num_splits <= kMaxSplits.
    constexpr size_t kMaxSplits = 8;
    const size_t per_split = info.num_seqs * info.num_heads * (info.head_size + 2) * sizeof(float);
    const size_t workspace_bytes = kMaxSplits * per_split;

    *desc_ptr = new Descriptor(
        new Opaque{reinterpret_cast<device::nvidia::Handle *>(handle)->internal()},
        info, workspace_bytes, handle->device, handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size,
    void *out, const void *q, const void *k_cache, const void *v_cache,
    const void *block_tables, const void *cache_lens, const void *alibi_slopes,
    void *stream_) const {

    bool need_workspace = false;
    if (const char *env = std::getenv("INFINIOP_FLASH_DECODE_SPLITKV")) {
        // "auto" may enable split-kv depending on the runtime heuristic.
        need_workspace = (std::strcmp(env, "auto") == 0) || (std::strcmp(env, "1") == 0) || (std::strcmp(env, "true") == 0);
    } else {
        // Keep hd64 behavior unchanged, but for hd128 we default to split-kv decode, which needs workspace.
        need_workspace = (_info.head_size == 128);
    }
    if (need_workspace && workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    auto stream = static_cast<cudaStream_t>(stream_);

    const float *alibi_ptr = (alibi_slopes == nullptr) ? nullptr : static_cast<const float *>(alibi_slopes);

    if (_info.index_dtype == INFINI_DTYPE_I64) {
        const auto *block_table_i64 = static_cast<const int64_t *>(block_tables);
        const auto *cache_lens_i64 = static_cast<const int64_t *>(cache_lens);
        switch (_info.head_size) {
        case 64:
            return launch_decode_hd64_i64(
                workspace, workspace_size,
                out, q, k_cache, v_cache, _info.dtype,
                block_table_i64, cache_lens_i64, alibi_ptr,
                _info.num_heads, _info.num_seqs, _info.num_kv_heads, _info.scale,
                _info.max_num_blocks_per_seq, _info.page_block_size,
                _info.q_stride, _info.k_batch_stride, _info.k_row_stride, _info.k_head_stride,
                _info.v_batch_stride, _info.v_row_stride, _info.v_head_stride,
                _info.o_stride, stream);
        case 128:
            return launch_decode_hd128_i64(
                workspace, workspace_size,
                out, q, k_cache, v_cache, _info.dtype,
                block_table_i64, cache_lens_i64, alibi_ptr,
                _info.num_heads, _info.num_seqs, _info.num_kv_heads, _info.scale,
                _info.max_num_blocks_per_seq, _info.page_block_size,
                _info.q_stride, _info.k_batch_stride, _info.k_row_stride, _info.k_head_stride,
                _info.v_batch_stride, _info.v_row_stride, _info.v_head_stride,
                _info.o_stride, stream);
        default:
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
    }

    if (_info.index_dtype == INFINI_DTYPE_I32) {
        const auto *block_table_i32 = static_cast<const int32_t *>(block_tables);
        const auto *cache_lens_i32 = static_cast<const int32_t *>(cache_lens);
        switch (_info.head_size) {
        case 64:
            return launch_decode_hd64_i32(
                workspace, workspace_size,
                out, q, k_cache, v_cache, _info.dtype,
                block_table_i32, cache_lens_i32, alibi_ptr,
                _info.num_heads, _info.num_seqs, _info.num_kv_heads, _info.scale,
                _info.max_num_blocks_per_seq, _info.page_block_size,
                _info.q_stride, _info.k_batch_stride, _info.k_row_stride, _info.k_head_stride,
                _info.v_batch_stride, _info.v_row_stride, _info.v_head_stride,
                _info.o_stride, stream);
        case 128:
            return launch_decode_hd128_i32(
                workspace, workspace_size,
                out, q, k_cache, v_cache, _info.dtype,
                block_table_i32, cache_lens_i32, alibi_ptr,
                _info.num_heads, _info.num_seqs, _info.num_kv_heads, _info.scale,
                _info.max_num_blocks_per_seq, _info.page_block_size,
                _info.q_stride, _info.k_batch_stride, _info.k_row_stride, _info.k_head_stride,
                _info.v_batch_stride, _info.v_row_stride, _info.v_head_stride,
                _info.o_stride, stream);
        default:
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
    }

    if (_info.index_dtype == INFINI_DTYPE_U32) {
        const auto *block_table_u32 = static_cast<const uint32_t *>(block_tables);
        const auto *cache_lens_u32 = static_cast<const uint32_t *>(cache_lens);
        switch (_info.head_size) {
        case 64:
            return launch_decode_hd64_u32(
                workspace, workspace_size,
                out, q, k_cache, v_cache, _info.dtype,
                block_table_u32, cache_lens_u32, alibi_ptr,
                _info.num_heads, _info.num_seqs, _info.num_kv_heads, _info.scale,
                _info.max_num_blocks_per_seq, _info.page_block_size,
                _info.q_stride, _info.k_batch_stride, _info.k_row_stride, _info.k_head_stride,
                _info.v_batch_stride, _info.v_row_stride, _info.v_head_stride,
                _info.o_stride, stream);
        case 128:
            return launch_decode_hd128_u32(
                workspace, workspace_size,
                out, q, k_cache, v_cache, _info.dtype,
                block_table_u32, cache_lens_u32, alibi_ptr,
                _info.num_heads, _info.num_seqs, _info.num_kv_heads, _info.scale,
                _info.max_num_blocks_per_seq, _info.page_block_size,
                _info.q_stride, _info.k_batch_stride, _info.k_row_stride, _info.k_head_stride,
                _info.v_batch_stride, _info.v_row_stride, _info.v_head_stride,
                _info.o_stride, stream);
        default:
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
    }

    return INFINI_STATUS_BAD_TENSOR_DTYPE;
}

} // namespace op::paged_attention::nvidia

// #include <cub/block/block_reduce.cuh>

// #include "../../../devices/nvidia/nvidia_common.cuh"
// #include "../../../devices/nvidia/nvidia_kernel_common.cuh"

// #include "../../../reduce/cuda/reduce.cuh"
// #include "../cuda/kernel.cuh"
// #include "paged_attention_nvidia.cuh"

// template <typename Tdata, typename Tcompute, size_t HEAD_SIZE, size_t NUM_THREADS>
// INFINIOP_CUDA_KERNEL pagedAttention(
//     Tdata *out, const Tdata *q, const Tdata *k_cache, const Tdata *v_cache,
//     const int64_t *block_tables, const int64_t *seq_lens, const float *alibi_slopes,
//     const size_t num_kv_heads, const float scale, const size_t max_num_blocks_per_seq,
//     const size_t block_size,
//     const ptrdiff_t q_stride,
//     const ptrdiff_t kv_block_stride,
//     const ptrdiff_t kv_head_stride,
//     const ptrdiff_t o_stride) {
//     op::paged_attention::cuda::pagedAttentionKernel<Tdata, Tcompute, HEAD_SIZE, NUM_THREADS>(
//         out, q, k_cache, v_cache, block_tables, seq_lens, alibi_slopes, num_kv_heads, scale,
//         max_num_blocks_per_seq, block_size, q_stride, kv_block_stride, kv_head_stride, o_stride);
// }

// namespace op::paged_attention::nvidia {

// struct Descriptor::Opaque {
//     std::shared_ptr<device::nvidia::Handle::Internal> internal;
// };

// Descriptor::~Descriptor() {
//     delete _opaque;
// }

// infiniStatus_t Descriptor::create(
//     infiniopHandle_t handle,
//     Descriptor **desc_ptr,
//     infiniopTensorDescriptor_t out_desc,
//     infiniopTensorDescriptor_t q_desc,
//     infiniopTensorDescriptor_t k_cache_desc,
//     infiniopTensorDescriptor_t v_cache_desc,
//     infiniopTensorDescriptor_t block_tables_desc,
//     infiniopTensorDescriptor_t seq_lens_desc,
//     const std::optional<infiniopTensorDescriptor_t> &alibi_slopes_desc,
//     float scale) {
//     auto info = PagedAttentionInfo::create(out_desc, q_desc, k_cache_desc, v_cache_desc, block_tables_desc, seq_lens_desc, alibi_slopes_desc, scale);
//     CHECK_RESULT(info);
//     *desc_ptr = new Descriptor(
//         new Opaque{reinterpret_cast<device::nvidia::Handle *>(handle)->internal()},
//         info.take(), 0, handle->device, handle->device_id);

//     return INFINI_STATUS_SUCCESS;
// }

// template <size_t HEAD_SIZE, size_t NUM_THREADS>
// infiniStatus_t launchKernel(void *out, const void *q, const void *k_cache, const void *v_cache,
//                             infiniDtype_t dtype,
//                             const void *block_tables, const void *seq_lens, const void *alibi_slopes,
//                             size_t num_heads, size_t num_seqs,
//                             size_t num_kv_heads, float scale, size_t max_num_blocks_per_seq, size_t block_size,
//                             ptrdiff_t q_stride, ptrdiff_t kv_block_stride, ptrdiff_t kv_head_stride, ptrdiff_t o_stride,
//                             cudaStream_t stream) {
//     dim3 grid(uint64_t(num_heads), uint64_t(num_seqs), 1);
//     dim3 block(NUM_THREADS);
//     size_t shared_mem_size = (HEAD_SIZE + max_num_blocks_per_seq * block_size + 2) * sizeof(float);

//     if (dtype == INFINI_DTYPE_F16) {
//         pagedAttention<half, float, HEAD_SIZE, NUM_THREADS>
//             <<<grid, block, shared_mem_size, stream>>>(
//                 (half *)out,
//                 (const half *)q, (const half *)k_cache, (const half *)v_cache,
//                 (const int64_t *)block_tables, (const int64_t *)seq_lens, (const float *)alibi_slopes, num_kv_heads,
//                 scale, max_num_blocks_per_seq, block_size,
//                 q_stride, kv_block_stride, kv_head_stride, o_stride);
//     } else if (dtype == INFINI_DTYPE_BF16) {
//         pagedAttention<__nv_bfloat16, float, HEAD_SIZE, NUM_THREADS>
//             <<<grid, block, shared_mem_size, stream>>>(
//                 (__nv_bfloat16 *)out, (const __nv_bfloat16 *)q, (const __nv_bfloat16 *)k_cache, (const __nv_bfloat16 *)v_cache,
//                 (const int64_t *)block_tables, (const int64_t *)seq_lens, (const float *)alibi_slopes, num_kv_heads,
//                 scale, max_num_blocks_per_seq, block_size,
//                 q_stride, kv_block_stride, kv_head_stride, o_stride);
//     } else if (dtype == INFINI_DTYPE_F32) {
//         pagedAttention<float, float, HEAD_SIZE, NUM_THREADS>
//             <<<grid, block, shared_mem_size, stream>>>(
//                 (float *)out, (const float *)q, (const float *)k_cache, (const float *)v_cache,
//                 (const int64_t *)block_tables, (const int64_t *)seq_lens, (const float *)alibi_slopes, num_kv_heads,
//                 scale, max_num_blocks_per_seq, block_size,
//                 q_stride, kv_block_stride, kv_head_stride, o_stride);
//     } else {
//         return INFINI_STATUS_BAD_TENSOR_DTYPE;
//     }
//     return INFINI_STATUS_SUCCESS;
// }

// infiniStatus_t Descriptor::calculate(
//     void *workspace, size_t workspace_size,
//     void *out, const void *q, const void *k_cache, const void *v_cache,
//     const void *block_tables, const void *seq_lens, const void *alibi_slopes,
//     void *stream_) const {
//     cudaStream_t stream = (cudaStream_t)stream_;

// #define LAUNCH_HEADSIZE_BLOCKSIZE(__H_SIZE, __B_SIZE)                                    \
//     launchKernel<__H_SIZE, __B_SIZE>(                                                    \
//         out, q, k_cache, v_cache, _info.dtype, block_tables, seq_lens, alibi_slopes,     \
//         _info.num_heads, _info.num_seqs,                                                 \
//         _info.num_kv_heads, _info.scale, _info.max_num_blocks_per_seq, _info.block_size, \
//         _info.q_stride, _info.kv_block_stride, _info.kv_head_stride, _info.o_stride,     \
//         stream);

// #define SWITCH_HEAD_SIZE(__B_SIZE)               \
//     switch (_info.head_size) {                   \
//     case 16:                                     \
//         LAUNCH_HEADSIZE_BLOCKSIZE(16, __B_SIZE)  \
//         break;                                   \
//     case 32:                                     \
//         LAUNCH_HEADSIZE_BLOCKSIZE(32, __B_SIZE)  \
//         break;                                   \
//     case 64:                                     \
//         LAUNCH_HEADSIZE_BLOCKSIZE(64, __B_SIZE)  \
//         break;                                   \
//     case 128:                                    \
//         LAUNCH_HEADSIZE_BLOCKSIZE(128, __B_SIZE) \
//         break;                                   \
//     case 256:                                    \
//         LAUNCH_HEADSIZE_BLOCKSIZE(256, __B_SIZE) \
//         break;                                   \
//     default:                                     \
//         return INFINI_STATUS_BAD_TENSOR_SHAPE;   \
//     }

//     if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_1024) {
//         SWITCH_HEAD_SIZE(CUDA_BLOCK_SIZE_1024)
//     } else if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_512) {
//         SWITCH_HEAD_SIZE(CUDA_BLOCK_SIZE_512)
//     } else if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_4096) {
//         SWITCH_HEAD_SIZE(CUDA_BLOCK_SIZE_4096)
//     } else {
//         return INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED;
//     }

// #undef LAUNCH_HEADSIZE_BLOCKSIZE
// #undef SWITCH_HEAD_SIZE

//     return INFINI_STATUS_SUCCESS;
// }

// } // namespace op::paged_attention::nvidia
