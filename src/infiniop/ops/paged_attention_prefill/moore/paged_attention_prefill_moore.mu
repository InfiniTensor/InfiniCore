#include <musa_fp16.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <cmath>

#include "../../../devices/moore/moore_common.h"
#include "../../../devices/moore/moore_kernel_common.h"
#include "paged_attention_prefill_kernel.h"
#include "paged_attention_prefill_moore.h"

template <typename Tindex, typename Tdata, int QK_HEAD_SIZE, int VALUE_SIZE>
INFINIOP_MOORE_KERNEL pagedAttentionPrefillMlaKernel(
    Tdata *out_, const Tdata *q_, const Tdata *k_cache_, const Tdata *v_cache_,
    const Tindex *block_tables_,
    const Tindex *total_kv_lens_,
    const Tindex *cum_seq_lens_q_,
    const float *alibi_slopes_,
    const size_t num_heads, const size_t num_kv_heads, const float scale,
    const size_t page_block_size,
    const ptrdiff_t block_table_batch_stride,
    const ptrdiff_t q_stride, const ptrdiff_t q_head_stride,
    const ptrdiff_t k_batch_stride, const ptrdiff_t k_row_stride, const ptrdiff_t k_head_stride,
    const ptrdiff_t v_batch_stride, const ptrdiff_t v_row_stride, const ptrdiff_t v_head_stride,
    const ptrdiff_t o_stride, const ptrdiff_t o_head_stride,
    const size_t num_seqs) {

    constexpr int WARP_SIZE = 32;
    constexpr int Q_DIMS_PER_LANE = QK_HEAD_SIZE / WARP_SIZE;
    constexpr int V_DIMS_PER_LANE = VALUE_SIZE / WARP_SIZE;

    const size_t global_token_idx = static_cast<size_t>(blockIdx.x);
    const size_t head_idx = static_cast<size_t>(blockIdx.y);
    const int lane = static_cast<int>(threadIdx.x);
    if (head_idx >= num_heads || lane >= WARP_SIZE) {
        return;
    }

    const size_t seq_idx = op::paged_attention_prefill::cuda::find_seq_id<Tindex>(global_token_idx, cum_seq_lens_q_, num_seqs);
    const size_t q_start = static_cast<size_t>(cum_seq_lens_q_[seq_idx]);
    const size_t q_end = static_cast<size_t>(cum_seq_lens_q_[seq_idx + 1]);
    const size_t q_token_idx = global_token_idx - q_start;
    const size_t q_len = q_end - q_start;
    const size_t total_kv_len = static_cast<size_t>(total_kv_lens_[seq_idx]);
    const size_t history_len = total_kv_len - q_len;
    const size_t causal_limit = history_len + q_token_idx;

    const size_t num_queries_per_kv = num_heads / num_kv_heads;
    const size_t kv_head_idx = head_idx / num_queries_per_kv;
    const Tindex *block_table = block_tables_ + static_cast<ptrdiff_t>(seq_idx) * block_table_batch_stride;
    const Tdata *q_vec = q_ + static_cast<ptrdiff_t>(global_token_idx) * q_stride + static_cast<ptrdiff_t>(head_idx) * q_head_stride;
    Tdata *out_vec = out_ + static_cast<ptrdiff_t>(global_token_idx) * o_stride + static_cast<ptrdiff_t>(head_idx) * o_head_stride;
    const float alibi_slope = (alibi_slopes_ == nullptr) ? 0.0f : alibi_slopes_[head_idx];

    float q_reg[Q_DIMS_PER_LANE];
    float acc[V_DIMS_PER_LANE];
#pragma unroll
    for (int i = 0; i < Q_DIMS_PER_LANE; ++i) {
        q_reg[i] = static_cast<float>(q_vec[lane * Q_DIMS_PER_LANE + i]);
    }
#pragma unroll
    for (int i = 0; i < V_DIMS_PER_LANE; ++i) {
        acc[i] = 0.0f;
    }

    float m = -INFINITY;
    float l = 0.0f;
    const int pbs = static_cast<int>(page_block_size);
    for (size_t t = 0; t <= causal_limit; ++t) {
        const size_t page = t / static_cast<size_t>(pbs);
        const size_t off = t - page * static_cast<size_t>(pbs);
        const ptrdiff_t phys = static_cast<ptrdiff_t>(block_table[page]);
        const Tdata *k_vec = k_cache_ + phys * k_batch_stride + static_cast<ptrdiff_t>(off) * k_row_stride + static_cast<ptrdiff_t>(kv_head_idx) * k_head_stride;
        const Tdata *v_vec = v_cache_ + phys * v_batch_stride + static_cast<ptrdiff_t>(off) * v_row_stride + static_cast<ptrdiff_t>(kv_head_idx) * v_head_stride;

        float qk = 0.0f;
#pragma unroll
        for (int i = 0; i < Q_DIMS_PER_LANE; ++i) {
            const int dim = lane * Q_DIMS_PER_LANE + i;
            qk += q_reg[i] * static_cast<float>(k_vec[dim]);
        }
        qk = op::paged_attention_prefill::cuda::warpReduceSum(qk, 0xffffffffu);

        float alpha = 1.0f;
        float beta = 0.0f;
        if (lane == 0) {
            float score = qk * scale;
            if (alibi_slope != 0.0f) {
                score += alibi_slope * static_cast<float>(t - causal_limit);
            }
            const float m_new = fmaxf(m, score);
            alpha = expf(m - m_new);
            beta = expf(score - m_new);
            l = l * alpha + beta;
            m = m_new;
        }
        alpha = __shfl_sync(0xffffffff, alpha, 0);
        beta = __shfl_sync(0xffffffff, beta, 0);

#pragma unroll
        for (int i = 0; i < V_DIMS_PER_LANE; ++i) {
            const int dim = lane * V_DIMS_PER_LANE + i;
            acc[i] = acc[i] * alpha + beta * static_cast<float>(v_vec[dim]);
        }
    }

    float inv_l = 0.0f;
    if (lane == 0) {
        inv_l = 1.0f / (l + 1e-6f);
    }
    inv_l = __shfl_sync(0xffffffff, inv_l, 0);

#pragma unroll
    for (int i = 0; i < V_DIMS_PER_LANE; ++i) {
        const int dim = lane * V_DIMS_PER_LANE + i;
        out_vec[dim] = static_cast<Tdata>(acc[i] * inv_l);
    }
}

template <typename Tindex, typename Tdata, typename Tcompute>
infiniStatus_t launchPagedAttentionPrefill(
    Tdata *out, const Tdata *q, const Tdata *k_cache, const Tdata *v_cache,
    const Tindex *block_tables,
    const Tindex *seq_lens,
    const Tindex *cum_seq_lens_q,
    const float *alibi_slopes,
    const size_t num_heads,
    const size_t num_seqs,
    const size_t num_kv_heads,
    const float scale,
    const size_t max_num_blocks_per_seq,
    const size_t page_block_size,
    const size_t total_q_tokens,
    const size_t head_size,
    const size_t value_size,
    const ptrdiff_t block_table_batch_stride,
    const ptrdiff_t k_batch_stride,
    const ptrdiff_t k_head_stride,
    const ptrdiff_t k_row_stride,
    const ptrdiff_t v_batch_stride,
    const ptrdiff_t v_row_stride,
    const ptrdiff_t v_head_stride,
    const ptrdiff_t o_stride,
    const ptrdiff_t o_head_stride,
    const ptrdiff_t q_stride,
    const ptrdiff_t q_head_stride,
    musaStream_t stream) {

    if (total_q_tokens == 0 || num_heads == 0) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    dim3 grid(total_q_tokens, num_heads);
    if (head_size == 576 && value_size == 512) {
        dim3 block(32);
        pagedAttentionPrefillMlaKernel<Tindex, Tdata, 576, 512><<<grid, block, 0, stream>>>(
            out, q, k_cache, v_cache, block_tables, seq_lens, cum_seq_lens_q, alibi_slopes,
            num_heads, num_kv_heads, scale, page_block_size,
            block_table_batch_stride, q_stride, q_head_stride,
            k_batch_stride, k_row_stride, k_head_stride,
            v_batch_stride, v_row_stride, v_head_stride,
            o_stride, o_head_stride, num_seqs);
        return INFINI_STATUS_SUCCESS;
    }
    dim3 block(head_size);

    op::paged_attention_prefill::cuda::pagedAttentionPrefillKernel<Tindex, Tdata, Tcompute>
        <<<grid, block, 0, stream>>>(
            out, q, k_cache, v_cache,
            block_tables, seq_lens, cum_seq_lens_q, alibi_slopes,
            num_heads, num_kv_heads, scale,
            max_num_blocks_per_seq, page_block_size,
            k_batch_stride, k_head_stride,
            q_stride, q_head_stride,
            head_size,
            num_seqs);

    return INFINI_STATUS_SUCCESS;
}

namespace op::paged_attention_prefill::moore {

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
    infiniopTensorDescriptor_t q_desc,
    infiniopTensorDescriptor_t k_cache_desc,
    infiniopTensorDescriptor_t v_cache_desc,
    infiniopTensorDescriptor_t block_tables_desc,
    infiniopTensorDescriptor_t seq_lens_desc,
    infiniopTensorDescriptor_t cum_seq_lens_q_desc,
    const std::optional<infiniopTensorDescriptor_t> &alibi_slopes_desc,
    float scale) {

    auto info = PagedAttentionPrefillInfo::create(
        out_desc, q_desc, k_cache_desc, v_cache_desc,
        block_tables_desc, seq_lens_desc,
        cum_seq_lens_q_desc,
        alibi_slopes_desc, scale);

    CHECK_RESULT(info);

    *desc_ptr = new Descriptor(
        new Opaque{reinterpret_cast<device::moore::Handle *>(handle)->internal()},
        info.take(), 0, handle->device, handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size,
    void *out, const void *q, const void *k_cache, const void *v_cache,
    const void *block_tables,
    const void *seq_lens,
    const void *cum_seq_lens_q,
    const void *alibi_slopes,
    void *stream_) const {

    musaStream_t stream = (musaStream_t)stream_;

#define DISPATCH_KERNEL(Tindex, Tdata, Tcompute)                                                             \
    return launchPagedAttentionPrefill<Tindex, Tdata, Tcompute>(                                                  \
        (Tdata *)out, (const Tdata *)q, (const Tdata *)k_cache, (const Tdata *)v_cache,            \
        static_cast<const Tindex *>(block_tables), static_cast<const Tindex *>(seq_lens), static_cast<const Tindex *>(cum_seq_lens_q), \
        (const float *)alibi_slopes,                                                               \
        _info.num_heads, _info.num_seqs, _info.num_kv_heads,                                       \
        _info.scale, _info.max_num_blocks_per_seq,                                                 \
        _info.page_block_size, _info.total_q_tokens,                                                    \
        _info.head_size, _info.value_size,                                                        \
        _info.block_table_batch_stride,                                                           \
        _info.k_batch_stride, _info.k_head_stride, _info.k_row_stride,                            \
        _info.v_batch_stride, _info.v_row_stride, _info.v_head_stride,                            \
        _info.o_stride, _info.o_head_stride,                                                      \
        _info.q_stride, _info.q_head_stride,                                                       \
        stream)

#define DISPATCH_INDEX(Tindex)                             \
    do {                                                   \
        if (_info.dtype == INFINI_DTYPE_F16) {             \
            DISPATCH_KERNEL(Tindex, half, float);          \
        }                                                  \
        if (_info.dtype == INFINI_DTYPE_BF16) {            \
            DISPATCH_KERNEL(Tindex, __nv_bfloat16, float); \
        }                                                  \
        return INFINI_STATUS_BAD_TENSOR_DTYPE;             \
    } while (false)

    if (_info.index_dtype == INFINI_DTYPE_I64){
        DISPATCH_INDEX(int64_t);
    } else if (_info.index_dtype == INFINI_DTYPE_I32){
        DISPATCH_INDEX(int32_t);
    } else if (_info.index_dtype == INFINI_DTYPE_U32){
        DISPATCH_INDEX(uint32_t);
    }

    return INFINI_STATUS_BAD_TENSOR_DTYPE;
}

} // namespace op::paged_attention_prefill::moore
