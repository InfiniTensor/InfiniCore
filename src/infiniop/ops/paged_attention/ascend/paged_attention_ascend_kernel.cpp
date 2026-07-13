#include "../../../devices/ascend/ascend_kernel_common.h"
#include <type_traits>

using namespace AscendC;

constexpr size_t PAGED_ATTENTION_TILE = 128;

template <typename T>
__aicore__ inline float dataToFloat(T value) {
    if constexpr (std::is_same<T, bfloat16_t>::value) {
        uint32_t bits = static_cast<uint32_t>(*reinterpret_cast<uint16_t *>(&value)) << 16;
        return *reinterpret_cast<float *>(&bits);
    } else {
        return static_cast<float>(value);
    }
}

template <typename Tindex>
__aicore__ inline int64_t loadIndex(GlobalTensor<Tindex> &tensor, ptrdiff_t offset) {
    return static_cast<int64_t>(tensor.GetValue(offset));
}

template <typename Tdata, typename Tindex>
class PagedAttentionKernel {
public:
    __aicore__ inline PagedAttentionKernel() {}

    __aicore__ inline void init(
        GM_ADDR out,
        GM_ADDR q,
        GM_ADDR k_cache,
        GM_ADDR v_cache,
        GM_ADDR block_tables,
        GM_ADDR cache_lens,
        GM_ADDR alibi_slopes,
        bool has_alibi,
        size_t num_heads,
        size_t num_seqs,
        size_t num_kv_heads,
        size_t head_size,
        float scale,
        size_t max_num_blocks_per_seq,
        size_t page_block_size,
        ptrdiff_t q_stride,
        ptrdiff_t k_batch_stride,
        ptrdiff_t k_row_stride,
        ptrdiff_t k_head_stride,
        ptrdiff_t v_batch_stride,
        ptrdiff_t v_row_stride,
        ptrdiff_t v_head_stride,
        ptrdiff_t o_stride,
        ptrdiff_t block_table_batch_stride,
        ptrdiff_t cache_lens_stride) {
        _num_heads = num_heads;
        _num_seqs = num_seqs;
        _num_kv_heads = num_kv_heads;
        _head_size = head_size;
        _scale = scale;
        _max_num_blocks_per_seq = max_num_blocks_per_seq;
        _page_block_size = page_block_size;
        _q_stride = q_stride;
        _k_batch_stride = k_batch_stride;
        _k_row_stride = k_row_stride;
        _k_head_stride = k_head_stride;
        _v_batch_stride = v_batch_stride;
        _v_row_stride = v_row_stride;
        _v_head_stride = v_head_stride;
        _o_stride = o_stride;
        _block_table_batch_stride = block_table_batch_stride;
        _cache_lens_stride = cache_lens_stride;
        _has_alibi = has_alibi;

        _out_gm.SetGlobalBuffer(reinterpret_cast<__gm__ Tdata *>(out));
        _q_gm.SetGlobalBuffer(reinterpret_cast<__gm__ Tdata *>(q));
        _k_cache_gm.SetGlobalBuffer(reinterpret_cast<__gm__ Tdata *>(k_cache));
        _v_cache_gm.SetGlobalBuffer(reinterpret_cast<__gm__ Tdata *>(v_cache));
        _block_tables_gm.SetGlobalBuffer(reinterpret_cast<__gm__ Tindex *>(block_tables));
        _cache_lens_gm.SetGlobalBuffer(reinterpret_cast<__gm__ Tindex *>(cache_lens));
        if (_has_alibi) {
            _alibi_slopes_gm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(alibi_slopes));
        }

        _pipe.InitBuffer(_q_buf, alignTileLen<float>(_head_size, BYTE_ALIGN) * sizeof(float));
        _pipe.InitBuffer(_acc_buf, alignTileLen<float>(_head_size, BYTE_ALIGN) * sizeof(float));
        _pipe.InitBuffer(_score_buf, PAGED_ATTENTION_TILE * sizeof(float));
        _pipe.InitBuffer(_out_buf, alignTileLen<Tdata>(_head_size, BYTE_ALIGN) * sizeof(Tdata));
    }

    __aicore__ inline void process() {
        const size_t work_idx = GetBlockIdx();
        const size_t head_idx = work_idx % _num_heads;
        const size_t seq_idx = work_idx / _num_heads;
        if (seq_idx >= _num_seqs) {
            return;
        }

        const int64_t seq_len_i64 = loadIndex(_cache_lens_gm, static_cast<ptrdiff_t>(seq_idx) * _cache_lens_stride);
        if (seq_len_i64 <= 0) {
            return;
        }
        const size_t seq_len = static_cast<size_t>(seq_len_i64);
        const size_t num_queries_per_kv = _num_heads / _num_kv_heads;
        const size_t kv_head_idx = head_idx / num_queries_per_kv;
        const float alibi_slope = _has_alibi ? _alibi_slopes_gm.GetValue(head_idx) : 0.0f;

        LocalTensor<float> q_local = _q_buf.Get<float>();
        LocalTensor<float> acc_local = _acc_buf.Get<float>();
        LocalTensor<float> score_local = _score_buf.Get<float>();

        const ptrdiff_t q_base = static_cast<ptrdiff_t>(seq_idx) * _q_stride
                               + static_cast<ptrdiff_t>(head_idx * _head_size);
        for (size_t d = 0; d < _head_size; ++d) {
            q_local.SetValue(d, dataToFloat(_q_gm.GetValue(q_base + static_cast<ptrdiff_t>(d))));
            acc_local.SetValue(d, 0.0f);
        }

        float max_score = -3.4028234663852886e38f;
        for (size_t token_idx = 0; token_idx < seq_len; ++token_idx) {
            const float score = computeScore(q_local, seq_idx, kv_head_idx, token_idx, alibi_slope, seq_len);
            if (score > max_score) {
                max_score = score;
            }
        }

        float sum_exp = 0.0f;
        for (size_t tile_begin = 0; tile_begin < seq_len; tile_begin += PAGED_ATTENTION_TILE) {
            size_t tile_count = seq_len - tile_begin;
            if (tile_count > PAGED_ATTENTION_TILE) {
                tile_count = PAGED_ATTENTION_TILE;
            }

            for (size_t i = 0; i < tile_count; ++i) {
                const float score = computeScore(q_local, seq_idx, kv_head_idx, tile_begin + i, alibi_slope, seq_len);
                score_local.SetValue(i, score - max_score);
            }
            Exp(score_local, score_local, static_cast<int32_t>(tile_count));

            for (size_t i = 0; i < tile_count; ++i) {
                const float prob = score_local.GetValue(i);
                sum_exp += prob;

                const ptrdiff_t v_base = valueBase(seq_idx, kv_head_idx, tile_begin + i);
                for (size_t d = 0; d < _head_size; ++d) {
                    const float acc = acc_local.GetValue(d);
                    const float v_val = dataToFloat(_v_cache_gm.GetValue(v_base + static_cast<ptrdiff_t>(d)));
                    acc_local.SetValue(d, acc + prob * v_val);
                }
            }
        }

        const float inv_sum = 1.0f / (sum_exp + 1e-6f);
        LocalTensor<Tdata> out_local = _out_buf.Get<Tdata>();
        for (size_t d = 0; d < _head_size; ++d) {
            acc_local.SetValue(d, acc_local.GetValue(d) * inv_sum);
        }
        Cast(out_local, acc_local, AscendC::RoundMode::CAST_RINT, static_cast<int32_t>(_head_size));

        const ptrdiff_t out_base = static_cast<ptrdiff_t>(seq_idx) * _o_stride
                                 + static_cast<ptrdiff_t>(head_idx * _head_size);
        for (size_t d = 0; d < _head_size; ++d) {
            _out_gm.SetValue(out_base + static_cast<ptrdiff_t>(d), out_local.GetValue(d));
        }
    }

private:
    __aicore__ inline ptrdiff_t keyBase(size_t seq_idx, size_t kv_head_idx, size_t token_idx) {
        const size_t block_idx = token_idx / _page_block_size;
        const size_t token_offset = token_idx % _page_block_size;
        const int64_t physical_block = loadIndex(
            _block_tables_gm,
            static_cast<ptrdiff_t>(seq_idx) * _block_table_batch_stride + static_cast<ptrdiff_t>(block_idx));
        return static_cast<ptrdiff_t>(physical_block) * _k_batch_stride
             + static_cast<ptrdiff_t>(kv_head_idx) * _k_head_stride
             + static_cast<ptrdiff_t>(token_offset) * _k_row_stride;
    }

    __aicore__ inline ptrdiff_t valueBase(size_t seq_idx, size_t kv_head_idx, size_t token_idx) {
        const size_t block_idx = token_idx / _page_block_size;
        const size_t token_offset = token_idx % _page_block_size;
        const int64_t physical_block = loadIndex(
            _block_tables_gm,
            static_cast<ptrdiff_t>(seq_idx) * _block_table_batch_stride + static_cast<ptrdiff_t>(block_idx));
        return static_cast<ptrdiff_t>(physical_block) * _v_batch_stride
             + static_cast<ptrdiff_t>(kv_head_idx) * _v_head_stride
             + static_cast<ptrdiff_t>(token_offset) * _v_row_stride;
    }

    __aicore__ inline float computeScore(
        LocalTensor<float> &q_local,
        size_t seq_idx,
        size_t kv_head_idx,
        size_t token_idx,
        float alibi_slope,
        size_t seq_len) {
        const ptrdiff_t k_base = keyBase(seq_idx, kv_head_idx, token_idx);
        float score = 0.0f;
        for (size_t d = 0; d < _head_size; ++d) {
            const float q_val = q_local.GetValue(d);
            const float k_val = dataToFloat(_k_cache_gm.GetValue(k_base + static_cast<ptrdiff_t>(d)));
            score += q_val * k_val;
        }
        score *= _scale;
        if (_has_alibi) {
            score += alibi_slope * static_cast<float>(static_cast<int64_t>(token_idx) - static_cast<int64_t>(seq_len) + 1);
        }
        return score;
    }

    GlobalTensor<Tdata> _out_gm;
    GlobalTensor<Tdata> _q_gm;
    GlobalTensor<Tdata> _k_cache_gm;
    GlobalTensor<Tdata> _v_cache_gm;
    GlobalTensor<Tindex> _block_tables_gm;
    GlobalTensor<Tindex> _cache_lens_gm;
    GlobalTensor<float> _alibi_slopes_gm;

    TPipe _pipe;
    TBuf<TPosition::VECCALC> _q_buf;
    TBuf<TPosition::VECCALC> _acc_buf;
    TBuf<TPosition::VECCALC> _score_buf;
    TBuf<TPosition::VECCALC> _out_buf;

    size_t _num_heads;
    size_t _num_seqs;
    size_t _num_kv_heads;
    size_t _head_size;
    float _scale;
    size_t _max_num_blocks_per_seq;
    size_t _page_block_size;
    ptrdiff_t _q_stride;
    ptrdiff_t _k_batch_stride;
    ptrdiff_t _k_row_stride;
    ptrdiff_t _k_head_stride;
    ptrdiff_t _v_batch_stride;
    ptrdiff_t _v_row_stride;
    ptrdiff_t _v_head_stride;
    ptrdiff_t _o_stride;
    ptrdiff_t _block_table_batch_stride;
    ptrdiff_t _cache_lens_stride;
    bool _has_alibi;
};

#define DEFINE_PAGED_ATTENTION_KERNEL(KERNEL_NAME, DATA_TYPE, INDEX_TYPE)              \
    extern "C" __global__ __aicore__ void KERNEL_NAME(                                 \
        GM_ADDR out, GM_ADDR q, GM_ADDR k_cache, GM_ADDR v_cache,                      \
        GM_ADDR block_tables, GM_ADDR cache_lens, GM_ADDR alibi_slopes,                \
        bool has_alibi, size_t num_heads, size_t num_seqs, size_t num_kv_heads,        \
        size_t head_size, float scale, size_t max_num_blocks_per_seq,                  \
        size_t page_block_size, ptrdiff_t q_stride, ptrdiff_t k_batch_stride,          \
        ptrdiff_t k_row_stride, ptrdiff_t k_head_stride, ptrdiff_t v_batch_stride,     \
        ptrdiff_t v_row_stride, ptrdiff_t v_head_stride, ptrdiff_t o_stride,           \
        ptrdiff_t block_table_batch_stride, ptrdiff_t cache_lens_stride) {             \
        PagedAttentionKernel<DATA_TYPE, INDEX_TYPE> op;                                \
        op.init(out, q, k_cache, v_cache, block_tables, cache_lens, alibi_slopes,      \
                has_alibi, num_heads, num_seqs, num_kv_heads, head_size, scale,        \
                max_num_blocks_per_seq, page_block_size, q_stride, k_batch_stride,     \
                k_row_stride, k_head_stride, v_batch_stride, v_row_stride,             \
                v_head_stride, o_stride, block_table_batch_stride, cache_lens_stride); \
        op.process();                                                                  \
    }

DEFINE_PAGED_ATTENTION_KERNEL(paged_attention_f16_i64, half, int64_t)
DEFINE_PAGED_ATTENTION_KERNEL(paged_attention_f16_i32, half, int32_t)
DEFINE_PAGED_ATTENTION_KERNEL(paged_attention_f16_u32, half, uint32_t)
DEFINE_PAGED_ATTENTION_KERNEL(paged_attention_bf16_i64, bfloat16_t, int64_t)
DEFINE_PAGED_ATTENTION_KERNEL(paged_attention_bf16_i32, bfloat16_t, int32_t)
DEFINE_PAGED_ATTENTION_KERNEL(paged_attention_bf16_u32, bfloat16_t, uint32_t)

#undef DEFINE_PAGED_ATTENTION_KERNEL

extern "C" infiniStatus_t paged_attention_kernel_launch(
    void *out,
    const void *q,
    const void *k_cache,
    const void *v_cache,
    const void *block_tables,
    const void *cache_lens,
    const void *alibi_slopes,
    infiniDtype_t dtype,
    infiniDtype_t index_dtype,
    size_t num_heads,
    size_t num_seqs,
    size_t num_kv_heads,
    size_t head_size,
    float scale,
    size_t max_num_blocks_per_seq,
    size_t page_block_size,
    ptrdiff_t q_stride,
    ptrdiff_t k_batch_stride,
    ptrdiff_t k_row_stride,
    ptrdiff_t k_head_stride,
    ptrdiff_t v_batch_stride,
    ptrdiff_t v_row_stride,
    ptrdiff_t v_head_stride,
    ptrdiff_t o_stride,
    ptrdiff_t block_table_batch_stride,
    ptrdiff_t cache_lens_stride,
    void *stream) {
    const size_t block_dim = num_seqs * num_heads;
    if (block_dim == 0) {
        return INFINI_STATUS_SUCCESS;
    }
    const bool has_alibi = (alibi_slopes != nullptr);

#define LAUNCH_PAGED_ATTENTION(KERNEL_NAME)                                    \
    KERNEL_NAME<<<block_dim, nullptr, stream>>>(                               \
        out, const_cast<void *>(q), const_cast<void *>(k_cache),               \
        const_cast<void *>(v_cache), const_cast<void *>(block_tables),         \
        const_cast<void *>(cache_lens), const_cast<void *>(alibi_slopes),      \
        has_alibi, num_heads, num_seqs, num_kv_heads, head_size, scale,        \
        max_num_blocks_per_seq, page_block_size, q_stride, k_batch_stride,     \
        k_row_stride, k_head_stride, v_batch_stride, v_row_stride,             \
        v_head_stride, o_stride, block_table_batch_stride, cache_lens_stride); \
    return INFINI_STATUS_SUCCESS

    if (dtype == INFINI_DTYPE_F16) {
        if (index_dtype == INFINI_DTYPE_I64) {
            LAUNCH_PAGED_ATTENTION(paged_attention_f16_i64);
        }
        if (index_dtype == INFINI_DTYPE_I32) {
            LAUNCH_PAGED_ATTENTION(paged_attention_f16_i32);
        }
        if (index_dtype == INFINI_DTYPE_U32) {
            LAUNCH_PAGED_ATTENTION(paged_attention_f16_u32);
        }
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    if (dtype == INFINI_DTYPE_BF16) {
        if (index_dtype == INFINI_DTYPE_I64) {
            LAUNCH_PAGED_ATTENTION(paged_attention_bf16_i64);
        }
        if (index_dtype == INFINI_DTYPE_I32) {
            LAUNCH_PAGED_ATTENTION(paged_attention_bf16_i32);
        }
        if (index_dtype == INFINI_DTYPE_U32) {
            LAUNCH_PAGED_ATTENTION(paged_attention_bf16_u32);
        }
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    return INFINI_STATUS_BAD_TENSOR_DTYPE;

#undef LAUNCH_PAGED_ATTENTION
}
