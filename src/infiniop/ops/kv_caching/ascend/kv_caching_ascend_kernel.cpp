
#include "../../../devices/ascend/ascend_kernel_common.h"

using namespace AscendC;

template <typename Tdata, typename Tidx>
class KVCachingKernel {
public:
    __aicore__ inline KVCachingKernel() {}

    __aicore__ inline void init(
        GM_ADDR k_cache,
        GM_ADDR v_cache,
        GM_ADDR k,
        GM_ADDR v,
        GM_ADDR past_kv_lengths,
        size_t batch_size,
        size_t num_kv_heads,
        size_t seq_len,
        size_t hidden_dim,
        ptrdiff_t k_cache_strides_0,
        ptrdiff_t k_cache_strides_1,
        ptrdiff_t k_cache_strides_2,
        ptrdiff_t k_cache_strides_3,
        ptrdiff_t v_cache_strides_0,
        ptrdiff_t v_cache_strides_1,
        ptrdiff_t v_cache_strides_2,
        ptrdiff_t v_cache_strides_3,
        ptrdiff_t k_strides_0,
        ptrdiff_t k_strides_1,
        ptrdiff_t k_strides_2,
        ptrdiff_t k_strides_3,
        ptrdiff_t v_strides_0,
        ptrdiff_t v_strides_1,
        ptrdiff_t v_strides_2,
        ptrdiff_t v_strides_3) {
        _batch_size = batch_size;
        _num_kv_heads = num_kv_heads;
        _seq_len = seq_len;
        _hidden_dim = hidden_dim;
        _k_cache_strides_0 = k_cache_strides_0;
        _k_cache_strides_1 = k_cache_strides_1;
        _k_cache_strides_2 = k_cache_strides_2;
        _k_cache_strides_3 = k_cache_strides_3;
        _v_cache_strides_0 = v_cache_strides_0;
        _v_cache_strides_1 = v_cache_strides_1;
        _v_cache_strides_2 = v_cache_strides_2;
        _v_cache_strides_3 = v_cache_strides_3;
        _k_strides_0 = k_strides_0;
        _k_strides_1 = k_strides_1;
        _k_strides_2 = k_strides_2;
        _k_strides_3 = k_strides_3;
        _v_strides_0 = v_strides_0;
        _v_strides_1 = v_strides_1;
        _v_strides_2 = v_strides_2;
        _v_strides_3 = v_strides_3;

        _k_cache_gm.SetGlobalBuffer(reinterpret_cast<__gm__ Tdata *>(k_cache));
        _v_cache_gm.SetGlobalBuffer(reinterpret_cast<__gm__ Tdata *>(v_cache));
        _k_gm.SetGlobalBuffer(reinterpret_cast<__gm__ Tdata *>(k));
        _v_gm.SetGlobalBuffer(reinterpret_cast<__gm__ Tdata *>(v));
        _past_kv_lengths_gm.SetGlobalBuffer(reinterpret_cast<__gm__ Tidx *>(past_kv_lengths));
    }

    __aicore__ inline void process() {
        const size_t work_idx = GetBlockIdx();
        const size_t total = _batch_size * _num_kv_heads * _seq_len;
        if (work_idx >= total) {
            return;
        }

        size_t idx = work_idx;
        size_t s = idx % _seq_len;
        idx /= _seq_len;
        size_t h = idx % _num_kv_heads;
        size_t b = idx / _num_kv_heads;

        ptrdiff_t cache_s = static_cast<ptrdiff_t>(_past_kv_lengths_gm.GetValue(b)) + static_cast<ptrdiff_t>(s);
        ptrdiff_t k_cache_base = static_cast<ptrdiff_t>(b) * _k_cache_strides_0
                               + static_cast<ptrdiff_t>(h) * _k_cache_strides_1
                               + cache_s * _k_cache_strides_2;
        ptrdiff_t v_cache_base = static_cast<ptrdiff_t>(b) * _v_cache_strides_0
                               + static_cast<ptrdiff_t>(h) * _v_cache_strides_1
                               + cache_s * _v_cache_strides_2;
        ptrdiff_t k_src_base = static_cast<ptrdiff_t>(b) * _k_strides_0
                             + static_cast<ptrdiff_t>(h) * _k_strides_1
                             + static_cast<ptrdiff_t>(s) * _k_strides_2;
        ptrdiff_t v_src_base = static_cast<ptrdiff_t>(b) * _v_strides_0
                             + static_cast<ptrdiff_t>(h) * _v_strides_1
                             + static_cast<ptrdiff_t>(s) * _v_strides_2;

        for (size_t d = 0; d < _hidden_dim; ++d) {
            ptrdiff_t d_offset = static_cast<ptrdiff_t>(d);
            _k_cache_gm.SetValue(k_cache_base + d_offset * _k_cache_strides_3,
                                 _k_gm.GetValue(k_src_base + d_offset * _k_strides_3));
            _v_cache_gm.SetValue(v_cache_base + d_offset * _v_cache_strides_3,
                                 _v_gm.GetValue(v_src_base + d_offset * _v_strides_3));
        }
    }

private:
    GlobalTensor<Tdata> _k_cache_gm;
    GlobalTensor<Tdata> _v_cache_gm;
    GlobalTensor<Tdata> _k_gm;
    GlobalTensor<Tdata> _v_gm;
    GlobalTensor<Tidx> _past_kv_lengths_gm;
    size_t _batch_size;
    size_t _num_kv_heads;
    size_t _seq_len;
    size_t _hidden_dim;
    ptrdiff_t _k_cache_strides_0;
    ptrdiff_t _k_cache_strides_1;
    ptrdiff_t _k_cache_strides_2;
    ptrdiff_t _k_cache_strides_3;
    ptrdiff_t _v_cache_strides_0;
    ptrdiff_t _v_cache_strides_1;
    ptrdiff_t _v_cache_strides_2;
    ptrdiff_t _v_cache_strides_3;
    ptrdiff_t _k_strides_0;
    ptrdiff_t _k_strides_1;
    ptrdiff_t _k_strides_2;
    ptrdiff_t _k_strides_3;
    ptrdiff_t _v_strides_0;
    ptrdiff_t _v_strides_1;
    ptrdiff_t _v_strides_2;
    ptrdiff_t _v_strides_3;
};

#define DEFINE_KV_CACHING_KERNEL(KERNEL_NAME, TYPE, IDX_TYPE)                            \
    extern "C" __global__ __aicore__ void KERNEL_NAME(                                   \
        GM_ADDR k_cache, GM_ADDR v_cache, GM_ADDR k, GM_ADDR v, GM_ADDR past_kv_lengths, \
        size_t batch_size, size_t num_kv_heads, size_t seq_len, size_t hidden_dim,       \
        ptrdiff_t k_cache_strides_0, ptrdiff_t k_cache_strides_1,                        \
        ptrdiff_t k_cache_strides_2, ptrdiff_t k_cache_strides_3,                        \
        ptrdiff_t v_cache_strides_0, ptrdiff_t v_cache_strides_1,                        \
        ptrdiff_t v_cache_strides_2, ptrdiff_t v_cache_strides_3,                        \
        ptrdiff_t k_strides_0, ptrdiff_t k_strides_1, ptrdiff_t k_strides_2,             \
        ptrdiff_t k_strides_3, ptrdiff_t v_strides_0, ptrdiff_t v_strides_1,             \
        ptrdiff_t v_strides_2, ptrdiff_t v_strides_3) {                                  \
        KVCachingKernel<TYPE, IDX_TYPE> op;                                              \
        op.init(k_cache, v_cache, k, v, past_kv_lengths, batch_size, num_kv_heads,       \
                seq_len, hidden_dim, k_cache_strides_0, k_cache_strides_1,               \
                k_cache_strides_2, k_cache_strides_3, v_cache_strides_0,                 \
                v_cache_strides_1, v_cache_strides_2, v_cache_strides_3,                 \
                k_strides_0, k_strides_1, k_strides_2, k_strides_3,                      \
                v_strides_0, v_strides_1, v_strides_2, v_strides_3);                     \
        op.process();                                                                    \
    }

DEFINE_KV_CACHING_KERNEL(kv_caching_kernel_f16_i32, half, int32_t)
DEFINE_KV_CACHING_KERNEL(kv_caching_kernel_f16_i64, half, int64_t)
DEFINE_KV_CACHING_KERNEL(kv_caching_kernel_bf16_i32, bfloat16_t, int32_t)
DEFINE_KV_CACHING_KERNEL(kv_caching_kernel_bf16_i64, bfloat16_t, int64_t)
DEFINE_KV_CACHING_KERNEL(kv_caching_kernel_f32_i32, float, int32_t)
DEFINE_KV_CACHING_KERNEL(kv_caching_kernel_f32_i64, float, int64_t)

#undef DEFINE_KV_CACHING_KERNEL

extern "C" infiniStatus_t kv_caching_kernel_launch(
    void *k_cache,
    void *v_cache,
    const void *k,
    const void *v,
    const void *past_kv_lengths,
    infiniDtype_t dtype,
    infiniDtype_t past_len_dtype,
    size_t batch_size,
    size_t num_kv_heads,
    size_t seq_len,
    size_t hidden_dim,
    ptrdiff_t k_cache_strides_0,
    ptrdiff_t k_cache_strides_1,
    ptrdiff_t k_cache_strides_2,
    ptrdiff_t k_cache_strides_3,
    ptrdiff_t v_cache_strides_0,
    ptrdiff_t v_cache_strides_1,
    ptrdiff_t v_cache_strides_2,
    ptrdiff_t v_cache_strides_3,
    ptrdiff_t k_strides_0,
    ptrdiff_t k_strides_1,
    ptrdiff_t k_strides_2,
    ptrdiff_t k_strides_3,
    ptrdiff_t v_strides_0,
    ptrdiff_t v_strides_1,
    ptrdiff_t v_strides_2,
    ptrdiff_t v_strides_3,
    void *stream) {
    const size_t block_dim = batch_size * num_kv_heads * seq_len;
    if (block_dim == 0) {
        return INFINI_STATUS_SUCCESS;
    }

#define LAUNCH_KV_CACHING(DTYPE_ENUM, IDX_ENUM, KERNEL_NAME)                \
    if (dtype == DTYPE_ENUM && past_len_dtype == IDX_ENUM) {                \
        KERNEL_NAME<<<block_dim, nullptr, stream>>>(                        \
            k_cache, v_cache, const_cast<void *>(k), const_cast<void *>(v), \
            const_cast<void *>(past_kv_lengths), batch_size, num_kv_heads,  \
            seq_len, hidden_dim, k_cache_strides_0, k_cache_strides_1,      \
            k_cache_strides_2, k_cache_strides_3, v_cache_strides_0,        \
            v_cache_strides_1, v_cache_strides_2, v_cache_strides_3,        \
            k_strides_0, k_strides_1, k_strides_2, k_strides_3,             \
            v_strides_0, v_strides_1, v_strides_2, v_strides_3);            \
        return INFINI_STATUS_SUCCESS;                                       \
    }

    LAUNCH_KV_CACHING(INFINI_DTYPE_F16, INFINI_DTYPE_I32, kv_caching_kernel_f16_i32)
    LAUNCH_KV_CACHING(INFINI_DTYPE_F16, INFINI_DTYPE_I64, kv_caching_kernel_f16_i64)
    LAUNCH_KV_CACHING(INFINI_DTYPE_BF16, INFINI_DTYPE_I32, kv_caching_kernel_bf16_i32)
    LAUNCH_KV_CACHING(INFINI_DTYPE_BF16, INFINI_DTYPE_I64, kv_caching_kernel_bf16_i64)
    LAUNCH_KV_CACHING(INFINI_DTYPE_F32, INFINI_DTYPE_I32, kv_caching_kernel_f32_i32)
    LAUNCH_KV_CACHING(INFINI_DTYPE_F32, INFINI_DTYPE_I64, kv_caching_kernel_f32_i64)

    return INFINI_STATUS_BAD_TENSOR_DTYPE;

#undef LAUNCH_KV_CACHING
}
