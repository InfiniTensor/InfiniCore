#include "../../../devices/ascend/ascend_kernel_common.h"

using namespace AscendC;

template <typename Tdata>
class PagedCachingKernel {
public:
    __aicore__ inline PagedCachingKernel() {}

    __aicore__ inline void init(
        GM_ADDR k_cache,
        GM_ADDR v_cache,
        GM_ADDR k,
        GM_ADDR v,
        GM_ADDR slot_mapping,
        size_t num_kv_heads,
        size_t head_size,
        size_t block_size,
        ptrdiff_t k_src_stride,
        ptrdiff_t v_src_stride,
        ptrdiff_t k_cache_block_stride,
        ptrdiff_t v_cache_block_stride,
        ptrdiff_t k_cache_head_stride,
        ptrdiff_t v_cache_head_stride,
        ptrdiff_t k_cache_slot_stride,
        ptrdiff_t v_cache_slot_stride) {
        _num_kv_heads = num_kv_heads;
        _head_size = head_size;
        _block_size = block_size;
        _k_src_stride = k_src_stride;
        _v_src_stride = v_src_stride;
        _k_cache_block_stride = k_cache_block_stride;
        _v_cache_block_stride = v_cache_block_stride;
        _k_cache_head_stride = k_cache_head_stride;
        _v_cache_head_stride = v_cache_head_stride;
        _k_cache_slot_stride = k_cache_slot_stride;
        _v_cache_slot_stride = v_cache_slot_stride;

        _k_cache_gm.SetGlobalBuffer(reinterpret_cast<__gm__ Tdata *>(k_cache));
        _v_cache_gm.SetGlobalBuffer(reinterpret_cast<__gm__ Tdata *>(v_cache));
        _k_gm.SetGlobalBuffer(reinterpret_cast<__gm__ Tdata *>(k));
        _v_gm.SetGlobalBuffer(reinterpret_cast<__gm__ Tdata *>(v));
        _slot_mapping_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(slot_mapping));
    }

    __aicore__ inline void process() {
        const size_t work_idx = GetBlockIdx();
        const size_t head_idx = work_idx % _num_kv_heads;
        const size_t token_idx = work_idx / _num_kv_heads;

        const int64_t slot_idx = _slot_mapping_gm.GetValue(token_idx);
        if (slot_idx < 0) {
            return;
        }

        const int64_t physical_block_idx = slot_idx / static_cast<int64_t>(_block_size);
        const int64_t block_offset = slot_idx % static_cast<int64_t>(_block_size);

        const ptrdiff_t k_src_base = static_cast<ptrdiff_t>(token_idx) * _k_src_stride
                                   + static_cast<ptrdiff_t>(head_idx * _head_size);
        const ptrdiff_t v_src_base = static_cast<ptrdiff_t>(token_idx) * _v_src_stride
                                   + static_cast<ptrdiff_t>(head_idx * _head_size);
        const ptrdiff_t k_dst_base = static_cast<ptrdiff_t>(physical_block_idx) * _k_cache_block_stride
                                   + static_cast<ptrdiff_t>(head_idx) * _k_cache_head_stride
                                   + static_cast<ptrdiff_t>(block_offset) * _k_cache_slot_stride;
        const ptrdiff_t v_dst_base = static_cast<ptrdiff_t>(physical_block_idx) * _v_cache_block_stride
                                   + static_cast<ptrdiff_t>(head_idx) * _v_cache_head_stride
                                   + static_cast<ptrdiff_t>(block_offset) * _v_cache_slot_stride;

        for (size_t d = 0; d < _head_size; ++d) {
            _k_cache_gm.SetValue(k_dst_base + static_cast<ptrdiff_t>(d), _k_gm.GetValue(k_src_base + static_cast<ptrdiff_t>(d)));
            _v_cache_gm.SetValue(v_dst_base + static_cast<ptrdiff_t>(d), _v_gm.GetValue(v_src_base + static_cast<ptrdiff_t>(d)));
        }
    }

private:
    GlobalTensor<Tdata> _k_cache_gm;
    GlobalTensor<Tdata> _v_cache_gm;
    GlobalTensor<Tdata> _k_gm;
    GlobalTensor<Tdata> _v_gm;
    GlobalTensor<int64_t> _slot_mapping_gm;

    size_t _num_kv_heads;
    size_t _head_size;
    size_t _block_size;
    ptrdiff_t _k_src_stride;
    ptrdiff_t _v_src_stride;
    ptrdiff_t _k_cache_block_stride;
    ptrdiff_t _v_cache_block_stride;
    ptrdiff_t _k_cache_head_stride;
    ptrdiff_t _v_cache_head_stride;
    ptrdiff_t _k_cache_slot_stride;
    ptrdiff_t _v_cache_slot_stride;
};

#define DEFINE_PAGED_CACHING_KERNEL(KERNEL_NAME, TYPE)                 \
    extern "C" __global__ __aicore__ void KERNEL_NAME(                 \
        GM_ADDR k_cache, GM_ADDR v_cache, GM_ADDR k, GM_ADDR v,        \
        GM_ADDR slot_mapping, size_t num_kv_heads, size_t head_size,   \
        size_t block_size, ptrdiff_t k_src_stride,                     \
        ptrdiff_t v_src_stride, ptrdiff_t k_cache_block_stride,        \
        ptrdiff_t v_cache_block_stride, ptrdiff_t k_cache_head_stride, \
        ptrdiff_t v_cache_head_stride, ptrdiff_t k_cache_slot_stride,  \
        ptrdiff_t v_cache_slot_stride) {                               \
        PagedCachingKernel<TYPE> op;                                   \
        op.init(k_cache, v_cache, k, v, slot_mapping, num_kv_heads,    \
                head_size, block_size, k_src_stride, v_src_stride,     \
                k_cache_block_stride, v_cache_block_stride,            \
                k_cache_head_stride, v_cache_head_stride,              \
                k_cache_slot_stride, v_cache_slot_stride);             \
        op.process();                                                  \
    }

DEFINE_PAGED_CACHING_KERNEL(paged_caching_kernel_f16, half)
DEFINE_PAGED_CACHING_KERNEL(paged_caching_kernel_bf16, bfloat16_t)
DEFINE_PAGED_CACHING_KERNEL(paged_caching_kernel_f32, float)

#undef DEFINE_PAGED_CACHING_KERNEL

extern "C" infiniStatus_t paged_caching_kernel_launch(
    void *k_cache,
    void *v_cache,
    const void *k,
    const void *v,
    const void *slot_mapping,
    infiniDtype_t dtype,
    size_t num_tokens,
    size_t num_kv_heads,
    size_t head_size,
    size_t block_size,
    ptrdiff_t k_src_stride,
    ptrdiff_t v_src_stride,
    ptrdiff_t k_cache_block_stride,
    ptrdiff_t v_cache_block_stride,
    ptrdiff_t k_cache_head_stride,
    ptrdiff_t v_cache_head_stride,
    ptrdiff_t k_cache_slot_stride,
    ptrdiff_t v_cache_slot_stride,
    void *stream) {
    const size_t block_dim = num_tokens * num_kv_heads;
    if (block_dim == 0) {
        return INFINI_STATUS_SUCCESS;
    }

#define LAUNCH_PAGED_CACHING(DTYPE_ENUM, KERNEL_NAME)                       \
    case DTYPE_ENUM:                                                        \
        KERNEL_NAME<<<block_dim, nullptr, stream>>>(                        \
            k_cache, v_cache, const_cast<void *>(k), const_cast<void *>(v), \
            const_cast<void *>(slot_mapping), num_kv_heads, head_size,      \
            block_size, k_src_stride, v_src_stride,                         \
            k_cache_block_stride, v_cache_block_stride,                     \
            k_cache_head_stride, v_cache_head_stride,                       \
            k_cache_slot_stride, v_cache_slot_stride);                      \
        return INFINI_STATUS_SUCCESS;

    switch (dtype) {
        LAUNCH_PAGED_CACHING(INFINI_DTYPE_F16, paged_caching_kernel_f16)
        LAUNCH_PAGED_CACHING(INFINI_DTYPE_BF16, paged_caching_kernel_bf16)
        LAUNCH_PAGED_CACHING(INFINI_DTYPE_F32, paged_caching_kernel_f32)
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

#undef LAUNCH_PAGED_CACHING
}
