#ifndef __PAGED_CACHING_ASCEND_H__
#define __PAGED_CACHING_ASCEND_H__

#include "../paged_caching.h"

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
    void *stream);

DESCRIPTOR(ascend)

#endif // __PAGED_CACHING_ASCEND_H__
