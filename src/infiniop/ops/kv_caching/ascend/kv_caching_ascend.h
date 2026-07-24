
#ifndef __KV_CACHING_ASCEND_API_H__
#define __KV_CACHING_ASCEND_API_H__

#include "../kv_caching.h"

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
    void *stream);

DESCRIPTOR(ascend)

#endif // __KV_CACHING_ASCEND_API_H__
