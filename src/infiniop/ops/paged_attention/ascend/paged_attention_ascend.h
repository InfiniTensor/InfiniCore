#ifndef __PAGED_ATTENTION_ASCEND_H__
#define __PAGED_ATTENTION_ASCEND_H__

#include "../paged_attention.h"

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
    void *stream);

DESCRIPTOR(ascend)

#endif // __PAGED_ATTENTION_ASCEND_H__
