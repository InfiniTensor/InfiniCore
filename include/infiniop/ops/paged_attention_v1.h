#ifndef __INFINIOP_PAGED_ATTENTION_V1_API_H__
#define __INFINIOP_PAGED_ATTENTION_V1_API_H__

#include "../operator_descriptor.h"
#include <stdint.h>

typedef struct InfiniopDescriptor *infiniopPagedAttentionV1Descriptor_t;

__C __export infiniStatus_t infiniopCreatePagedAttentionV1Descriptor(
    infiniopHandle_t handle,
    infiniopPagedAttentionV1Descriptor_t *desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t query_desc,
    infiniopTensorDescriptor_t key_cache_desc,
    infiniopTensorDescriptor_t value_cache_desc,
    infiniopTensorDescriptor_t block_tables_desc,
    infiniopTensorDescriptor_t seq_lens_desc,
    infiniopTensorDescriptor_t alibi_slopes_desc,
    double scale);

__C __export infiniStatus_t infiniopGetPagedAttentionV1WorkspaceSize(
    infiniopPagedAttentionV1Descriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopPagedAttentionV1(
    infiniopPagedAttentionV1Descriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *out,            // [num_seqs, num_heads, head_size]
    void *query,          // [num_seqs, num_heads, head_size]
    void *key_cache,      // [num_blocks, num_heads, head_size/x, block_size, x]
    void *value_cache,    // [num_blocks, num_heads, head_size, block_size]
    int64_t num_kv_heads, // [num_heads]
    double scale,
    void *block_tables, // [num_seqs, max_num_blocks_per_seq]
    void *seq_lens,     // [num_seqs]
    int64_t block_size, // *** 不向下传递
    int64_t max_seq_len,
    const void *alibi_slopes,   // 注意cpp中是 std::optional
    const char *kv_cache_dtype, //  *** 不向下传递
    void *k_scale,
    void *v_scale,
    const int64_t tp_rank,
    const int64_t blocksparse_local_blocks,
    const int64_t blocksparse_vert_stride,
    const int64_t blocksparse_block_size,
    const int64_t blocksparse_head_sliding_step,
    void *stream);

__C __export infiniStatus_t infiniopDestroyPagedAttentionV1Descriptor(
    infiniopPagedAttentionV1Descriptor_t desc);

#endif // __INFINIOP_PAGED_ATTENTION_API_H__
