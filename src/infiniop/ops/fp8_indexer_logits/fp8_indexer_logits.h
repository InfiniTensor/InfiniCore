#ifndef __FP8_INDEXER_LOGITS_H__
#define __FP8_INDEXER_LOGITS_H__

#include "../../../utils.h"
#include "../../operator.h"

#define DESCRIPTOR(NAMESPACE)                                  \
    namespace op::fp8_indexer_logits::NAMESPACE {              \
    class Descriptor final : public InfiniopDescriptor {       \
        size_t _num_tokens;                                    \
        size_t _num_heads;                                     \
        size_t _head_dim;                                      \
        size_t _num_cache_blocks;                              \
        size_t _block_size;                                    \
        size_t _cache_stride;                                  \
        size_t _num_requests;                                  \
        size_t _max_blocks_per_request;                        \
        size_t _max_context_len;                               \
                                                               \
        Descriptor(                                            \
            size_t num_tokens,                                 \
            size_t num_heads,                                  \
            size_t head_dim,                                   \
            size_t num_cache_blocks,                           \
            size_t block_size,                                 \
            size_t cache_stride,                               \
            size_t num_requests,                               \
            size_t max_blocks_per_request,                     \
            size_t max_context_len,                            \
            infiniDevice_t device_type,                        \
            int device_id)                                     \
            : InfiniopDescriptor{device_type, device_id},      \
              _num_tokens(num_tokens),                         \
              _num_heads(num_heads),                           \
              _head_dim(head_dim),                             \
              _num_cache_blocks(num_cache_blocks),             \
              _block_size(block_size),                         \
              _cache_stride(cache_stride),                     \
              _num_requests(num_requests),                     \
              _max_blocks_per_request(max_blocks_per_request), \
              _max_context_len(max_context_len) {}             \
                                                               \
    public:                                                    \
        static infiniStatus_t create(                          \
            infiniopHandle_t handle,                           \
            Descriptor **desc_ptr,                             \
            infiniopTensorDescriptor_t logits_desc,            \
            infiniopTensorDescriptor_t q_fp8_desc,             \
            infiniopTensorDescriptor_t kv_cache_desc,          \
            infiniopTensorDescriptor_t block_tables_desc,      \
            infiniopTensorDescriptor_t weights_fp32_desc,      \
            infiniopTensorDescriptor_t positions_desc,         \
            infiniopTensorDescriptor_t request_ids_desc);      \
                                                               \
        infiniStatus_t calculate(                              \
            void *logits,                                      \
            const void *q_fp8,                                 \
            const void *kv_cache,                              \
            const void *block_tables,                          \
            const void *weights_fp32,                          \
            const void *positions,                             \
            const void *request_ids,                           \
            void *stream) const;                               \
    };                                                         \
    }

#endif
