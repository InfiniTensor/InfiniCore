#ifndef PAGED_ATTENTION_V1_H
#define PAGED_ATTENTION_V1_H

#include "../../operator.h"
#include "info.h"

#define DESCRIPTOR(NAMESPACE)                                                   \
                                                                                \
    namespace op::paged_attention_v1::NAMESPACE {                               \
    class Descriptor final : public InfiniopDescriptor {                        \
        struct Opaque;                                                          \
        Opaque *_opaque;                                                        \
        PagedAttentionV1Info _info;                                             \
        size_t _workspace_size;                                                 \
                                                                                \
        Descriptor(                                                             \
            Opaque *opaque,                                                     \
            PagedAttentionV1Info info,                                          \
            size_t workspace_size,                                              \
            infiniDevice_t device_type,                                         \
            int device_id)                                                      \
            : InfiniopDescriptor{device_type, device_id},                       \
              _opaque(opaque),                                                  \
              _info(info),                                                      \
              _workspace_size(workspace_size) {}                                \
                                                                                \
    public:                                                                     \
        ~Descriptor();                                                          \
                                                                                \
        size_t workspaceSize() const { return _workspace_size; }                \
                                                                                \
        static infiniStatus_t create(                                           \
            infiniopHandle_t handle,                                            \
            Descriptor **desc_ptr,                                              \
            infiniopTensorDescriptor_t out_desc,                                \
            infiniopTensorDescriptor_t query_desc,                              \
            infiniopTensorDescriptor_t key_cache_desc,                          \
            infiniopTensorDescriptor_t value_cache_desc,                        \
            infiniopTensorDescriptor_t block_tables_desc,                       \
            infiniopTensorDescriptor_t seq_lens_desc,                           \
            const std::optional<infiniopTensorDescriptor_t> &alibi_slopes_desc, \
            double scale);                                                      \
                                                                                \
        infiniStatus_t calculate(                                               \
            void *workspace,                                                    \
            size_t workspace_size,                                              \
            void *out,                                                          \
            void *query,                                                        \
            void *key_cache,                                                    \
            void *value_cache,                                                  \
            int64_t num_kv_heads,                                               \
            double scale,                                                       \
            int64_t *block_tables,                                              \
            int64_t *seq_lens,                                                  \
            int64_t block_size,                                                 \
            int64_t max_seq_len,                                                \
            const void *alibi_slopes,                                           \
            const char *kv_cache_dtype,                                         \
            float *k_scale,                                                     \
            float *v_scale,                                                     \
            const int64_t tp_rank,                                              \
            const int64_t blocksparse_local_blocks,                             \
            const int64_t blocksparse_vert_stride,                              \
            const int64_t blocksparse_block_size,                               \
            const int64_t blocksparse_head_sliding_step,                        \
            void *stream) const;                                                \
    };                                                                          \
    }

#endif // PAGED_ATTENTION_1_H
