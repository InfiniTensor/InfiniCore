#ifndef __FP8_SPARSE_MLA_H__
#define __FP8_SPARSE_MLA_H__

#include "../../../utils.h"
#include "../../operator.h"

#define DESCRIPTOR(NAMESPACE)                                    \
    namespace op::fp8_sparse_mla::NAMESPACE {                    \
    class Descriptor final : public InfiniopDescriptor {         \
        size_t _num_tokens;                                      \
        size_t _num_heads;                                       \
        size_t _head_dim;                                        \
        size_t _value_dim;                                       \
        size_t _num_cache_tokens;                                \
        size_t _cache_stride;                                    \
        size_t _topk;                                            \
        size_t _groups;                                          \
        size_t _workspace_size;                                  \
        float _scale;                                            \
                                                                 \
        Descriptor(                                              \
            size_t num_tokens,                                   \
            size_t num_heads,                                    \
            size_t head_dim,                                     \
            size_t value_dim,                                    \
            size_t num_cache_tokens,                             \
            size_t cache_stride,                                 \
            size_t topk,                                         \
            size_t groups,                                       \
            size_t workspace_size,                               \
            float scale,                                         \
            infiniDevice_t device_type,                          \
            int device_id)                                       \
            : InfiniopDescriptor{device_type, device_id},        \
              _num_tokens(num_tokens),                           \
              _num_heads(num_heads),                             \
              _head_dim(head_dim),                               \
              _value_dim(value_dim),                             \
              _num_cache_tokens(num_cache_tokens),               \
              _cache_stride(cache_stride),                       \
              _topk(topk),                                       \
              _groups(groups),                                   \
              _workspace_size(workspace_size),                   \
              _scale(scale) {}                                   \
                                                                 \
    public:                                                      \
        static infiniStatus_t create(                            \
            infiniopHandle_t handle,                             \
            Descriptor **desc_ptr,                               \
            infiniopTensorDescriptor_t output_desc,              \
            infiniopTensorDescriptor_t query_desc,               \
            infiniopTensorDescriptor_t kv_cache_desc,            \
            infiniopTensorDescriptor_t indices_desc,             \
            infiniopTensorDescriptor_t topk_lens_desc,           \
            float scale);                                        \
                                                                 \
        size_t workspaceSize() const { return _workspace_size; } \
                                                                 \
        infiniStatus_t calculate(                                \
            void *workspace,                                     \
            size_t workspace_size,                               \
            void *output,                                        \
            const void *query,                                   \
            const void *kv_cache,                                \
            const void *indices,                                 \
            const void *topk_lens,                               \
            void *stream) const;                                 \
    };                                                           \
    }

#endif
