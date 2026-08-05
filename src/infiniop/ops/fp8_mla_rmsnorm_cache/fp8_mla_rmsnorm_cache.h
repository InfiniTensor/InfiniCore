#ifndef __FP8_MLA_RMSNORM_CACHE_H__
#define __FP8_MLA_RMSNORM_CACHE_H__

#include "../../../utils.h"
#include "../../operator.h"

#define DESCRIPTOR(NAMESPACE)                              \
    namespace op::fp8_mla_rmsnorm_cache::NAMESPACE {       \
    class Descriptor final : public InfiniopDescriptor {   \
        size_t _num_tokens;                                \
        size_t _num_cache_blocks;                          \
        size_t _block_size;                                \
        bool _write_vendor_cache;                          \
        float _eps;                                        \
                                                           \
        Descriptor(                                        \
            size_t num_tokens,                             \
            size_t num_cache_blocks,                       \
            size_t block_size,                             \
            bool write_vendor_cache,                       \
            double eps,                                    \
            infiniDevice_t device_type,                    \
            int device_id)                                 \
            : InfiniopDescriptor{device_type, device_id},  \
              _num_tokens(num_tokens),                     \
              _num_cache_blocks(num_cache_blocks),         \
              _block_size(block_size),                     \
              _write_vendor_cache(write_vendor_cache),     \
              _eps(static_cast<float>(eps)) {}             \
                                                           \
    public:                                                \
        static infiniStatus_t create(                      \
            infiniopHandle_t handle,                       \
            Descriptor **desc_ptr,                         \
            infiniopTensorDescriptor_t cache_desc,         \
            infiniopTensorDescriptor_t vendor_cache_desc,  \
            infiniopTensorDescriptor_t compressed_kv_desc, \
            infiniopTensorDescriptor_t norm_weight_desc,   \
            infiniopTensorDescriptor_t rope_desc,          \
            infiniopTensorDescriptor_t slot_mapping_desc,  \
            double eps);                                   \
                                                           \
        infiniStatus_t calculate(                          \
            void *cache,                                   \
            void *vendor_cache,                            \
            const void *compressed_kv,                     \
            const void *norm_weight,                       \
            const void *rope,                              \
            const void *slot_mapping,                      \
            void *stream) const;                           \
    };                                                     \
    }

#endif
