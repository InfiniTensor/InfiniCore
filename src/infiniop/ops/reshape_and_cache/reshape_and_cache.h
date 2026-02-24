#ifndef RESHAPE_AND_CACHE_H
#define RESHAPE_AND_CACHE_H

#include "../../operator.h"
#include "info.h"

#define DESCRIPTOR(NAMESPACE)                                    \
                                                                 \
    namespace op::reshape_and_cache::NAMESPACE {                 \
    class Descriptor final : public InfiniopDescriptor {         \
        struct Opaque;                                           \
        Opaque *_opaque;                                         \
        ReshapeAndCacheInfo _info;                               \
        size_t _workspace_size;                                  \
                                                                 \
        Descriptor(                                              \
            Opaque *opaque,                                      \
            ReshapeAndCacheInfo info,                            \
            size_t workspace_size,                               \
            infiniDevice_t device_type,                          \
            int device_id)                                       \
            : InfiniopDescriptor{device_type, device_id},        \
              _opaque(opaque),                                   \
              _info(info),                                       \
              _workspace_size(workspace_size) {}                 \
                                                                 \
    public:                                                      \
        ~Descriptor();                                           \
                                                                 \
        size_t workspaceSize() const { return _workspace_size; } \
                                                                 \
        static infiniStatus_t create(                            \
            infiniopHandle_t handle,                             \
            Descriptor **desc_ptr,                               \
            infiniopTensorDescriptor_t key_desc,                 \
            infiniopTensorDescriptor_t value_desc,               \
            infiniopTensorDescriptor_t key_cache_desc,           \
            infiniopTensorDescriptor_t value_cache_desc,         \
            infiniopTensorDescriptor_t slot_mapping_desc,        \
            const char *kv_cache_dtype);                         \
                                                                 \
        infiniStatus_t calculate(                                \
            void *workspace,                                     \
            size_t workspace_size,                               \
            void *key,                                           \
            void *value,                                         \
            void *key_cache,                                     \
            void *value_cache,                                   \
            const void *slot_mapping,                            \
            const char *kv_cache_dtype,                          \
            void *k_scale,                                       \
            void *v_scale,                                       \
            void *stream) const;                                 \
    };                                                           \
    }

#endif // RESHAPE_AND_CACHE_H
