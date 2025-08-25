#ifndef FLASH_ATTENTION_H
#define FLASH_ATTENTION_H

#include "../../operator.h"
#include "info.h"

#define DESCRIPTOR(NAMESPACE)                                    \
                                                                 \
    namespace op::flash_attention::NAMESPACE {                   \
    class Descriptor final : public InfiniopDescriptor {         \
        struct Opaque;                                           \
        Opaque *_opaque;                                         \
        FlashAttentionInfo _info;                                \
        size_t _workspace_size;                                  \
                                                                 \
        Descriptor(                                              \
            FlashAttentionInfo info,                             \
            size_t workspace_size_,                              \
            Opaque *opaque,                                      \
            infiniDevice_t device_type,                          \
            int device_id)                                       \
            : InfiniopDescriptor{device_type, device_id},        \
              _opaque(opaque),                                   \
              _info(info),                                       \
              _workspace_size(workspace_size_) {}                \
                                                                 \
    public:                                                      \
        ~Descriptor();                                           \
                                                                 \
        size_t workspaceSize() const { return _workspace_size; } \
                                                                 \
        static infiniStatus_t create(                            \
            infiniopHandle_t handle,                             \
            Descriptor **desc_ptr,                               \
            infiniopTensorDescriptor_t out_desc,                 \
            infiniopTensorDescriptor_t l_desc,                   \
            infiniopTensorDescriptor_t q_desc,                   \
            infiniopTensorDescriptor_t k_desc,                   \
            infiniopTensorDescriptor_t v_desc,                   \
            infiniopTensorDescriptor_t mask_desc,                \
            infiniopAttentionMaskType_t mask_type);              \
                                                                 \
        infiniStatus_t calculate(                                \
            void *workspace,                                     \
            size_t workspace_size,                               \
            void *out,                                           \
            void *l,                                             \
            const void *q,                                       \
            const void *k,                                       \
            const void *v,                                       \
            const void *mask,                                    \
            void *stream) const;                                 \
    };                                                           \
    }

#endif // FLASH_ATTENTION_H
