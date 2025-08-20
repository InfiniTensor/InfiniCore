#ifndef FLASH_ATTENTION_BACKWARD_H
#define FLASH_ATTENTION_BACKWARD_H

#include "../../operator.h"
#include "info.h"

#define DESCRIPTOR(NAMESPACE)                                    \
                                                                 \
    namespace op::flash_attention_backward::NAMESPACE {          \
    class Descriptor final : public InfiniopDescriptor {         \
        struct Opaque;                                           \
        Opaque *_opaque;                                         \
        FlashAttentionBackwardInfo _info;                        \
        size_t _workspace_size;                                  \
                                                                 \
        Descriptor(                                              \
            FlashAttentionBackwardInfo info,                     \
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
            infiniopTensorDescriptor_t grad_q_desc,              \
            infiniopTensorDescriptor_t grad_k_desc,              \
            infiniopTensorDescriptor_t grad_v_desc,              \
            infiniopTensorDescriptor_t q_desc,                   \
            infiniopTensorDescriptor_t k_desc,                   \
            infiniopTensorDescriptor_t v_desc,                   \
            infiniopTensorDescriptor_t grad_out_desc,            \
            infiniopTensorDescriptor_t mask_desc,                \
            infiniopAttentionMaskType_t mask_type);              \
                                                                 \
        infiniStatus_t calculate(                                \
            void *workspace,                                     \
            size_t workspace_size,                               \
            void *grad_q,                                        \
            void *grad_k,                                        \
            void *grad_v,                                        \
            const void *q,                                       \
            const void *k,                                       \
            const void *v,                                       \
            const void *grad_out,                                \
            const void *mask,                                    \
            void *stream) const;                                 \
    };                                                           \
    }

#endif // FLASH_ATTENTION_BACKWARD_H
