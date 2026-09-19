// infiniop/ops/lightning_attention.h

#ifndef __INFINIOP_LIGHTNING_ATTENTION_H__
#define __INFINIOP_LIGHTNING_ATTENTION_H__

#include "../../operator.h"
#include "info.h"

#define DESCRIPTOR(NAMESPACE)                                        \
                                                                     \
    namespace op::lightning_attention::NAMESPACE {                   \
    class Descriptor final : public InfiniopDescriptor {             \
        struct Opaque;                                               \
        Opaque *_opaque;                                             \
        LightningAttentionInfo _info;                                \
        size_t _workspace_size;                                      \
                                                                     \
        Descriptor(                                                  \
            Opaque *opaque,                                          \
            LightningAttentionInfo info,                             \
            size_t workspace_size,                                   \
            infiniDevice_t device_type,                              \
            int device_id)                                           \
            : InfiniopDescriptor{device_type, device_id},            \
              _opaque(opaque),                                       \
              _info(info),                                           \
              _workspace_size(workspace_size) {}                     \
                                                                     \
    public:                                                          \
        ~Descriptor();                                               \
                                                                     \
        size_t workspaceSize() const { return _workspace_size; }     \
                                                                     \
        static infiniStatus_t create(                                \
            infiniopHandle_t handle,                                 \
            Descriptor **desc_ptr,                                   \
            infiniopTensorDescriptor_t out_desc,                     \
            infiniopTensorDescriptor_t initial_state_desc,           \
            infiniopTensorDescriptor_t q_desc,                       \
            infiniopTensorDescriptor_t k_desc,                       \
            infiniopTensorDescriptor_t v_desc,                       \
            infiniopTensorDescriptor_t slope_desc,                   \
            infiniopTensorDescriptor_t initial_state_indices_desc,   \
            infiniopTensorDescriptor_t final_state_indices_desc);    \
                                                                     \
        infiniStatus_t calculate(                                    \
            void *workspace, size_t workspace_size,                  \
            void *out, void *initial_state,                          \
            const void *q, const void *k, const void *v,             \
            const void *slope,                                       \
            const void *initial_state_indices,                       \
            const void *final_state_indices,                         \
            void *stream) const;                                     \
    };                                                               \
    }

#endif // __INFINIOP_LIGHTNING_ATTENTION_H__
