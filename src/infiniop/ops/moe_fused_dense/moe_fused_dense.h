#ifndef __MOE_FUSED_DENSE_H__
#define __MOE_FUSED_DENSE_H__

#include "../../operator.h"
#include "info.h"

#define DESCRIPTOR(NAMESPACE)                                                                                    \
    namespace op::moe_fused_dense::NAMESPACE {                                                                   \
    class Descriptor final : public InfiniopDescriptor {                                                         \
        struct Opaque;                                                                                           \
        Opaque *_opaque;                                                                                         \
        MoeFusedDenseInfo _info;                                                                                 \
        size_t _workspace_size;                                                                                  \
                                                                                                                 \
        Descriptor(Opaque *opaque, MoeFusedDenseInfo info, size_t workspace_size, infiniDevice_t device, int id) \
            : InfiniopDescriptor{device, id}, _opaque(opaque), _info(info), _workspace_size(workspace_size) {}   \
                                                                                                                 \
    public:                                                                                                      \
        ~Descriptor();                                                                                           \
                                                                                                                 \
        size_t workspaceSize() const { return _workspace_size; }                                                 \
                                                                                                                 \
        static infiniStatus_t create(infiniopHandle_t handle, Descriptor **desc_ptr,                             \
                                     infiniopTensorDescriptor_t output_desc,                                     \
                                     infiniopTensorDescriptor_t hidden_states_desc,                              \
                                     infiniopTensorDescriptor_t w13_desc,                                        \
                                     infiniopTensorDescriptor_t w2_desc,                                         \
                                     infiniopTensorDescriptor_t topk_weights_desc,                               \
                                     infiniopTensorDescriptor_t topk_ids_desc,                                   \
                                     infiniopTensorDescriptor_t sorted_token_ids_desc,                           \
                                     infiniopTensorDescriptor_t expert_ids_desc,                                 \
                                     infiniopTensorDescriptor_t num_tokens_post_padded_desc);                    \
                                                                                                                 \
        infiniStatus_t calculate(void *workspace, size_t workspace_size,                                         \
                                 void *output, const void *hidden_states, const void *w13, const void *w2,       \
                                 const void *topk_weights, const void *topk_ids,                                 \
                                 const void *sorted_token_ids, const void *expert_ids,                           \
                                 const void *num_tokens_post_padded, void *stream) const;                        \
    };                                                                                                           \
    }

#endif
