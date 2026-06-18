#ifndef __MOE_ALIGN_H__
#define __MOE_ALIGN_H__

#include "../../operator.h"
#include "info.h"

#define DESCRIPTOR(NAMESPACE)                                                                                                  \
                                                                                                                               \
    namespace op::moe_align::NAMESPACE {                                                                                       \
    class Descriptor final : public InfiniopDescriptor {                                                                       \
        struct Opaque;                                                                                                         \
        Opaque *_opaque;                                                                                                       \
        MoeAlignInfo _info;                                                                                                    \
        size_t _workspace_size;                                                                                                \
                                                                                                                               \
        Descriptor(Opaque *opaque, MoeAlignInfo info, size_t workspace_size, infiniDevice_t device, int id)                    \
            : InfiniopDescriptor{device, id}, _opaque(opaque), _info(info), _workspace_size(workspace_size) {}                 \
                                                                                                                               \
    public:                                                                                                                    \
        ~Descriptor();                                                                                                         \
                                                                                                                               \
        size_t workspaceSize() const { return _workspace_size; }                                                               \
                                                                                                                               \
        static infiniStatus_t create(infiniopHandle_t handle, Descriptor **desc_ptr,                                           \
                                     infiniopTensorDescriptor_t sorted_token_ids_desc,                                         \
                                     infiniopTensorDescriptor_t expert_ids_desc,                                               \
                                     infiniopTensorDescriptor_t num_tokens_post_padded_desc,                                   \
                                     infiniopTensorDescriptor_t topk_ids_desc,                                                 \
                                     size_t num_experts, size_t block_size);                                                   \
                                                                                                                               \
        infiniStatus_t calculate(void *workspace, size_t workspace_size,                                                       \
                                 void *sorted_token_ids, void *expert_ids, void *num_tokens_post_padded,                       \
                                 const void *topk_ids, const void *expert_map, bool pad_sorted_token_ids, void *stream) const; \
    };                                                                                                                         \
    }

#endif
