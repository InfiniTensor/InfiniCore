#ifndef __MOE_ARGSORT_BINCOUNT_H__
#define __MOE_ARGSORT_BINCOUNT_H__

#include "../../../utils.h"
#include "../../operator.h"

#define DESCRIPTOR(NAMESPACE)                                       \
    namespace op::moe_argsort_bincount::NAMESPACE {                 \
    class Descriptor final : public InfiniopDescriptor {            \
        size_t _total;                                              \
        size_t _num_experts;                                        \
        size_t _workspace_size;                                     \
                                                                    \
        Descriptor(size_t total,                                    \
                   size_t num_experts,                              \
                   infiniDevice_t device_type,                      \
                   int device_id)                                   \
            : InfiniopDescriptor{device_type, device_id},           \
              _total(total),                                        \
              _num_experts(num_experts),                            \
              _workspace_size(2 * num_experts * sizeof(int32_t)) {} \
                                                                    \
    public:                                                         \
        size_t workspaceSize() const { return _workspace_size; }    \
        static infiniStatus_t create(                               \
            infiniopHandle_t handle,                                \
            Descriptor **desc_ptr,                                  \
            infiniopTensorDescriptor_t tokens_per_experts_desc,     \
            infiniopTensorDescriptor_t sorted_indices_desc,         \
            infiniopTensorDescriptor_t inv_pos_desc,                \
            infiniopTensorDescriptor_t topk_ids_desc,               \
            size_t num_experts);                                    \
                                                                    \
        infiniStatus_t calculate(void *workspace,                   \
                                 size_t workspace_size,             \
                                 void *tokens_per_experts,          \
                                 void *sorted_indices,              \
                                 void *inv_pos,                     \
                                 const void *topk_ids,              \
                                 void *stream) const;               \
    };                                                              \
    }

#endif
