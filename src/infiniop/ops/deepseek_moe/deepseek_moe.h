#ifndef DEEPSEEK_MOE_H
#define DEEPSEEK_MOE_H

#include "../../operator.h"
#include "info.h"

#define DESCRIPTOR(NAMESPACE)                                                                            \
    namespace op::deepseek_moe::NAMESPACE {                                                              \
    class Descriptor final : public InfiniopDescriptor {                                                 \
        struct Opaque;                                                                                   \
        Opaque *_opaque;                                                                                 \
        DeepseekMoeInfo _info;                                                                           \
        size_t _workspace_size;                                                                          \
                                                                                                         \
        Descriptor(Opaque *opaque, DeepseekMoeInfo info, size_t workspace_size,                          \
                   infiniDevice_t device_type, int device_id)                                            \
            : InfiniopDescriptor{device_type, device_id}, _opaque(opaque),                               \
              _info(info), _workspace_size(workspace_size) {}                                            \
                                                                                                         \
    public:                                                                                              \
        ~Descriptor();                                                                                   \
        size_t workspaceSize() const { return _workspace_size; }                                         \
        static infiniStatus_t create(infiniopHandle_t handle, Descriptor **desc_ptr,                     \
                                     infiniopTensorDescriptor_t out_desc,                                \
                                     infiniopTensorDescriptor_t hidden_desc,                             \
                                     infiniopTensorDescriptor_t topk_indices_desc,                       \
                                     infiniopTensorDescriptor_t topk_weights_desc,                       \
                                     size_t intermediate_size, size_t num_experts);                      \
        infiniStatus_t calculate(void *workspace, size_t workspace_size, void *out,                      \
                                 const void *hidden, const void *topk_indices,                           \
                                 const void *topk_weights, const void *const *gate_weights,              \
                                 const void *const *up_weights, const void *const *down_weights,         \
                                 void *stream) const;                                                    \
        infiniStatus_t calculateWithDevicePtrs(void *workspace, size_t workspace_size,                   \
                                               void *out, const void *hidden, const void *topk_indices,  \
                                               const void *topk_weights, const void *gate_weight_ptrs,   \
                                               const void *up_weight_ptrs, const void *down_weight_ptrs, \
                                               void *stream) const;                                      \
    };                                                                                                   \
    }

#endif
