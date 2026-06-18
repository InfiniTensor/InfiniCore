#ifndef __MOE_FUSED_GATE_H__
#define __MOE_FUSED_GATE_H__

#include "../../operator.h"
#include "info.h"

#define DESCRIPTOR(NAMESPACE)                                                                                    \
    namespace op::moe_fused_gate::NAMESPACE {                                                                    \
    class Descriptor final : public InfiniopDescriptor {                                                         \
        struct Opaque;                                                                                           \
        Opaque *_opaque;                                                                                         \
        MoeFusedGateInfo _info;                                                                                  \
        size_t _workspace_size;                                                                                  \
        Descriptor(Opaque *opaque, MoeFusedGateInfo info, size_t workspace_size, infiniDevice_t device, int id)  \
            : InfiniopDescriptor{device, id}, _opaque(opaque), _info(info), _workspace_size(workspace_size) {}   \
                                                                                                                 \
    public:                                                                                                      \
        ~Descriptor();                                                                                           \
        size_t workspaceSize() const { return _workspace_size; }                                                 \
        static infiniStatus_t create(infiniopHandle_t handle, Descriptor **desc_ptr,                             \
                                     infiniopTensorDescriptor_t topk_weights_desc,                               \
                                     infiniopTensorDescriptor_t topk_indices_desc,                               \
                                     infiniopTensorDescriptor_t input_desc,                                      \
                                     infiniopTensorDescriptor_t bias_desc,                                       \
                                     size_t num_expert_group, size_t topk_group,                                 \
                                     size_t num_fused_shared_experts, float routed_scaling_factor,               \
                                     bool apply_routed_scaling_factor_on_output);                                \
        infiniStatus_t calculate(void *workspace, size_t workspace_size, void *topk_weights, void *topk_indices, \
                                 const void *input, const void *bias, void *stream) const;                       \
    };                                                                                                           \
    }

#endif
