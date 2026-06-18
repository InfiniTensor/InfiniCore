#ifndef __MOE_TOPK_SIGMOID_H__
#define __MOE_TOPK_SIGMOID_H__

#include "../../operator.h"
#include "info.h"

#define DESCRIPTOR(NAMESPACE)                                                                                     \
    namespace op::moe_topk_sigmoid::NAMESPACE {                                                                   \
    class Descriptor final : public InfiniopDescriptor {                                                          \
        struct Opaque;                                                                                            \
        Opaque *_opaque;                                                                                          \
        MoeTopkSigmoidInfo _info;                                                                                 \
        size_t _workspace_size;                                                                                   \
        Descriptor(Opaque *opaque, MoeTopkSigmoidInfo info, size_t workspace_size, infiniDevice_t device, int id) \
            : InfiniopDescriptor{device, id}, _opaque(opaque), _info(info), _workspace_size(workspace_size) {}    \
                                                                                                                  \
    public:                                                                                                       \
        ~Descriptor();                                                                                            \
        size_t workspaceSize() const { return _workspace_size; }                                                  \
        static infiniStatus_t create(infiniopHandle_t handle, Descriptor **desc_ptr,                              \
                                     infiniopTensorDescriptor_t topk_weights_desc,                                \
                                     infiniopTensorDescriptor_t topk_indices_desc,                                \
                                     infiniopTensorDescriptor_t gating_output_desc,                               \
                                     infiniopTensorDescriptor_t correction_bias_desc,                             \
                                     bool renormalize);                                                           \
        infiniStatus_t calculate(void *workspace, size_t workspace_size, void *topk_weights, void *topk_indices,  \
                                 const void *gating_output, const void *correction_bias, void *stream) const;     \
    };                                                                                                            \
    }

#endif
