#ifndef __FUSED_MOE_W4A8_H__
#define __FUSED_MOE_W4A8_H__

#include "../../operator.h"
#include "info.h"

#define DESCRIPTOR(NAMESPACE)                                          \
    namespace op::fused_moe_w4a8::NAMESPACE {                          \
    class Descriptor final : public InfiniopDescriptor {               \
        struct Opaque;                                                 \
        Opaque *_opaque;                                               \
        FusedMoeW4A8Info _info;                                        \
                                                                       \
        Descriptor(Opaque *opaque, FusedMoeW4A8Info info,              \
                   infiniDevice_t device_type, int device_id)          \
            : InfiniopDescriptor{device_type, device_id},              \
              _opaque(opaque), _info(info) {}                          \
                                                                       \
    public:                                                            \
        ~Descriptor();                                                 \
        size_t workspaceSize() const { return _info.workspaceSize(); } \
                                                                       \
        static infiniStatus_t create(                                  \
            infiniopHandle_t handle, Descriptor **desc_ptr,            \
            infiniopTensorDescriptor_t output_desc,                    \
            infiniopTensorDescriptor_t input_desc,                     \
            infiniopTensorDescriptor_t selected_experts_desc,          \
            infiniopTensorDescriptor_t routing_weights_desc,           \
            infiniopTensorDescriptor_t w13_packed_desc,                \
            infiniopTensorDescriptor_t w13_scale_desc,                 \
            infiniopTensorDescriptor_t w2_packed_desc,                 \
            infiniopTensorDescriptor_t w2_scale_desc,                  \
            infiniopFusedMoeActivation_t activation,                   \
            bool weights_are_aiter_shuffled);                          \
                                                                       \
        infiniStatus_t calculate(                                      \
            void *workspace, size_t workspace_size, void *output,      \
            const void *input, const void *selected_experts,           \
            const void *routing_weights, const void *w13_packed,       \
            const void *w13_scale, const void *w2_packed,              \
            const void *w2_scale, void *stream) const;                 \
    };                                                                 \
    }

#endif
