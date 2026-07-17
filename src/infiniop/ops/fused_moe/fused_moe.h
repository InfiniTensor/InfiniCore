#ifndef __INFINIOP_FUSED_MOE_H__
#define __INFINIOP_FUSED_MOE_H__

#include "../../operator.h"
#include "info.h"

#define DESCRIPTOR(NAMESPACE)                                               \
                                                                            \
    namespace op::fused_moe::NAMESPACE {                                    \
    class Descriptor final : public InfiniopDescriptor {                    \
        struct Opaque;                                                      \
        Opaque *_opaque;                                                    \
        FusedMoeInfo _info;                                                 \
        size_t _workspace_size;                                             \
                                                                            \
        Descriptor(Opaque *opaque, FusedMoeInfo info, size_t workspace_size,\
                   infiniDevice_t device_type, int device_id)               \
            : InfiniopDescriptor{device_type, device_id},                   \
              _opaque(opaque), _info(info), _workspace_size(workspace_size) {}\
                                                                            \
    public:                                                                 \
        ~Descriptor();                                                      \
        size_t workspaceSize() const { return _workspace_size; }            \
        static infiniStatus_t create(                                       \
            infiniopHandle_t handle, Descriptor **desc_ptr,                 \
            infiniopTensorDescriptor_t out_desc,                            \
            infiniopTensorDescriptor_t input_desc,                          \
            infiniopTensorDescriptor_t token_selected_experts_desc,         \
            infiniopTensorDescriptor_t token_final_scales_desc,             \
            infiniopTensorDescriptor_t w1_desc,                             \
            infiniopTensorDescriptor_t w2_desc,                             \
            infiniopTensorDescriptor_t b1_desc,                             \
            infiniopTensorDescriptor_t b2_desc,                             \
            infiniopFusedMoeActivation_t activation);                       \
        infiniStatus_t calculate(                                           \
            void *workspace, size_t workspace_size,                         \
            void *out, const void *input,                                   \
            const void *token_selected_experts,                             \
            const void *token_final_scales,                                 \
            const void *w1, const void *w2,                                 \
            const void *b1, const void *b2, void *stream) const;            \
    };                                                                      \
    }

#endif // __INFINIOP_FUSED_MOE_H__
