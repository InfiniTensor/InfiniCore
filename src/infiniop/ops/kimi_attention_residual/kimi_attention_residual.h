#ifndef __INFINIOP_KIMI_ATTENTION_RESIDUAL_H__
#define __INFINIOP_KIMI_ATTENTION_RESIDUAL_H__

#include "../../operator.h"
#include "info.h"

#define DESCRIPTOR(NAMESPACE)                                                       \
                                                                                    \
    namespace op::kimi_attention_residual::NAMESPACE {                              \
    class Descriptor final : public InfiniopDescriptor {                            \
        struct Opaque;                                                              \
        Opaque *_opaque;                                                            \
        KimiAttentionResidualInfo _info;                                            \
        size_t _workspace_size;                                                     \
                                                                                    \
        Descriptor(Opaque *opaque,                                                  \
                   KimiAttentionResidualInfo info,                                  \
                   size_t workspace_size,                                           \
                   infiniDevice_t device_type,                                      \
                   int device_id)                                                   \
            : InfiniopDescriptor{device_type, device_id},                           \
              _opaque(opaque),                                                      \
              _info(std::move(info)),                                               \
              _workspace_size(workspace_size) {}                                    \
                                                                                    \
    public:                                                                         \
        ~Descriptor();                                                              \
                                                                                    \
        size_t workspaceSize() const { return _workspace_size; }                    \
                                                                                    \
        static infiniStatus_t create(                                               \
            infiniopHandle_t handle,                                                \
            Descriptor **desc_ptr,                                                  \
            infiniopTensorDescriptor_t output_desc,                                 \
            infiniopTensorDescriptor_t values_desc,                                 \
            infiniopTensorDescriptor_t norm_weight_desc,                            \
            infiniopTensorDescriptor_t projection_weight_desc,                      \
            float epsilon);                                                         \
                                                                                    \
        infiniStatus_t calculate(                                                   \
            void *workspace, size_t workspace_size,                                 \
            void *output, const void *values,                                       \
            const void *norm_weight, const void *projection_weight,                 \
            void *stream) const;                                                    \
    };                                                                              \
    }

#endif
