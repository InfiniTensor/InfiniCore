#ifndef __LAYER_NORM_BACKWARD_H__
#define __LAYER_NORM_BACKWARD_H__

#include "../../../utils.h"
#include "../../operator.h"
#include "../../tensor.h"
#include "info.h"

#define DESCRIPTOR(NAMESPACE)                                         \
    namespace op::layer_norm_backward::NAMESPACE {                    \
    class Descriptor final : public InfiniopDescriptor {              \
        struct Opaque;                                                \
        Opaque *_opaque;                                              \
        LayerNormBackwardInfo _info;                                  \
        size_t _workspace_size;                                       \
        Descriptor(                                                   \
            infiniDtype_t dtype,                                      \
            LayerNormBackwardInfo info,                               \
            size_t workspace_size_,                                   \
            Opaque *opaque,                                           \
            infiniDevice_t device_type,                               \
            int device_id                                             \
        ) : InfiniopDescriptor{device_type, device_id},               \
              _opaque(opaque),                                        \
              _info(info),                                            \
              _workspace_size(workspace_size_) {}                     \
    public:                                                           \
        ~Descriptor();                                                \
        size_t workspaceSize() const { return _workspace_size; }      \
        static infiniStatus_t create(                                 \
            infiniopHandle_t handle,                                  \
            Descriptor **desc_ptr,                                    \
            infiniopTensorDescriptor_t grad_input_desc,               \
            infiniopTensorDescriptor_t grad_weight_desc,              \
            infiniopTensorDescriptor_t grad_bias_desc,                \
            infiniopTensorDescriptor_t grad_output_desc,              \
            infiniopTensorDescriptor_t weight_desc,                   \
            infiniopTensorDescriptor_t input_standardization_desc,    \
            infiniopTensorDescriptor_t input_std_deviation_desc       \
        );                                                            \
        infiniStatus_t calculate(                                     \
            void *workspace,                                          \
            size_t workspace_size,                                    \
            void * grad_input,                                        \
            void * grad_weight,                                       \
            void * grad_bias,                                         \
            const void * grad_output,                                 \
            const void * weight,                                      \
            const void * input_standardization,                       \
            const void * input_std_deviation,                         \
            void *stream                                              \
        ) const;                                                      \
    };                                                                \
    }

#endif