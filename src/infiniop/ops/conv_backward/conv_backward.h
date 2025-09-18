#ifndef __CONV_BACKWARD_H__
#define __CONV_BACKWARD_H__

#include "../../operator.h"

#define DESCRIPTOR(NAMESPACE)                                     \
    namespace op::conv_backward::NAMESPACE {                      \
    class Descriptor final : public InfiniopDescriptor {          \
        struct Opaque;                                            \
        Opaque *_opaque;                                          \
        infiniDtype_t _dtype;                                     \
        size_t _workspace_size;                                   \
        Descriptor(                                               \
            infiniDtype_t dtype,                                  \
            size_t workspace_size_,                               \
            Opaque *opaque,                                       \
            infiniDevice_t device_type,                           \
            int device_id)                                        \
            : InfiniopDescriptor{device_type, device_id},         \
              _opaque(opaque),                                    \
              _dtype(dtype),                                      \
              _workspace_size(workspace_size_) {}                 \
                                                                  \
    public:                                                       \
        ~Descriptor();                                            \
        size_t workspaceSize() const { return _workspace_size; }  \
        static infiniStatus_t create(                             \
            infiniopHandle_t handle,                              \
            Descriptor **desc_ptr,                                \
            infiniopTensorDescriptor_t grad_output_desc,          \
            infiniopTensorDescriptor_t input_desc,                \
            infiniopTensorDescriptor_t weight_desc,               \
            infiniopTensorDescriptor_t bias_desc,                 \
            void *pads,                                           \
            void *strides,                                        \
            void *dilations,                                      \
            size_t groups);                                       \
        infiniStatus_t calculate(                                 \
            void *workspace, size_t workspace_size,               \
            void *grad_input, void *grad_weight, void *grad_bias, \
            const void *grad_output,                              \
            const void *input, const void *weight,                \
            void *stream) const;                                  \
    };                                                            \
    }

#endif // __CONV_BACKWARD_H__
