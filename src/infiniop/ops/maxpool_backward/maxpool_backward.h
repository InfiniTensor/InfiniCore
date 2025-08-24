#ifndef __MAXPOOL_BACKWARD_H__
#define __MAXPOOL_BACKWARD_H__

#include "../../operator.h"
#include "info.h"

#define DESCRIPTOR(NAMESPACE)                                    \
                                                                 \
    namespace op::maxpool_backward::NAMESPACE {                  \
    class Descriptor final : public InfiniopDescriptor {         \
        struct Opaque;                                           \
        Opaque *_opaque;                                         \
        infiniDtype_t _dtype;                                    \
        MaxPoolBackwardInfo _info;                               \
        size_t _workspace_size;                                  \
                                                                 \
        Descriptor(                                              \
            infiniDtype_t dtype,                                 \
            MaxPoolBackwardInfo info,                            \
            size_t workspace_size_,                              \
            Opaque *opaque,                                      \
            infiniDevice_t device_type,                          \
            int device_id)                                       \
            : InfiniopDescriptor{device_type, device_id},        \
              _opaque(opaque),                                   \
              _dtype(dtype),                                     \
              _info(info),                                       \
              _workspace_size(workspace_size_) {}                \
                                                                 \
    public:                                                      \
        ~Descriptor();                                           \
                                                                 \
        size_t workspaceSize() const { return _workspace_size; } \
                                                                 \
        static infiniStatus_t create(                            \
            infiniopHandle_t handle,                             \
            Descriptor **desc_ptr,                               \
            infiniopTensorDescriptor_t grad_input_desc,          \
            infiniopTensorDescriptor_t grad_output_desc,         \
            infiniopTensorDescriptor_t input_desc,               \
            void *kernel_size,                                   \
            void *strides,                                       \
            void *pads,                                          \
            bool ceil_mode);                                     \
                                                                 \
        infiniStatus_t calculate(                                \
            void *workspace, size_t workspace_size,              \
            void *grad_input,                                    \
            const void *grad_output,                             \
            const void *input,                                   \
            void *stream) const;                                 \
    };                                                           \
    }

#endif // __MAXPOOL_BACKWARD_H__
