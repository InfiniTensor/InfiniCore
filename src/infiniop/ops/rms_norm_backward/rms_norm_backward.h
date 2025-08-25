#ifndef RMS_NORM_BACKWARD_H
#define RMS_NORM_BACKWARD_H

#include "../../operator.h"
#include "info.h"

#define DESCRIPTOR(NAMESPACE)                                    \
                                                                 \
    namespace op::rms_norm_backward::NAMESPACE {                 \
    class Descriptor final : public InfiniopDescriptor {         \
        struct Opaque;                                           \
        Opaque *_opaque;                                         \
        RMSNormBackwardInfo _info;                               \
        size_t _workspace_size;                                  \
                                                                 \
        Descriptor(                                              \
            Opaque *opaque,                                      \
            RMSNormBackwardInfo info,                            \
            size_t workspace_size,                               \
            infiniDevice_t device_type,                          \
            int device_id)                                       \
            : InfiniopDescriptor{device_type, device_id},        \
              _opaque(opaque),                                   \
              _info(info),                                       \
              _workspace_size(workspace_size) {}                 \
                                                                 \
    public:                                                      \
        ~Descriptor();                                           \
                                                                 \
        size_t workspaceSize() const { return _workspace_size; } \
                                                                 \
        static infiniStatus_t create(                           \
            infiniopHandle_t handle,                             \
            Descriptor **desc_ptr,                               \
            infiniopTensorDescriptor_t grad_x_desc,              \
            infiniopTensorDescriptor_t grad_w_desc,              \
            infiniopTensorDescriptor_t grad_y_desc,              \
            infiniopTensorDescriptor_t x_desc,                   \
            infiniopTensorDescriptor_t w_desc,                   \
            float epsilon                                        \
        );                                                       \
                                                                 \
        infiniStatus_t calculate(                                \
            void *workspace, size_t workspace_size,              \
            void *grad_x,                                        \
            void *grad_w,                                        \
            const void *grad_y,                                  \
            const void *x,                                       \
            const void *w,                                       \
            void *stream) const;                                 \
    };                                                           \
    }

#endif // RMS_NORM_BACKWARD_H