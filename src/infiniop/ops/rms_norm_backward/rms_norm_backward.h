#ifndef __RMS_NORM_BACKWARD_H__
#define __RMS_NORM_BACKWARD_H__

#include "../../../utils.h"
#include "../../operator.h"
#include "../../tensor.h"
#include "info.h"

#define DESCRIPTOR(NAMESPACE)                                         \
    namespace op::rms_norm_backward::NAMESPACE {                      \
    class Descriptor final : public InfiniopDescriptor {              \
        struct Opaque;                                                \
        Opaque *_opaque;                                              \
        RMSNormBackwardInfo _info;                                    \
        size_t _workspace_size;                                       \
        Descriptor(                                                   \
            infiniDtype_t dtype,                                      \
            RMSNormBackwardInfo info,                                 \
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
            infiniopTensorDescriptor_t grad_x_desc,                   \
            infiniopTensorDescriptor_t grad_w_desc,                   \
            infiniopTensorDescriptor_t grad_y_desc,                   \
            infiniopTensorDescriptor_t x_desc,                        \
            infiniopTensorDescriptor_t w_desc                         \
        );                                                            \
        infiniStatus_t calculate(                                     \
            void *workspace,                                          \
            size_t workspace_size,                                    \
            void * grad_x,                                            \
            void * grad_w,                                            \
            const void * grad_y,                                      \
            const void * x,                                           \
            const void * w,                                           \
            void *stream                                              \
        ) const;                                                      \
    };                                                                \
    }

#endif