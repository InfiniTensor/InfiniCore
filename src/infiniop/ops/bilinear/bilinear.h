#ifndef __BILINEAR_H__
#define __BILINEAR_H__

#include "../../operator.h"

#define DESCRIPTOR(NAMESPACE)                                    \
                                                                 \
    namespace op::bilinear::NAMESPACE {                              \
    class Descriptor final : public InfiniopDescriptor {         \
        struct Opaque;                                           \
        Opaque *_opaque;                                         \
        size_t _workspace_size;                                  \
                                                                 \
        Descriptor(                                              \
            Opaque *opaque,                                      \
            size_t workspace_size_,                              \
            infiniDevice_t device_type,                          \
            int device_id)                                       \
            : InfiniopDescriptor{device_type, device_id},        \
              _opaque(opaque),                                   \
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
            infiniopTensorDescriptor_t out_desc,                 \
            infiniopTensorDescriptor_t x1_desc,                  \
            infiniopTensorDescriptor_t x2_desc,                  \
            infiniopTensorDescriptor_t weight_desc,              \
            infiniopTensorDescriptor_t bias_desc);               \
                                                                 \
    }
#endif // __BILINEAR_H__
