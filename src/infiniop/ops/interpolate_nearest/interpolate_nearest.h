#ifndef __INTERPOLATE_NEAREST_H__
#define __INTERPOLATE_NEAREST_H__

#include "../../operator.h"
#include "info.h"

#define DESCRIPTOR(NAMESPACE)                                    \
                                                                 \
    namespace op::interpolate_nearest::NAMESPACE {               \
    class Descriptor final : public InfiniopDescriptor {         \
        struct Opaque;                                           \
        Opaque *_opaque;                                         \
                                                                 \
        InterpolateNearestInfo _info;                            \
        infiniDtype_t _dtype;                                    \
        size_t _workspace_size;                                  \
                                                                 \
        Descriptor(                                              \
            infiniDtype_t dtype,                                 \
            InterpolateNearestInfo info,                         \
            size_t workspace_size,                               \
            Opaque *opaque,                                      \
            infiniDevice_t device_type,                          \
            int device_id)                                       \
            : InfiniopDescriptor{device_type, device_id},        \
              _opaque(opaque),                                   \
              _info(info),                                       \
              _dtype(dtype),                                     \
              _workspace_size(workspace_size) {}                 \
                                                                 \
    public:                                                      \
        ~Descriptor();                                           \
                                                                 \
        static infiniStatus_t create(                            \
            infiniopHandle_t handle,                             \
            Descriptor **desc_ptr,                               \
            infiniopTensorDescriptor_t output_desc,              \
            infiniopTensorDescriptor_t input_desc);              \
                                                                 \
        size_t workspaceSize() const { return _workspace_size; } \
                                                                 \
        infiniStatus_t calculate(                                \
            void *workspace,                                     \
            size_t workspace_size,                               \
            void *output,                                        \
            const void *input,                                   \
            void *stream) const;                                 \
    };                                                           \
    }

#endif // __INTERPOLATE_NEAREST_H__
