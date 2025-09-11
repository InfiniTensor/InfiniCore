#ifndef __CONV1D_H__
#define __CONV1D_H__

#include "../../operator.h"
#include "info.h"

#define DESCRIPTOR(NAMESPACE)                                    \
                                                                 \
    namespace op::conv1d::NAMESPACE {                            \
    class Descriptor final : public InfiniopDescriptor {         \
        struct Opaque;                                           \
        Opaque *_opaque;                                         \
        infiniDtype_t _dtype;                                    \
        Conv1dInfo _info;                                        \
        size_t _workspace_size;                                  \
                                                                 \
        Descriptor(                                              \
            infiniDtype_t dtype,                                 \
            Conv1dInfo info,                                     \
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
        inline infiniDtype_t dtype() const { return _dtype; }   \
        inline const Conv1dInfo& info() const { return _info; } \
        size_t workspaceSize() const { return _workspace_size; } \
                                                                 \
        static infiniStatus_t create(                            \
            infiniopHandle_t handle,                             \
            Descriptor **desc_ptr,                               \
            infiniopTensorDescriptor_t y,                        \
            infiniopTensorDescriptor_t x,                        \
            infiniopTensorDescriptor_t w,                        \
            infiniopTensorDescriptor_t b,                        \
            const void *pads,                                    \
            const void *strides,                                 \
            const void *dilations,                               \
            size_t n);                                           \
                                                                 \
        infiniStatus_t calculate(                                \
            void *workspace, size_t workspace_size,              \
            void *y,                                             \
            const void *x,                                       \
            const void *w,                                       \
            const void *bias,                                    \
            void *stream) const;                                 \
    };                                                           \
    }

#endif // __CONV1D_H__
