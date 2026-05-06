#ifndef __BLAS_COPY_H__
#define __BLAS_COPY_H__

#include "../../operator.h"
#include "info.h"

#define DESCRIPTOR(NAMESPACE)                                    \
                                                                 \
    namespace op::blas_copy::NAMESPACE {                         \
    class Descriptor final : public InfiniopDescriptor {         \
        struct Opaque;                                           \
        Opaque *_opaque;                                         \
        BlasCopyInfo _info;                                      \
        size_t _workspace_size;                                  \
                                                                 \
        Descriptor(                                              \
            BlasCopyInfo info,                                   \
            size_t workspace_size_,                              \
            Opaque *opaque,                                      \
            infiniDevice_t device_type,                          \
            int device_id)                                       \
            : InfiniopDescriptor{device_type, device_id},        \
              _opaque(opaque),                                   \
              _info(std::move(info)),                            \
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
            infiniopTensorDescriptor_t x_desc,                   \
            infiniopTensorDescriptor_t y_desc);                  \
                                                                 \
        infiniStatus_t calculate(                                \
            void *workspace,                                     \
            size_t workspace_size,                               \
            const void *x,                                       \
            void *y,                                             \
            void *stream) const;                                 \
    };                                                           \
    }

#endif // __BLAS_COPY_H__
