#ifndef __BLAS_DOT_H__
#define __BLAS_DOT_H__

#include "../../operator.h"
#include "info.h"

#define DESCRIPTOR(NAMESPACE)                                    \
                                                                 \
    namespace op::blas_dot::NAMESPACE {                          \
    class Descriptor final : public InfiniopDescriptor {         \
        struct Opaque;                                           \
        Opaque *_opaque;                                         \
        BlasDotInfo _info;                                       \
        size_t _workspace_size;                                  \
                                                                 \
        Descriptor(                                              \
            BlasDotInfo info,                                    \
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
            infiniopTensorDescriptor_t y_desc,                   \
            infiniopTensorDescriptor_t result_desc);             \
                                                                 \
        infiniStatus_t calculate(                                \
            void *workspace,                                     \
            size_t workspace_size,                               \
            const void *x,                                       \
            const void *y,                                       \
            void *result,                                        \
            void *stream) const;                                 \
    };                                                           \
    }

#endif // __BLAS_DOT_H__
