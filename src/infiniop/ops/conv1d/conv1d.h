#ifndef __CONV1D_INTERNAL_H__
#define __CONV1D_INTERNAL_H__

#include "../../operator.h"

// Internal descriptor signature for backend implementations
#define DESCRIPTOR(NAMESPACE)                                                                 \
                                                                                                \
    namespace op::conv1d::NAMESPACE {                                                   \
    class Descriptor final : public InfiniopDescriptor {                                       \
        struct Opaque;                                                                         \
        Opaque *_opaque;                                                                       \
        infiniDtype_t _dtype;                                                                  \
        size_t _B, _C, _L, _K, _L_padded;                                                      \
        size_t _workspace_size;                                                                \
                                                                                                \
        Descriptor(infiniDtype_t dtype, size_t B, size_t C, size_t L, size_t K, size_t L_padded, \
                   size_t workspace_size, Opaque *opaque,                                      \
                   infiniDevice_t device_type, int device_id)                                  \
            : InfiniopDescriptor{device_type, device_id}, _opaque(opaque), _dtype(dtype),      \
              _B(B), _C(C), _L(L), _K(K), _L_padded(L_padded), _workspace_size(workspace_size) {}                   \
                                                                                                \
    public:                                                                                    \
        ~Descriptor();                                                                         \
                                                                                                \
        inline size_t B() const { return _B; }                                                 \
        inline size_t C() const { return _C; }                                                 \
        inline size_t L() const { return _L; }                                                 \
        inline size_t K() const { return _K; }                                                 \
        inline infiniDtype_t dtype() const { return _dtype; }                                  \
        size_t workspaceSize() const { return _workspace_size; }                               \
                                                                                                \
        static infiniStatus_t create(                                                          \
            infiniopHandle_t handle,                                                           \
            Descriptor **desc_ptr,                                                             \
            infiniopTensorDescriptor_t y_desc,                                                 \
            infiniopTensorDescriptor_t x_desc,                                                 \
            infiniopTensorDescriptor_t w_desc,                                                 \
            size_t kernel_size);                                                               \
                                                                                                \
        infiniStatus_t fn(                                                                     \
            void *workspace, size_t workspace_size,                                            \
            void *y, const void *x, const void *w,                                             \
            void *stream) const;                                             \
                                                                                                \
        infiniStatus_t update(void *params_void) const;                                        \
    };                                                                                         \
    }

#endif // __CONV1D_INTERNAL_H__
