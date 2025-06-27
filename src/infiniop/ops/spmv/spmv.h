#ifndef __SPMV_H__
#define __SPMV_H__

#include "../../operator.h"
#include "info.h"

#define DESCRIPTOR(NAMESPACE)                             \
                                                          \
    namespace op::spmv::NAMESPACE {                       \
    class Descriptor final : public InfiniopDescriptor {  \
        struct Opaque;                                    \
        Opaque *_opaque;                                  \
        infiniDtype_t _dtype;                             \
        SpMVInfo _info;                                   \
                                                          \
        Descriptor(                                       \
            infiniDtype_t dtype,                          \
            SpMVInfo info,                                \
            Opaque *opaque,                               \
            infiniDevice_t device_type,                   \
            int device_id)                                \
            : InfiniopDescriptor{device_type, device_id}, \
              _opaque(opaque),                            \
              _dtype(dtype),                              \
              _info(info) {}                              \
                                                          \
    public:                                               \
        ~Descriptor();                                    \
                                                          \
        static infiniStatus_t create(                     \
            infiniopHandle_t handle,                      \
            Descriptor **desc_ptr,                        \
            size_t num_cols,                              \
            size_t num_rows,                              \
            size_t nnz,                                   \
            infiniDtype_t dtype);                         \
                                                          \
        infiniStatus_t calculate(                         \
            void *y,                                      \
            const void *x,                                \
            const void *values,                           \
            const void *row_ptr,                          \
            const void *col_indices,                      \
            void *stream) const;                          \
    };                                                    \
    }

#endif // __SPMV_H__
