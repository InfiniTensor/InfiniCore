#ifndef __LINEAR_H__
#define __LINEAR_H__

#include "../../operator.h"
#include "info.h"

#define DESCRIPTOR(NAMESPACE)                                                     \
                                                                                  \
    namespace op::linear::NAMESPACE {                                             \
    class Descriptor final : public InfiniopDescriptor {                          \
        struct Opaque;                                                            \
        Opaque *_opaque;                                                          \
        infiniDtype_t _dtype;                                                     \
        op::linear::MatmulInfo _info;                                             \
        size_t _workspace_size;                                                   \
                                                                                  \
        Descriptor(infiniDtype_t dtype, op::linear::MatmulInfo info,              \
                   size_t workspace_size_, Opaque *opaque,                        \
                   infiniDevice_t device_type, int device_id)                     \
            : InfiniopDescriptor{device_type, device_id}, _opaque(opaque),        \
              _dtype(dtype), _info(info), _workspace_size(workspace_size_) {}     \
                                                                                  \
    public:                                                                       \
        ~Descriptor();                                                            \
                                                                                  \
        size_t workspaceSize() const { return _workspace_size; }                  \
                                                                                  \
        static infiniStatus_t create(infiniopHandle_t handle,                     \
                                     Descriptor **desc_ptr,                       \
                                     infiniopTensorDescriptor_t d_desc,           \
                                     infiniopTensorDescriptor_t a_desc,           \
                                     infiniopTensorDescriptor_t b_desc,           \
                                     infiniopTensorDescriptor_t c_desc);          \
                                                                                  \
        infiniStatus_t calculate(float alpha, const void *a, const void *a_scale, \
                                 const void *b, const void *b_scale, float beta,  \
                                 const void *c, const void *c_scale,              \
                                 const void *bias, void *d, const void *d_scale,  \
                                 bool is_blockwise, bool is_a_1d_scaled,          \
                                 bool is_b_1d_scaled, void *workspace,            \
                                 size_t workspace_size, void *stream) const;      \
    };                                                                            \
    }

#endif // __LINEAR_H__
