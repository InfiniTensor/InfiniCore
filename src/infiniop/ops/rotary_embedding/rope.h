#ifndef __ROPE_H__
#define __ROPE_H__

#include "../../operator.h"
#include "../../tensor.h"

#define DESCRIPTOR(NAMESPACE)                             \
                                                          \
    namespace op::rope::NAMESPACE {                       \
    class Descriptor final : public InfiniopDescriptor {  \
        struct Opaque;                                    \
        Opaque *_opaque;                                  \
        RoPEInfo _info;                                   \
                                                          \
        Descriptor(                                       \
            RoPEInfo info,                                \
            size_t workspace_size_,                       \
            Opaque *opaque,                               \
            infiniDevice_t device_type,                   \
            int device_id)                                \
            : InfiniopDescriptor{device_type, device_id}, \
              _opaque(opaque),                            \
              _info(info),                                \
              workspace_size(workspace_size_) {}          \
                                                          \
    public:                                               \
        size_t workspace_size;                            \
                                                          \
        ~Descriptor();                                    \
                                                          \
        static infiniStatus_t create(                     \
            infiniopHandle_t handle,                      \
            Descriptor **desc_ptr,                        \
            infiniopTensorDescriptor_t t_desc,            \
            infiniopTensorDescriptor_t pos_desc,          \
            infiniopTensorDescriptor_t sin_desc,          \
            infiniopTensorDescriptor_t cos_desc);         \
                                                          \
        infiniStatus_t calculate(                         \
            void *workspace,                              \
            size_t workspace_size,                        \
            void *t,                                      \
            const void *pos_ids,                          \
            const float *sin_table,                       \
            const float *cos_table,                       \
            void *stream) const;                          \
    };                                                    \
    }

// struct that stores data needed for RoPE
struct RoPEInfo {
    infiniDtype_t ty_t, ty_pos;
    size_t nt, nh, dh, nsin, ncos;
    ptrdiff_t
        s_nt,
        s_nh,
        s_np,
        s_nsin, s_ncos,
        s_dsin, s_dcos;
};

inline infiniStatus_t createRoPEInfo(
    RoPEInfo &info,
    infiniopTensorDescriptor_t t_desc,
    infiniopTensorDescriptor_t pos_desc,
    infiniopTensorDescriptor_t sin_desc,
    infiniopTensorDescriptor_t cos_desc) {

    if (!t_desc || !pos_desc || !sin_desc || !cos_desc) {
        return INFINI_STATUS_BAD_PARAM;
    }

    const infiniDtype_t t_dtype = t_desc->dtype();
    const infiniDtype_t pos_dtype = pos_desc->dtype();
    const auto nt = t_desc->dim(0),
               nh = t_desc->dim(1),
               dh = t_desc->dim(2),
               np = pos_desc->dim(0),
               nsin = sin_desc->dim(0),
               ncos = cos_desc->dim(0);

    if (sin_desc->dtype() != INFINI_DTYPE_F32 || cos_desc->dtype() != INFINI_DTYPE_F32) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    if (t_desc->ndim() != 3 || pos_desc->ndim() != 1 || sin_desc->ndim() != 2 || cos_desc->ndim() != 2) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    if (nt != np || dh % 2 != 0) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }
    if (t_desc->strides()[2] != 1) {
        return INFINI_STATUS_BAD_TENSOR_STRIDES;
    }

    // Initializing RoPEInfo struct
    info.ty_t = t_dtype;
    info.ty_pos = pos_dtype;
    info.nt = nt;
    info.nh = nh;
    info.dh = dh;
    info.nsin = nsin;
    info.ncos = ncos;
    info.s_nt = t_desc->stride(0);
    info.s_nh = t_desc->stride(1);
    info.s_np = pos_desc->stride(0);
    info.s_nsin = sin_desc->stride(0);
    info.s_ncos = cos_desc->stride(0);
    info.s_dsin = sin_desc->stride(1);
    info.s_dcos = cos_desc->stride(1);

    return INFINI_STATUS_SUCCESS;
}

#endif // __GEMM_H__
