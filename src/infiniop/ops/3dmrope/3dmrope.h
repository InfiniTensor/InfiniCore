#ifndef __3DMROPE_H__
#define __3DMROPE_H__

#include "../../../utils.h"
#include "../../operator.h"
#include "../../tensor.h"

#define DESCRIPTOR(NAMESPACE)                                    \
                                                                 \
    namespace op::mrope3d::NAMESPACE {                           \
    class Descriptor final : public InfiniopDescriptor {        \
        struct Opaque;                                           \
        Opaque *_opaque;                                         \
        MRoPE3DInfo _info;                                       \
        size_t _workspace_size;                                  \
                                                                 \
        Descriptor(                                              \
            MRoPE3DInfo info,                                    \
            size_t workspace_size_,                              \
            Opaque *opaque,                                      \
            infiniDevice_t device_type,                          \
            int device_id)                                       \
            : InfiniopDescriptor{device_type, device_id},        \
              _opaque(opaque),                                   \
              _info(info),                                       \
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
            infiniopTensorDescriptor_t y_desc,                   \
            infiniopTensorDescriptor_t x_desc,                   \
            infiniopTensorDescriptor_t pos_desc,                 \
            infiniopTensorDescriptor_t sin_desc,                 \
            infiniopTensorDescriptor_t cos_desc,                 \
            infiniopTensorDescriptor_t rope_section_desc);       \
                                                                 \
        infiniStatus_t calculate(                                \
            void *workspace,                                     \
            size_t workspace_size,                               \
            void *y,                                             \
            const void *x,                                       \
            const void *pos_ids,                                 \
            const void *sin_table,                               \
            const void *cos_table,                               \
            const void *rope_section,                            \
            void *stream) const;                                 \
    };                                                           \
    }

class MRoPE3DInfo {
private:
    MRoPE3DInfo() = default;

public:
    infiniDtype_t data_type, pos_type, section_type;
    size_t seqlen, nhead, dhead, table_len, table_dim;
    ptrdiff_t
        y_stride_seqlen,
        y_stride_nhead,
        x_stride_seqlen,
        x_stride_nhead;

    static utils::Result<MRoPE3DInfo> createMRoPE3DInfo(
        infiniopTensorDescriptor_t y_desc,
        infiniopTensorDescriptor_t x_desc,
        infiniopTensorDescriptor_t pos_desc,
        infiniopTensorDescriptor_t sin_desc,
        infiniopTensorDescriptor_t cos_desc,
        infiniopTensorDescriptor_t rope_section_desc) {
        CHECK_OR_RETURN(
            y_desc != nullptr && x_desc != nullptr && pos_desc != nullptr && sin_desc != nullptr && cos_desc != nullptr && rope_section_desc != nullptr,
            INFINI_STATUS_NULL_POINTER);

        const infiniDtype_t data_type = y_desc->dtype();
        const infiniDtype_t pos_type = pos_desc->dtype();
        const infiniDtype_t section_type = rope_section_desc->dtype();
        CHECK_OR_RETURN(data_type == x_desc->dtype(),
                        INFINI_STATUS_BAD_TENSOR_DTYPE);
        // // sin_table and cos_table should be float32 for precision
        // CHECK_OR_RETURN(sin_desc->dtype() == INFINI_DTYPE_F32 && cos_desc->dtype() == INFINI_DTYPE_F32,
        //                 INFINI_STATUS_BAD_TENSOR_DTYPE);
        CHECK_OR_RETURN(data_type == x_desc->dtype() && data_type == sin_desc->dtype() && data_type == cos_desc->dtype(),
        INFINI_STATUS_BAD_TENSOR_DTYPE);
        CHECK_DTYPE(data_type, INFINI_DTYPE_F16, INFINI_DTYPE_BF16, INFINI_DTYPE_F32, INFINI_DTYPE_F64);
        CHECK_DTYPE_ANY_INT(pos_type);
        CHECK_DTYPE_ANY_INT(section_type);

        CHECK_OR_RETURN(y_desc->ndim() == 3
                            && x_desc->ndim() == 3
                            && pos_desc->ndim() == 2
                            && sin_desc->ndim() == 2
                            && cos_desc->ndim() == 2
                            && rope_section_desc->ndim() == 1,
                        INFINI_STATUS_BAD_TENSOR_SHAPE);

        const auto nhead = y_desc->dim(0),
                   seqlen = y_desc->dim(1),
                   dhead = y_desc->dim(2),
                   table_len = sin_desc->dim(0),
                   table_dim = sin_desc->dim(1);

        CHECK_OR_RETURN(nhead == x_desc->dim(0)
                            && seqlen == x_desc->dim(1) && seqlen == pos_desc->dim(0)
                            && dhead == x_desc->dim(2)
                            && table_len == cos_desc->dim(0) && table_dim == cos_desc->dim(1)
                            && pos_desc->dim(1) == 3
                            && rope_section_desc->dim(0) == 3,
                        INFINI_STATUS_BAD_TENSOR_SHAPE);

        CHECK_OR_RETURN(dhead == table_dim * 2, INFINI_STATUS_BAD_TENSOR_SHAPE);
        // Last dimension of x and y must be contiguous
        CHECK_OR_RETURN(y_desc->stride(2) == 1 && x_desc->stride(2) == 1, INFINI_STATUS_BAD_TENSOR_STRIDES);
        // sin table and cos table must be totally contiguous
        CHECK_OR_RETURN(sin_desc->isContiguous() && cos_desc->isContiguous(), INFINI_STATUS_BAD_TENSOR_STRIDES);
        // pos_ids and rope_section must be contiguous
        CHECK_OR_RETURN(pos_desc->isContiguous() && rope_section_desc->isContiguous(), INFINI_STATUS_BAD_TENSOR_STRIDES);

        return utils::Result<MRoPE3DInfo>(MRoPE3DInfo{
            data_type,
            pos_type,
            section_type,
            seqlen,
            nhead,
            dhead,
            table_len,
            table_dim,
            y_desc->stride(1),
            y_desc->stride(0),
            x_desc->stride(1),
            x_desc->stride(0),
        });
    }
};

#endif
