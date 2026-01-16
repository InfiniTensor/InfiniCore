#ifndef __2DMROPE_H__
#define __2DMROPE_H__

#include "../../../utils.h"
#include "../../operator.h"
#include "../../tensor.h"
#include <cstdio>

#define DESCRIPTOR(NAMESPACE)                                    \
                                                                 \
    namespace op::mrope2d::NAMESPACE {                           \
    class Descriptor final : public InfiniopDescriptor {         \
        struct Opaque;                                           \
        Opaque *_opaque;                                         \
        MRoPE2DInfo _info;                                       \
        size_t _workspace_size;                                  \
                                                                 \
        Descriptor(                                              \
            MRoPE2DInfo info,                                    \
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
            infiniopTensorDescriptor_t cos_desc);                \
                                                                 \
        infiniStatus_t calculate(                                \
            void *workspace,                                     \
            size_t workspace_size,                               \
            void *y,                                             \
            const void *x,                                       \
            const void *pos_ids,                                 \
            const void *sin_table,                               \
            const void *cos_table,                               \
            void *stream) const;                                 \
    };                                                           \
    }

class MRoPE2DInfo {
private:
    MRoPE2DInfo() = default;

public:
    infiniDtype_t data_type, pos_type;
    size_t seqlen, nhead, dhead, table_len, table_dim;
    ptrdiff_t
        y_stride_seqlen,
        y_stride_nhead,
        x_stride_seqlen,
        x_stride_nhead;

    static utils::Result<MRoPE2DInfo> createMRoPE2DInfo(
        infiniopTensorDescriptor_t y_desc,
        infiniopTensorDescriptor_t x_desc,
        infiniopTensorDescriptor_t pos_desc,
        infiniopTensorDescriptor_t sin_desc,
        infiniopTensorDescriptor_t cos_desc) {
        CHECK_OR_RETURN(
            y_desc != nullptr && x_desc != nullptr && pos_desc != nullptr && sin_desc != nullptr && cos_desc != nullptr,
            INFINI_STATUS_NULL_POINTER);

        const infiniDtype_t data_type = y_desc->dtype();
        const infiniDtype_t pos_type = pos_desc->dtype();
        CHECK_OR_RETURN(data_type == x_desc->dtype(),
                        INFINI_STATUS_BAD_TENSOR_DTYPE);
        // // sin_table and cos_table should be float32 for precision
        // CHECK_OR_RETURN(sin_desc->dtype() == INFINI_DTYPE_F32 && cos_desc->dtype() == INFINI_DTYPE_F32,
        //                 INFINI_STATUS_BAD_TENSOR_DTYPE);
        CHECK_OR_RETURN(data_type == x_desc->dtype() && data_type == sin_desc->dtype() && data_type == cos_desc->dtype(),
                        INFINI_STATUS_BAD_TENSOR_DTYPE);
        CHECK_DTYPE(data_type, INFINI_DTYPE_F16, INFINI_DTYPE_BF16, INFINI_DTYPE_F32, INFINI_DTYPE_F64);
        CHECK_DTYPE_ANY_INT(pos_type);

        CHECK_OR_RETURN(y_desc->ndim() == 3
                            && x_desc->ndim() == 3
                            && pos_desc->ndim() == 2
                            && sin_desc->ndim() == 2
                            && cos_desc->ndim() == 2,
                        INFINI_STATUS_BAD_TENSOR_SHAPE);

        const auto nhead = y_desc->dim(0),
                   seqlen = y_desc->dim(1),
                   dhead = y_desc->dim(2),
                   table_len = sin_desc->dim(0),
                   table_dim = sin_desc->dim(1);
        printf("y_desc->dim(0): %zu, y_desc->dim(1): %zu, y_desc->dim(2): %zu\n", y_desc->dim(0), y_desc->dim(1), y_desc->dim(2));
        printf("x_desc->dim(0): %zu, x_desc->dim(1): %zu, x_desc->dim(2): %zu\n", x_desc->dim(0), x_desc->dim(1), x_desc->dim(2));
        printf("pos_desc->dim(0): %zu, pos_desc->dim(1): %zu\n", pos_desc->dim(0), pos_desc->dim(1));
        printf("sin_desc->dim(0): %zu, sin_desc->dim(1): %zu\n", sin_desc->dim(0), sin_desc->dim(1));
        printf("cos_desc->dim(0): %zu, cos_desc->dim(1): %zu\n", cos_desc->dim(0), cos_desc->dim(1));
        printf("nhead: %zu, seqlen: %zu, dhead: %zu, table_len: %zu, table_dim: %zu\n", nhead, seqlen, dhead, table_len, table_dim);

        CHECK_OR_RETURN(nhead == x_desc->dim(0)
                            && seqlen == x_desc->dim(1) && seqlen == pos_desc->dim(0)
                            && dhead == x_desc->dim(2)
                            && table_len == cos_desc->dim(0) && table_dim == cos_desc->dim(1)
                            && pos_desc->dim(1) == 2,
                        INFINI_STATUS_BAD_TENSOR_SHAPE);

        CHECK_OR_RETURN(dhead == table_dim * 4, INFINI_STATUS_BAD_TENSOR_SHAPE); // 2D MRoPE: dhead = table_dim * 4
        // Last dimension of x and y must be contiguous
        CHECK_OR_RETURN(y_desc->stride(2) == 1 && x_desc->stride(2) == 1, INFINI_STATUS_BAD_TENSOR_STRIDES);
        // sin table and cos table must be totally contiguous
        CHECK_OR_RETURN(sin_desc->isContiguous() && cos_desc->isContiguous(), INFINI_STATUS_BAD_TENSOR_STRIDES);
        // pos_ids must be contiguous
        CHECK_OR_RETURN(pos_desc->isContiguous(), INFINI_STATUS_BAD_TENSOR_STRIDES);

        return utils::Result<MRoPE2DInfo>(MRoPE2DInfo{
            data_type,
            pos_type,
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
