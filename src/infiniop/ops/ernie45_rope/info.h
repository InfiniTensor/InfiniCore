#ifndef __ERNIE45_ROPE_INFO_H__
#define __ERNIE45_ROPE_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"

namespace op::ernie45_rope {

struct QKInfo {
    infiniDtype_t data_type;
    infiniDtype_t pos_type;
    size_t seqlen;
    size_t q_heads;
    size_t k_heads;
    size_t head_dim;
    ptrdiff_t q_stride_seq;
    ptrdiff_t q_stride_head;
    ptrdiff_t k_stride_seq;
    ptrdiff_t k_stride_head;
    ptrdiff_t pos_stride_seq;
    ptrdiff_t pos_stride_axis;
    bool pos_axis_first;
    double rope_theta;
    size_t section_h;
    size_t section_w;
    size_t section_t;

    static utils::Result<QKInfo> create(infiniopTensorDescriptor_t q_desc,
                                        infiniopTensorDescriptor_t k_desc,
                                        infiniopTensorDescriptor_t pos_desc,
                                        double rope_theta,
                                        size_t section_h,
                                        size_t section_w,
                                        size_t section_t) {
        CHECK_OR_RETURN(q_desc != nullptr && k_desc != nullptr && pos_desc != nullptr, INFINI_STATUS_NULL_POINTER);
        CHECK_OR_RETURN(q_desc->ndim() == 3 && k_desc->ndim() == 3, INFINI_STATUS_BAD_TENSOR_SHAPE);
        auto dtype = q_desc->dtype();
        CHECK_OR_RETURN(dtype == k_desc->dtype(), INFINI_STATUS_BAD_TENSOR_DTYPE);
        CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_BF16, INFINI_DTYPE_F32);
        CHECK_DTYPE_ANY_INT(pos_desc->dtype());

        size_t seqlen = q_desc->dim(0);
        size_t q_heads = q_desc->dim(1);
        size_t head_dim = q_desc->dim(2);
        CHECK_OR_RETURN(k_desc->dim(0) == seqlen && k_desc->dim(2) == head_dim, INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN((head_dim % 2) == 0, INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(section_h + section_w + section_t == head_dim / 2, INFINI_STATUS_BAD_PARAM);
        CHECK_OR_RETURN(q_desc->stride(2) == 1 && k_desc->stride(2) == 1, INFINI_STATUS_BAD_TENSOR_STRIDES);

        bool axis_first = false;
        ptrdiff_t pos_stride_seq = 0;
        ptrdiff_t pos_stride_axis = 0;
        if (pos_desc->ndim() == 2 && pos_desc->dim(0) == 3 && pos_desc->dim(1) == seqlen) {
            axis_first = true;
            pos_stride_axis = pos_desc->stride(0);
            pos_stride_seq = pos_desc->stride(1);
        } else if (pos_desc->ndim() == 2 && pos_desc->dim(0) == seqlen && pos_desc->dim(1) == 3) {
            pos_stride_seq = pos_desc->stride(0);
            pos_stride_axis = pos_desc->stride(1);
        } else if (pos_desc->ndim() == 3 && pos_desc->dim(0) == 1 && pos_desc->dim(1) == seqlen && pos_desc->dim(2) == 3) {
            pos_stride_seq = pos_desc->stride(1);
            pos_stride_axis = pos_desc->stride(2);
        } else {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        return utils::Result<QKInfo>(QKInfo{
            dtype,
            pos_desc->dtype(),
            seqlen,
            q_heads,
            k_desc->dim(1),
            head_dim,
            q_desc->stride(0),
            q_desc->stride(1),
            k_desc->stride(0),
            k_desc->stride(1),
            pos_stride_seq,
            pos_stride_axis,
            axis_first,
            rope_theta,
            section_h,
            section_w,
            section_t});
    }
};

struct VisionInfo {
    infiniDtype_t data_type;
    infiniDtype_t pos_type;
    size_t seqlen;
    size_t q_heads;
    size_t k_heads;
    size_t head_dim;
    ptrdiff_t q_stride_seq;
    ptrdiff_t q_stride_head;
    ptrdiff_t k_stride_seq;
    ptrdiff_t k_stride_head;
    ptrdiff_t pos_stride_seq;
    ptrdiff_t pos_stride_axis;
    double rope_theta;

    static utils::Result<VisionInfo> create(infiniopTensorDescriptor_t q_desc,
                                            infiniopTensorDescriptor_t k_desc,
                                            infiniopTensorDescriptor_t pos_desc,
                                            double rope_theta) {
        CHECK_OR_RETURN(q_desc != nullptr && k_desc != nullptr && pos_desc != nullptr, INFINI_STATUS_NULL_POINTER);
        CHECK_OR_RETURN(q_desc->ndim() == 3 && k_desc->ndim() == 3, INFINI_STATUS_BAD_TENSOR_SHAPE);
        auto dtype = q_desc->dtype();
        CHECK_OR_RETURN(dtype == k_desc->dtype(), INFINI_STATUS_BAD_TENSOR_DTYPE);
        CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_BF16, INFINI_DTYPE_F32);
        CHECK_DTYPE_ANY_INT(pos_desc->dtype());

        size_t seqlen = q_desc->dim(0);
        size_t head_dim = q_desc->dim(2);
        CHECK_OR_RETURN(k_desc->dim(0) == seqlen && k_desc->dim(2) == head_dim, INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN((head_dim % 4) == 0, INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(q_desc->stride(2) == 1 && k_desc->stride(2) == 1, INFINI_STATUS_BAD_TENSOR_STRIDES);
        CHECK_OR_RETURN(pos_desc->ndim() == 2 && pos_desc->dim(0) == seqlen && pos_desc->dim(1) == 2, INFINI_STATUS_BAD_TENSOR_SHAPE);

        return utils::Result<VisionInfo>(VisionInfo{
            dtype,
            pos_desc->dtype(),
            seqlen,
            q_desc->dim(1),
            k_desc->dim(1),
            head_dim,
            q_desc->stride(0),
            q_desc->stride(1),
            k_desc->stride(0),
            k_desc->stride(1),
            pos_desc->stride(0),
            pos_desc->stride(1),
            rope_theta});
    }
};

} // namespace op::ernie45_rope

#endif
