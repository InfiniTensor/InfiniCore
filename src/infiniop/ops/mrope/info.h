#ifndef __MROPE_INFO_H__
#define __MROPE_INFO_H__

#include "../../../utils.h"
#include "../../operator.h"
#include "../../tensor.h"

class MRoPEInfo {
private:
    MRoPEInfo() = default;

public:
    infiniDtype_t data_type, position_type;
    size_t num_tokens, num_q_heads, num_kv_heads, head_size, rotary_dim, half_rotary_dim;
    size_t max_position_embeddings;
    size_t section_t, section_h, section_w;
    ptrdiff_t q_out_stride_token, q_out_stride_head, k_out_stride_token, k_out_stride_head;
    ptrdiff_t q_stride_token, q_stride_head, k_stride_token, k_stride_head;
    ptrdiff_t cos_stride_position, sin_stride_position;
    ptrdiff_t positions_stride_axis, positions_stride_token;
    bool positions_has_axes;
    bool interleaved;

    static utils::Result<MRoPEInfo>
    create(
        infiniopTensorDescriptor_t q_out_desc,
        infiniopTensorDescriptor_t k_out_desc,
        infiniopTensorDescriptor_t q_desc,
        infiniopTensorDescriptor_t k_desc,
        infiniopTensorDescriptor_t cos_desc,
        infiniopTensorDescriptor_t sin_desc,
        infiniopTensorDescriptor_t positions_desc,
        int head_size,
        int rotary_dim,
        int section_t,
        int section_h,
        int section_w,
        bool interleaved) {
        CHECK_OR_RETURN(q_out_desc != nullptr && k_out_desc != nullptr && q_desc != nullptr && k_desc != nullptr && cos_desc != nullptr && sin_desc != nullptr && positions_desc != nullptr,
                        INFINI_STATUS_NULL_POINTER);
        CHECK_OR_RETURN(head_size > 0 && rotary_dim > 0 && section_t > 0 && section_h > 0 && section_w > 0,
                        INFINI_STATUS_BAD_PARAM);
        CHECK_OR_RETURN(rotary_dim <= head_size && rotary_dim % 2 == 0, INFINI_STATUS_BAD_PARAM);
        CHECK_OR_RETURN(size_t(section_t + section_h + section_w) == size_t(rotary_dim / 2), INFINI_STATUS_BAD_PARAM);

        const infiniDtype_t data_type = q_desc->dtype();
        CHECK_OR_RETURN(data_type == k_desc->dtype() && data_type == q_out_desc->dtype() && data_type == k_out_desc->dtype() && data_type == cos_desc->dtype() && data_type == sin_desc->dtype(),
                        INFINI_STATUS_BAD_TENSOR_DTYPE);
        CHECK_DTYPE(data_type, INFINI_DTYPE_F16, INFINI_DTYPE_BF16, INFINI_DTYPE_F32, INFINI_DTYPE_F64);
        CHECK_DTYPE(positions_desc->dtype(), INFINI_DTYPE_I64, INFINI_DTYPE_I32);

        CHECK_OR_RETURN(q_desc->ndim() == 2 && k_desc->ndim() == 2 && q_out_desc->ndim() == 2 && k_out_desc->ndim() == 2,
                        INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(cos_desc->ndim() == 2 && sin_desc->ndim() == 2, INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(cos_desc->dim(0) == sin_desc->dim(0), INFINI_STATUS_BAD_TENSOR_SHAPE);

        const size_t num_tokens = q_desc->dim(0);
        CHECK_OR_RETURN(k_desc->dim(0) == num_tokens && q_out_desc->dim(0) == num_tokens && k_out_desc->dim(0) == num_tokens,
                        INFINI_STATUS_BAD_TENSOR_SHAPE);
        bool positions_has_axes = false;
        ptrdiff_t positions_stride_axis = 0;
        ptrdiff_t positions_stride_token = 1;
        if (positions_desc->ndim() == 1) {
            CHECK_OR_RETURN(positions_desc->dim(0) == num_tokens, INFINI_STATUS_BAD_TENSOR_SHAPE);
            positions_stride_token = positions_desc->stride(0);
        } else if (positions_desc->ndim() == 2) {
            CHECK_OR_RETURN(positions_desc->dim(0) == 3 && positions_desc->dim(1) == num_tokens, INFINI_STATUS_BAD_TENSOR_SHAPE);
            positions_has_axes = true;
            positions_stride_axis = positions_desc->stride(0);
            positions_stride_token = positions_desc->stride(1);
        } else {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        CHECK_OR_RETURN(cos_desc->dim(1) == size_t(rotary_dim / 2) && sin_desc->dim(1) == size_t(rotary_dim / 2),
                        INFINI_STATUS_BAD_TENSOR_SHAPE);

        CHECK_OR_RETURN(q_desc->dim(1) % size_t(head_size) == 0 && k_desc->dim(1) % size_t(head_size) == 0,
                        INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(q_out_desc->dim(1) == q_desc->dim(1) && k_out_desc->dim(1) == k_desc->dim(1),
                        INFINI_STATUS_BAD_TENSOR_SHAPE);

        CHECK_OR_RETURN(q_desc->stride(1) == 1 && k_desc->stride(1) == 1 && q_out_desc->stride(1) == 1 && k_out_desc->stride(1) == 1,
                        INFINI_STATUS_BAD_TENSOR_STRIDES);
        CHECK_OR_RETURN(cos_desc->stride(1) == 1 && sin_desc->stride(1) == 1, INFINI_STATUS_BAD_TENSOR_STRIDES);

        return utils::Result<MRoPEInfo>(MRoPEInfo{
            data_type,
            positions_desc->dtype(),
            num_tokens,
            q_desc->dim(1) / size_t(head_size),
            k_desc->dim(1) / size_t(head_size),
            size_t(head_size),
            size_t(rotary_dim),
            size_t(rotary_dim / 2),
            cos_desc->dim(0),
            size_t(section_t),
            size_t(section_h),
            size_t(section_w),
            q_out_desc->stride(0),
            ptrdiff_t(head_size),
            k_out_desc->stride(0),
            ptrdiff_t(head_size),
            q_desc->stride(0),
            ptrdiff_t(head_size),
            k_desc->stride(0),
            ptrdiff_t(head_size),
            cos_desc->stride(0),
            sin_desc->stride(0),
            positions_stride_axis,
            positions_stride_token,
            positions_has_axes,
            interleaved,
        });
    }
};

#endif // __MROPE_INFO_H__
