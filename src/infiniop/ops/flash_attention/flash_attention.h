#ifndef __FLASH_ATTENTION_H__
#define __FLASH_ATTENTION_H__

#include "../../../utils.h"
#include "../../operator.h"
#include "../../tensor.h"
#include <algorithm>

namespace op::flash_attention {
class FlashAttentionInfo {
    FlashAttentionInfo() = default;

public:
    size_t b, q_len, kv_len, nh, nkvh, d_qk, d_v;
    int has_mask;
    infiniDtype_t dtype;

    static utils::Result<FlashAttentionInfo> create(
        infiniopTensorDescriptor_t out_desc,
        infiniopTensorDescriptor_t q_desc,
        infiniopTensorDescriptor_t k_desc,
        infiniopTensorDescriptor_t v_desc,
        infiniopTensorDescriptor_t mask_desc) {
        auto dtype = out_desc->dtype();
        CHECK_OR_RETURN(
            dtype == q_desc->dtype()
                && dtype == k_desc->dtype()
                && dtype == v_desc->dtype(),
            INFINI_STATUS_BAD_TENSOR_DTYPE);
        CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);
        size_t b_ = 1;
        size_t i_len_ = 0;
        size_t i_head_ = 1;
        size_t i_dim_ = 2;
        int has_mask_ = mask_desc == nullptr ? 0 : 1;
        size_t ndim_ = out_desc->ndim();
        CHECK_OR_RETURN(ndim_ == 3 || ndim_ == 4, INFINI_STATUS_BAD_TENSOR_SHAPE);
        if (ndim_ == 4) {
            b_ = out_desc->dim(0);
            i_len_++;
            i_head_++;
            i_dim_++;
            CHECK_OR_RETURN(b_ == q_desc->dim(0) && b_ == k_desc->dim(0) && b_ == v_desc->dim(0), INFINI_STATUS_BAD_TENSOR_SHAPE);
        }

        size_t q_len_ = q_desc->dim(i_len_);
        CHECK_OR_RETURN(q_len_ == out_desc->dim(i_len_), INFINI_STATUS_BAD_TENSOR_SHAPE);

        size_t kv_len_ = k_desc->dim(i_len_);
        CHECK_OR_RETURN(kv_len_ == v_desc->dim(i_len_), INFINI_STATUS_BAD_TENSOR_SHAPE);

        size_t nh_ = out_desc->dim(i_head_);
        CHECK_OR_RETURN(nh_ == out_desc->dim(i_head_), INFINI_STATUS_BAD_TENSOR_SHAPE);

        size_t nkvh_ = k_desc->dim(i_head_);
        CHECK_OR_RETURN(nkvh_ == v_desc->dim(i_head_), INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(nh_ % nkvh_ == 0, INFINI_STATUS_BAD_TENSOR_SHAPE);

        size_t d_qk_ = q_desc->dim(i_dim_);
        CHECK_OR_RETURN(d_qk_ == k_desc->dim(i_dim_), INFINI_STATUS_BAD_TENSOR_SHAPE);

        size_t d_v_ = v_desc->dim(i_dim_);
        CHECK_OR_RETURN(d_v_ == out_desc->dim(i_dim_), INFINI_STATUS_BAD_TENSOR_SHAPE);

        CHECK_OR_RETURN(out_desc->isContiguous(), INFINI_STATUS_BAD_TENSOR_STRIDES);
        CHECK_OR_RETURN(q_desc->isContiguous(), INFINI_STATUS_BAD_TENSOR_STRIDES);
        CHECK_OR_RETURN(k_desc->isContiguous(), INFINI_STATUS_BAD_TENSOR_STRIDES);
        CHECK_OR_RETURN(v_desc->isContiguous(), INFINI_STATUS_BAD_TENSOR_STRIDES);

        if (has_mask_) {
            CHECK_OR_RETURN(mask_desc->ndim() == 2, INFINI_STATUS_BAD_TENSOR_SHAPE);
            CHECK_OR_RETURN(mask_desc->dim(0) == q_len_ && mask_desc->dim(1) == kv_len_, INFINI_STATUS_BAD_TENSOR_SHAPE);
            CHECK_OR_RETURN(mask_desc->isContiguous(), INFINI_STATUS_BAD_TENSOR_STRIDES);
        }

        return utils::Result<FlashAttentionInfo>{
            FlashAttentionInfo{b_, q_len_, kv_len_, nh_, nkvh_, d_qk_, d_v_, has_mask_, dtype}};
    }
};
} // namespace op::flash_attention

#endif //__FLASH_ATTENTION_H__
