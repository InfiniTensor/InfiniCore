#ifndef __REPETITION_PENALTY_INFO_H__
#define __REPETITION_PENALTY_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"

namespace op::repetition_penalty {

struct RepetitionPenaltyInfo {
    infiniDtype_t dt_logits;
    size_t num_seqs;
    size_t vocab_size;

    static utils::Result<RepetitionPenaltyInfo> create(
        infiniopTensorDescriptor_t logits_desc,
        infiniopTensorDescriptor_t mask_desc) {

        auto dt_logits = logits_desc->dtype();
        auto dt_mask = mask_desc->dtype();

        // Check logits dtype (should be float types)
        CHECK_DTYPE(dt_logits, INFINI_DTYPE_F16, INFINI_DTYPE_BF16, INFINI_DTYPE_F32, INFINI_DTYPE_F64);

        // Check mask dtype (should be bool)
        CHECK_OR_RETURN(dt_mask == INFINI_DTYPE_BOOL, INFINI_STATUS_BAD_TENSOR_DTYPE);

        // Check shapes: both should be [num_seqs, vocab_size]
        CHECK_OR_RETURN(logits_desc->ndim() == 2, INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(mask_desc->ndim() == 2, INFINI_STATUS_BAD_TENSOR_SHAPE);

        auto num_seqs = logits_desc->dim(0);
        auto vocab_size = logits_desc->dim(1);
        auto mask_num_seqs = mask_desc->dim(0);
        auto mask_vocab_size = mask_desc->dim(1);

        CHECK_OR_RETURN(num_seqs == mask_num_seqs, INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(vocab_size == mask_vocab_size, INFINI_STATUS_BAD_TENSOR_SHAPE);

        // Check strides: should be contiguous
        CHECK_OR_RETURN(logits_desc->stride(1) == 1, INFINI_STATUS_BAD_TENSOR_STRIDES);
        CHECK_OR_RETURN(mask_desc->stride(1) == 1, INFINI_STATUS_BAD_TENSOR_STRIDES);

        return utils::Result<RepetitionPenaltyInfo>({dt_logits, num_seqs, vocab_size});
    }
};

} // namespace op::repetition_penalty

#endif // __REPETITION_PENALTY_INFO_H__
