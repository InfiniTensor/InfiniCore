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

        CHECK_OR_RETURN(logits_desc->ndim() == 2, INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(mask_desc->ndim() == 2, INFINI_STATUS_BAD_TENSOR_SHAPE);

        auto num_seqs = logits_desc->dim(0);
        auto vocab_size = logits_desc->dim(1);

        CHECK_OR_RETURN(mask_desc->dim(0) == num_seqs, INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(mask_desc->dim(1) == vocab_size, INFINI_STATUS_BAD_TENSOR_SHAPE);

        CHECK_DTYPE(logits_desc->dtype(), INFINI_DTYPE_F16, INFINI_DTYPE_BF16, INFINI_DTYPE_F32, INFINI_DTYPE_F64);
        CHECK_OR_RETURN(mask_desc->dtype() == INFINI_DTYPE_BOOL, INFINI_STATUS_BAD_TENSOR_DTYPE);

        return utils::Result<RepetitionPenaltyInfo>({
            logits_desc->dtype(),
            num_seqs,
            vocab_size
        });
    }
};

} // namespace op::repetition_penalty

#endif // __REPETITION_PENALTY_INFO_H__
