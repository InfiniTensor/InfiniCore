#ifndef __MOE_ALIGN_INFO_H__
#define __MOE_ALIGN_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"

namespace op::moe_align {

class MoeAlignInfo {
    MoeAlignInfo() = default;

public:
    size_t num_tokens;
    size_t topk;
    size_t numel;
    size_t num_experts;
    size_t block_size;
    size_t max_num_tokens_padded;
    size_t max_num_blocks;

    static utils::Result<MoeAlignInfo> create(
        infiniopTensorDescriptor_t sorted_token_ids_desc,
        infiniopTensorDescriptor_t expert_ids_desc,
        infiniopTensorDescriptor_t num_tokens_post_padded_desc,
        infiniopTensorDescriptor_t topk_ids_desc,
        size_t num_experts,
        size_t block_size) {
        if (topk_ids_desc->dtype() != INFINI_DTYPE_I32 || sorted_token_ids_desc->dtype() != INFINI_DTYPE_I32 || expert_ids_desc->dtype() != INFINI_DTYPE_I32 || num_tokens_post_padded_desc->dtype() != INFINI_DTYPE_I32) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        if (topk_ids_desc->ndim() != 2 || sorted_token_ids_desc->ndim() != 1 || expert_ids_desc->ndim() != 1 || num_tokens_post_padded_desc->ndim() != 1 || num_tokens_post_padded_desc->shape()[0] != 1 || num_experts == 0 || block_size == 0) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        if (topk_ids_desc->strides()[1] != 1 || sorted_token_ids_desc->strides()[0] != 1 || expert_ids_desc->strides()[0] != 1 || num_tokens_post_padded_desc->strides()[0] != 1) {
            return INFINI_STATUS_BAD_TENSOR_STRIDES;
        }

        const size_t num_tokens = topk_ids_desc->shape()[0];
        const size_t topk = topk_ids_desc->shape()[1];
        const size_t numel = num_tokens * topk;
        const size_t align_num_experts = num_experts + 1;
        const size_t min_num_tokens_padded = numel < align_num_experts
                                               ? numel * block_size
                                               : numel + align_num_experts * (block_size - 1);
        const size_t min_num_blocks = (min_num_tokens_padded + block_size - 1) / block_size;
        if (sorted_token_ids_desc->shape()[0] < min_num_tokens_padded || expert_ids_desc->shape()[0] < min_num_blocks) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        return utils::Result<MoeAlignInfo>(MoeAlignInfo{
            num_tokens,
            topk,
            numel,
            num_experts,
            block_size,
            sorted_token_ids_desc->shape()[0],
            expert_ids_desc->shape()[0],
        });
    }
};

} // namespace op::moe_align

#endif
