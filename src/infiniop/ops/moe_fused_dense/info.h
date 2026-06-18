#ifndef __MOE_FUSED_DENSE_INFO_H__
#define __MOE_FUSED_DENSE_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"

namespace op::moe_fused_dense {

struct MoeFusedDenseInfo {
    size_t num_tokens;
    size_t hidden_size;
    size_t num_experts;
    size_t intermediate_size;
    size_t topk;
    size_t max_num_tokens_padded;
    size_t max_num_blocks;
    infiniDtype_t dtype;

    static utils::Result<MoeFusedDenseInfo> create(
        infiniopTensorDescriptor_t output_desc,
        infiniopTensorDescriptor_t hidden_states_desc,
        infiniopTensorDescriptor_t w13_desc,
        infiniopTensorDescriptor_t w2_desc,
        infiniopTensorDescriptor_t topk_weights_desc,
        infiniopTensorDescriptor_t topk_ids_desc,
        infiniopTensorDescriptor_t sorted_token_ids_desc,
        infiniopTensorDescriptor_t expert_ids_desc,
        infiniopTensorDescriptor_t num_tokens_post_padded_desc) {
        if (output_desc->dtype() != hidden_states_desc->dtype() || output_desc->dtype() != w13_desc->dtype() || output_desc->dtype() != w2_desc->dtype() || topk_weights_desc->dtype() != INFINI_DTYPE_F32 || topk_ids_desc->dtype() != INFINI_DTYPE_I32 || sorted_token_ids_desc->dtype() != INFINI_DTYPE_I32 || expert_ids_desc->dtype() != INFINI_DTYPE_I32 || num_tokens_post_padded_desc->dtype() != INFINI_DTYPE_I32) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        auto dtype = output_desc->dtype();
        if (dtype != INFINI_DTYPE_F16 && dtype != INFINI_DTYPE_BF16) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        if (output_desc->ndim() != 2 || hidden_states_desc->ndim() != 2 || w13_desc->ndim() != 3 || w2_desc->ndim() != 3 || topk_weights_desc->ndim() != 2 || topk_ids_desc->ndim() != 2 || sorted_token_ids_desc->ndim() != 1 || expert_ids_desc->ndim() != 1 || num_tokens_post_padded_desc->ndim() != 1) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        const size_t num_tokens = hidden_states_desc->shape()[0];
        const size_t hidden_size = hidden_states_desc->shape()[1];
        const size_t num_experts = w13_desc->shape()[0];
        const size_t w13_rows = w13_desc->shape()[1];
        const size_t w13_cols = w13_desc->shape()[2];
        const size_t w2_experts = w2_desc->shape()[0];
        const size_t w2_rows = w2_desc->shape()[1];
        const size_t w2_cols = w2_desc->shape()[2];
        const size_t topk = topk_ids_desc->shape()[1];
        const size_t pairs = num_tokens * topk;
        const size_t max_num_tokens_padded = sorted_token_ids_desc->shape()[0];
        const size_t max_num_blocks = expert_ids_desc->shape()[0];

        if (w13_rows % 2 != 0 || num_tokens == 0 || hidden_size == 0 || num_experts == 0 || topk == 0 || output_desc->shape()[0] != num_tokens || output_desc->shape()[1] != hidden_size || w13_cols != hidden_size || w2_experts != num_experts || w2_rows != hidden_size || w2_cols * 2 != w13_rows || topk_weights_desc->shape()[0] != num_tokens || topk_weights_desc->shape()[1] != topk || topk_ids_desc->shape()[0] != num_tokens || max_num_tokens_padded < pairs || max_num_blocks == 0 || num_tokens_post_padded_desc->shape()[0] != 1) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        if (output_desc->strides()[1] != 1 || hidden_states_desc->strides()[1] != 1 || w13_desc->strides()[2] != 1 || w2_desc->strides()[2] != 1 || topk_weights_desc->strides()[1] != 1 || topk_ids_desc->strides()[1] != 1 || sorted_token_ids_desc->strides()[0] != 1 || expert_ids_desc->strides()[0] != 1 || num_tokens_post_padded_desc->strides()[0] != 1) {
            return INFINI_STATUS_BAD_TENSOR_STRIDES;
        }

        return utils::Result<MoeFusedDenseInfo>(MoeFusedDenseInfo{
            num_tokens,
            hidden_size,
            num_experts,
            w2_cols,
            topk,
            max_num_tokens_padded,
            max_num_blocks,
            dtype,
        });
    }
};

} // namespace op::moe_fused_dense

#endif
