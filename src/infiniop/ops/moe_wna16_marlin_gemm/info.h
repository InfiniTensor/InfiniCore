#ifndef __MOE_WNA16_MARLIN_GEMM_INFO_H__
#define __MOE_WNA16_MARLIN_GEMM_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include <vector>

#include <cassert>

namespace op::moe_wna16_marlin_gemm {

class MoeWna16MarlinGemmInfo {
    MoeWna16MarlinGemmInfo() = default;

public:
    infiniDtype_t dtype;
    int size_m, size_n, size_k, top_k, moe_block_size;
    size_t num_groups, sorted_token_ids_size_0, b_q_weight_size_1, b_q_weight_size_2, b_zeros_size_1, b_zeros_size_2, c_size_0;
    bool has_act_order, has_bias, has_zp;

    static utils::Result<MoeWna16MarlinGemmInfo> create(
        infiniopTensorDescriptor_t c_desc,
        infiniopTensorDescriptor_t a_desc,
        infiniopTensorDescriptor_t b_q_weight_desc,
        infiniopTensorDescriptor_t b_bias_desc,
        infiniopTensorDescriptor_t b_scales_desc,
        infiniopTensorDescriptor_t global_scales_desc,
        infiniopTensorDescriptor_t b_zeros_desc,
        infiniopTensorDescriptor_t g_idx_desc,
        infiniopTensorDescriptor_t perm_desc,
        infiniopTensorDescriptor_t sorted_token_desc,
        infiniopTensorDescriptor_t expert_ids_desc,
        infiniopTensorDescriptor_t num_tokens_post_padded_desc,
        infiniopTensorDescriptor_t topk_weights_desc, int size_m, int size_n, int size_k, int top_k, int moe_block_size) {
        CHECK_OR_RETURN(
            c_desc != nullptr && a_desc != nullptr && b_q_weight_desc != nullptr && b_scales_desc != nullptr,
            INFINI_STATUS_NULL_POINTER);
        const infiniDtype_t dtype = a_desc->dtype();

        CHECK_OR_RETURN(a_desc->dim(0) == static_cast<size_t>(size_m)
                            && a_desc->dim(1) == static_cast<size_t>(size_k)
                            && c_desc->dim(1) == static_cast<size_t>(size_n),
                        INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(b_scales_desc->ndim() == 3
                            && b_scales_desc->dim(2) == static_cast<size_t>(size_n),
                        INFINI_STATUS_BAD_TENSOR_SHAPE);
        size_t num_groups = b_scales_desc->dim(1);
        bool has_act_order = false;
        bool has_bias = (b_bias_desc != nullptr);
        bool has_zp = (b_zeros_desc != nullptr);
        if (g_idx_desc != nullptr && perm_desc != nullptr) {
            CHECK_OR_RETURN(g_idx_desc->dim(g_idx_desc->ndim() - 1) == static_cast<size_t>(size_k)
                                && perm_desc->dim(perm_desc->ndim() - 1) == static_cast<size_t>(size_k),
                            INFINI_STATUS_BAD_TENSOR_SHAPE);
            has_act_order = true;
        }
        if (num_groups > 1) {
            CHECK_OR_RETURN(static_cast<size_t>(size_k) % num_groups == 0,
                            INFINI_STATUS_BAD_TENSOR_SHAPE);
        }
        if (b_bias_desc != nullptr) {
            CHECK_OR_RETURN(b_bias_desc->dim(1) == static_cast<size_t>(size_n)
                                && b_bias_desc->strides()[1] == 1,
                            INFINI_STATUS_BAD_TENSOR_SHAPE);
        }

        size_t sorted_token_ids_size_0 = sorted_token_desc->dim(0);
        size_t b_q_weight_size_1 = b_q_weight_desc->dim(1);
        size_t b_q_weight_size_2 = b_q_weight_desc->dim(2);
        size_t b_zeros_size_1 = 0;
        size_t b_zeros_size_2 = 0;
        if (b_zeros_desc != nullptr) {
            b_zeros_size_1 = b_zeros_desc->dim(1);
            b_zeros_size_2 = b_zeros_desc->dim(2);
        }
        size_t c_size_0 = c_desc->dim(0);
        return utils::Result<MoeWna16MarlinGemmInfo>(
            MoeWna16MarlinGemmInfo{dtype, size_m, size_n, size_k, top_k, moe_block_size, num_groups, sorted_token_ids_size_0, b_q_weight_size_1, b_q_weight_size_2, b_zeros_size_1, b_zeros_size_2, has_act_order, has_bias, has_zp});
    }
};

} // namespace op::moe_wna16_marlin_gemm

#endif // __MOE_WNA16_MARLIN_GEMM_INFO_H__
