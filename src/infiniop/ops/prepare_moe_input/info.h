#ifndef __PREPARE_MOE_INPUT_INFO_H__
#define __PREPARE_MOE_INPUT_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"

namespace op::prepare_moe_input {

class PrepareMoeInputInfo {
    PrepareMoeInputInfo() = default;

public:
    size_t topk_length;
    size_t topk;
    size_t num_experts;
    size_t n;
    size_t k;
    bool has_blockscale_offsets;

    static utils::Result<PrepareMoeInputInfo> create(
        infiniopTensorDescriptor_t expert_offsets_desc,
        infiniopTensorDescriptor_t blockscale_offsets_desc,
        infiniopTensorDescriptor_t problem_sizes1_desc,
        infiniopTensorDescriptor_t problem_sizes2_desc,
        infiniopTensorDescriptor_t input_permutation_desc,
        infiniopTensorDescriptor_t output_permutation_desc,
        infiniopTensorDescriptor_t topk_ids_desc,
        size_t num_experts,
        size_t n,
        size_t k) {
        if (num_experts == 0 || n == 0 || k == 0) {
            return INFINI_STATUS_BAD_PARAM;
        }
        if (topk_ids_desc->dtype() != INFINI_DTYPE_I32 || expert_offsets_desc->dtype() != INFINI_DTYPE_I32 || problem_sizes1_desc->dtype() != INFINI_DTYPE_I32 || problem_sizes2_desc->dtype() != INFINI_DTYPE_I32 || input_permutation_desc->dtype() != INFINI_DTYPE_I32 || output_permutation_desc->dtype() != INFINI_DTYPE_I32 || (blockscale_offsets_desc != nullptr && blockscale_offsets_desc->dtype() != INFINI_DTYPE_I32)) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        if (topk_ids_desc->ndim() != 2 || expert_offsets_desc->ndim() != 1 || input_permutation_desc->ndim() != 1 || output_permutation_desc->ndim() != 1 || !is_problem_sizes_shape(problem_sizes1_desc, num_experts) || !is_problem_sizes_shape(problem_sizes2_desc, num_experts) || expert_offsets_desc->shape()[0] != num_experts + 1) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        if (blockscale_offsets_desc != nullptr && (blockscale_offsets_desc->ndim() != 1 || blockscale_offsets_desc->shape()[0] != num_experts + 1)) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        const size_t topk_length = topk_ids_desc->shape()[0] * topk_ids_desc->shape()[1];
        if (input_permutation_desc->shape()[0] < topk_length || output_permutation_desc->shape()[0] < topk_length) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        if (topk_ids_desc->strides()[1] != 1 || expert_offsets_desc->strides()[0] != 1 || input_permutation_desc->strides()[0] != 1 || output_permutation_desc->strides()[0] != 1 || !is_problem_sizes_contiguous(problem_sizes1_desc) || !is_problem_sizes_contiguous(problem_sizes2_desc) || (blockscale_offsets_desc != nullptr && blockscale_offsets_desc->strides()[0] != 1)) {
            return INFINI_STATUS_BAD_TENSOR_STRIDES;
        }

        return utils::Result<PrepareMoeInputInfo>(PrepareMoeInputInfo{
            topk_length,
            topk_ids_desc->shape()[1],
            num_experts,
            n,
            k,
            blockscale_offsets_desc != nullptr,
        });
    }

private:
    static bool is_problem_sizes_shape(infiniopTensorDescriptor_t desc, size_t num_experts) {
        return (desc->ndim() == 1 && desc->shape()[0] == num_experts * 3) || (desc->ndim() == 2 && desc->shape()[0] == num_experts && desc->shape()[1] == 3);
    }

    static bool is_problem_sizes_contiguous(infiniopTensorDescriptor_t desc) {
        return (desc->ndim() == 1 && desc->strides()[0] == 1) || (desc->ndim() == 2 && desc->strides()[1] == 1 && desc->strides()[0] == 3);
    }
};

} // namespace op::prepare_moe_input

#endif
