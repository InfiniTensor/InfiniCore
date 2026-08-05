#include "infinicore/ops/moe_argsort_bincount.hpp"
#include "../../utils.hpp"
#include "../vendor_ops/vendor_ops_dispatch.hpp"
#include <stdexcept>

namespace infinicore::op {

void moe_argsort_bincount_with_inv_pos_(Tensor tokens_per_experts, Tensor sorted_indices, Tensor inv_pos, const Tensor &topk_ids, int64_t num_experts) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(tokens_per_experts, sorted_indices, inv_pos, topk_ids);
    if (num_experts <= 0 || num_experts > 512) {
        throw std::runtime_error("moe_argsort_bincount_with_inv_pos expects 0 < num_experts <= 512");
    }
    if (topk_ids->dtype() != DataType::I32 || tokens_per_experts->dtype() != DataType::I32 || sorted_indices->dtype() != DataType::I32 || inv_pos->dtype() != DataType::I32) {
        throw std::runtime_error("moe_argsort_bincount_with_inv_pos expects int32 tensors");
    }
    if (tokens_per_experts->ndim() != 1 || tokens_per_experts->numel() != static_cast<size_t>(num_experts)) {
        throw std::runtime_error("moe_argsort_bincount_with_inv_pos tokens_per_experts shape mismatch");
    }
    if (sorted_indices->ndim() != 1 || inv_pos->ndim() != 1 || sorted_indices->numel() != topk_ids->numel() || inv_pos->numel() != topk_ids->numel()) {
        throw std::runtime_error("moe_argsort_bincount_with_inv_pos sorted_indices/inv_pos shape mismatch");
    }
    if (!topk_ids->is_contiguous() || !tokens_per_experts->is_contiguous() || !sorted_indices->is_contiguous() || !inv_pos->is_contiguous()) {
        throw std::runtime_error("moe_argsort_bincount_with_inv_pos expects contiguous tensors");
    }
    auto kernel = vendor_ops::lookup(
        vendor_ops::moe_argsort_dispatcher(),
        topk_ids->device().getType(),
        "moe_argsort_bincount_with_inv_pos");
    kernel(tokens_per_experts, sorted_indices, inv_pos, topk_ids, num_experts);
}

} // namespace infinicore::op
