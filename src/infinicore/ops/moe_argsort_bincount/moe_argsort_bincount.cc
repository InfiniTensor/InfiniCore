#include "infinicore/ops/moe_argsort_bincount.hpp"
#include "../../utils.hpp"
#include <stdexcept>
#if defined(ENABLE_ATEN) && defined(ENABLE_ILUVATAR_API)
#include "infinicore/adaptor/aten_adaptor.hpp"
#include "infinicore/adaptor/vllm_iluvatar_adaptor.hpp"
#endif

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
#if defined(ENABLE_ATEN) && defined(ENABLE_ILUVATAR_API)
    if (topk_ids->device().getType() == Device::Type::ILUVATAR) {
        if (!adaptor::vllm_iluvatar::argsort_bincount_with_inv_pos_available()) {
            throw std::runtime_error("moe_argsort_bincount_with_inv_pos requires vllm_iluvatar perf extension");
        }
        auto ids = adaptor::to_aten_tensor(topk_ids);
        auto tpe = adaptor::to_aten_tensor(tokens_per_experts);
        auto sorted = adaptor::to_aten_tensor(sorted_indices);
        auto inv = adaptor::to_aten_tensor(inv_pos);
        adaptor::vllm_iluvatar::argsort_bincount_with_inv_pos(ids, tpe, sorted, inv, num_experts);
        return;
    }
#endif
    throw std::runtime_error("moe_argsort_bincount_with_inv_pos currently supports only Iluvatar builds with ATen and vllm_iluvatar");
}

} // namespace infinicore::op
