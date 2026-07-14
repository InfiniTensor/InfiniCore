#include "infinicore/ops/grouped_topk_vllm.hpp"
#include "../../utils.hpp"
#include <stdexcept>
#if defined(ENABLE_ATEN) && defined(ENABLE_ILUVATAR_API)
#include "infinicore/adaptor/aten_adaptor.hpp"
#include "infinicore/adaptor/vllm_iluvatar_adaptor.hpp"
#endif
namespace infinicore::op {
void grouped_topk_vllm_(Tensor topk_weights, Tensor topk_ids, const Tensor &scores, int64_t num_expert_group, int64_t topk_group, bool renormalize, float routed_scaling_factor, const Tensor &bias, const std::string &scoring_func) {
    if (!topk_weights || !topk_ids || !scores) {
        throw std::runtime_error("grouped_topk_vllm expects non-empty topk_weights, topk_ids, scores");
    }
    if (bias) {
        INFINICORE_ASSERT_TENSORS_SAME_DEVICE(topk_weights, topk_ids, scores, bias);
    } else {
        INFINICORE_ASSERT_TENSORS_SAME_DEVICE(topk_weights, topk_ids, scores);
    }
    if (scores->ndim() != 2 || topk_weights->ndim() != 2 || topk_ids->ndim() != 2) {
        throw std::runtime_error("grouped_topk_vllm expects 2D tensors");
    }
    const auto tokens = scores->size(0), experts = scores->size(1), topk = topk_weights->size(1);
    if (topk_weights->size(0) != tokens || topk_ids->size(0) != tokens || topk_ids->size(1) != topk) {
        throw std::runtime_error("grouped_topk_vllm expects outputs (tokens, topk)");
    }
    if (num_expert_group != 1 && num_expert_group != 8) {
        throw std::runtime_error("grouped_topk_vllm currently supports num_expert_group 1 or 8");
    }
    if (topk_group < 1 || topk_group > num_expert_group) {
        throw std::runtime_error("grouped_topk_vllm expects 1 <= topk_group <= num_expert_group");
    }
    if (!(experts == 64 || experts == 128 || experts == 160 || experts == 192 || experts == 256 || experts == 384)) {
        throw std::runtime_error("grouped_topk_vllm supports expert counts 64/128/160/192/256/384");
    }
    if (experts % static_cast<size_t>(num_expert_group) != 0) {
        throw std::runtime_error("grouped_topk_vllm expects experts divisible by num_expert_group");
    }
    if (topk < 1 || topk > 32 || topk > experts) {
        throw std::runtime_error("grouped_topk_vllm expects topk in [1,32] and <= experts");
    }
    if (scores->dtype() != DataType::F16 && scores->dtype() != DataType::BF16) {
        throw std::runtime_error("grouped_topk_vllm perf supports only fp16/bfloat16 scores");
    }
    if (!bias) {
        throw std::runtime_error("grouped_topk_vllm currently requires correction bias; vllm_iluvatar perf no-bias path mismatches reference");
    }
    if (topk_weights->dtype() != DataType::F32) {
        throw std::runtime_error("grouped_topk_vllm expects topk_weights float32");
    }
    if (topk_ids->dtype() != DataType::I32 && topk_ids->dtype() != DataType::I64) {
        throw std::runtime_error("grouped_topk_vllm expects topk_ids int32/int64");
    }
    if (bias && (bias->numel() != experts || bias->dtype() != scores->dtype())) {
        throw std::runtime_error("grouped_topk_vllm expects bias shape (experts,) and same dtype as scores");
    }
    if (scoring_func != "softmax" && scoring_func != "sigmoid") {
        throw std::runtime_error("grouped_topk_vllm scoring_func must be softmax or sigmoid");
    }
    if (!topk_weights->is_contiguous() || !topk_ids->is_contiguous() || !scores->is_contiguous() || (bias && !bias->is_contiguous())) {
        throw std::runtime_error("grouped_topk_vllm expects contiguous tensors");
    }
#if defined(ENABLE_ATEN) && defined(ENABLE_ILUVATAR_API)
    if (scores->device().getType() == Device::Type::ILUVATAR) {
        if (!adaptor::vllm_iluvatar::grouped_topk_available()) {
            throw std::runtime_error("grouped_topk requires vllm_iluvatar perf extension");
        }
        auto w = adaptor::to_aten_tensor(topk_weights);
        auto ids = adaptor::to_aten_tensor(topk_ids);
        auto s = adaptor::to_aten_tensor(scores);
        std::optional<at::Tensor> b;
        if (bias) {
            b = adaptor::to_aten_tensor(bias);
        }
        adaptor::vllm_iluvatar::grouped_topk(w, ids, s, b, num_expert_group, topk_group, scoring_func, renormalize);
        if (routed_scaling_factor != 1.0f) {
            w.mul_(routed_scaling_factor);
        }
        return;
    }
#endif
    throw std::runtime_error("grouped_topk_vllm currently supports only Iluvatar builds with ATen and vllm_iluvatar");
}
} // namespace infinicore::op
