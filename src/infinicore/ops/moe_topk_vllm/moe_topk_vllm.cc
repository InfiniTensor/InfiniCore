#include "infinicore/ops/moe_topk_vllm.hpp"
#include "../../utils.hpp"
#include <stdexcept>
#if defined(ENABLE_ATEN) && defined(ENABLE_ILUVATAR_API)
#include "infinicore/adaptor/aten_adaptor.hpp"
#include "infinicore/adaptor/vllm_iluvatar_adaptor.hpp"
#endif
namespace infinicore::op {
namespace {
void validate_vllm_topk(Tensor topk_weights, Tensor topk_ids, Tensor token_expert_indices, const Tensor &gating_output, const Tensor &correction_bias, const char *name) {
    if (!topk_weights || !topk_ids || !token_expert_indices || !gating_output) {
        throw std::runtime_error(std::string(name) + " expects non-empty tensors");
    }
    if (correction_bias) {
        INFINICORE_ASSERT_TENSORS_SAME_DEVICE(topk_weights, topk_ids, token_expert_indices, gating_output, correction_bias);
    } else {
        INFINICORE_ASSERT_TENSORS_SAME_DEVICE(topk_weights, topk_ids, token_expert_indices, gating_output);
    }
    if (gating_output->ndim() != 2 || topk_weights->ndim() != 2 || topk_ids->ndim() != 2 || token_expert_indices->ndim() != 2) {
        throw std::runtime_error(std::string(name) + " expects 2D tensors");
    }
    const auto tokens = gating_output->size(0), experts = gating_output->size(1), topk = topk_weights->size(1);
    if (topk_weights->size(0) != tokens || topk_ids->size(0) != tokens || token_expert_indices->size(0) != tokens || topk_ids->size(1) != topk || token_expert_indices->size(1) != topk) {
        throw std::runtime_error(std::string(name) + " expects output shapes (tokens, topk)");
    }
    if (topk < 1 || topk > 32 || topk > experts || experts < 1 || experts > 512) {
        throw std::runtime_error(std::string(name) + " supports experts in [1,512], topk in [1,32], topk<=experts");
    }
    if ((gating_output->dtype() == DataType::F16 || gating_output->dtype() == DataType::BF16) && (experts % 2) != 0) {
        throw std::runtime_error(std::string(name) + " requires even experts for fp16/bfloat16 gating");
    }
    if (gating_output->dtype() != DataType::F16 && gating_output->dtype() != DataType::BF16 && gating_output->dtype() != DataType::F32) {
        throw std::runtime_error(std::string(name) + " expects fp16/bfloat16/fp32 gating");
    }
    if (topk_weights->dtype() != DataType::F32 && topk_weights->dtype() != gating_output->dtype()) {
        throw std::runtime_error(std::string(name) + " expects topk_weights dtype float32 or same as gating");
    }
    if (topk_ids->dtype() != DataType::I32 && topk_ids->dtype() != DataType::I64) {
        throw std::runtime_error(std::string(name) + " expects topk_ids int32/int64");
    }
    if (token_expert_indices->dtype() != DataType::I32) {
        throw std::runtime_error(std::string(name) + " expects token_expert_indices int32");
    }
    if (correction_bias && (correction_bias->numel() != experts || correction_bias->dtype() != gating_output->dtype())) {
        throw std::runtime_error(std::string(name) + " expects correction_bias shape (experts,) and same dtype as gating");
    }
    if (!topk_weights->is_contiguous() || !topk_ids->is_contiguous() || !token_expert_indices->is_contiguous() || !gating_output->is_contiguous() || (correction_bias && !correction_bias->is_contiguous())) {
        throw std::runtime_error(std::string(name) + " expects contiguous tensors");
    }
}
} // namespace
void moe_topk_softmax_vllm_(Tensor topk_weights, Tensor topk_ids, Tensor token_expert_indices, const Tensor &gating_output, bool renormalize, const Tensor &correction_bias) {
    validate_vllm_topk(topk_weights, topk_ids, token_expert_indices, gating_output, correction_bias, "moe_topk_softmax_vllm");
#if defined(ENABLE_ATEN) && defined(ENABLE_ILUVATAR_API)
    if (gating_output->device().getType() == Device::Type::ILUVATAR) {
        if (!adaptor::vllm_iluvatar::topk_softmax_available()) {
            throw std::runtime_error("topk_softmax requires vllm_iluvatar perf extension");
        }
        auto w = adaptor::to_aten_tensor(topk_weights);
        auto ids = adaptor::to_aten_tensor(topk_ids);
        auto src = adaptor::to_aten_tensor(token_expert_indices);
        auto gate = adaptor::to_aten_tensor(gating_output);
        std::optional<at::Tensor> bias;
        if (correction_bias) {
            bias = adaptor::to_aten_tensor(correction_bias);
        }
        adaptor::vllm_iluvatar::topk_softmax(w, ids, src, gate, renormalize, bias);
        return;
    }
#endif
    throw std::runtime_error("moe_topk_softmax_vllm currently supports only Iluvatar builds with ATen and vllm_iluvatar");
}
void moe_topk_sigmoid_vllm_(Tensor topk_weights, Tensor topk_ids, Tensor token_expert_indices, const Tensor &gating_output, bool renormalize, const Tensor &correction_bias) {
    validate_vllm_topk(topk_weights, topk_ids, token_expert_indices, gating_output, correction_bias, "moe_topk_sigmoid_vllm");
#if defined(ENABLE_ATEN) && defined(ENABLE_ILUVATAR_API)
    if (gating_output->device().getType() == Device::Type::ILUVATAR) {
        if (!adaptor::vllm_iluvatar::topk_sigmoid_available()) {
            throw std::runtime_error("topk_sigmoid requires vllm_iluvatar perf extension");
        }
        auto w = adaptor::to_aten_tensor(topk_weights);
        auto ids = adaptor::to_aten_tensor(topk_ids);
        auto src = adaptor::to_aten_tensor(token_expert_indices);
        auto gate = adaptor::to_aten_tensor(gating_output);
        std::optional<at::Tensor> bias;
        if (correction_bias) {
            bias = adaptor::to_aten_tensor(correction_bias);
        }
        adaptor::vllm_iluvatar::topk_sigmoid(w, ids, src, gate, renormalize, bias);
        return;
    }
#endif
    throw std::runtime_error("moe_topk_sigmoid_vllm currently supports only Iluvatar builds with ATen and vllm_iluvatar");
}
} // namespace infinicore::op
