#include "infinicore/ops/moe_expand_input.hpp"
#include "../../utils.hpp"
#include <stdexcept>
#if defined(ENABLE_ATEN) && defined(ENABLE_ILUVATAR_API)
#include "infinicore/adaptor/aten_adaptor.hpp"
#include "infinicore/adaptor/vllm_iluvatar_adaptor.hpp"
#endif

namespace infinicore::op {

void moe_expand_input_with_inv_pos_(Tensor expand_states, std::optional<Tensor> expand_scales, const Tensor &hidden_states, const Tensor &inv_pos, int64_t top_k, int64_t group_size, int64_t format) {
    if (expand_scales) {
        INFINICORE_ASSERT_TENSORS_SAME_DEVICE(expand_states, *expand_scales, hidden_states, inv_pos);
    } else {
        INFINICORE_ASSERT_TENSORS_SAME_DEVICE(expand_states, hidden_states, inv_pos);
    }
    if (top_k <= 0 || top_k > 32) {
        throw std::runtime_error("moe_expand_input_with_inv_pos expects 1 <= top_k <= 32");
    }
    if (group_size != 64 && group_size != 128) {
        throw std::runtime_error("moe_expand_input_with_inv_pos expects group_size 64 or 128");
    }
    if (format < 0 || format > 2) {
        throw std::runtime_error("moe_expand_input_with_inv_pos format must be 0 normal, 1 quant, or 2 packed");
    }
    if (hidden_states->ndim() != 2 || expand_states->ndim() != 2 || inv_pos->ndim() != 1) {
        throw std::runtime_error("moe_expand_input_with_inv_pos expects 2D hidden/expand and 1D inv_pos");
    }
    if (hidden_states->dtype() != DataType::F16 && hidden_states->dtype() != DataType::BF16) {
        throw std::runtime_error("moe_expand_input_with_inv_pos expects fp16/bfloat16 hidden_states");
    }
    if (inv_pos->dtype() != DataType::I32) {
        throw std::runtime_error("moe_expand_input_with_inv_pos expects int32 inv_pos");
    }
    if (!hidden_states->is_contiguous() || !expand_states->is_contiguous() || !inv_pos->is_contiguous() || (expand_scales && !(*expand_scales)->is_contiguous())) {
        throw std::runtime_error("moe_expand_input_with_inv_pos expects contiguous tensors");
    }
    const size_t m = hidden_states->size(0);
    const size_t n = hidden_states->size(1);
    const size_t total = m * static_cast<size_t>(top_k);
    if (inv_pos->numel() != total || expand_states->size(0) != total) {
        throw std::runtime_error("moe_expand_input_with_inv_pos total token shape mismatch");
    }
    if (format == 0) {
        if (expand_scales) {
            throw std::runtime_error("moe_expand_input_with_inv_pos normal format does not take expand_scales");
        }
        if (expand_states->dtype() != hidden_states->dtype() || expand_states->size(1) != n) {
            throw std::runtime_error("moe_expand_input_with_inv_pos normal output shape/dtype mismatch");
        }
        if (n > 16384) {
            throw std::runtime_error("moe_expand_input_with_inv_pos normal format supports hidden <= 16384");
        }
    } else {
        const size_t n_out = ((n + static_cast<size_t>(group_size) - 1) / static_cast<size_t>(group_size)) * static_cast<size_t>(group_size);
        if (!expand_scales) {
            throw std::runtime_error("moe_expand_input_with_inv_pos quant/packed format requires expand_scales");
        }
        if (expand_states->dtype() != DataType::I8 || (*expand_scales)->dtype() != DataType::F32) {
            throw std::runtime_error("moe_expand_input_with_inv_pos quant/packed expects int8 output and fp32 scales");
        }
        if (expand_states->size(1) != n_out || (*expand_scales)->ndim() != 2 || (*expand_scales)->size(0) != total || (*expand_scales)->size(1) != 1) {
            throw std::runtime_error("moe_expand_input_with_inv_pos quant/packed output shape mismatch");
        }
        if (n_out > 32768) {
            throw std::runtime_error("moe_expand_input_with_inv_pos quant/packed supports padded hidden <= 32768");
        }
        if (format == 2 && (n_out % 64) != 0) {
            throw std::runtime_error("moe_expand_input_with_inv_pos packed requires padded hidden % 64 == 0");
        }
    }
#if defined(ENABLE_ATEN) && defined(ENABLE_ILUVATAR_API)
    if (hidden_states->device().getType() == Device::Type::ILUVATAR) {
        if (!adaptor::vllm_iluvatar::expand_moe_input_with_inv_pos_available()) {
            throw std::runtime_error("moe_expand_input_with_inv_pos requires vllm_iluvatar perf extension");
        }
        auto out = adaptor::to_aten_tensor(expand_states);
        auto hs = adaptor::to_aten_tensor(hidden_states);
        auto inv = adaptor::to_aten_tensor(inv_pos);
        std::optional<at::Tensor> scales;
        if (expand_scales) {
            scales = adaptor::to_aten_tensor(*expand_scales);
        }
        adaptor::vllm_iluvatar::expand_moe_input_with_inv_pos(out, scales, hs, inv, top_k, group_size, format);
        return;
    }
#endif
    throw std::runtime_error("moe_expand_input_with_inv_pos currently supports only Iluvatar builds with ATen and vllm_iluvatar");
}

} // namespace infinicore::op
