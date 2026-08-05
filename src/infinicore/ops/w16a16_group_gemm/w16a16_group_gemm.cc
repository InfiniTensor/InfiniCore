#include "infinicore/ops/w16a16_group_gemm.hpp"
#include "../../utils.hpp"
#include "../vendor_ops/vendor_ops_dispatch.hpp"
#include <stdexcept>

namespace infinicore::op {
void w16a16_group_gemm_(Tensor out,
                        const Tensor &input,
                        const Tensor &weight,
                        const Tensor &tokens_per_experts,
                        std::optional<Tensor> sorted_token_ids,
                        std::optional<Tensor> bias,
                        bool trans_weight,
                        bool is_decode) {
    if (bias && sorted_token_ids) {
        INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, input, weight, *sorted_token_ids, *bias);
    } else if (bias) {
        INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, input, weight, *bias);
    } else if (sorted_token_ids) {
        INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, input, weight, *sorted_token_ids);
    } else {
        INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, input, weight);
    }
    if (!trans_weight) {
        throw std::runtime_error("w16a16_group_gemm currently supports only trans_weight=True (TN layout)");
    }
    if (out->ndim() != 2 || input->ndim() != 2 || weight->ndim() != 3 || tokens_per_experts->ndim() != 1) {
        throw std::runtime_error("w16a16_group_gemm expects out/input 2D, weight 3D, tokens_per_experts 1D");
    }
    if (out->dtype() != DataType::F16 && out->dtype() != DataType::BF16) {
        throw std::runtime_error("w16a16_group_gemm expects fp16/bfloat16 tensors");
    }
    if (input->dtype() != out->dtype() || weight->dtype() != out->dtype()) {
        throw std::runtime_error("w16a16_group_gemm expects input, weight, and out to have the same dtype");
    }
    if (tokens_per_experts->dtype() != DataType::I32) {
        throw std::runtime_error("w16a16_group_gemm expects int32 tokens_per_experts");
    }
    if (is_decode && tokens_per_experts->device() != out->device()) {
        throw std::runtime_error("w16a16_group_gemm decode expects GPU tokens_per_experts on the output device");
    }
    if (!is_decode && tokens_per_experts->device().getType() != Device::Type::CPU) {
        throw std::runtime_error("w16a16_group_gemm prefill expects CPU tokens_per_experts");
    }
    if (sorted_token_ids && (*sorted_token_ids)->dtype() != DataType::I32) {
        throw std::runtime_error("w16a16_group_gemm expects int32 sorted_token_ids");
    }
    if (bias && ((*bias)->ndim() != 2 || (*bias)->dtype() != out->dtype())) {
        throw std::runtime_error("w16a16_group_gemm expects bias shape (E,N) and same dtype as out");
    }
    if (!out->is_contiguous() || !input->is_contiguous() || !weight->is_contiguous() || !tokens_per_experts->is_contiguous() || (sorted_token_ids && !(*sorted_token_ids)->is_contiguous()) || (bias && !(*bias)->is_contiguous())) {
        throw std::runtime_error("w16a16_group_gemm expects contiguous tensors");
    }

    const size_t experts = weight->size(0);
    const size_t rows = input->size(0);
    const size_t output_features = weight->size(1);
    if (tokens_per_experts->numel() != experts) {
        throw std::runtime_error("w16a16_group_gemm expert count mismatch");
    }
    if (input->size(1) != weight->size(2) || out->size(0) != rows || out->size(1) != output_features) {
        throw std::runtime_error("w16a16_group_gemm TN shape mismatch");
    }
    if (sorted_token_ids && (*sorted_token_ids)->numel() != rows) {
        throw std::runtime_error("w16a16_group_gemm sorted_token_ids length mismatch");
    }
    if (bias && ((*bias)->size(0) != experts || (*bias)->size(1) != output_features)) {
        throw std::runtime_error("w16a16_group_gemm bias shape mismatch");
    }

    auto kernel = vendor_ops::lookup(
        vendor_ops::w16a16_group_gemm_dispatcher(),
        out->device().getType(),
        "w16a16_group_gemm");
    kernel(out, input, weight, tokens_per_experts, sorted_token_ids, bias, trans_weight, is_decode);
}
} // namespace infinicore::op
