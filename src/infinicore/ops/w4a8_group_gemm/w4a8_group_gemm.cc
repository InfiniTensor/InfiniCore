#include "infinicore/ops/w4a8_group_gemm.hpp"
#include "../../utils.hpp"
#include <stdexcept>
#if defined(ENABLE_ATEN) && defined(ENABLE_ILUVATAR_API)
#include "infinicore/adaptor/aten_adaptor.hpp"
#include "infinicore/adaptor/vllm_iluvatar_adaptor.hpp"
#endif
namespace infinicore::op {
void w4a8_group_gemm_(Tensor out, const Tensor &input, const Tensor &weight, const Tensor &input_scale, const Tensor &weight_scale, const Tensor &tokens_per_experts, std::optional<Tensor> sorted_token_ids, std::optional<Tensor> bias, bool trans_weight, bool is_decode) {
    if (bias) {
        INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, input, weight, input_scale, weight_scale, *bias);
    } else {
        INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, input, weight, input_scale, weight_scale);
    }
    if (sorted_token_ids && (*sorted_token_ids)->device().getType() != Device::Type::CPU) {
        INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, *sorted_token_ids);
    }
    if (out->ndim() != 2 || input->ndim() != 2 || weight->ndim() != 3 || input_scale->ndim() != 2 || weight_scale->ndim() != 3 || tokens_per_experts->ndim() != 1) {
        throw std::runtime_error("w4a8_group_gemm expects out/input 2D, weight/weight_scale 3D, tokens_per_experts 1D");
    }
    if (out->dtype() != DataType::F16 && out->dtype() != DataType::BF16) {
        throw std::runtime_error("w4a8_group_gemm expects fp16/bfloat16 output");
    }
    if (input->dtype() != DataType::I8 || weight->dtype() != DataType::I8) {
        throw std::runtime_error("w4a8_group_gemm expects int8 input and packed int8 weight");
    }
    if (input_scale->dtype() != DataType::F32 || weight_scale->dtype() != DataType::F32) {
        throw std::runtime_error("w4a8_group_gemm expects float32 scales");
    }
    if (tokens_per_experts->dtype() != DataType::I32) {
        throw std::runtime_error("w4a8_group_gemm expects int32 tokens_per_experts");
    }
    if (sorted_token_ids && (*sorted_token_ids)->dtype() != DataType::I32) {
        throw std::runtime_error("w4a8_group_gemm expects int32 sorted_token_ids");
    }
    if (bias && ((*bias)->ndim() != 2 || (*bias)->dtype() != out->dtype())) {
        throw std::runtime_error("w4a8_group_gemm expects bias shape (E,N) and same dtype as out");
    }
    if (!out->is_contiguous() || !input->is_contiguous() || !weight->is_contiguous() || !input_scale->is_contiguous() || !weight_scale->is_contiguous() || !tokens_per_experts->is_contiguous() || (sorted_token_ids && !(*sorted_token_ids)->is_contiguous()) || (bias && !(*bias)->is_contiguous())) {
        throw std::runtime_error("w4a8_group_gemm expects contiguous tensors");
    }
    const size_t e = weight->size(0);
    if (tokens_per_experts->numel() != e || weight_scale->size(0) != e) {
        throw std::runtime_error("w4a8_group_gemm expert count mismatch");
    }
    if (trans_weight) {
        if (weight_scale->size(1) != weight->size(1) || weight_scale->size(2) != 1 || input->size(1) != weight->size(2) * 2 || out->size(1) != weight->size(1)) {
            throw std::runtime_error("w4a8_group_gemm TN shape mismatch");
        }
    } else {
        if (weight_scale->size(0) != e || input->size(1) != weight->size(1) || out->size(1) != weight->size(2) * 2) {
            throw std::runtime_error("w4a8_group_gemm NN shape mismatch");
        }
    }
    if (out->size(0) != input->size(0) || input_scale->size(0) != input->size(0) || input_scale->size(1) != 1) {
        throw std::runtime_error("w4a8_group_gemm input/output scale shape mismatch");
    }
    if (sorted_token_ids && (*sorted_token_ids)->numel() != out->size(0)) {
        throw std::runtime_error("w4a8_group_gemm sorted_token_ids length mismatch");
    }
    if (bias && ((*bias)->size(0) != e || (*bias)->size(1) != out->size(1))) {
        throw std::runtime_error("w4a8_group_gemm bias shape mismatch");
    }
#if defined(ENABLE_ATEN) && defined(ENABLE_ILUVATAR_API)
    if (out->device().getType() == Device::Type::ILUVATAR) {
        if (!adaptor::vllm_iluvatar::w4a8_group_gemm_available()) {
            throw std::runtime_error("w4a8_group_gemm requires vllm_iluvatar cuinfer extension");
        }
        auto o = adaptor::to_aten_tensor(out);
        auto in = adaptor::to_aten_tensor(input);
        auto w = adaptor::to_aten_tensor(weight);
        auto is = adaptor::to_aten_tensor(input_scale);
        auto ws = adaptor::to_aten_tensor(weight_scale);
        auto tpe = adaptor::to_aten_tensor(tokens_per_experts);
        std::optional<at::Tensor> sti;
        if (sorted_token_ids) {
            sti = adaptor::to_aten_tensor(*sorted_token_ids);
        }
        std::optional<at::Tensor> bi;
        if (bias) {
            bi = adaptor::to_aten_tensor(*bias);
        }
        adaptor::vllm_iluvatar::w4a8_group_gemm(o, in, w, is, ws, tpe, sti, bi, trans_weight, is_decode);
        return;
    }
#endif
    throw std::runtime_error("w4a8_group_gemm currently supports only Iluvatar builds with ATen and vllm_iluvatar");
}
} // namespace infinicore::op
