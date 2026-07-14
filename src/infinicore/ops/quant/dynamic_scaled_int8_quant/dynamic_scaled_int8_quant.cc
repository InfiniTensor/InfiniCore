#include "infinicore/ops/dynamic_scaled_int8_quant.hpp"
#include "../../../utils.hpp"

#include <stdexcept>
#include <string>

#if defined(ENABLE_ATEN) && defined(ENABLE_ILUVATAR_API)
#include "infinicore/adaptor/aten_adaptor.hpp"
#include "infinicore/adaptor/vllm_iluvatar_adaptor.hpp"
#endif

namespace infinicore::op {

namespace {

void validate_dynamic_scaled_int8_quant(Tensor output, const Tensor &input, Tensor input_scales) {
    if (!output || !input || !input_scales) {
        throw std::runtime_error("dynamic_scaled_int8_quant expects non-empty output, input, and input_scales tensors");
    }
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, input, input_scales);
    if (output->dtype() != DataType::I8) {
        throw std::runtime_error("dynamic_scaled_int8_quant expects output dtype int8");
    }
    if (input_scales->dtype() != DataType::F32) {
        throw std::runtime_error("dynamic_scaled_int8_quant expects input_scales dtype float32");
    }
    if (input->dtype() != DataType::F16 && input->dtype() != DataType::BF16) {
        throw std::runtime_error("dynamic_scaled_int8_quant expects input dtype float16 or bfloat16");
    }
    if (input->ndim() == 0 || input->size(input->ndim() - 1) == 0) {
        throw std::runtime_error("dynamic_scaled_int8_quant expects input with a non-empty hidden dimension");
    }
    if (output->numel() != input->numel()) {
        throw std::runtime_error("dynamic_scaled_int8_quant expects output numel to equal input numel");
    }
    const auto hidden_size = input->size(input->ndim() - 1);
    const auto num_tokens = input->numel() / hidden_size;
    if (input_scales->numel() != num_tokens) {
        throw std::runtime_error("dynamic_scaled_int8_quant expects input_scales numel to equal input.numel / input.shape[-1]");
    }
    if (!input->is_contiguous() || !output->is_contiguous() || !input_scales->is_contiguous()) {
        throw std::runtime_error("dynamic_scaled_int8_quant expects contiguous input, output, and input_scales tensors");
    }
}

} // namespace

void dynamic_scaled_int8_quant_(Tensor output, const Tensor &input, Tensor input_scales) {
    validate_dynamic_scaled_int8_quant(output, input, input_scales);

#if defined(ENABLE_ATEN) && defined(ENABLE_ILUVATAR_API)
    if (input->device().getType() == Device::Type::ILUVATAR) {
        if (!adaptor::vllm_iluvatar::dynamic_scaled_int8_quant_available()) {
            throw std::runtime_error("dynamic_scaled_int8_quant requires vllm_iluvatar perf extension on Iluvatar");
        }
        auto output_at = adaptor::to_aten_tensor(output);
        auto scales_at = adaptor::to_aten_tensor(input_scales);
        auto input_at = adaptor::to_aten_tensor(input);
        adaptor::vllm_iluvatar::dynamic_scaled_int8_quant(output_at, scales_at, input_at);
        return;
    }
#endif

    throw std::runtime_error("dynamic_scaled_int8_quant currently supports only Iluvatar builds with ATen and vllm_iluvatar");
}

Tensor dynamic_scaled_int8_quant(const Tensor &input, Tensor input_scales) {
    auto output = Tensor::empty(input->shape(), DataType::I8, input->device());
    dynamic_scaled_int8_quant_(output, input, input_scales);
    return output;
}

} // namespace infinicore::op
