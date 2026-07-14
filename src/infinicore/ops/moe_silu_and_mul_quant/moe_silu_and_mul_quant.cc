#include "infinicore/ops/moe_silu_and_mul_quant.hpp"
#include "../../utils.hpp"
#include <stdexcept>
#if defined(ENABLE_ATEN) && defined(ENABLE_ILUVATAR_API)
#include "infinicore/adaptor/aten_adaptor.hpp"
#include "infinicore/adaptor/vllm_iluvatar_adaptor.hpp"
#endif

namespace infinicore::op {

void moe_silu_and_mul_quant_(Tensor output, std::optional<Tensor> output_scale, const Tensor &input, int64_t format) {
    if (output_scale) {
        INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, *output_scale, input);
    } else {
        INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, input);
    }
    if (format < 0 || format > 2) {
        throw std::runtime_error("moe_silu_and_mul_quant format must be 0 normal, 1 quant, or 2 packed");
    }
    if (input->ndim() != 2 || output->ndim() != 2) {
        throw std::runtime_error("moe_silu_and_mul_quant expects 2D tensors");
    }
    if (input->dtype() != DataType::F16 && input->dtype() != DataType::BF16) {
        throw std::runtime_error("moe_silu_and_mul_quant expects fp16/bfloat16 input");
    }
    if ((input->size(1) % 2) != 0) {
        throw std::runtime_error("moe_silu_and_mul_quant input last dim must be even");
    }
    if (!input->is_contiguous() || !output->is_contiguous() || (output_scale && !(*output_scale)->is_contiguous())) {
        throw std::runtime_error("moe_silu_and_mul_quant expects contiguous tensors");
    }
    const size_t m = input->size(0);
    const size_t n = input->size(1) / 2;
    if (output->size(0) != m || output->size(1) != n) {
        throw std::runtime_error("moe_silu_and_mul_quant output shape mismatch");
    }
    if (format == 0) {
        if (output_scale) {
            throw std::runtime_error("moe_silu_and_mul_quant normal format does not take output_scale");
        }
        if (output->dtype() != input->dtype()) {
            throw std::runtime_error("moe_silu_and_mul_quant normal output dtype mismatch");
        }
        if ((n % 2) != 0 || n > 16384) {
            throw std::runtime_error("moe_silu_and_mul_quant normal format requires N % 2 == 0 and N <= 16384");
        }
    } else {
        if (!output_scale) {
            throw std::runtime_error("moe_silu_and_mul_quant quant/packed requires output_scale");
        }
        if (output->dtype() != DataType::I8 || (*output_scale)->dtype() != DataType::F32) {
            throw std::runtime_error("moe_silu_and_mul_quant quant/packed expects int8 output and fp32 scale");
        }
        if ((*output_scale)->ndim() != 2 || (*output_scale)->size(0) != m || (*output_scale)->size(1) != 1) {
            throw std::runtime_error("moe_silu_and_mul_quant scale shape mismatch");
        }
        if ((n % 4) != 0 || n > 32768) {
            throw std::runtime_error("moe_silu_and_mul_quant quant/packed requires N % 4 == 0 and N <= 32768");
        }
        if (format == 2 && (n % 64) != 0) {
            throw std::runtime_error("moe_silu_and_mul_quant packed requires N % 64 == 0");
        }
    }
#if defined(ENABLE_ATEN) && defined(ENABLE_ILUVATAR_API)
    if (input->device().getType() == Device::Type::ILUVATAR) {
        if (!adaptor::vllm_iluvatar::silu_and_mul_quant_available()) {
            throw std::runtime_error("moe_silu_and_mul_quant requires vllm_iluvatar perf extension");
        }
        auto out = adaptor::to_aten_tensor(output);
        auto in = adaptor::to_aten_tensor(input);
        std::optional<at::Tensor> scale;
        if (output_scale) {
            scale = adaptor::to_aten_tensor(*output_scale);
        }
        adaptor::vllm_iluvatar::silu_and_mul_quant(out, scale, in, format);
        return;
    }
#endif
    throw std::runtime_error("moe_silu_and_mul_quant currently supports only Iluvatar builds with ATen and vllm_iluvatar");
}

} // namespace infinicore::op
