#include "infinicore/ops/moe_sum_vllm.hpp"
#include "../../utils.hpp"
#include <stdexcept>
#if defined(ENABLE_ATEN) && defined(ENABLE_ILUVATAR_API)
#include "infinicore/adaptor/aten_adaptor.hpp"
#include "infinicore/adaptor/vllm_iluvatar_adaptor.hpp"
#endif

namespace infinicore::op {

void moe_sum_vllm_(Tensor output, const Tensor &input, std::optional<Tensor> topk_weights, std::optional<Tensor> extra_residual, double routed_scale, double residual_scale) {
    if (topk_weights && extra_residual) {
        INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, input, *topk_weights, *extra_residual);
    } else if (topk_weights) {
        INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, input, *topk_weights);
    } else if (extra_residual) {
        INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, input, *extra_residual);
    } else {
        INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, input);
    }
    if (input->ndim() != 3 || output->ndim() != 2) {
        throw std::runtime_error("moe_sum_vllm expects input 3D and output 2D");
    }
    if (input->dtype() != DataType::F16 && input->dtype() != DataType::BF16) {
        throw std::runtime_error("moe_sum_vllm expects fp16/bfloat16 input");
    }
    if (output->dtype() != input->dtype()) {
        throw std::runtime_error("moe_sum_vllm output dtype must match input");
    }
    if (!input->is_contiguous() || !output->is_contiguous() || (topk_weights && !(*topk_weights)->is_contiguous()) || (extra_residual && !(*extra_residual)->is_contiguous())) {
        throw std::runtime_error("moe_sum_vllm expects contiguous tensors");
    }
    const size_t n = input->size(0), t = input->size(1), h = input->size(2);
    if (output->size(0) != n || output->size(1) != h) {
        throw std::runtime_error("moe_sum_vllm output shape mismatch");
    }
    if (h == 0 || (h % 2) != 0 || h > 16384) {
        throw std::runtime_error("moe_sum_vllm requires 0 < H <= 16384 and H % 2 == 0");
    }
    if (topk_weights && ((*topk_weights)->dtype() != DataType::F32 || (*topk_weights)->ndim() != 2 || (*topk_weights)->size(0) != n || (*topk_weights)->size(1) != t)) {
        throw std::runtime_error("moe_sum_vllm topk_weights must be float32 [N,T]");
    }
    if (extra_residual && ((*extra_residual)->dtype() != output->dtype() || (*extra_residual)->ndim() != 2 || (*extra_residual)->size(0) != n || (*extra_residual)->size(1) != h)) {
        throw std::runtime_error("moe_sum_vllm extra_residual shape/dtype mismatch");
    }
#if defined(ENABLE_ATEN) && defined(ENABLE_ILUVATAR_API)
    if (output->device().getType() == Device::Type::ILUVATAR) {
        if (!adaptor::vllm_iluvatar::moe_sum_vllm_available()) {
            throw std::runtime_error("moe_sum_vllm requires vllm_iluvatar perf extension");
        }
        auto out = adaptor::to_aten_tensor(output);
        auto in = adaptor::to_aten_tensor(input);
        std::optional<at::Tensor> tw;
        if (topk_weights) {
            tw = adaptor::to_aten_tensor(*topk_weights);
        }
        std::optional<at::Tensor> er;
        if (extra_residual) {
            er = adaptor::to_aten_tensor(*extra_residual);
        }
        adaptor::vllm_iluvatar::moe_sum_vllm(out, in, tw, er, routed_scale, residual_scale);
        return;
    }
#endif
    throw std::runtime_error("moe_sum_vllm currently supports only Iluvatar builds with ATen and vllm_iluvatar");
}

} // namespace infinicore::op
