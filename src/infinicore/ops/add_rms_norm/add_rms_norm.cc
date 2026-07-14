#include "infinicore/ops/add_rms_norm.hpp"

#if defined(ENABLE_ATEN) && defined(ENABLE_ILUVATAR_API)
#include "infinicore/adaptor/aten_adaptor.hpp"
#include "infinicore/adaptor/vllm_iluvatar_adaptor.hpp"
#endif

#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(AddRMSNorm);

AddRMSNorm::AddRMSNorm(Tensor y, Tensor residual_out, const Tensor &a, const Tensor &b, const Tensor &weight, float epsilon) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(y, residual_out, a, b, weight);
    INFINICORE_GRAPH_OP_DISPATCH(y->device().getType(), y, residual_out, a, b, weight, epsilon);
}

void AddRMSNorm::execute(Tensor y, Tensor residual_out, const Tensor &a, const Tensor &b, const Tensor &weight, float epsilon) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(AddRMSNorm, y, residual_out, a, b, weight, epsilon);
}

std::pair<Tensor, Tensor> add_rms_norm(const Tensor &a, const Tensor &b, const Tensor &weight, float epsilon) {
    auto y = Tensor::empty(a->shape(), a->dtype(), a->device());
    auto residual_out = Tensor::empty(a->shape(), a->dtype(), a->device());
    add_rms_norm_(y, residual_out, a, b, weight, epsilon);
    return std::make_pair(y, residual_out);
}

void add_rms_norm_(Tensor out, Tensor residual, const Tensor &a, const Tensor &b, const Tensor &weight, float epsilon) {
    AddRMSNorm::execute(out, residual, a, b, weight, epsilon);
}

void add_rms_norm_inplace(Tensor input, Tensor residual, const Tensor &weight, float epsilon) {
#if defined(ENABLE_ATEN) && defined(ENABLE_ILUVATAR_API)
    if (input->device().getType() == Device::Type::ILUVATAR && adaptor::vllm_iluvatar::available()) {
        INFINICORE_ASSERT_TENSORS_SAME_DEVICE(input, residual, weight);
        auto input_at = adaptor::to_aten_tensor(input);
        auto residual_at = adaptor::to_aten_tensor(residual);
        auto weight_at = adaptor::to_aten_tensor(weight);
        adaptor::vllm_iluvatar::fused_add_rms_norm(input_at, residual_at, weight_at, epsilon);
        return;
    }
#endif
    add_rms_norm_(input, residual, input, residual, weight, epsilon);
}

} // namespace infinicore::op
