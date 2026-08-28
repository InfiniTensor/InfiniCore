#include "../../../utils.hpp"
#include "infinicore/ops/per_tensor_dequant_fp8.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(PerTensorDequantFp8);

PerTensorDequantFp8::PerTensorDequantFp8(const Tensor &x, const Tensor &x_packed, const Tensor &x_scale) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(x, x_packed, x_scale);
    INFINICORE_GRAPH_OP_DISPATCH(x->device().getType(), x, x_packed, x_scale);
}

void PerTensorDequantFp8::execute(const Tensor &x, const Tensor &x_packed, const Tensor &x_scale) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(PerTensorDequantFp8, x, x_packed, x_scale);
}

void per_tensor_dequant_fp8_(Tensor x, const Tensor &x_packed, const Tensor &x_scale) {
    PerTensorDequantFp8::execute(x, x_packed, x_scale);
}
} // namespace infinicore::op
