#include "../../../utils.hpp"
#include "infinicore/ops/per_tensor_quant_fp8.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(PerTensorQuantFp8);

PerTensorQuantFp8::PerTensorQuantFp8(const Tensor &x, Tensor x_packed, Tensor x_scale, bool is_static) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(x, x_packed, x_scale);
    INFINICORE_GRAPH_OP_DISPATCH(x->device().getType(), x, x_packed, x_scale, is_static);
}

void PerTensorQuantFp8::execute(const Tensor &x, Tensor x_packed, Tensor x_scale, bool is_static) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(PerTensorQuantFp8, x, x_packed, x_scale, is_static);
}

void per_tensor_quant_fp8_(const Tensor &x, Tensor x_packed, Tensor x_scale, bool is_static) {
    PerTensorQuantFp8::execute(x, x_packed, x_scale, is_static);
}

Tensor per_tensor_quant_fp8(const Tensor &x, Tensor x_scale, bool is_static) {
    auto x_packed = Tensor::strided_empty(x->shape(), x->strides(), infinicore::DataType::F8, x->device());
    PerTensorQuantFp8::execute(x, x_packed, x_scale, is_static);
    return x_packed;
}
} // namespace infinicore::op
