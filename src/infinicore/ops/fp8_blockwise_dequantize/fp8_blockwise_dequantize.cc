#include "infinicore/ops/fp8_blockwise_dequantize.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Fp8BlockwiseDequantize);

Fp8BlockwiseDequantize::Fp8BlockwiseDequantize(Tensor output,
                                               const Tensor &q,
                                               const Tensor &scales) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, q, scales);
    INFINICORE_GRAPH_OP_DISPATCH(output->device().getType(), output, q, scales);
}

void Fp8BlockwiseDequantize::execute(Tensor output,
                                     const Tensor &q,
                                     const Tensor &scales) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Fp8BlockwiseDequantize, output, q, scales);
}

Tensor fp8_blockwise_dequantize(const Tensor &q,
                                const Tensor &scales,
                                const DataType &output_dtype) {
    auto output = Tensor::empty(q->shape(), output_dtype, q->device());
    Fp8BlockwiseDequantize::execute(output, q, scales);
    return output;
}

void fp8_blockwise_dequantize_(Tensor output,
                               const Tensor &q,
                               const Tensor &scales) {
    Fp8BlockwiseDequantize::execute(output, q, scales);
}

} // namespace infinicore::op
