#include "infinicore/ops/mxfp4_dequantize.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(MXFP4Dequantize);

MXFP4Dequantize::MXFP4Dequantize(Tensor output,
                                 const Tensor &packed,
                                 const Tensor &scales) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, packed, scales);
    INFINICORE_GRAPH_OP_DISPATCH(output->device().getType(), output, packed, scales);
}

void MXFP4Dequantize::execute(Tensor output,
                              const Tensor &packed,
                              const Tensor &scales) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(MXFP4Dequantize, output, packed, scales);
}

Tensor mxfp4_dequantize(const Tensor &packed,
                        const Tensor &scales,
                        const DataType &output_dtype) {
    auto output_shape = packed->shape();
    output_shape.back() *= 2;
    auto output = Tensor::empty(output_shape, output_dtype, packed->device());
    MXFP4Dequantize::execute(output, packed, scales);
    return output;
}

void mxfp4_dequantize_(Tensor output,
                       const Tensor &packed,
                       const Tensor &scales) {
    MXFP4Dequantize::execute(output, packed, scales);
}

} // namespace infinicore::op
