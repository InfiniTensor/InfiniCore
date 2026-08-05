#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(MXFP4Dequantize, Tensor, const Tensor &, const Tensor &);

Tensor mxfp4_dequantize(const Tensor &packed,
                        const Tensor &scales,
                        const DataType &output_dtype);
void mxfp4_dequantize_(Tensor output,
                       const Tensor &packed,
                       const Tensor &scales);

} // namespace infinicore::op
