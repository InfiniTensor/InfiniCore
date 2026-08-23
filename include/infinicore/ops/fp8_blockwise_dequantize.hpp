#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(Fp8BlockwiseDequantize, Tensor, const Tensor &, const Tensor &);

Tensor fp8_blockwise_dequantize(const Tensor &q,
                                const Tensor &scales,
                                const DataType &output_dtype);
void fp8_blockwise_dequantize_(Tensor output,
                               const Tensor &q,
                               const Tensor &scales);

} // namespace infinicore::op
