#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(Fp8BlockwiseGemm, Tensor, const Tensor &, const Tensor &, const Tensor &);

Tensor fp8_blockwise_gemm(const Tensor &a,
                          const Tensor &q,
                          const Tensor &scales);
void fp8_blockwise_gemm_(Tensor output,
                         const Tensor &a,
                         const Tensor &q,
                         const Tensor &scales);

} // namespace infinicore::op
