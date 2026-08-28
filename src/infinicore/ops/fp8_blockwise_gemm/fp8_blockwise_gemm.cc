#include "infinicore/ops/fp8_blockwise_gemm.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Fp8BlockwiseGemm);

Fp8BlockwiseGemm::Fp8BlockwiseGemm(Tensor output,
                                   const Tensor &a,
                                   const Tensor &q,
                                   const Tensor &scales) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, a, q, scales);
    INFINICORE_GRAPH_OP_DISPATCH(output->device().getType(), output, a, q, scales);
}

void Fp8BlockwiseGemm::execute(Tensor output,
                               const Tensor &a,
                               const Tensor &q,
                               const Tensor &scales) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Fp8BlockwiseGemm, output, a, q, scales);
}

Tensor fp8_blockwise_gemm(const Tensor &a,
                          const Tensor &q,
                          const Tensor &scales) {
    const auto M = a->size(0);
    const auto N = q->size(0);
    auto output = Tensor::empty({M, N}, a->dtype(), a->device());
    Fp8BlockwiseGemm::execute(output, a, q, scales);
    return output;
}

void fp8_blockwise_gemm_(Tensor output,
                         const Tensor &a,
                         const Tensor &q,
                         const Tensor &scales) {
    Fp8BlockwiseGemm::execute(output, a, q, scales);
}

} // namespace infinicore::op
