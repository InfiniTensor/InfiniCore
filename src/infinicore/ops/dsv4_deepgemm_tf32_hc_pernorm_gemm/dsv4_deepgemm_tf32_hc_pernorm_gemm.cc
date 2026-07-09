#include "infinicore/ops/dsv4_deepgemm_tf32_hc_pernorm_gemm.hpp"

#include "../../utils.hpp"
namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Dsv4DeepgemmTf32HcPernormGemm);

Dsv4DeepgemmTf32HcPernormGemm::Dsv4DeepgemmTf32HcPernormGemm(const Tensor &a, const Tensor &b, Tensor d, Tensor sqr_sum, int64_t num_splits) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(a, b, d, sqr_sum);
    INFINICORE_GRAPH_OP_DISPATCH(a->device().getType(), a, b, d, sqr_sum, num_splits);
}

void Dsv4DeepgemmTf32HcPernormGemm::execute(const Tensor &a, const Tensor &b, Tensor d, Tensor sqr_sum, int64_t num_splits) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Dsv4DeepgemmTf32HcPernormGemm, a, b, d, sqr_sum, num_splits);
}

void dsv4_deepgemm_tf32_hc_pernorm_gemm_(const Tensor &a, const Tensor &b, Tensor d, Tensor sqr_sum, int64_t num_splits) {
    Dsv4DeepgemmTf32HcPernormGemm::execute(a, b, d, sqr_sum, num_splits);
}

} // namespace infinicore::op
