#include "infinicore/ops/grouped_gemm.hpp"

#include "../../utils.hpp"

namespace infinicore::op {
INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(GroupedGemm);

GroupedGemm::GroupedGemm(Tensor c,
                         const Tensor &a,
                         const Tensor &b,
                         const Tensor &group_sizes,
                         float alpha,
                         float beta,
                         const int32_t *group_sizes_host) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(c, a, b, group_sizes);
    INFINICORE_GRAPH_OP_DISPATCH(c->device().getType(), c, a, b, group_sizes, alpha, beta, group_sizes_host);
}

void GroupedGemm::execute(Tensor c,
                          const Tensor &a,
                          const Tensor &b,
                          const Tensor &group_sizes,
                          float alpha,
                          float beta,
                          const int32_t *group_sizes_host) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(GroupedGemm, c, a, b, group_sizes, alpha, beta, group_sizes_host);
}

Tensor grouped_gemm(const Tensor &a,
                    const Tensor &b,
                    const Tensor &group_sizes,
                    float alpha,
                    float beta,
                    const int32_t *group_sizes_host) {
    // a: [M_total, K], b: [num_groups, N, K] -> c: [M_total, N].
    Shape shape = a->shape();
    shape[shape.size() - 1] = b->size(1);
    auto c = Tensor::empty(shape, a->dtype(), a->device());
    grouped_gemm_(c, a, b, group_sizes, alpha, beta, group_sizes_host);
    return c;
}

void grouped_gemm_(Tensor c,
                   const Tensor &a,
                   const Tensor &b,
                   const Tensor &group_sizes,
                   float alpha,
                   float beta,
                   const int32_t *group_sizes_host) {
    GroupedGemm::execute(c, a, b, group_sizes, alpha, beta, group_sizes_host);
}

} // namespace infinicore::op
