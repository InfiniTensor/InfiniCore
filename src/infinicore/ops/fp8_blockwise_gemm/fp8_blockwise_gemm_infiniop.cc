#include "../../utils.hpp"
#include "../infiniop_impl.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/common/cache.hpp"
#include "infinicore/ops/fp8_blockwise_gemm.hpp"

#include <infiniop.h>

namespace infinicore::op::fp8_blockwise_gemm_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, Fp8BlockwiseGemm, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor output, a, q, scales;
};

void *plan(Tensor output, const Tensor &a, const Tensor &q, const Tensor &scales) {
    const size_t seed = hash_combine(output, a, q, scales);
    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, Fp8BlockwiseGemm,
        seed, output->desc(), a->desc(), q->desc(), scales->desc());
    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(output),
        graph::GraphTensor(a),
        graph::GraphTensor(q),
        graph::GraphTensor(scales)};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);
    INFINICORE_CHECK_ERROR(infiniopFp8BlockwiseGemm(
        planned->descriptor->desc,
        nullptr, 0,
        planned->output->data(),
        planned->a->data(),
        planned->q->data(),
        planned->scales->data(),
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(Fp8BlockwiseGemm, &plan, &run, &cleanup);

} // namespace infinicore::op::fp8_blockwise_gemm_impl::infiniop
