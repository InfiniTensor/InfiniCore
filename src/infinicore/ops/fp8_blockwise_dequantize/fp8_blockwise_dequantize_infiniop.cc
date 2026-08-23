#include "../../utils.hpp"
#include "../infiniop_impl.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/common/cache.hpp"
#include "infinicore/ops/fp8_blockwise_dequantize.hpp"

#include <infiniop.h>

namespace infinicore::op::fp8_blockwise_dequantize_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, Fp8BlockwiseDequantize, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor output, q, scales;
};

void *plan(Tensor output, const Tensor &q, const Tensor &scales) {
    const size_t seed = hash_combine(output, q, scales);
    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, Fp8BlockwiseDequantize,
        seed, output->desc(), q->desc(), scales->desc());
    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(output),
        graph::GraphTensor(q),
        graph::GraphTensor(scales)};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);
    INFINICORE_CHECK_ERROR(infiniopFp8BlockwiseDequantize(
        planned->descriptor->desc,
        nullptr, 0,
        planned->output->data(),
        planned->q->data(),
        planned->scales->data(),
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(Fp8BlockwiseDequantize, &plan, &run, &cleanup);

} // namespace infinicore::op::fp8_blockwise_dequantize_impl::infiniop
