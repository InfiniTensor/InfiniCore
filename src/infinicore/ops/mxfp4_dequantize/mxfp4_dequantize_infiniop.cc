#include "../../utils.hpp"
#include "../infiniop_impl.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/common/cache.hpp"
#include "infinicore/ops/mxfp4_dequantize.hpp"

#include <infiniop.h>

namespace infinicore::op::mxfp4_dequantize_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, Mxfp4Dequantize, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor output, packed, scales;
};

void *plan(Tensor output, const Tensor &packed, const Tensor &scales) {
    const size_t seed = hash_combine(output, packed, scales);
    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, Mxfp4Dequantize,
        seed, output->desc(), packed->desc(), scales->desc());
    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(output),
        graph::GraphTensor(packed),
        graph::GraphTensor(scales)};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);
    INFINICORE_CHECK_ERROR(infiniopMxfp4Dequantize(
        planned->descriptor->desc,
        nullptr, 0,
        planned->output->data(),
        planned->packed->data(),
        planned->scales->data(),
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(MXFP4Dequantize, &plan, &run, &cleanup);

} // namespace infinicore::op::mxfp4_dequantize_impl::infiniop
