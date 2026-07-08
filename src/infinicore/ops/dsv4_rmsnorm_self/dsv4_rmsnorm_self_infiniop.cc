#include "../infiniop_impl.hpp"
#include "infinicore/ops/dsv4_rmsnorm_self.hpp"
#include "infiniop/ops/dsv4_rmsnorm_self.h"

namespace infinicore::op::dsv4_rmsnorm_self_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, Dsv4RMSNormSelf, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, y, x;
};

void *plan(Tensor y, const Tensor &x, float epsilon) {
    size_t seed = hash_combine(y, x, epsilon);
    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(Descriptor, descriptor, Dsv4RMSNormSelf, seed, y->desc(), x->desc(), epsilon);
    INFINIOP_WORKSPACE_TENSOR(workspace, Dsv4RMSNormSelf, descriptor);
    return new PlannedMeta{descriptor, graph::GraphTensor(workspace), graph::GraphTensor(y), graph::GraphTensor(x)};
}

void run(void *planned_meta) {
    auto p = reinterpret_cast<PlannedMeta *>(planned_meta);
    INFINICORE_CHECK_ERROR(infiniopDsv4RMSNormSelf(p->descriptor->desc, p->workspace->data(), p->workspace->numel(), p->y->data(), p->x->data(), context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(Dsv4RMSNormSelf, &plan, &run, &cleanup);

} // namespace infinicore::op::dsv4_rmsnorm_self_impl::infiniop
